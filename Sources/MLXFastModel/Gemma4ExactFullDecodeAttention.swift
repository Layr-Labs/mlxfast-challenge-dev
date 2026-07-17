import MLX

/// Exact single-token full-attention decode specialized for Gemma 4 31B.
///
/// The stock D=512 SDPA fallback launches a BF16 GEMV for QK, a precise
/// softmax, and a second BF16 GEMV for PV.  One 1,024-thread threadgroup owns a
/// query head here and emulates those three stock kernels in order:
///
/// - each SIMDgroup uses the stock `gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4`
///   lane map and reduction tree for QK;
/// - the first `ceil(keyCount / 128)` SIMDgroups use the stock precise block
///   softmax lane map, `simd_max` / `simd_sum` trees, fast exponential, and
///   BF16 probability boundary; and
/// - groups of four SIMDgroups use the stock
///   `gemv_t_bfloat16_bm1_bn4_sm8_sn4_tm4_tn4` lane map and reduction tree
///   for PV.
///
/// Keeping the intermediate scores and probabilities in BF16 threadgroup
/// memory is intentional: they are observable arithmetic boundaries in the
/// reference fallback.  The kernel writes the token-major `[1, 1, 16384]`
/// layout consumed by the output projection, avoiding the reference
/// transpose/reshape materialization without changing values.
let gemma4ExactFullDecodeAttentionMaximumKeyCount = 2_048

func gemma4ExactFullDecodeAttentionShapeIsSupported(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float
) -> Bool {
    let keyCount = keys.ndim == 4 ? keys.dim(2) : 0
    return scale == 1.0
        && queries.dtype == .bfloat16
        && keys.dtype == .bfloat16
        && values.dtype == .bfloat16
        && queries.shape == [1, 32, 1, 512]
        && keys.ndim == 4
        && keys.dim(0) == 1
        && keys.dim(1) == 4
        && keys.dim(3) == 512
        && values.shape == keys.shape
        // For shorter rows MLX selects a different QK GEMV specialization.
        && keyCount >= 33
        // At most 16 virtual precise-softmax SIMDgroups remain live in the
        // 32 physical SIMDgroups while retaining their four FP32 reads.
        && keyCount <= gemma4ExactFullDecodeAttentionMaximumKeyCount
}

let gemma4ExactFullDecodeAttentionKernelSource = """
        constexpr uint kHeadDimension = 512;
        constexpr uint kQueryHeads = 32;
        constexpr uint kKVHeads = 4;
        constexpr uint kGQAFactor = 8;
        constexpr uint kThreads = 1024;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroups = 32;
        constexpr uint kSoftmaxReads = 4;
        constexpr uint kMaximumKeyCount = 2048;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint kv_head = query_head / kGQAFactor;
        const uint key_count = keys_shape[2];

        const device bfloat* query = queries
            + static_cast<int64_t>(query_head) * queries_strides[1];
        const device bfloat* key = keys
            + static_cast<int64_t>(kv_head) * keys_strides[1];
        const device bfloat* value = values
            + static_cast<int64_t>(kv_head) * values_strides[1];
        device bfloat* head_output = attention_output
            + query_head * kHeadDimension;

        // This buffer intentionally serves first as the stock BF16 QK output
        // and then, in place, as the stock BF16 precise-softmax output.
        threadgroup bfloat scores[kMaximumKeyCount];
        threadgroup float local_maximum[kSIMDSize];
        threadgroup float local_normalizer[kSIMDSize];

        // QK: one physical SIMDgroup exactly emulates one stock QK SIMDgroup.
        // The reference uses four SIMDgroups per 16-score threadgroup, but
        // there is no cross-SIMDgroup arithmetic in its BN=1 specialization.
        const uint score_row_blocks = (key_count + 3) / 4;
        for (uint row_block = simd_group;
             row_block < score_row_blocks;
             row_block += kSIMDGroups) {
            uint output_row = row_block * 4;
            // Reproduce GEMVKernel's tail shift.  It can overlap an earlier
            // SIMDgroup, but both writers compute identical BF16 values.
            output_row = output_row + 4 <= key_count
                ? output_row
                : key_count - 4;

            float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            uint column = simd_lane * 4;
            for (uint iteration = 0; iteration < 4; ++iteration) {
                float query_values[4];
                bfloat key_values[4];
                for (uint inner = 0; inner < 4; ++inner) {
                    query_values[inner] = static_cast<float>(
                        query[column + inner]);
                }
                for (uint row = 0; row < 4; ++row) {
                    const device bfloat* key_row = key
                        + static_cast<int64_t>(output_row + row)
                            * keys_strides[2];
                    for (uint inner = 0; inner < 4; ++inner) {
                        key_values[inner] = key_row[column + inner];
                    }
                    for (uint inner = 0; inner < 4; ++inner) {
                        result[row] += static_cast<float>(key_values[inner])
                            * query_values[inner];
                    }
                }
                column += 128;
            }
            for (ushort offset = 16; offset >= 1; offset >>= 1) {
                for (uint row = 0; row < 4; ++row) {
                    result[row] += simd_shuffle_down(result[row], offset);
                }
            }
            if (simd_lane == 0) {
                for (uint row = 0; row < 4; ++row) {
                    scores[output_row + row] = static_cast<bfloat>(result[row]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Precise block softmax.  Stock launches one SIMDgroup per 128 BF16
        // values, with four adjacent reads per lane.  key_count <= 2048 means
        // every virtual stock SIMDgroup maps one-to-one to a physical group,
        // allowing its four FP32 values to stay live across the exact global
        // maximum and normalizer barriers.
        const uint softmax_groups = (key_count + 127) / 128;
        const bool active_softmax_group = simd_group < softmax_groups;
        const uint read_offset =
            (simd_group * kSIMDSize + simd_lane) * kSoftmaxReads;
        float loaded[kSoftmaxReads];

        if (simd_group == 0) {
            local_maximum[simd_lane] =
                -metal::numeric_limits<float>::infinity();
            local_normalizer[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_softmax_group) {
            for (uint read = 0; read < kSoftmaxReads; ++read) {
                loaded[read] = read_offset + read < key_count
                    ? static_cast<float>(scores[read_offset + read])
                    : -metal::numeric_limits<float>::infinity();
            }
            float maximum = -metal::numeric_limits<float>::max();
            for (uint read = 0; read < kSoftmaxReads; ++read) {
                maximum = maximum < loaded[read] ? loaded[read] : maximum;
            }
            maximum = simd_max(maximum);
            if (simd_lane == 0) {
                local_maximum[simd_group] = maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float maximum = simd_max(local_maximum[simd_lane]);
            if (simd_lane == 0) {
                local_maximum[0] = maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float maximum = local_maximum[0];

        if (active_softmax_group) {
            float normalizer = 0.0f;
            for (uint read = 0; read < kSoftmaxReads; ++read) {
                const float exponential = fast::exp(loaded[read] - maximum);
                loaded[read] = exponential;
                normalizer += exponential;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane == 0) {
                local_normalizer[simd_group] = normalizer;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float normalizer = simd_sum(local_normalizer[simd_lane]);
            if (simd_lane == 0) {
                local_normalizer[0] = normalizer;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inverse_normalizer = 1.0f / local_normalizer[0];
        if (active_softmax_group) {
            for (uint read = 0; read < kSoftmaxReads; ++read) {
                if (read_offset + read < key_count) {
                    scores[read_offset + read] = static_cast<bfloat>(
                        loaded[read] * inverse_normalizer);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PV: four adjacent physical SIMDgroups exactly emulate one stock
        // 128-thread GEMV-T group.  All eight virtual groups run together to
        // cover the fixed 512 output columns.
        const uint virtual_group = simd_group / 4;
        const uint stock_simd_group = simd_group % 4;
        const uint thread_row = simd_lane / 4;
        const uint thread_column = simd_lane % 4;
        {
            const uint output_block = virtual_group;
            const uint output_column = output_block * 64
                + (stock_simd_group * 4 + thread_column) * 4;
            uint input_row = thread_row * 4;
            float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint full_iterations = key_count / 32;
            const uint leftover = key_count - full_iterations * 32;

            for (uint iteration = 0;
                 iteration < full_iterations;
                 ++iteration) {
                // The stock GEMV-T includes this barrier at the beginning of
                // every 32-row block.  The fused group is larger, but no
                // arithmetic crosses a virtual 128-thread group.
                threadgroup_barrier(mem_flags::mem_none);
                float probability[4];
                bfloat value_elements[4];
                for (uint row = 0; row < 4; ++row) {
                    probability[row] = static_cast<float>(
                        scores[input_row + row]);
                }
                for (uint row = 0; row < 4; ++row) {
                    const device bfloat* value_row = value
                        + static_cast<int64_t>(input_row + row)
                            * values_strides[2];
                    for (uint column = 0; column < 4; ++column) {
                        value_elements[column] =
                            value_row[output_column + column];
                    }
                    for (uint column = 0; column < 4; ++column) {
                        result[column] += probability[row]
                            * static_cast<float>(value_elements[column]);
                    }
                }
                input_row += 32;
            }

            if (leftover > 0) {
                for (uint row = 0;
                     row < 4 && input_row + row < key_count;
                     ++row) {
                    const float probability = static_cast<float>(
                        scores[input_row + row]);
                    const device bfloat* value_row = value
                        + static_cast<int64_t>(input_row + row)
                            * values_strides[2];
                    bfloat value_elements[4];
                    for (uint column = 0; column < 4; ++column) {
                        value_elements[column] =
                            value_row[output_column + column];
                    }
                    for (uint column = 0; column < 4; ++column) {
                        result[column] += probability
                            * static_cast<float>(value_elements[column]);
                    }
                }
            }

            for (uint column = 0; column < 4; ++column) {
                for (ushort offset = 16; offset >= 4; offset >>= 1) {
                    result[column] += simd_shuffle_down(
                        result[column], offset);
                }
            }
            if (thread_row == 0) {
                for (uint column = 0; column < 4; ++column) {
                    head_output[output_column + column] =
                        static_cast<bfloat>(result[column]);
                }
            }
        }
        """

private let gemma4ExactFullDecodeAttentionKernel = MLXFast.metalKernel(
    name: "gemma4_exact_full_decode_attention_d512_1024t_v1",
    inputNames: ["queries", "keys", "values"],
    outputNames: ["attention_output"],
    source: gemma4ExactFullDecodeAttentionKernelSource,
    header: """
        #include <metal_stdlib>
        using namespace metal;

        """,
    ensureRowContiguous: false
)

/// Returns token-major `[1, 1, 16384]` attention output.
func gemma4ExactFullDecodeAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(
        gemma4ExactFullDecodeAttentionShapeIsSupported(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale
        )
    )
    return gemma4ExactFullDecodeAttentionKernel(
        [queries, keys, values],
        grid: (1_024, 32, 1),
        threadGroup: (1_024, 1, 1),
        outputShapes: [[1, 1, 32 * 512]],
        outputDTypes: [.bfloat16]
    )[0]
}
