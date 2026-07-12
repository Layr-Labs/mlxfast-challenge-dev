import MLX

/// Exact Gemma 4 full-attention decode suffix for D=512.
///
/// QK remains the stock MLX BF16 matmul. The custom kernel starts at the
/// precise-softmax boundary, stages probabilities as BF16, and reproduces the
/// pinned GEMVT BM1/BN4/SM8/SN4/TM4/TN4 reduction topology for PV. One
/// 1,024-thread threadgroup owns a query head; its 32 SIMDgroups are split into
/// the eight independent four-SIMDgroup clusters used by stock GEMVT.
private let gemma4FullDecodeSoftmaxPV = MLXFast.metalKernel(
    name: "gemma4_full_decode_precise_softmax_gemvt_512_v1",
    inputNames: ["scores", "values"],
    outputNames: ["output"],
    source: """
        constexpr uint kHeadDim = 512;
        constexpr uint kReads = 4;
        constexpr uint kSIMDSize = 32;
        constexpr uint kMaxSequenceLength = 2048;
        constexpr uint kQueryHeadsPerKVHead = 8;

        const uint query_head = threadgroup_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.y * kSIMDSize
            + thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint n = static_cast<uint>(values_shape[2]);
        const uint softmax_simds =
            ((n + kReads - 1) / kReads + kSIMDSize - 1) / kSIMDSize;
        const bool softmax_active = simd_group < softmax_simds;

        // Clone softmax_single_row<bfloat, float, 4>. SG0 initializes all 32
        // slots, so SIMDgroups beyond the stock dynamic threadgroup size stay
        // inert while every physical thread can still reach each barrier.
        threadgroup float local_max[kSIMDSize];
        threadgroup float local_normalizer[kSIMDSize];
        threadgroup bfloat probabilities[kMaxSequenceLength];

        float loaded[kReads];
        const float negative_infinity =
            -metal::numeric_limits<float>::infinity();
        const float finite_minimum =
            -metal::numeric_limits<float>::max();
        for (uint index = 0; index < kReads; ++index) {
            loaded[index] = negative_infinity;
        }
        if (softmax_active) {
            const uint base = lid * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint position = base + index;
                loaded[index] = position < n
                    ? static_cast<float>(scores[query_head * n + position])
                    : negative_infinity;
            }
        }

        if (simd_group == 0) {
            local_max[simd_lane] = negative_infinity;
            local_normalizer[simd_lane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float max_value = finite_minimum;
        if (softmax_active) {
            for (uint index = 0; index < kReads; ++index) {
                max_value = max_value < loaded[index]
                    ? loaded[index]
                    : max_value;
            }
            max_value = simd_max(max_value);
            if (simd_lane == 0) {
                local_max[simd_group] = max_value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            max_value = simd_max(local_max[simd_lane]);
            if (simd_lane == 0) {
                local_max[0] = max_value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_value = local_max[0];

        float normalizer = 0;
        if (softmax_active) {
            for (uint index = 0; index < kReads; ++index) {
                const float exponential = fast::exp(loaded[index] - max_value);
                loaded[index] = exponential;
                normalizer += exponential;
            }
            normalizer = simd_sum(normalizer);
            if (simd_lane == 0) {
                local_normalizer[simd_group] = normalizer;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            normalizer = simd_sum(local_normalizer[simd_lane]);
            if (simd_lane == 0) {
                local_normalizer[0] = normalizer;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer = 1 / local_normalizer[0];

        if (softmax_active) {
            const uint base = lid * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint position = base + index;
                if (position < n) {
                    probabilities[position] = static_cast<bfloat>(
                        loaded[index] * normalizer);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Clone GEMVTKernel<bfloat,1,4,8,4,4,4,false>. Each group of four
        // SIMDgroups is one original GEMVT threadgroup/output tile. Keeping the
        // eight clusters independent preserves every per-output reduction.
        const uint cluster = simd_group / 4;
        const uint local_simd_group = simd_group - cluster * 4;
        const uint thread_row = simd_lane / 4;
        const uint thread_column = simd_lane - thread_row * 4;
        uint input_row = thread_row * 4;
        const uint output_column = cluster * 64
            + (local_simd_group * 4 + thread_column) * 4;

        const uint kv_head = query_head / kQueryHeadsPerKVHead;
        const int64_t head_stride = values_strides[1];
        const int64_t sequence_stride = values_strides[2];
        const int64_t dimension_stride = values_strides[3];
        const device bfloat* value_head = values + kv_head * head_stride;

        float result[4] = {0};
        bfloat matrix_values[4];
        float probability_values[4];
        const uint iterations = n / 32;
        const uint last_iteration = iterations * 32;
        const uint leftover = n - last_iteration;

        for (uint iteration = 0; iteration < iterations; ++iteration) {
            // This barrier is present in the pinned GEMVT kernel's main loop.
            threadgroup_barrier(mem_flags::mem_none);
            for (uint row = 0; row < 4; ++row) {
                probability_values[row] = static_cast<float>(
                    probabilities[input_row + row]);
            }
            for (uint row = 0; row < 4; ++row) {
                const float coefficient = probability_values[row];
                const device bfloat* matrix_row = value_head
                    + static_cast<int64_t>(input_row + row) * sequence_stride
                    + static_cast<int64_t>(output_column) * dimension_stride;
                for (uint column = 0; column < 4; ++column) {
                    matrix_values[column] = matrix_row[
                        static_cast<int64_t>(column) * dimension_stride];
                }
                for (uint column = 0; column < 4; ++column) {
                    result[column] += coefficient * matrix_values[column];
                }
            }
            input_row += 32;
        }

        if (leftover > 0) {
            for (uint row = 0; row < 4 && input_row + row < n; ++row) {
                probability_values[row] = static_cast<float>(
                    probabilities[input_row + row]);
                const device bfloat* matrix_row = value_head
                    + static_cast<int64_t>(input_row + row) * sequence_stride
                    + static_cast<int64_t>(output_column) * dimension_stride;
                for (uint column = 0; column < 4; ++column) {
                    matrix_values[column] = matrix_row[
                        static_cast<int64_t>(column) * dimension_stride];
                }
                for (uint column = 0; column < 4; ++column) {
                    result[column] += probability_values[row]
                        * matrix_values[column];
                }
            }
        }

        for (uint column = 0; column < 4; ++column) {
            result[column] += simd_shuffle_down(result[column], 16);
            result[column] += simd_shuffle_down(result[column], 8);
            result[column] += simd_shuffle_down(result[column], 4);
        }

        if (thread_row == 0) {
            device bfloat* output_row = output + query_head * kHeadDim;
            for (uint column = 0; column < 4; ++column) {
                output_row[output_column + column] = static_cast<bfloat>(
                    result[column]);
            }
        }
        """,
    header: "using namespace metal;",
    ensureRowContiguous: false
)

struct Gemma4FullDecodeAttention: @unchecked Sendable {
    private let verifyBits: Bool

    init(verifyBits: Bool) {
        self.verifyBits = verifyBits
    }

    func supports(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) -> Bool {
        guard queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              queries.shape == [1, 32, 1, 512],
              keys.ndim == 4,
              values.ndim == 4,
              keys.shape == values.shape,
              keys.dim(0) == 1,
              keys.dim(1) == 4,
              keys.dim(3) == 512,
              (1...2048).contains(keys.dim(2))
        else {
            return false
        }

        return true
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) -> MLXArray {
        precondition(supports(queries: queries, keys: keys, values: values))

        // This is exactly the fallback SDPA QK layout. Scale is exactly one,
        // so elide only the otherwise bit-preserving BF16 multiply dispatch.
        let groupedQueries = queries.reshaped([1, 4, 8, 1, 512])
        let groupedKeys = expandedDimensions(keys, axis: 2)
        let scores = matmul(
            groupedQueries,
            groupedKeys.swappedAxes(-1, -2)
        )

        let candidate = gemma4FullDecodeSoftmaxPV(
            [scores, values],
            grid: (32, 32 * 32, 1),
            threadGroup: (32, 32, 1),
            outputShapes: [[1, 32, 1, 512]],
            outputDTypes: [.bfloat16]
        )[0]

        if verifyBits {
            let reference = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: 1.0,
                mask: .none
            )
            let matches = arrayEqual(
                candidate.view(dtype: .uint16),
                reference.view(dtype: .uint16)
            )
            eval(matches)
            precondition(
                matches.item(Bool.self),
                "fused full D512 decode attention differs from stock SDPA"
            )
        }

        return candidate
    }
}
