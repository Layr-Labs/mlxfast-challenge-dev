import MLX

/// Exact-shape MPP prototype for Gemma 4 full-attention prefill.
///
/// This is the D=512, four-KV-head counterpart of the verified sliding
/// kernel. Q and K already contain the full layer's partial-RoPE result, while
/// V contains the distinct no-weight RMS result derived from the shared K/V
/// projection. The kernel consumes those prepared arrays without assuming
/// that the postprocessed K and V bytes are identical.
private let gemma4StagedFullPrefill512Kernel = MLXFast.metalKernel(
    name: "gemma4_staged_full_prefill_16x512x512_mpp_v1",
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: """
        constexpr uint kLength = 512;
        constexpr uint kHeadDim = 512;
        constexpr uint kQueryRows = 16;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 4;
        constexpr uint kGQAFactor = 8;
        constexpr uint kThreads = 128;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroups = 4;
        constexpr uint kSoftmaxReads = 4;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint query_block = threadgroup_position_in_grid.z;
        const uint query_start = query_block * kQueryRows;
        const uint kv_head = query_head / kGQAFactor;

        threadgroup bfloat scores[kQueryRows * kLength];

        // QK: four SIMDgroups cover independent 32-key tiles. Inputs remain
        // in device memory so MPP and the Apple GPU cache own source reuse;
        // only the complete 16x512 BF16 score tile enters threadgroup memory.
        constexpr auto qk_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            qk_descriptor, metal::execution_simdgroup> qk;
        device bfloat* mutable_queries = const_cast<device bfloat*>(queries)
            + static_cast<int64_t>(query_head) * queries_strides[1]
            + static_cast<int64_t>(query_start) * queries_strides[2];
        for (uint key_block = simd_group;
             key_block < kLength / 32;
             key_block += kSIMDGroups) {
            const uint key_start = key_block * 32;
            device bfloat* mutable_keys = const_cast<device bfloat*>(keys)
                + static_cast<int64_t>(kv_head) * keys_strides[1]
                + static_cast<int64_t>(key_start) * keys_strides[2];
            auto q_tensor = metal::tensor(
                mutable_queries,
                metal::dextents<int, 2>{512, 16},
                metal::array<int64_t, 2>{
                    queries_strides[3], queries_strides[2]});
            auto k_tensor = metal::tensor(
                mutable_keys,
                metal::dextents<int, 2>{512, 32},
                metal::array<int64_t, 2>{keys_strides[3], keys_strides[2]});
            auto score_tensor = metal::tensor(
                scores + key_start,
                metal::dextents<int, 2>{32, 16},
                metal::array<int, 2>{1, 512});
            qk.run(q_tensor, k_tensor, score_tensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Offset-zero full attention has the same causal visibility as stock:
        // no sliding lower bound, and no key may be newer than its query.
        for (uint index = thread_index;
             index < kQueryRows * kLength;
             index += kThreads) {
            const uint row = index / kLength;
            const uint key_position = index - row * kLength;
            if (key_position > query_start + row) {
                scores[index] = metal::numeric_limits<bfloat>::lowest();
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exact MLX precise block-softmax topology. A physical SIMDgroup owns
        // one row and emulates the stock kernel's four virtual SIMDgroups.
        // Lane/read mapping, float accumulator order, reduction intrinsics,
        // fast::exp, reciprocal, and BF16 cast points are unchanged.
        for (uint row_group = 0; row_group < kQueryRows;
             row_group += kSIMDGroups) {
            const uint row = row_group + simd_group;
            const uint row_offset = row * kLength;
            float loaded[kSIMDGroups][kSoftmaxReads];
            float virtual_maxima[kSIMDGroups];

            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                const uint read_offset =
                    virtual_group * kSIMDSize * kSoftmaxReads
                    + simd_lane * kSoftmaxReads;
                float maximum = metal::numeric_limits<float>::lowest();
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    const float value = static_cast<float>(
                        scores[row_offset + read_offset + read]);
                    loaded[virtual_group][read] = value;
                    maximum = maximum < value ? value : maximum;
                }
                virtual_maxima[virtual_group] = simd_max(maximum);
            }

            float maximum = simd_lane < kSIMDGroups
                ? virtual_maxima[simd_lane]
                : metal::numeric_limits<float>::lowest();
            maximum = simd_max(maximum);

            float virtual_normalizers[kSIMDGroups];
            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                float normalizer = 0.0f;
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    const float exponential = fast::exp(
                        loaded[virtual_group][read] - maximum);
                    loaded[virtual_group][read] = exponential;
                    normalizer += exponential;
                }
                virtual_normalizers[virtual_group] = simd_sum(normalizer);
            }
            float normalizer = simd_lane < kSIMDGroups
                ? virtual_normalizers[simd_lane]
                : 0.0f;
            normalizer = 1.0f / simd_sum(normalizer);

            for (uint virtual_group = 0;
                 virtual_group < kSIMDGroups;
                 ++virtual_group) {
                const uint read_offset =
                    virtual_group * kSIMDSize * kSoftmaxReads
                    + simd_lane * kSoftmaxReads;
                for (uint read = 0; read < kSoftmaxReads; ++read) {
                    scores[row_offset + read_offset + read] =
                        static_cast<bfloat>(
                            loaded[virtual_group][read] * normalizer);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PV: Dv=512 means four SIMDgroups cover sixteen 32-column tiles in
        // four waves. K/V share a source projection in Gemma 4 full layers,
        // but the separately prepared V pointer and strides are used here.
        constexpr auto pv_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_descriptor, metal::execution_simdgroup> pv;
        for (uint value_block = simd_group;
             value_block < kHeadDim / 32;
             value_block += kSIMDGroups) {
            const uint value_start = value_block * 32;
            device bfloat* mutable_values = const_cast<device bfloat*>(values)
                + static_cast<int64_t>(kv_head) * values_strides[1]
                + static_cast<int64_t>(value_start) * values_strides[3];
            auto probability_tensor = metal::tensor(
                scores,
                metal::dextents<int, 2>{512, 16},
                metal::array<int, 2>{1, 512});
            auto value_tensor = metal::tensor(
                mutable_values,
                metal::dextents<int, 2>{32, 512},
                metal::array<int64_t, 2>{values_strides[3], values_strides[2]});
            device bfloat* output_tile = output
                + (query_head * kLength + query_start) * kHeadDim
                + value_start;
            auto output_tensor = metal::tensor(
                output_tile,
                metal::dextents<int, 2>{32, 16},
                metal::array<int, 2>{1, 512});
            pv.run(probability_tensor, value_tensor, output_tensor);
        }
        """,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

func gemma4StagedFullPrefill512(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray
) -> MLXArray {
    precondition(queries.dtype == .bfloat16)
    precondition(keys.dtype == .bfloat16)
    precondition(values.dtype == .bfloat16)
    precondition(queries.shape == [1, 32, 512, 512])
    precondition(keys.shape == [1, 4, 512, 512])
    precondition(values.shape == [1, 4, 512, 512])

    return gemma4StagedFullPrefill512Kernel(
        [queries, keys, values],
        grid: (128, 32, 32),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 32, 512, 512]],
        outputDTypes: [.bfloat16]
    )[0]
}
