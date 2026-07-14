import MLX

/// Exact-shape full-attention prefill experiment. A threadgroup owns eight
/// query rows and retains its complete BF16 score/probability stripe on chip.
private let gemma4StripedFullPrefill512Kernel = MLXFast.metalKernel(
    name: "gemma4_striped_full_prefill_8x512x512_mpp_v1",
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: """
        constexpr uint kLength = 512;
        constexpr uint kHeadDim = 512;
        constexpr uint kRows = 8;
        constexpr uint kThreads = 128;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroups = 4;
        constexpr uint kReads = 4;

        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint qhead = threadgroup_position_in_grid.y;
        const uint qstart = threadgroup_position_in_grid.z * kRows;
        const uint kvhead = qhead / 8;
        threadgroup bfloat scores[kRows * kLength];

        constexpr auto qkd = mpp::tensor_ops::matmul2d_descriptor(
            8, 32, 512, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<qkd, metal::execution_simdgroup> qk;
        device bfloat* qbase = const_cast<device bfloat*>(queries)
            + static_cast<int64_t>(qhead) * queries_strides[1]
            + static_cast<int64_t>(qstart) * queries_strides[2];
        for (uint block = sg; block < 16; block += kSIMDGroups) {
            const uint kstart = block * 32;
            device bfloat* kbase = const_cast<device bfloat*>(keys)
                + static_cast<int64_t>(kvhead) * keys_strides[1]
                + static_cast<int64_t>(kstart) * keys_strides[2];
            auto qt = metal::tensor(qbase, metal::dextents<int, 2>{512, 8},
                metal::array<int64_t, 2>{queries_strides[3], queries_strides[2]});
            auto kt = metal::tensor(kbase, metal::dextents<int, 2>{512, 32},
                metal::array<int64_t, 2>{keys_strides[3], keys_strides[2]});
            auto st = metal::tensor(scores + kstart,
                metal::dextents<int, 2>{32, 8}, metal::array<int, 2>{1, 512});
            qk.run(qt, kt, st);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < kRows * kLength; i += kThreads) {
            const uint row = i / kLength;
            const uint key = i - row * kLength;
            if (key > qstart + row)
                scores[i] = metal::numeric_limits<bfloat>::lowest();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Preserve the promoted staged kernel's complete-row precise topology.
        for (uint wave = 0; wave < kRows; wave += kSIMDGroups) {
            const uint row = wave + sg;
            const uint ro = row * kLength;
            float loaded[kSIMDGroups][kReads];
            float maxima[kSIMDGroups];
            for (uint vg = 0; vg < kSIMDGroups; ++vg) {
                const uint off = vg * kSIMDSize * kReads + lane * kReads;
                float m = metal::numeric_limits<float>::lowest();
                for (uint r = 0; r < kReads; ++r) {
                    const float x = static_cast<float>(scores[ro + off + r]);
                    loaded[vg][r] = x;
                    m = m < x ? x : m;
                }
                maxima[vg] = simd_max(m);
            }
            float m = lane < kSIMDGroups ? maxima[lane]
                : metal::numeric_limits<float>::lowest();
            m = simd_max(m);
            float sums[kSIMDGroups];
            for (uint vg = 0; vg < kSIMDGroups; ++vg) {
                float sum = 0.0f;
                for (uint r = 0; r < kReads; ++r) {
                    const float x = fast::exp(loaded[vg][r] - m);
                    loaded[vg][r] = x;
                    sum += x;
                }
                sums[vg] = simd_sum(sum);
            }
            float inv = lane < kSIMDGroups ? sums[lane] : 0.0f;
            inv = 1.0f / simd_sum(inv);
            for (uint vg = 0; vg < kSIMDGroups; ++vg) {
                const uint off = vg * kSIMDSize * kReads + lane * kReads;
                for (uint r = 0; r < kReads; ++r)
                    scores[ro + off + r] = static_cast<bfloat>(loaded[vg][r] * inv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto pvd = mpp::tensor_ops::matmul2d_descriptor(
            8, 32, 512, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<pvd, metal::execution_simdgroup> pv;
        for (uint block = sg; block < 16; block += kSIMDGroups) {
            const uint col = block * 32;
            device bfloat* vbase = const_cast<device bfloat*>(values)
                + static_cast<int64_t>(kvhead) * values_strides[1]
                + static_cast<int64_t>(col) * values_strides[3];
            auto pt = metal::tensor(scores, metal::dextents<int, 2>{512, 8},
                metal::array<int, 2>{1, 512});
            auto vt = metal::tensor(vbase, metal::dextents<int, 2>{32, 512},
                metal::array<int64_t, 2>{values_strides[3], values_strides[2]});
            device bfloat* obase = output
                + (qhead * kLength + qstart) * kHeadDim + col;
            auto ot = metal::tensor(obase, metal::dextents<int, 2>{32, 8},
                metal::array<int, 2>{1, 512});
            pv.run(pt, vt, ot);
        }
        """,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

func gemma4StripedFullPrefill512(
    queries: MLXArray, keys: MLXArray, values: MLXArray
) -> MLXArray {
    precondition(queries.dtype == .bfloat16 && keys.dtype == .bfloat16
        && values.dtype == .bfloat16)
    precondition(queries.shape == [1, 32, 512, 512])
    precondition(keys.shape == [1, 4, 512, 512])
    precondition(values.shape == [1, 4, 512, 512])
    return gemma4StripedFullPrefill512Kernel(
        [queries, keys, values], grid: (128, 32, 64),
        threadGroup: (128, 1, 1), outputShapes: [[1, 32, 512, 512]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Test/verification-only stock reference for the exact production contract.
func gemma4StripedFullPrefill512Reference(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: 1, mask: mask)
}

func gemma4CanUseStripedFullPrefill(
    batch: Int, length: Int, offset: Int, isSliding: Bool, useKEqV: Bool,
    queries: MLXArray, keys: MLXArray, values: MLXArray
) -> Bool {
    batch == 1 && length == 512 && offset == 0 && !isSliding && useKEqV
        && queries.dtype == .bfloat16 && keys.dtype == .bfloat16
        && values.dtype == .bfloat16
        && queries.shape == [1, 32, 512, 512]
        && keys.shape == [1, 4, 512, 512]
        && values.shape == [1, 4, 512, 512]
}
