import Foundation
import MLX

/// Rollback switch for the staged prefill causal QK tile skip (P1).
///
/// Default ON. When enabled, the staged sliding prefill kernel never computes
/// or loads QK score tiles that lie entirely above the causal diagonal (every
/// element masked). Skipped tiles are bit-exact by construction: the causal
/// mask-fill pass overwrites every element of a fully-masked tile with
/// `bfloat::lowest()` regardless, so the threadgroup score contents entering
/// softmax are identical with the skip on or off. Set
/// `DARKBLOOM_STAGED_PREFILL_CAUSAL_TILE_SKIP=0` to restore the full-grid QK
/// computation.
let gemma4StagedPrefillCausalTileSkipEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_PREFILL_CAUSAL_TILE_SKIP"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Rollback switch for token-major staged prefill output emission (P3).
///
/// Default ON. When enabled, the staged sliding prefill kernel writes its
/// attention output directly in the merged token-major layout
/// `[1, 512, 32*256]` that `o_proj` consumes, and the engine skips the
/// `transposed(0, 2, 1, 3).reshaped(B, L, -1)` materialization (~8.4 MB per
/// sliding layer). The written values are bit-identical; only the memory
/// layout of the intermediate changes. Set
/// `DARKBLOOM_STAGED_PREFILL_TOKEN_MAJOR_OUTPUT=0` to restore head-major
/// `[1, 32, 512, 256]` emission plus the downstream transpose-reshape.
let gemma4StagedPrefillTokenMajorOutputEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_PREFILL_TOKEN_MAJOR_OUTPUT"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// CPU-side mirror of the staged sliding prefill kernel's tile geometry.
///
/// The staged kernel runs the exact ranked shape: B=1, L=512, 32 query heads,
/// 16 KV heads, head dim 256, sliding window 1024, cache offset 0. Tests use
/// these mirrors to prove, without touching the MLX runtime, that
/// - the P1 skip bound skips exactly the fully-masked QK tile set, and
/// - the P3 token-major tile addressing is the exact transpose bijection of
///   the head-major addressing (same logical element per tile coordinate).
///
/// Any change to the Metal-side constants or addressing below must be
/// reflected here; the arithmetic is duplicated intentionally so the tests
/// stay pure CPU.
enum Gemma4StagedPrefillGeometry {
    /// Sequence length of the staged prefill (queries and keys).
    static let length = 512
    /// Per-head attention dimension.
    static let headDim = 256
    /// Query rows owned by one threadgroup (M tile).
    static let queryRows = 16
    /// Number of query heads.
    static let queryHeads = 32
    /// Key columns covered by one QK tensor-op tile (N tile).
    static let keyTileColumns = 32
    /// Output columns covered by one PV tensor-op tile (N tile).
    static let valueTileColumns = 32
    /// Sliding window of the ranked sliding layers.
    static let windowSize = 1024
    /// Cache offset of the staged prefill dispatch.
    static let cacheOffset = 0

    /// Total 32-key QK tiles per query block.
    static var keyBlocksTotal: Int { length / keyTileColumns }
    /// Total 16-query blocks per head.
    static var queryBlocksTotal: Int { length / queryRows }

    /// Additive-mask keep predicate for the ranked staged shape.
    ///
    /// Mirrors MLXLMCommon `createCausalMask(n:offset:windowSize:)` at
    /// offset 0: a (query, key) score survives iff `key <= query` (causal,
    /// `linds .>= rinds`) AND `query - key < windowSize` (sliding,
    /// `linds .< rinds + windowSize`). For length 512 and window 1024 the
    /// sliding bound never bites (`query - key <= 511 < 1024`), which is why
    /// the kernel's mask fill is pure causal; the tests verify that
    /// equivalence exhaustively instead of assuming it.
    static func maskKeeps(queryPosition: Int, keyPosition: Int) -> Bool {
        keyPosition <= queryPosition + cacheOffset
            && (queryPosition + cacheOffset) - keyPosition < windowSize
    }

    /// P1 bound: number of leading 32-key QK tiles computed for the 16-query
    /// tile `queryBlock`; every tile index at or beyond this bound is skipped.
    ///
    /// Derivation: the tile starting at `key_start = 32*keyBlock` is fully
    /// masked iff its smallest key exceeds the tile's largest query position,
    /// i.e. `32*keyBlock > 16*queryBlock + 15`, equivalently
    /// `32*keyBlock >= 16*(queryBlock + 1)`, so the first fully-masked tile
    /// index is `ceil((queryBlock + 1) / 2) == (queryBlock + 2) / 2` in
    /// integer division. Must match the Metal-side `key_block_limit`.
    static func computedKeyBlockCount(queryBlock: Int) -> Int {
        (queryBlock + 2) / 2
    }

    /// Token-major linear offset of logical element (token, head, dim) in the
    /// P3 output layout `[1, length, queryHeads * headDim]`.
    static func tokenMajorOffset(token: Int, head: Int, dim: Int) -> Int {
        (token * queryHeads + head) * headDim + dim
    }

    /// Head-major linear offset of logical element (head, token, dim) in the
    /// legacy output layout `[1, queryHeads, length, headDim]`.
    static func headMajorOffset(head: Int, token: Int, dim: Int) -> Int {
        (head * length + token) * headDim + dim
    }

    /// Mirrors the kernel's PV output tile addressing: linear offset written
    /// by tile coordinate (queryHead, queryStart, valueStart) at tile-local
    /// (row, lane), where `row` indexes the 16 query rows and `lane` the 32
    /// output columns. Must match the Metal-side `output_tile` base and the
    /// output tensor strides (`{1, kOutputRowStride}`).
    static func outputTileElementOffset(
        tokenMajor: Bool,
        queryHead: Int,
        queryStart: Int,
        valueStart: Int,
        row: Int,
        lane: Int
    ) -> Int {
        if tokenMajor {
            let base = queryStart * (queryHeads * headDim)
                + queryHead * headDim
                + valueStart
            return base + row * (queryHeads * headDim) + lane
        }
        let base = (queryHead * length + queryStart) * headDim + valueStart
        return base + row * headDim + lane
    }
}

/// Exact-shape MPP kernel for Gemma 4 sliding prefill.
///
/// One 128-thread threadgroup owns one `(query head, 16-query)` tile. QK
/// writes the causally-live prefix of a 16x512 BF16 score tile to threadgroup
/// memory (fully-masked 32-key tiles are skipped and mask-filled instead --
/// P1), the same four-SIMD/4-read reduction topology as MLX's precise block
/// softmax rewrites that tile in place, and PV consumes it without
/// materializing either scores or probabilities in device memory. Output is
/// emitted token-major `[1, 512, 32*256]` (P3), directly in the merged layout
/// `o_proj` consumes; both P1 and P3 fall back to the v1 behavior via their
/// DARKBLOOM env switches above.
private let gemma4StagedSlidingPrefill512Kernel = MLXFast.metalKernel(
    name: "gemma4_staged_sliding_prefill_16x512x256_mpp_v2"
        + "_skip\(gemma4StagedPrefillCausalTileSkipEnabled ? 1 : 0)"
        + "_tokmaj\(gemma4StagedPrefillTokenMajorOutputEnabled ? 1 : 0)",
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: """
        constexpr uint kLength = 512;
        constexpr uint kHeadDim = 256;
        constexpr uint kQueryRows = 16;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 16;
        constexpr uint kGQAFactor = 2;
        constexpr uint kThreads = 128;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroups = 4;
        constexpr uint kSoftmaxReads = 4;
        constexpr bool kCausalTileSkip =
            \(gemma4StagedPrefillCausalTileSkipEnabled);
        constexpr bool kTokenMajorOutput =
            \(gemma4StagedPrefillTokenMajorOutputEnabled);

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint query_block = threadgroup_position_in_grid.z;
        const uint query_start = query_block * kQueryRows;
        const uint kv_head = query_head / kGQAFactor;

        threadgroup bfloat scores[kQueryRows * kLength];

        // Four SIMDgroups cooperatively cover 32-key column tiles. MPP keeps
        // the 256-wide reduction inside the tensor operation, and writes only
        // BF16 scores to on-chip threadgroup storage. Gemma 4's attention
        // scale is exactly 1.0, so Q can come directly from device memory;
        // Apple's MPP guidance recommends relying on cache instead of staging
        // GEMM sources through threadgroup memory.
        //
        // P1 causal tile skip: the 32-key tile at key_start = 32*key_block is
        // fully masked iff its smallest key exceeds this query tile's largest
        // position, i.e. 32*key_block > query_start + kQueryRows - 1, so the
        // first fully-masked tile index is (query_block + 2) / 2 (integer
        // ceil((query_block + 1) / 2)). Tiles at or beyond that bound are
        // never computed or loaded; the causal fill below overwrites every
        // one of their score elements with bfloat lowest anyway, so the
        // threadgroup contents entering softmax are bit-identical. Partially
        // masked tiles keep the unchanged mask arithmetic.
        const uint key_block_limit = kCausalTileSkip
            ? (query_block + 2) / 2
            : kLength / 32;
        constexpr auto qk_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 256, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            qk_descriptor, metal::execution_simdgroup> qk;
        device bfloat* mutable_queries = const_cast<device bfloat*>(queries)
            + static_cast<int64_t>(query_head) * queries_strides[1]
            + static_cast<int64_t>(query_start) * queries_strides[2];
        for (uint key_block = simd_group;
             key_block < key_block_limit;
             key_block += kSIMDGroups) {
            const uint key_start = key_block * 32;
            device bfloat* mutable_keys = const_cast<device bfloat*>(keys)
                + static_cast<int64_t>(kv_head) * keys_strides[1]
                + static_cast<int64_t>(key_start) * keys_strides[2];
            auto q_tensor = metal::tensor(
                mutable_queries,
                metal::dextents<int, 2>{256, 16},
                metal::array<int64_t, 2>{
                    queries_strides[3], queries_strides[2]});
            auto k_tensor = metal::tensor(
                mutable_keys,
                metal::dextents<int, 2>{256, 32},
                metal::array<int64_t, 2>{keys_strides[3], keys_strides[2]});
            auto score_tensor = metal::tensor(
                scores + key_start,
                metal::dextents<int, 2>{32, 16},
                metal::array<int, 2>{1, 512});
            qk.run(q_tensor, k_tensor, score_tensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Offset-zero, L=512 sliding attention (window=1024) is causal only.
        // Match the stock boolean-mask fill value in the BF16 score dtype.
        // This fill covers every element of every P1-skipped tile: a skipped
        // tile satisfies key_start > query_start + row for all of its rows.
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

        // Reproduce MLX precise block_softmax for axis size 512 exactly while
        // running four rows concurrently. One physical SIMDgroup owns a row
        // and emulates stock's four virtual SIMDgroups. Each virtual group
        // retains the identical lane/read mapping and simd_max/simd_sum; the
        // second reduction places its four partials in lanes 0...3 before the
        // same SIMD intrinsic. This removes all softmax threadgroup barriers
        // without changing reduction or cast order.
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

        // Consume the on-chip BF16 probabilities directly. Four SIMDgroups
        // cover 32 output columns each, in two waves, and write only the final
        // 16x256 attention output to device memory.
        //
        // P3 token-major emission: element (row, lane) of the tile is the
        // output for token (query_start + row), head query_head, dimension
        // (value_start + lane). Token-major places it at
        //   (query_start + row) * kQHeads * kHeadDim
        //     + query_head * kHeadDim + value_start + lane
        // (output shape [1, 512, 32*256], row stride 8192); head-major keeps
        // the v1 addressing (output shape [1, 32, 512, 256], row stride 256).
        // Identical computed values, different destination addresses only.
        constexpr int kOutputRowStride = kTokenMajorOutput
            ? int(kQHeads * kHeadDim)
            : int(kHeadDim);
        constexpr auto pv_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_descriptor, metal::execution_simdgroup> pv;
        const uint output_tile_base = kTokenMajorOutput
            ? query_start * (kQHeads * kHeadDim) + query_head * kHeadDim
            : (query_head * kLength + query_start) * kHeadDim;
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
                + output_tile_base
                + value_start;
            auto output_tensor = metal::tensor(
                output_tile,
                metal::dextents<int, 2>{32, 16},
                metal::array<int, 2>{1, kOutputRowStride});
            pv.run(probability_tensor, value_tensor, output_tensor);
        }
        """,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

/// Runs the staged sliding prefill attention kernel on the ranked shape.
///
/// Returns `[1, 512, 32*256]` (token-major, already the merged layout
/// `o_proj` consumes) when `gemma4StagedPrefillTokenMajorOutputEnabled`,
/// otherwise the legacy head-major `[1, 32, 512, 256]` that the caller merges
/// via `transposed(0, 2, 1, 3).reshaped(B, L, -1)`.
func gemma4StagedSlidingPrefill512(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray
) -> MLXArray {
    precondition(queries.dtype == .bfloat16)
    precondition(keys.dtype == .bfloat16)
    precondition(values.dtype == .bfloat16)
    precondition(queries.shape == [1, 32, 512, 256])
    precondition(keys.shape == [1, 16, 512, 256])
    precondition(values.shape == [1, 16, 512, 256])

    let outputShape: [Int] = gemma4StagedPrefillTokenMajorOutputEnabled
        ? [1, 512, 32 * 256]
        : [1, 32, 512, 256]
    return gemma4StagedSlidingPrefill512Kernel(
        [queries, keys, values],
        grid: (128, 32, 32),
        threadGroup: (128, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.bfloat16]
    )[0]
}
