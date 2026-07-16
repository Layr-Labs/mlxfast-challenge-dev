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

/// Rollback switch for the staged prefill PV causal column skip (P5).
///
/// Default ON. When enabled, both staged prefill kernels (sliding and full)
/// bound the PV reduction at the causal 32-key block limit instead of
/// consuming all 512 probability columns. Every skipped column is causally
/// masked for all 16 rows of the query tile, so its softmax probability is
/// exactly +0.0 bf16 (the `bfloat::lowest()` mask fill underflows `exp` to
/// +0.0f, and `+0.0f * normalizer` stays +0.0f); the only effect the skipped
/// columns' `acc += (+0.0) * v` terms can have on an IEEE-754 f32 accumulator
/// is canonicalizing -0.0 to +0.0, which the truncated path reproduces with a
/// +0.0f-initialized cooperative f32 accumulator plus a trailing `+ 0.0f`
/// before the identical `static_cast<bfloat>` rounding (see
/// `notes/agent-p5-pv-skip-2026-07-15.md` for the exactness argument). Query
/// blocks with no skipped tail keep the untouched full-width PV run.
/// Set `DARKBLOOM_STAGED_PREFILL_PV_TILE_SKIP=0` to restore the full-width
/// PV reduction everywhere.
let gemma4StagedPrefillPVTileSkipEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_PREFILL_PV_TILE_SKIP"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Rollback switch for the staged prefill softmax causal column truncation
/// (P7).
///
/// Default ON. Effective only while the P5 PV column skip is also enabled
/// (the interpolated Metal condition is `kSoftmaxTruncation && kPVTileSkip`).
/// When active, both staged prefill kernels bound the softmax stage (row
/// max, exp, exp-sum, normalize stores) at the same causal 32-key block
/// limit the P5 PV stage consumes, instead of processing all 512 score
/// columns per 16-query row tile. Columns at or beyond the limit hold
/// exactly the `bfloat::lowest()` causal fill for every row of the tile, so
/// - MAX is bit-identical unconditionally: the truncated path replaces each
///   fully-masked 4-column lane read with the exact constant the full fold
///   provably produces for it (`max(float lowest, fill x4) == fill`), so
///   every `simd_max` consumes bit-identical lane values;
/// - EXP-SUM is truncated only behind an in-kernel guard (`row_max_live`:
///   row maximum strictly above the fill). Under the guard every skipped
///   term is `fast::exp` of an argument at most `-2^120` (or `-inf`), which
///   underflows to exactly +0.0f, so skipped per-lane normalizers are
///   exactly +0.0f. If the guard fails (every live score saturated at or
///   below the fill -- adversarial-only inputs), the row falls back to the
///   original full-width fold instead of assuming the corner away;
/// - the skipped NORMALIZE stores (which would write exactly +0.0 bf16) are
///   never read: the P5-truncated PV consumes only columns below the same
///   bound, which is why P7 is inert without P5.
/// See `notes/agent-p7-softmax-truncation-2026-07-16.md` for the exactness
/// argument. Set `DARKBLOOM_STAGED_PREFILL_SOFTMAX_TRUNCATION=0` to restore
/// the full-width softmax everywhere.
let gemma4StagedPrefillSoftmaxTruncationEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_STAGED_PREFILL_SOFTMAX_TRUNCATION"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Exact Metal text of the P5 truncated-PV accumulator zero initialization.
///
/// Interpolated verbatim into both staged prefill kernel sources and pinned
/// by CPU tests: the +0.0f initialization is part of the signed-zero
/// exactness argument (a +0.0-seeded IEEE round-to-nearest accumulation can
/// never produce -0.0), so it must survive in the exact source string handed
/// to MLX's Metal compiler.
let gemma4StagedPrefillPVSkipZeroInitializationLine =
    "accumulator[i] = 0.0f;"

/// Exact Metal text of the P5 trailing signed-zero canonicalization.
///
/// Interpolated verbatim into both staged prefill kernel sources and pinned
/// by CPU tests. The `+ 0.0f` reproduces the net effect of the skipped
/// masked columns' `acc += (+0.0) * v` terms (canonicalizing -0.0 to +0.0,
/// identity on everything else). MLX compiles custom kernels with fast math
/// disabled (`options->setFastMathEnabled(false)` in
/// mlx/backend/metal/device.cpp, reached from custom_kernel.cpp), and under
/// IEEE semantics `x + 0.0f` is not an identity (it maps -0.0 to +0.0), so
/// the compiler must preserve the operation.
let gemma4StagedPrefillPVSkipCanonicalizationLine =
    "const float canonicalized = accumulator[i] + 0.0f;"

/// Exact Metal text of the P5 truncation guard. Query blocks whose causal
/// bound covers all 512 columns (no skipped tail) must keep the untouched
/// full-width PV run: the full loop applies no masked `+ (+0.0)` terms for
/// them, so the trailing canonicalization must not be applied either.
let gemma4StagedPrefillPVSkipGuardLine =
    "if (kPVTileSkip && pv_key_column_limit < kLength) {"

/// Exact Metal text of the P7 softmax column bound header, pinned by CPU
/// tests. The truncation is active only when the P5 PV skip is also active:
/// the truncated softmax stops writing the masked +0.0 probability columns,
/// and only the P5-truncated PV (bounded at the same limit) provably never
/// reads them. With either flag off the bound is `kLength` and the original
/// full-width softmax loop below runs untouched.
let gemma4StagedPrefillSoftmaxTruncationLimitLine =
    "const uint softmax_column_limit = (kSoftmaxTruncation && kPVTileSkip)"

/// Exact Metal text of the P7 max-stage substitution, pinned by CPU tests.
/// A fully-masked 4-column lane chunk holds exactly the causal fill
/// (`bfloat::lowest()` widened to f32) in all four columns, and the original
/// per-lane fold over it -- seeded with `float` lowest, stepped with
/// `maximum < value ? value : maximum` -- provably produces exactly that
/// fill value. Substituting the constant therefore hands every `simd_max`
/// bit-identical lane values, unconditionally, for any score contents.
let gemma4StagedPrefillSoftmaxTruncationSubstitutionLine =
    "maximum = softmax_mask_fill;"

/// Exact Metal text of the P7 in-kernel corner guard, pinned by CPU tests.
/// Truncation always substitutes at least one masked lane (the bound is at
/// most 480 when active), so the row maximum is at least the fill; equality
/// means every live score sits at or below the fill (a saturated or -inf
/// row), where the skipped exp terms would be `exp(0) == 1`, not +0.0. The
/// guard routes such rows to the original full-width fold instead of
/// relying on their unreachability from real bf16 Q/K.
let gemma4StagedPrefillSoftmaxTruncationGuardLine =
    "const bool row_max_live = maximum > softmax_mask_fill;"

/// Metal source of the softmax stage, shared verbatim by both staged prefill
/// kernels (their score tiles are the same 16 rows x 512 bf16 columns, and
/// the stage references only symbols both kernels define identically:
/// kLength, kQueryRows, kSIMDGroups, kSIMDSize, kSoftmaxReads,
/// kSoftmaxTruncation, kPVTileSkip, query_block, simd_lane, simd_group,
/// scores). Sharing one constant keeps the two kernels' softmax text
/// provably in sync and lets CPU tests pin a single implementation.
let gemma4StagedPrefillSoftmaxStageSource: String = """
        // Reproduce MLX precise block_softmax for axis size 512 exactly while
        // running four rows concurrently. One physical SIMDgroup owns a row
        // and emulates stock's four virtual SIMDgroups. Each virtual group
        // retains the identical lane/read mapping and simd_max/simd_sum; the
        // second reduction places its four partials in lanes 0...3 before the
        // same SIMD intrinsic. This removes all softmax threadgroup barriers
        // without changing reduction or cast order.
        //
        // P7 softmax causal column truncation: when the P5 PV skip is also
        // active, score columns at or beyond softmax_column_limit
        // = 32 * ((query_block + 2) / 2) are causally masked for every row
        // of this tile -- each holds exactly the bfloat-lowest causal fill --
        // and the P5-truncated PV never reads their probabilities. The
        // truncated path below is bit-exact for the consumed output:
        // - MAX: each fully-masked 4-column lane chunk is replaced by the
        //   constant the full fold provably produces for it (the fill;
        //   max(float lowest, fill x4) == fill; the 32-aligned bound never
        //   splits a 4-aligned chunk), so every simd_max consumes
        //   bit-identical lane values and the row maximum is bit-identical
        //   unconditionally.
        // - EXP-SUM: masked lanes are skipped only behind the row_max_live
        //   guard (row maximum strictly above the fill). Under the guard the
        //   maximum is a live bf16 score at least one bf16 ULP (2^120) above
        //   the fill, so each skipped term is fast::exp of an argument at
        //   most -2^120 (or -inf after f32 subtraction overflow), which
        //   underflows to exactly +0.0f; the skipped per-lane fold
        //   0.0f + 4x(+0.0f) is exactly +0.0f, so substituting 0.0f hands
        //   every simd_sum bit-identical lane values. When the guard fails
        //   (row maximum equals the fill exactly: every live score saturated
        //   at or below bfloat lowest -- adversarial-only, guarded in-kernel
        //   instead of assumed away), the fallback re-reads the untouched
        //   masked tail from threadgroup memory and runs the original
        //   full-width fold bit-for-bit.
        // - NORMALIZE: masked-tail stores are skipped under the same guard.
        //   They would write exactly +0.0 bf16 (+0.0f times a positive
        //   normal normalizer), and nothing reads them: the P5-truncated PV
        //   consumes only columns below the same bound -- which is why the
        //   truncation requires kPVTileSkip. On guard failure the fallback
        //   writes all 512 columns like the original loop.
        // Query blocks without a skipped tail (query_block >= 30) and every
        // configuration with P7 or P5 disabled run the untouched original
        // loop in the else branch. See
        // notes/agent-p7-softmax-truncation-2026-07-16.md for the exactness
        // argument.
        \(gemma4StagedPrefillSoftmaxTruncationLimitLine)
            ? ((query_block + 2) / 2) * 32
            : kLength;
        const float softmax_mask_fill =
            static_cast<float>(metal::numeric_limits<bfloat>::lowest());
        if (softmax_column_limit < kLength) {
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
                    if (read_offset < softmax_column_limit) {
                        for (uint read = 0; read < kSoftmaxReads; ++read) {
                            const float value = static_cast<float>(
                                scores[row_offset + read_offset + read]);
                            loaded[virtual_group][read] = value;
                            maximum = maximum < value ? value : maximum;
                        }
                    } else {
                        \(gemma4StagedPrefillSoftmaxTruncationSubstitutionLine)
                    }
                    virtual_maxima[virtual_group] = simd_max(maximum);
                }

                float maximum = simd_lane < kSIMDGroups
                    ? virtual_maxima[simd_lane]
                    : metal::numeric_limits<float>::lowest();
                maximum = simd_max(maximum);

                \(gemma4StagedPrefillSoftmaxTruncationGuardLine)

                float virtual_normalizers[kSIMDGroups];
                for (uint virtual_group = 0;
                     virtual_group < kSIMDGroups;
                     ++virtual_group) {
                    const uint read_offset =
                        virtual_group * kSIMDSize * kSoftmaxReads
                        + simd_lane * kSoftmaxReads;
                    float normalizer = 0.0f;
                    if (read_offset < softmax_column_limit) {
                        for (uint read = 0; read < kSoftmaxReads; ++read) {
                            const float exponential = fast::exp(
                                loaded[virtual_group][read] - maximum);
                            loaded[virtual_group][read] = exponential;
                            normalizer += exponential;
                        }
                    } else if (!row_max_live) {
                        // Corner fallback: the truncated MAX stage never
                        // loaded the masked tail; re-read it (untouched
                        // since the causal fill -- softmax only writes its
                        // own row, later) and run the original fold.
                        for (uint read = 0; read < kSoftmaxReads; ++read) {
                            const float value = static_cast<float>(
                                scores[row_offset + read_offset + read]);
                            const float exponential = fast::exp(
                                value - maximum);
                            loaded[virtual_group][read] = exponential;
                            normalizer += exponential;
                        }
                    }
                    virtual_normalizers[virtual_group] =
                        simd_sum(normalizer);
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
                    if (read_offset < softmax_column_limit
                        || !row_max_live) {
                        for (uint read = 0; read < kSoftmaxReads; ++read) {
                            scores[row_offset + read_offset + read] =
                                static_cast<bfloat>(
                                    loaded[virtual_group][read]
                                        * normalizer);
                        }
                    }
                }
            }
        } else {
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
                    virtual_normalizers[virtual_group] =
                        simd_sum(normalizer);
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
        }
        """

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

    /// P5 bound: number of leading probability columns the PV stage consumes
    /// for the 16-query tile `queryBlock`; every key column at or beyond this
    /// bound is causally masked for all 16 rows of the tile, so its softmax
    /// probability is exactly +0.0 bf16.
    ///
    /// Granularity: the PV matmul reduces over K with a single dynamic-K
    /// tensor op, so any column bound would be dispatchable; the bound is
    /// kept at the same 32-key block granularity as the P1 QK skip
    /// (`keyTileColumns * computedKeyBlockCount`), which is conservative
    /// (partially-live blocks stay fully included) and keeps the truncated
    /// reduction length 32-aligned. Must match the Metal-side
    /// `pv_key_column_limit`.
    static func pvKeyColumnLimit(queryBlock: Int) -> Int {
        keyTileColumns * computedKeyBlockCount(queryBlock: queryBlock)
    }

    /// P7 bound: number of leading score columns the softmax stage processes
    /// for the 16-query tile `queryBlock` while the truncation is active
    /// (P7 AND P5 enabled). Identical to the P5 PV bound by construction:
    /// the truncated softmax stops writing the masked +0.0 probability
    /// columns, so the consumer (the P5-truncated PV) must never read at or
    /// beyond the same limit. Must match the Metal-side
    /// `softmax_column_limit` in the active configuration.
    static func softmaxColumnLimit(queryBlock: Int) -> Int {
        pvKeyColumnLimit(queryBlock: queryBlock)
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

/// Pipeline name of the staged sliding prefill kernel. Internal so CPU tests
/// can pin the DARKBLOOM flag/name-suffix contract (`_skip`/`_tokmaj`/
/// `_pvskip`/`_smtrunc`); the suffix keeps differently-configured pipelines
/// from aliasing in MLX's kernel cache.
let gemma4StagedSlidingPrefill512KernelName: String =
    "gemma4_staged_sliding_prefill_16x512x256_mpp_v2"
    + "_skip\(gemma4StagedPrefillCausalTileSkipEnabled ? 1 : 0)"
    + "_tokmaj\(gemma4StagedPrefillTokenMajorOutputEnabled ? 1 : 0)"
    + "_pvskip\(gemma4StagedPrefillPVTileSkipEnabled ? 1 : 0)"
    + "_smtrunc\(gemma4StagedPrefillSoftmaxTruncationEnabled ? 1 : 0)"

/// Metal source of the staged sliding prefill kernel. Internal so CPU tests
/// can prove the P5 zero-initialization and `+ 0.0f` canonicalization text
/// survives verbatim in the exact source string handed to MLX's Metal
/// compiler (which disables fast math, so the compiler cannot elide
/// `x + 0.0f` -- it is not an identity under IEEE signed zeros).
let gemma4StagedSlidingPrefill512KernelSource: String = """
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
        constexpr bool kPVTileSkip =
            \(gemma4StagedPrefillPVTileSkipEnabled);
        constexpr bool kSoftmaxTruncation =
            \(gemma4StagedPrefillSoftmaxTruncationEnabled);

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
        // running four rows concurrently (P7-truncated at the causal column
        // bound when active); stage source shared verbatim with the staged
        // full-attention kernel.
        \(gemma4StagedPrefillSoftmaxStageSource)
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
        //
        // P5 PV causal column skip: probability columns at or beyond
        // pv_key_column_limit = 32 * ((query_block + 2) / 2) are causally
        // masked for every row of this query tile, so softmax wrote exactly
        // +0.0 bf16 there (fast::exp underflows the bfloat lowest fill to
        // +0.0f and +0.0f * normalizer stays +0.0f). In the full-width
        // reduction each such column contributes acc += (+0.0) * v, and
        // under IEEE-754 round-to-nearest a signed-zero term can only
        // canonicalize a -0.0 accumulator to +0.0 (when the product is
        // +0.0-signed) -- it can never change a nonzero accumulator. The
        // truncated path reproduces that net effect exactly: it reduces only
        // the causally live prefix into a +0.0f-initialized f32 cooperative
        // accumulator (a +0.0-seeded round-to-nearest accumulation can never
        // produce -0.0), applies the same trailing + 0.0f the skipped
        // columns would have applied, and rounds once to bf16 via
        // static_cast, the same final conversion the full-width run
        // performs. Query blocks without a skipped tail (query_block >= 30)
        // keep the untouched full-width run: the full loop applies no masked
        // terms for them, so no canonicalization may be applied either. See
        // notes/agent-p5-pv-skip-2026-07-15.md for the exactness argument,
        // including chunk-granularity accumulation.
        constexpr int kOutputRowStride = kTokenMajorOutput
            ? int(kQHeads * kHeadDim)
            : int(kHeadDim);
        constexpr auto pv_descriptor = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 512, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_descriptor, metal::execution_simdgroup> pv;
        constexpr auto pv_truncated_descriptor =
            mpp::tensor_ops::matmul2d_descriptor(
                16, 32, static_cast<int>(metal::dynamic_extent),
                false, false, false,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<
            pv_truncated_descriptor, metal::execution_simdgroup> pv_truncated;
        const uint pv_key_column_limit = kPVTileSkip
            ? ((query_block + 2) / 2) * 32
            : kLength;
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
            device bfloat* output_tile = output
                + output_tile_base
                + value_start;
            \(gemma4StagedPrefillPVSkipGuardLine)
                auto probability_tensor = metal::tensor(
                    scores,
                    metal::dextents<int, 2>{int(pv_key_column_limit), 16},
                    metal::array<int, 2>{1, 512});
                auto value_tensor = metal::tensor(
                    mutable_values,
                    metal::dextents<int, 2>{32, int(pv_key_column_limit)},
                    metal::array<int64_t, 2>{
                        values_strides[3], values_strides[2]});
                auto accumulator =
                    pv_truncated.get_destination_cooperative_tensor<
                        decltype(probability_tensor),
                        decltype(value_tensor),
                        float>();
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
                    if (accumulator.is_valid_element(i)) {
                        \(gemma4StagedPrefillPVSkipZeroInitializationLine)
                    }
                }
                pv_truncated.run(
                    probability_tensor, value_tensor, accumulator);
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
                    if (accumulator.is_valid_element(i)) {
                        \(gemma4StagedPrefillPVSkipCanonicalizationLine)
                        const auto coords =
                            accumulator.get_multidimensional_index(i);
                        output_tile[coords[0] + coords[1] * kOutputRowStride] =
                            static_cast<bfloat>(canonicalized);
                    }
                }
            } else {
                auto probability_tensor = metal::tensor(
                    scores,
                    metal::dextents<int, 2>{512, 16},
                    metal::array<int, 2>{1, 512});
                auto value_tensor = metal::tensor(
                    mutable_values,
                    metal::dextents<int, 2>{32, 512},
                    metal::array<int64_t, 2>{
                        values_strides[3], values_strides[2]});
                auto output_tensor = metal::tensor(
                    output_tile,
                    metal::dextents<int, 2>{32, 16},
                    metal::array<int, 2>{1, kOutputRowStride});
                pv.run(probability_tensor, value_tensor, output_tensor);
            }
        }
        """

/// Exact-shape MPP kernel for Gemma 4 sliding prefill.
///
/// One 128-thread threadgroup owns one `(query head, 16-query)` tile. QK
/// writes the causally-live prefix of a 16x512 BF16 score tile to threadgroup
/// memory (fully-masked 32-key tiles are skipped and mask-filled instead --
/// P1), the same four-SIMD/4-read reduction topology as MLX's precise block
/// softmax rewrites the causally live prefix of that tile in place (P7),
/// and PV consumes only the causally live prefix of it (P5) without
/// materializing either scores or probabilities in device memory. Output is
/// emitted token-major `[1, 512, 32*256]` (P3), directly in the merged
/// layout `o_proj` consumes; P1, P3, P5, and P7 fall back to the v1 behavior
/// via their DARKBLOOM env switches above.
private let gemma4StagedSlidingPrefill512Kernel = MLXFast.metalKernel(
    name: gemma4StagedSlidingPrefill512KernelName,
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: gemma4StagedSlidingPrefill512KernelSource,
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
