import Foundation
import MLX

/// Rollback switch for the sliding-prefill Q-preparation fusion (C-3).
///
/// Default ON. At the ranked sliding prefill shape (B=1, L=512, offset 0) the
/// combined Q/K/V prefill preparation kernel's Q output is consumed by exactly
/// one staged-attention threadgroup per 16-query tile, so the staged kernel
/// can apply the Q-row RMS norm + RoPE in-kernel instead of reading a
/// prepped device buffer: the preparation kernel then runs its established
/// K/V-only mode (`callCombinedPrefill`) and the 16.8 MB/layer Q write+read
/// round trip through device memory disappears.
///
/// Bit-exactness: the in-kernel preparation replicates the direct combined
/// Q/K/V preparation kernel's Q body exactly -- the same per-thread
/// 4-element ascending partials, the same `simd_sum` lane mapping, the same
/// 32-lane second-level exchange with +0.0f padding, the same
/// `metal::precise::rsqrt(sum / kHeadDim + 1.0e-6f)`, and the same
/// normalize/weight/RoPE epilogue expression with identical bf16 rounding
/// points (phase A additionally stashes the raw bf16 row in the on-chip Q
/// tile, a bit-preserving store/load, so phase C reads raw values from
/// threadgroup memory instead of re-reading the device projection). The QK
/// MPP tensor op consumes the same bf16 Q tile values from threadgroup
/// storage; softmax/PV are untouched. Raw-bit verified against the
/// production chain on the 12-case stress suite (multiple seeds, exact-tie
/// rows, subnormals, huge scores) for both layer types (see
/// notes/agent-c3-qprep-fusion-2026-07-18.md).
///
/// Full-attention layers are deliberately NOT fused: the winning m32wcf full
/// kernel's 32x512 bf16 score tile already occupies the entire 32 KB
/// threadgroup budget, so a 32-row Q tile (another 32 KB) cannot coexist,
/// and the m16 fallback geometry that would fit costs ~1.79x the m32wcf
/// attention time -- far more than the Q write+read saving (measured: the
/// fused full chain is ~0.75x the production chain). Set
/// `DARKBLOOM_PREFILL_Q_PREP_FUSION=0` to restore the separate Q preparation.
let gemma4PrefillQPrepFusionEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_Q_PREP_FUSION"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Raw-bit verify switch for the Q-preparation fusion. Default OFF. When
/// enabled, the engine additionally runs the production chain (combined
/// Q/K/V preparation + staged sliding attention) and preconditions that the
/// fused chain's attention output and live K/V cache rows are bit-identical.
let gemma4VerifyPrefillQPrepFusionBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_PREFILL_Q_PREP_FUSION_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Metal text of the fused Q-preparation phase inserted into the staged
/// sliding prefill kernel. The per-row reduction replicates the direct
/// combined Q/K/V preparation kernel's 64-thread-per-row topology (2
/// SIMDgroups per row, virtual thread x = (group % 2) * 32 + lane), its
/// 32-lane `local_sums` exchange (two partials in lanes 0...1, +0.0f
/// padding), and its epilogue expression character-for-character; only the
/// destination (threadgroup Q tile) and the per-row inverse-mean slot
/// differ. Internal so CPU tests can pin the exactness-critical markers.
let gemma4FusedQPrepSlidingPhaseSource: String = """
        threadgroup bfloat q_tile[kQueryRows * kHeadDim]
            __attribute__((aligned(16)));
        threadgroup float qprep_partials[32];
        threadgroup float qprep_inverse_mean[kQueryRows];
        constexpr uint kPrepPairs = 128;
        constexpr uint kPrepReads = 4;
        const int64_t q_dim_stride = raw_q_strides[2];
        // Phase A: per-virtual-thread partial sums of squares, replicating
        // the preparation kernel's 64-thread-per-row topology: virtual
        // thread x accumulates elements x*4..x*4+3 in ascending order, then
        // simd_sum over its 32-lane group. The raw bf16 values are also
        // stashed in the Q tile (a bit-preserving store/load) so phase C
        // reads them from threadgroup memory instead of re-reading the
        // device projection.
        for (uint pair_iter = 0; pair_iter < 8; ++pair_iter) {
            const uint row = pair_iter * 2 + simd_group / 2;
            const uint prep_x = (simd_group % 2) * kSIMDSize + simd_lane;
            const uint token = query_start + row;
            const device bfloat* input = raw_q
                + static_cast<int64_t>(token) * raw_q_strides[1]
                + static_cast<int64_t>(query_head * kHeadDim) * q_dim_stride
                + static_cast<int64_t>(prep_x * kPrepReads) * q_dim_stride;
            float accumulator = 0;
            if (prep_x * kPrepReads + kPrepReads <= kHeadDim) {
                for (uint index = 0; index < kPrepReads; ++index) {
                    const bfloat raw_value = input[
                        static_cast<int64_t>(index) * q_dim_stride];
                    q_tile[row * kHeadDim + prep_x * kPrepReads + index]
                        = raw_value;
                    const float value = raw_value;
                    accumulator += value * value;
                }
            }
            accumulator = simd_sum(accumulator);
            if (simd_lane == 0) {
                qprep_partials[pair_iter * 4 + simd_group] = accumulator;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase B: second-level reduction, replicating the preparation
        // kernel's local_sums exchange exactly: 32 lanes hold the row's
        // 2 group partials in order and +0.0f padding, one simd_sum, then
        // precise::rsqrt(sum / kHeadDim + 1.0e-6f).
        for (uint row = simd_group; row < kQueryRows; row += kSIMDGroups) {
            const float lane_value = simd_lane < 2
                ? qprep_partials[(row / 2) * 4 + (row % 2) * 2 + simd_lane]
                : 0.0f;
            float accumulator = simd_sum(lane_value);
            if (simd_lane == 0) {
                qprep_inverse_mean[row] = metal::precise::rsqrt(
                    accumulator / kHeadDim + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase C: normalize + weight + RoPE, per-element arithmetic
        // character-identical to the direct combined Q/K/V preparation
        // kernel's Q body; only the raw-value source (the threadgroup
        // stash), the destination (the same Q tile), and the per-row
        // inverse-mean slot differ.
        for (uint element = thread_index;
             element < kQueryRows * kPrepPairs;
             element += kThreads) {
            const uint row = element / kPrepPairs;
            const uint pair = element - row * kPrepPairs;
            const uint token = query_start + row;
            const bfloat raw_left = q_tile[row * kHeadDim + pair];
            const bfloat raw_right =
                q_tile[row * kHeadDim + pair + kPrepPairs];
            const bfloat normalized_left = static_cast<bfloat>(
                raw_left * qprep_inverse_mean[row]);
            const bfloat normalized_right = static_cast<bfloat>(
                raw_right * qprep_inverse_mean[row]);
            const bfloat weighted_left = q_weight[pair] * normalized_left;
            const bfloat weighted_right =
                q_weight[pair + kPrepPairs] * normalized_right;
            const uint rope_index =
                (static_cast<uint>(start_position) + token) * kPrepPairs
                + pair;
            const float cosine = rope_cosines[rope_index];
            const float sine = rope_sines[rope_index];
            const float left = static_cast<float>(weighted_left);
            const float right = static_cast<float>(weighted_right);
            q_tile[row * kHeadDim + pair] = static_cast<bfloat>(
                left * cosine - right * sine);
            q_tile[row * kHeadDim + pair + kPrepPairs] = static_cast<bfloat>(
                left * sine + right * cosine);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

"""

/// Builds the fused-Q staged sliding kernel source from the production
/// staged source: inserts the fused Q-preparation phase after the score-tile
/// declaration and redirects the QK MPP Q tensor at the threadgroup Q tile
/// (same logical tile the device buffer held: extents {256, 16}, dim stride
/// 1, row stride 256). Every other line -- QK descriptor and loop, causal
/// fill, softmax reduction topology, PV -- is byte-identical to production.
func gemma4FusedQStagedSlidingPrefillSource(_ reference: String) -> String {
    var source = reference
    let anchor = "threadgroup bfloat scores[kQueryRows * kLength];"
    precondition(source.components(separatedBy: anchor).count == 2)
    source = source.replacingOccurrences(
        of: anchor,
        with: anchor + "\n\n" + gemma4FusedQPrepSlidingPhaseSource)

    var lines = source.components(separatedBy: "\n")
    guard let qTensorLine = lines.firstIndex(where: {
        $0.contains("auto q_tensor = metal::tensor(")
    }) else {
        preconditionFailure("missing q_tensor declaration")
    }
    precondition(lines[qTensorLine + 1].contains("mutable_queries,"))
    precondition(lines[qTensorLine + 2].contains("metal::dextents<int, 2>{"))
    precondition(lines[qTensorLine + 3].contains("metal::array<int64_t, 2>{"))
    precondition(
        lines[qTensorLine + 4].contains(
            "queries_strides[3], queries_strides[2]});"))
    let indent = String(lines[qTensorLine].prefix(while: { $0 == " " }))
    let innerIndent = String(
        lines[qTensorLine + 1].prefix(while: { $0 == " " }))
    lines.replaceSubrange(
        qTensorLine...(qTensorLine + 4),
        with: [
            "\(indent)auto q_tensor = metal::tensor(",
            "\(innerIndent)q_tile,",
            "\(innerIndent)metal::dextents<int, 2>{256, 16},",
            "\(innerIndent)metal::array<int, 2>{1, 256});",
        ])

    guard let mutableLine = lines.firstIndex(where: {
        $0.contains("device bfloat* mutable_queries =")
    }) else {
        preconditionFailure("missing mutable_queries declaration")
    }
    precondition(lines[mutableLine + 1].contains("queries_strides[1]"))
    precondition(lines[mutableLine + 2].contains("queries_strides[2];"))
    lines.removeSubrange(mutableLine...(mutableLine + 2))
    return lines.joined(separator: "\n")
}

/// Pipeline name of the fused-Q staged sliding prefill kernel. Carries the
/// same flag suffixes as the production staged kernel plus `_qfuse`, so
/// differently-configured pipelines never alias in MLX's kernel cache.
let gemma4FusedQStagedSlidingPrefillKernelName: String =
    "gemma4_staged_sliding_prefill_16x512x256_mpp_v2_qfuse"
    + "_skip\(gemma4StagedPrefillCausalTileSkipEnabled ? 1 : 0)"
    + "_tokmaj\(gemma4StagedPrefillTokenMajorOutputEnabled ? 1 : 0)"
    + "_pvskip\(gemma4StagedPrefillPVTileSkipEnabled ? 1 : 0)"

let gemma4FusedQStagedSlidingPrefillKernelSource: String =
    gemma4FusedQStagedSlidingPrefillSource(
        gemma4StagedSlidingPrefill512KernelSource)

/// Exact-shape fused-Q MPP kernel for Gemma 4 sliding prefill (C-3).
///
/// Identical to the production staged sliding kernel except the 16-query Q
/// tile is prepared on-chip from the raw combined-projection Q slice (RMS
/// norm + RoPE, bit-identical to the preparation kernel's Q output) instead
/// of being read from a prepped device buffer. Inputs are the raw strided Q
/// slice `[1, 512, 32*256]`, the Q RMS norm weight, the 0-d start position,
/// the sliding RoPE cosine/sine tables, and the staged kernel's strided K/V
/// cache views. Output layout matches the production staged kernel
/// (token-major `[1, 512, 32*256]` under the ranked flags).
private let gemma4FusedQStagedSlidingPrefillKernel = MLXFast.metalKernel(
    name: gemma4FusedQStagedSlidingPrefillKernelName,
    inputNames: [
        "raw_q", "q_weight", "start_position", "rope_cosines", "rope_sines",
        "keys", "values",
    ],
    outputNames: ["output"],
    source: gemma4FusedQStagedSlidingPrefillKernelSource,
    header: """
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """,
    ensureRowContiguous: false
)

/// Runs the fused-Q staged sliding prefill kernel on the ranked shape.
///
/// Returns the same layout as `gemma4StagedSlidingPrefill512`: token-major
/// `[1, 512, 32*256]` when `gemma4StagedPrefillTokenMajorOutputEnabled`,
/// otherwise head-major `[1, 32, 512, 256]`.
func gemma4StagedSlidingPrefill512FusedQ(
    rawQueries: MLXArray,
    fusedAttentionRMS: FusedAttentionRMSPreparation,
    offset: Int,
    keys: MLXArray,
    values: MLXArray
) -> MLXArray {
    precondition(rawQueries.dtype == .bfloat16)
    precondition(keys.dtype == .bfloat16)
    precondition(values.dtype == .bfloat16)
    precondition(rawQueries.shape == [1, 512, 32 * 256])
    precondition(keys.shape == [1, 16, 512, 256])
    precondition(values.shape == [1, 16, 512, 256])
    precondition(offset == 0)
    precondition(fusedAttentionRMS.isSliding && fusedAttentionRMS.headDim == 256)

    let outputShape: [Int] = gemma4StagedPrefillTokenMajorOutputEnabled
        ? [1, 512, 32 * 256]
        : [1, 32, 512, 256]
    return gemma4FusedQStagedSlidingPrefillKernel(
        [
            rawQueries, fusedAttentionRMS.qNormWeight,
            fusedAttentionRMS.positionViews[offset],
            fusedAttentionRMS.ropeCosines, fusedAttentionRMS.ropeSines,
            keys, values,
        ],
        grid: (128, 32, 32),
        threadGroup: (128, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.bfloat16]
    )[0]
}
