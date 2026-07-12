import Foundation
import MLX
import MLXLMCommon

private func gemma4SlidingPrepSDPAEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool = false
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4FusedSlidingPrepSDPAFeatureEnabled =
    gemma4SlidingPrepSDPAEnvironmentFlag(
        "DARKBLOOM_FUSED_SLIDING_PREP_SDPA",
        default: true)

private let gemma4VerifyFusedSlidingPrepSDPABits =
    gemma4SlidingPrepSDPAEnvironmentFlag(
        "DARKBLOOM_VERIFY_FUSED_SLIDING_PREP_SDPA_BITS")

@inline(__always)
func gemma4FusedSlidingPrepSDPAEnabled() -> Bool {
    gemma4FusedSlidingPrepSDPAFeatureEnabled
}

@inline(__always)
func gemma4FusedSlidingPrepSDPAVerificationEnabled() -> Bool {
    gemma4VerifyFusedSlidingPrepSDPABits
}

/// Single-token sliding attention preparation plus MLX's pinned one-pass
/// D=256 `sdpa_vector` arithmetic. Each query-head threadgroup prepares its
/// own current K/V. Only even query heads publish their disjoint KV-head slabs.
private let gemma4FusedSlidingPrepSDPAKernel = MLXFast.metalKernel(
    name: "gemma4_fused_sliding_prep_sdpa_256_v1",
    inputNames: [
        "raw_q", "raw_k", "raw_v", "q_weight", "k_weight", "position",
        "old_combined_kv", "old_length", "old_capacity", "attention_scale",
        "rope_cosines", "rope_sines",
    ],
    outputNames: ["attention", "append_kv"],
    source: """
        constexpr uint kHeadDim = 256;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 16;
        constexpr uint kGQAFactor = 2;
        constexpr uint kPrepReads = 4;
        constexpr uint kPrepThreads = 64;
        constexpr uint kSIMDSize = 32;
        constexpr uint kAttentionReads = 8;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint kv_head = query_head / kGQAFactor;

        threadgroup bfloat prepared_query[kHeadDim];
        threadgroup bfloat prepared_key[kHeadDim];
        threadgroup bfloat prepared_value[kHeadDim];
        threadgroup float prep_inverse_mean[1];

        gemma4_prepare_sliding_row_256(
            raw_q + query_head * kHeadDim,
            q_weight,
            true,
            prepared_query,
            prep_inverse_mean,
            simd_group,
            simd_lane);
        gemma4_apply_sliding_rope_256(
            prepared_query,
            static_cast<uint>(position),
            rope_cosines,
            rope_sines,
            simd_group,
            simd_lane);

        gemma4_prepare_sliding_row_256(
            raw_k + kv_head * kHeadDim,
            k_weight,
            true,
            prepared_key,
            prep_inverse_mean,
            simd_group,
            simd_lane);
        gemma4_apply_sliding_rope_256(
            prepared_key,
            static_cast<uint>(position),
            rope_cosines,
            rope_sines,
            simd_group,
            simd_lane);

        gemma4_prepare_sliding_row_256(
            raw_v + kv_head * kHeadDim,
            k_weight,
            false,
            prepared_value,
            prep_inverse_mean,
            simd_group,
            simd_lane);

        // SIMDgroup zero exactly emulates the promoted two-SIMD preparation
        // trees for all three rows. Publish its private work once before the
        // other 31 SIMDgroups consume current Q/K/V in attention.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // There is no cross-threadgroup ordering. Both GQA partners retain
        // private current K/V, while the even partner is the sole writer for
        // this KV head's two append slabs.
        if ((query_head & 1u) == 0u && thread_index < kPrepThreads) {
            const uint base = thread_index * kPrepReads;
            const uint value_slab = kKVHeads * kHeadDim;
            for (uint index = 0; index < kPrepReads; ++index) {
                const uint dimension = base + index;
                append_kv[kv_head * kHeadDim + dimension] =
                    prepared_key[dimension];
                append_kv[value_slab + kv_head * kHeadDim + dimension] =
                    prepared_value[dimension];
            }
        }

        // Clone pinned sdpa_vector<bfloat,256,256>: 32 SIMDgroups each own
        // key positions gid, gid+32, ... and eight final output dimensions.
        const int cache_length = static_cast<int>(old_length);
        const int cache_capacity = static_cast<int>(old_capacity);
        const int key_count = cache_length + 1;
        const size_t old_value_slab =
            static_cast<size_t>(kKVHeads) * cache_capacity * kHeadDim;
        const size_t old_head_base =
            static_cast<size_t>(kv_head) * cache_capacity * kHeadDim;

        thread float query_values[kAttentionReads];
        thread float key_values[kAttentionReads];
        thread float output_values[kAttentionReads];
        const uint lane_dimension = simd_lane * kAttentionReads;
        const float query_scale = static_cast<float>(attention_scale);
        for (uint index = 0; index < kAttentionReads; ++index) {
            query_values[index] = query_scale * static_cast<float>(
                prepared_query[lane_dimension + index]);
            output_values[index] = 0.0f;
        }

        float max_score = -metal::numeric_limits<float>::max();
        float sum_exp_score = 0.0f;
        for (int key_index = static_cast<int>(simd_group);
             key_index < key_count;
             key_index += static_cast<int>(kSIMDSize)) {
            const bool is_current = key_index == cache_length;
            for (uint index = 0; index < kAttentionReads; ++index) {
                const uint dimension = lane_dimension + index;
                key_values[index] = is_current
                    ? static_cast<float>(prepared_key[dimension])
                    : static_cast<float>(old_combined_kv[
                        old_head_base
                        + static_cast<size_t>(key_index) * kHeadDim
                        + dimension]);
            }

            float score = 0.0f;
            for (uint index = 0; index < kAttentionReads; ++index) {
                score += query_values[index] * key_values[index];
            }
            score = simd_sum(score);

            const float new_max = max(max_score, score);
            const float factor = metal::fast::exp(max_score - new_max);
            const float exp_score = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            for (uint index = 0; index < kAttentionReads; ++index) {
                const uint dimension = lane_dimension + index;
                const float value = is_current
                    ? static_cast<float>(prepared_value[dimension])
                    : static_cast<float>(old_combined_kv[
                        old_value_slab
                        + old_head_base
                        + static_cast<size_t>(key_index) * kHeadDim
                        + dimension]);
                output_values[index] = output_values[index] * factor
                    + exp_score * value;
            }
        }

        threadgroup float partial_outputs[kSIMDSize * kSIMDSize];
        threadgroup float max_scores[kSIMDSize];
        threadgroup float sum_exp_scores[kSIMDSize];
        if (simd_lane == 0) {
            max_scores[simd_group] = max_score;
            sum_exp_scores[simd_group] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lane];
        const float new_max = simd_max(max_score);
        const float factor = metal::fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lane] * factor);

        for (uint index = 0; index < kAttentionReads; ++index) {
            partial_outputs[simd_lane * kSIMDSize + simd_group] =
                output_values[index];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            output_values[index] = simd_sum(
                partial_outputs[simd_group * kSIMDSize + simd_lane] * factor);
            output_values[index] = sum_exp_score == 0.0f
                ? output_values[index]
                : output_values[index] / sum_exp_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lane == 0) {
            const uint output_base =
                query_head * kHeadDim + simd_group * kAttentionReads;
            for (uint index = 0; index < kAttentionReads; ++index) {
                attention[output_base + index] = static_cast<bfloat>(
                    output_values[index]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline void gemma4_prepare_sliding_row_256(
            const device bfloat* input,
            const device bfloat* weight,
            bool has_weight,
            threadgroup bfloat* prepared,
            threadgroup float* inverse_mean,
            uint simd_group,
            uint simd_lane
        ) {
            constexpr uint kHeadDim = 256;
            constexpr uint kReads = 4;
            constexpr uint kHalf = kHeadDim / 2;
            if (simd_group != 0) {
                return;
            }

            // Reproduce the original two 32-lane first-stage reductions. Each
            // lane owns the same four elements that its promoted counterpart
            // owned in SIMDgroup zero and SIMDgroup one respectively.
            const uint lower_base = simd_lane * kReads;
            const uint upper_base = kHalf + lower_base;
            float lower_accumulator = 0.0f;
            float upper_accumulator = 0.0f;
            for (uint index = 0; index < kReads; ++index) {
                const float lower = input[lower_base + index];
                const float upper = input[upper_base + index];
                lower_accumulator += lower * lower;
                upper_accumulator += upper * upper;
            }
            lower_accumulator = simd_sum(lower_accumulator);
            upper_accumulator = simd_sum(upper_accumulator);

            // Feed those two sums into lanes zero/one of the same 32-lane
            // second stage, with the remaining lanes exactly zero-padded.
            float accumulator = simd_lane == 0
                ? lower_accumulator
                : (simd_lane == 1 ? upper_accumulator : 0.0f);
            accumulator = simd_sum(accumulator);
            if (simd_lane == 0) {
                inverse_mean[0] = metal::precise::rsqrt(
                    accumulator / kHeadDim + 1.0e-6f);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint index = 0; index < kReads; ++index) {
                const uint lower_dimension = lower_base + index;
                const uint upper_dimension = upper_base + index;
                const bfloat lower_normalized = static_cast<bfloat>(
                    input[lower_dimension] * inverse_mean[0]);
                const bfloat upper_normalized = static_cast<bfloat>(
                    input[upper_dimension] * inverse_mean[0]);
                prepared[lower_dimension] = has_weight
                    ? weight[lower_dimension] * lower_normalized
                    : static_cast<bfloat>(1.0f) * lower_normalized;
                prepared[upper_dimension] = has_weight
                    ? weight[upper_dimension] * upper_normalized
                    : static_cast<bfloat>(1.0f) * upper_normalized;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        inline void gemma4_apply_sliding_rope_256(
            threadgroup bfloat* prepared,
            uint position,
            const device float* rope_cosines,
            const device float* rope_sines,
            uint simd_group,
            uint simd_lane
        ) {
            constexpr uint kPairs = 128;
            constexpr uint kSIMDSize = 32;
            if (simd_group == 0) {
                for (uint pair = simd_lane;
                     pair < kPairs;
                     pair += kSIMDSize) {
                    const uint rope_index = position * kPairs + pair;
                    const float cosine = rope_cosines[rope_index];
                    const float sine = rope_sines[rope_index];
                    const float left = static_cast<float>(prepared[pair]);
                    const float right = static_cast<float>(
                        prepared[pair + kPairs]);
                    prepared[pair] = static_cast<bfloat>(
                        left * cosine - right * sine);
                    prepared[pair + kPairs] = static_cast<bfloat>(
                        left * sine + right * cosine);
                }
            }
        }
        """,
    ensureRowContiguous: true
)

struct Gemma4FusedSlidingPrepSDPAResult {
    let attention: MLXArray
    let appendKV: MLXArray
}

struct Gemma4FusedSlidingPrepSDPA: @unchecked Sendable {
    let qNormWeight: MLXArray
    let kNormWeight: MLXArray
    let positionViews: [MLXArray]
    let ropeCosines: MLXArray
    let ropeSines: MLXArray
    let attentionScale: MLXArray

    init?(preparation: FusedAttentionRMSPreparation) {
        guard preparation.isSliding,
              preparation.headDim == 256,
              preparation.kvHeads == 16,
              preparation.qNormWeight.dtype == .bfloat16,
              preparation.qNormWeight.shape == [256],
              preparation.kNormWeight.dtype == .bfloat16,
              preparation.kNormWeight.shape == [256],
              preparation.positionViews.count >= 1_024,
              preparation.ropeCosines.dtype == .float32,
              preparation.ropeCosines.shape == [4_096, 128],
              preparation.ropeSines.dtype == .float32,
              preparation.ropeSines.shape == [4_096, 128]
        else {
            return nil
        }
        self.qNormWeight = preparation.qNormWeight
        self.kNormWeight = preparation.kNormWeight
        self.positionViews = preparation.positionViews
        self.ropeCosines = preparation.ropeCosines
        self.ropeSines = preparation.ropeSines
        self.attentionScale = MLXArray(Float(1.0), dtype: .float32)
    }

    func supports(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray,
        offset: Int,
        snapshot: Gemma4CombinedKVSlidingDecodeSnapshot
    ) -> Bool {
        rawQueries.dtype == .bfloat16
            && rawQueries.shape == [1, 1, 8_192]
            && rawKeys.dtype == .bfloat16
            && rawKeys.shape == [1, 1, 4_096]
            && rawValues.dtype == .bfloat16
            && rawValues.shape == [1, 1, 4_096]
            && offset == snapshot.length
            && (1..<1_023).contains(offset)
            && snapshot.capacity >= snapshot.length
            && snapshot.capacity <= 1_024
            && snapshot.storage.dtype == .bfloat16
            && snapshot.storage.shape == [2, 1, 16, snapshot.capacity, 256]
    }

    func callAsFunction(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray,
        offset: Int,
        snapshot: Gemma4CombinedKVSlidingDecodeSnapshot
    ) -> Gemma4FusedSlidingPrepSDPAResult {
        precondition(supports(
            rawQueries: rawQueries,
            rawKeys: rawKeys,
            rawValues: rawValues,
            offset: offset,
            snapshot: snapshot
        ))
        let outputs = gemma4FusedSlidingPrepSDPAKernel(
            [
                rawQueries, rawKeys, rawValues, qNormWeight, kNormWeight,
                positionViews[offset], snapshot.storage,
                MLXArray(Int32(snapshot.length)),
                MLXArray(Int32(snapshot.capacity)), attentionScale,
                ropeCosines, ropeSines,
            ],
            grid: (1_024, 32, 1),
            threadGroup: (1_024, 1, 1),
            outputShapes: [[1, 32, 1, 256], [2, 1, 16, 1, 256]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return Gemma4FusedSlidingPrepSDPAResult(
            attention: outputs[0],
            appendKV: outputs[1]
        )
    }

    func verifyRawBits(
        _ candidate: Gemma4FusedSlidingPrepSDPAResult,
        promotedAttention: MLXArray,
        promotedAppendKV: MLXArray
    ) {
        precondition(promotedAttention.dtype == .bfloat16)
        precondition(promotedAttention.shape == [1, 32, 1, 256])
        precondition(promotedAppendKV.dtype == .bfloat16)
        precondition(promotedAppendKV.shape == [2, 1, 16, 1, 256])

        let attentionMatches = arrayEqual(
            candidate.attention.view(dtype: .uint16),
            promotedAttention.view(dtype: .uint16)
        )
        let appendMatches = arrayEqual(
            candidate.appendKV.view(dtype: .uint16),
            promotedAppendKV.view(dtype: .uint16)
        )
        eval(attentionMatches, appendMatches)
        precondition(
            attentionMatches.item(Bool.self),
            "fused sliding prep+SDPA attention differs from promoted path"
        )
        precondition(
            appendMatches.item(Bool.self),
            "fused sliding prep+SDPA append KV differs from promoted path"
        )
    }
}

extension Gemma4FastLayer {
    func fusedSlidingPrepSDPAIfSupported(
        rawQueries: MLXArray,
        rawKeys: MLXArray,
        rawValues: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray? {
        guard let fusedSlidingPrepSDPA,
              let fusedAttentionRMS,
              let rawValues,
              gemma4CombinedKVDirectEnabled(),
              case .none = mask,
              let combinedCache = cache as? Gemma4CombinedKVCache,
              let snapshot = combinedCache.directSlidingDecodeSnapshot(),
              fusedSlidingPrepSDPA.supports(
                  rawQueries: rawQueries,
                  rawKeys: rawKeys,
                  rawValues: rawValues,
                  offset: offset,
                  snapshot: snapshot
              )
        else {
            return nil
        }

        let candidate = fusedSlidingPrepSDPA(
            rawQueries: rawQueries,
            rawKeys: rawKeys,
            rawValues: rawValues,
            offset: offset,
            snapshot: snapshot
        )
        if gemma4FusedSlidingPrepSDPAVerificationEnabled() {
            let promoted = fusedAttentionRMS.callCombined(
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset
            )
            guard let previewCache = combinedCache.copy()
                as? Gemma4CombinedKVCache
            else {
                preconditionFailure("combined cache copy changed concrete type")
            }
            let preview = previewCache.updateCombined(promoted.combinedKV)
            let promotedAttention = MLXFast.scaledDotProductAttention(
                queries: promoted.queries,
                keys: preview.0,
                values: preview.1,
                scale: 1.0,
                mask: .none
            )
            // Force both complete raw-bit comparisons before mutating the live
            // cache. Verification always returns the promoted attention path.
            fusedSlidingPrepSDPA.verifyRawBits(
                candidate,
                promotedAttention: promotedAttention,
                promotedAppendKV: promoted.combinedKV
            )
            _ = combinedCache.updateCombined(promoted.combinedKV)
            return promotedAttention
        }

        _ = combinedCache.updateCombined(candidate.appendKV)
        return candidate.attention
    }
}
