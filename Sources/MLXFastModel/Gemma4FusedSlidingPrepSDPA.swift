import Foundation
import MLX
import MLXLMCommon

private func gemma4SlidingPrepSDPAEnvironmentFlag(_ name: String) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4FusedSlidingPrepSDPAFeatureEnabled: Bool = {
    guard ProcessInfo.processInfo.environment[
        "DARKBLOOM_FUSED_SLIDING_PREP_SDPA"
    ] != nil else {
        // Enabled after raw-bit verification across all 50 sliding layers for
        // the complete scored 128-token decode window. Keep the environment
        // override so paired qualification can still force the reference arm.
        return true
    }
    return gemma4SlidingPrepSDPAEnvironmentFlag(
        "DARKBLOOM_FUSED_SLIDING_PREP_SDPA")
}()

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

/// First pass of a split exact clone of MLX's D=256 `sdpa_vector`. Two
/// 512-thread threadgroups own each KV head; each owns 16 of the original 32
/// SIMDgroup streams and computes both grouped-query heads from one K/V load.
private let gemma4FusedSlidingPrepSDPAKernel = MLXFast.metalKernel(
    name: "gemma4_fused_sliding_prep_sdpa_gqa2_split2_direct_256_v4",
    inputNames: [
        "raw_q", "raw_k", "raw_v", "q_weight", "k_weight", "position",
        "old_combined_kv", "old_length", "old_capacity", "attention_scale",
        "rope_cosines", "rope_sines",
    ],
    outputNames: ["partial_outputs", "max_scores", "sum_exp_scores"],
    source: """
        constexpr uint kHeadDim = 256;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 16;
        constexpr uint kGQAFactor = 2;
        constexpr uint kPrepReads = 4;
        constexpr uint kPrepThreads = 64;
        constexpr uint kSIMDSize = 32;
        constexpr uint kSIMDGroupsPerBlock = 16;
        constexpr uint kAttentionReads = 8;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint block = threadgroup_position_in_grid.y & 1u;
        const uint kv_head = threadgroup_position_in_grid.y >> 1;
        const uint query_head_0 = kv_head * kGQAFactor;
        const uint query_head_1 = query_head_0 + 1;
        const uint global_simd_group =
            block * kSIMDGroupsPerBlock + simd_group;

        threadgroup bfloat prepared_query_0[kHeadDim];
        threadgroup bfloat prepared_query_1[kHeadDim];
        threadgroup bfloat prepared_key[kHeadDim];
        threadgroup bfloat prepared_value[kHeadDim];
        threadgroup float prep_inverse_mean[1];

        gemma4_prepare_sliding_row_256(
            raw_q + query_head_0 * kHeadDim,
            q_weight,
            true,
            prepared_query_0,
            prep_inverse_mean,
            simd_group,
            simd_lane);
        gemma4_apply_sliding_rope_256(
            prepared_query_0,
            static_cast<uint>(position),
            rope_cosines,
            rope_sines,
            simd_group,
            simd_lane);

        gemma4_prepare_sliding_row_256(
            raw_q + query_head_1 * kHeadDim,
            q_weight,
            true,
            prepared_query_1,
            prep_inverse_mean,
            simd_group,
            simd_lane);
        gemma4_apply_sliding_rope_256(
            prepared_query_1,
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

        // SIMDgroup zero separately emulates the promoted two-SIMD trees for
        // Q0/Q1/K/V. Publish all four completed rows once before attention.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int cache_length = static_cast<int>(old_length);
        const int cache_capacity = static_cast<int>(old_capacity);
        const int key_count = cache_length + 1;
        device bfloat* combined_cache_mutable =
            const_cast<device bfloat*>(old_combined_kv);
        const size_t old_value_slab =
            static_cast<size_t>(kKVHeads) * cache_capacity * kHeadDim;
        const size_t old_head_base =
            static_cast<size_t>(kv_head) * cache_capacity * kHeadDim;

        // Block zero is the sole owner and writer of its KV head. Mirror the
        // promoted direct preparation path: write the supplied token straight
        // into the cache slab at the physical pre-wrap position, eliminating
        // both the 16 KiB staging output and the subsequent slice update.
        if (block == 0u && thread_index < kPrepThreads) {
            const uint base = thread_index * kPrepReads;
            for (uint index = 0; index < kPrepReads; ++index) {
                const uint dimension = base + index;
                combined_cache_mutable[
                    old_head_base
                    + static_cast<size_t>(cache_length) * kHeadDim
                    + dimension
                ] =
                    prepared_key[dimension];
                combined_cache_mutable[
                    old_value_slab + old_head_base
                    + static_cast<size_t>(cache_length) * kHeadDim
                    + dimension
                ] =
                    prepared_value[dimension];
            }
        }

        // Clone pinned sdpa_vector<bfloat,256,256>: 32 SIMDgroups each own
        // key positions gid, gid+32, ... and eight final output dimensions.
        thread float query_values_0[kAttentionReads];
        thread float query_values_1[kAttentionReads];
        thread float key_values[kAttentionReads];
        thread float output_values_0[kAttentionReads];
        thread float output_values_1[kAttentionReads];
        const uint lane_dimension = simd_lane * kAttentionReads;
        const float query_scale = static_cast<float>(attention_scale);
        for (uint index = 0; index < kAttentionReads; ++index) {
            query_values_0[index] = query_scale * static_cast<float>(
                prepared_query_0[lane_dimension + index]);
            query_values_1[index] = query_scale * static_cast<float>(
                prepared_query_1[lane_dimension + index]);
            output_values_0[index] = 0.0f;
            output_values_1[index] = 0.0f;
        }

        float max_score_0 = -metal::numeric_limits<float>::max();
        float max_score_1 = -metal::numeric_limits<float>::max();
        float sum_exp_score_0 = 0.0f;
        float sum_exp_score_1 = 0.0f;
        for (int key_index = static_cast<int>(global_simd_group);
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

            float score_0 = 0.0f;
            for (uint index = 0; index < kAttentionReads; ++index) {
                score_0 += query_values_0[index] * key_values[index];
            }
            score_0 = simd_sum(score_0);

            const float new_max_0 = max(max_score_0, score_0);
            const float factor_0 = metal::fast::exp(
                max_score_0 - new_max_0);
            const float exp_score_0 = metal::fast::exp(
                score_0 - new_max_0);
            max_score_0 = new_max_0;
            sum_exp_score_0 = sum_exp_score_0 * factor_0 + exp_score_0;

            float score_1 = 0.0f;
            for (uint index = 0; index < kAttentionReads; ++index) {
                score_1 += query_values_1[index] * key_values[index];
            }
            score_1 = simd_sum(score_1);

            const float new_max_1 = max(max_score_1, score_1);
            const float factor_1 = metal::fast::exp(
                max_score_1 - new_max_1);
            const float exp_score_1 = metal::fast::exp(
                score_1 - new_max_1);
            max_score_1 = new_max_1;
            sum_exp_score_1 = sum_exp_score_1 * factor_1 + exp_score_1;

            for (uint index = 0; index < kAttentionReads; ++index) {
                const uint dimension = lane_dimension + index;
                const float value = is_current
                    ? static_cast<float>(prepared_value[dimension])
                    : static_cast<float>(old_combined_kv[
                        old_value_slab
                        + old_head_base
                        + static_cast<size_t>(key_index) * kHeadDim
                        + dimension]);
                output_values_0[index] = output_values_0[index] * factor_0
                    + exp_score_0 * value;
                output_values_1[index] = output_values_1[index] * factor_1
                    + exp_score_1 * value;
            }
        }

        const size_t state_base_0 =
            (static_cast<size_t>(query_head_0) * kSIMDSize
                + global_simd_group) * kHeadDim;
        const size_t state_base_1 =
            (static_cast<size_t>(query_head_1) * kSIMDSize
                + global_simd_group) * kHeadDim;
        for (uint index = 0; index < kAttentionReads; ++index) {
            partial_outputs[state_base_0 + lane_dimension + index] =
                output_values_0[index];
            partial_outputs[state_base_1 + lane_dimension + index] =
                output_values_1[index];
        }
        if (simd_lane == 0) {
            const size_t scalar_base_0 =
                static_cast<size_t>(query_head_0) * kSIMDSize
                    + global_simd_group;
            const size_t scalar_base_1 =
                static_cast<size_t>(query_head_1) * kSIMDSize
                    + global_simd_group;
            max_scores[scalar_base_0] = max_score_0;
            max_scores[scalar_base_1] = max_score_1;
            sum_exp_scores[scalar_base_0] = sum_exp_score_0;
            sum_exp_scores[scalar_base_1] = sum_exp_score_1;
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
            // lane owns the same four elements as SIMDgroups zero and one.
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

            // Feed the two sums into lanes zero/one of the original zero-
            // padded 32-lane second stage without changing its reduction tree.
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

/// Reconstruct the original 32-way `sdpa_vector` final reductions from the
/// FP32 online-softmax states emitted under their original SIMDgroup IDs.
private let gemma4FusedSlidingPrepSDPAMergeKernel = MLXFast.metalKernel(
    name: "gemma4_fused_sliding_prep_sdpa_gqa2_split2_merge_256_v1",
    inputNames: ["partial_states", "max_states", "sum_exp_states"],
    outputNames: ["attention"],
    source: """
        constexpr uint kHeadDim = 256;
        constexpr uint kSIMDSize = 32;
        constexpr uint kAttentionReads = 8;

        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint lane_dimension = simd_lane * kAttentionReads;
        const size_t scalar_base =
            static_cast<size_t>(query_head) * kSIMDSize;
        const size_t state_base =
            (scalar_base + simd_group) * kHeadDim + lane_dimension;

        thread float output_values[kAttentionReads];
        for (uint index = 0; index < kAttentionReads; ++index) {
            output_values[index] = partial_states[state_base + index];
        }

        float max_score = max_states[scalar_base + simd_lane];
        const float new_max = simd_max(max_score);
        const float factor = metal::fast::exp(max_score - new_max);
        const float sum_exp_score = simd_sum(
            sum_exp_states[scalar_base + simd_lane] * factor);

        threadgroup float partial_outputs[kSIMDSize * kSIMDSize];
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
    header: "using namespace metal;",
    ensureRowContiguous: true
)

struct Gemma4FusedSlidingPrepSDPAResult {
    let attention: MLXArray
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
              preparation.positionViews.count > 1_024,
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
            && snapshot.capacity > snapshot.length
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
                positionViews[snapshot.length],
                positionViews[snapshot.capacity], attentionScale,
                ropeCosines, ropeSines,
            ],
            grid: (512, 32, 1),
            threadGroup: (512, 1, 1),
            outputShapes: [
                [32, 32, 256], [32, 32], [32, 32],
            ],
            outputDTypes: [.float32, .float32, .float32]
        )
        let attention = gemma4FusedSlidingPrepSDPAMergeKernel(
            [outputs[0], outputs[1], outputs[2]],
            grid: (1_024, 32, 1),
            threadGroup: (1_024, 1, 1),
            outputShapes: [[1, 32, 1, 256]],
            outputDTypes: [.bfloat16]
        )
        return Gemma4FusedSlidingPrepSDPAResult(attention: attention[0])
    }

    func verifyRawBits(
        _ candidate: Gemma4FusedSlidingPrepSDPAResult,
        candidateCacheStorage: MLXArray,
        candidateWritePosition: Int,
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
        // Sequence the undeclared cache side effect behind an output of the
        // candidate first pass, exactly as the promoted direct-write verifier
        // does. Verifier mode fully materializes its staging reference before
        // this read observes the candidate's live direct write.
        let sequencingZero = candidate.attention.sum().asType(.int32) * Int32(0)
        let candidateAppendBits = candidateCacheStorage[
            0..., 0..., 0...,
            candidateWritePosition..<(candidateWritePosition + 1), 0...
        ].view(dtype: .uint16).asType(.int32) + sequencingZero
        let promotedAppendBits = promotedAppendKV.view(dtype: .uint16)
            .asType(.int32) + sequencingZero
        let appendMatches = arrayEqual(candidateAppendBits, promotedAppendBits)
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
              let directTarget = combinedCache.directDecodeWriteTarget(),
              directTarget.position == snapshot.length,
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
            // Materialize the complete staging reference before the candidate
            // mutates the live cache slab as an undeclared Metal side effect.
            eval(promotedAttention, promoted.combinedKV)

            let candidate = fusedSlidingPrepSDPA(
                rawQueries: rawQueries,
                rawKeys: rawKeys,
                rawValues: rawValues,
                offset: offset,
                snapshot: snapshot
            )
            _ = combinedCache.adoptDirectDecodeWrite(
                position: directTarget.position
            )
            fusedSlidingPrepSDPA.verifyRawBits(
                candidate,
                candidateCacheStorage: directTarget.storage,
                candidateWritePosition: directTarget.position,
                promotedAttention: promotedAttention,
                promotedAppendKV: promoted.combinedKV
            )
            return candidate.attention
        }

        let candidate = fusedSlidingPrepSDPA(
            rawQueries: rawQueries,
            rawKeys: rawKeys,
            rawValues: rawValues,
            offset: offset,
            snapshot: snapshot
        )
        _ = combinedCache.adoptDirectDecodeWrite(position: directTarget.position)
        return candidate.attention
    }
}
