import Foundation
import MLX
import Testing

// @testable on both: the stub subclasses BaseKVCache, whose initializer and
// maxSize are internal/non-open, and the probe builds a LagunaModel from an
// in-memory LagunaConfiguration.
@testable import MLXLLM
@testable import MLXLMCommon

// Regression tests for the sliding-window WRAP SEAM in DFlash rollback.
//
// Block decode was broken at the seam and the ranked window is exactly where it
// bites: Laguna's sliding window is 512 and the scored window is a 512-token
// seed plus 128 decode steps, so the first block already crosses the ring
// boundary. It went unnoticed through bring-up because every measurement used a
// 26-68 token prompt.
//
// Mechanism. `RotatingKVCache.isTrimmable` is `offset < maxCacheSize`, which is
// CORRECT and not merely conservative: after the ring wraps, rolling the offset
// back would need the entries the wrap just overwrote, and those are the oldest
// rows still inside the window. A wrapped cache must therefore be rolled back by
// snapshot and replay. The bug was that the snapshot decision happens BEFORE the
// block is written while the trim happens AFTER, so the one round that starts
// trimmable and ends wrapped got neither a snapshot nor a usable trim.
//
// These tests exercise the decision function directly with stub caches, so they
// need no model and no GPU and can run in CI.
@Suite
struct DFlashRollbackSeamTests {
    /// Minimal cache stub: only the properties the snapshot decision reads
    /// (`offset`, `maxSize`, `isTrimmable`) plus the protocol's required members.
    private final class SeamStubCache: BaseKVCache {
        private let capacity: Int?
        private var trimmedRows = 0

        init(offset: Int, maxSize: Int?) {
            self.capacity = maxSize
            super.init()
            self.offset = offset
        }

        override var maxSize: Int? { capacity }

        // Mirrors RotatingKVCache: a ring stops being trimmable once it wraps.
        override var isTrimmable: Bool {
            guard let capacity else { return true }
            return offset < capacity
        }

        override func trim(_ n: Int) -> Int {
            let trimmed = min(offset, n)
            offset -= trimmed
            trimmedRows += trimmed
            return trimmed
        }

        override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            (keys, values)
        }

        override func copy() -> any KVCache {
            SeamStubCache(offset: offset, maxSize: capacity)
        }
    }

    /// The decision under test is a `DFlashTargetModel` protocol extension, and
    /// `DFlashTargetModel: LLMModel`, so it needs a real conformer rather than a
    /// hand-rolled stub. A tiny `LagunaModel` built from an in-memory config is
    /// enough: `makeDefaultDFlashCacheRollbackState` only inspects the caches it
    /// is handed, so no weights, no tokenizer and no GPU work are involved.
    private static let tinyConfigJSON = """
        {
          "model_type": "laguna",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "max_position_embeddings": 4096,
          "rms_norm_eps": 1e-6,
          "attention_bias": false,
          "qkv_bias": false,
          "tie_word_embeddings": false,
          "rope_theta": 500000.0,
          "sliding_window": 8,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
          "gating": "per-head",
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "shared_expert_intermediate_size": 8,
          "moe_routed_scaling_factor": 1.0,
          "norm_topk_prob": true,
          "decoder_sparse_step": 1,
          "moe_router_logit_softcapping": 0.0
        }
        """

    private func decisionProbe() throws -> LagunaModel {
        LagunaModel(
            try JSONDecoder.json5().decode(
                LagunaConfiguration.self, from: Data(Self.tinyConfigJSON.utf8)
            )
        )
    }

    // A round that stays comfortably inside the ring needs no snapshot: rolling
    // back is just moving the offset, and copying the cache every round would be
    // pure overhead on the hot path.
    @Test
    func roundWellInsideTheRingNeedsNoSnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 100, maxSize: 512)]
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state == nil)
    }

    // THE REGRESSION. Trimmable at the start (500 < 512) but the block write
    // lands past the boundary (500 + 16 >= 512). Before the fix this returned
    // nil and the round then failed with untrimmableCache; it must now snapshot.
    @Test
    func roundThatCrossesTheRingBoundaryTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 500, maxSize: 512)]
        #expect(cache[0].isTrimmable, "precondition: trimmable BEFORE the write")

        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 16
        )
        #expect(
            state is DFlashCopiedTargetRollbackState,
            "a round that ends wrapped must carry a snapshot; without one it can neither trim nor replay"
        )
    }

    // Exactly-at-the-boundary is the same case: offset + width == maxSize means
    // the last written row occupies the final slot and the next write wraps, so
    // the conservative side is the correct side.
    @Test
    func roundLandingExactlyOnTheBoundaryTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 504, maxSize: 512)]
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state is DFlashCopiedTargetRollbackState)
    }

    // An already-wrapped cache was always handled (it reports untrimmable, so the
    // old code snapshotted too). Pinned so a future optimisation cannot regress
    // it back to trusting isTrimmable alone.
    @Test
    func alreadyWrappedCacheTakesASnapshot() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 900, maxSize: 512)]
        #expect(!cache[0].isTrimmable)
        let state = try decisionProbe().makeDefaultDFlashCacheRollbackState(
            cache: cache, plannedWriteCount: 8
        )
        #expect(state is DFlashCopiedTargetRollbackState)
    }

    // Laguna interleaves sliding-window and full-attention layers. A full
    // attention cache (no maxSize) never wraps, so the decision must be driven by
    // the ring-bounded members: one crossing cache forces the snapshot for all.
    @Test
    func mixedFullAttentionAndSlidingCachesFollowTheRingBoundedMember() throws {
        let unbounded = SeamStubCache(offset: 5_000, maxSize: nil)
        #expect(unbounded.isTrimmable, "an unbounded cache trims exactly")

        let safe: [KVCache] = [unbounded, SeamStubCache(offset: 100, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(
                cache: safe, plannedWriteCount: 8
            ) == nil
        )

        let crossing: [KVCache] = [unbounded, SeamStubCache(offset: 508, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(
                cache: crossing, plannedWriteCount: 8
            ) is DFlashCopiedTargetRollbackState
        )
    }

    // Callers that cannot know the width keep the old behaviour, so this fix is
    // additive: only rounds that declare a planned write get the new protection.
    @Test
    func omittingThePlannedWriteCountPreservesTheOldDecision() throws {
        let cache: [KVCache] = [SeamStubCache(offset: 500, maxSize: 512)]
        #expect(
            try decisionProbe().makeDefaultDFlashCacheRollbackState(cache: cache) == nil
        )
    }
}
