import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative

/// One block-decode round as the trusted parent sees it.
///
/// `tokens` is the target-confirmed block: the accepted draft prefix followed by
/// one target token (so `1...maxBlockSize` entries). `targetCacheOffset` is the
/// physical KV position after the round, which the worker cross-checks against
/// its own logical ledger before answering.
public struct LagunaDFlashBlockResult {
    public let tokens: [Int]
    public let targetCacheOffset: Int
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    public let declaredRows: Int
    public let perRowHiddenDigest: [String]
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
}

/// Re-entrant DFlash block-decode session for the `laguna-xs-2.1-dflash-v1`
/// track's runtime worker.
///
/// The vendored DFlash entry points (`DFlashTokenIterator`,
/// `generateDFlashTokens`) are one-shot: they own the whole generation loop and
/// return when it finishes. The ranked protocol is the opposite shape -- the
/// trusted parent drives one `dflash_decode_block` request at a time, chooses
/// each round's width, and owns the token budget and the timer. This session is
/// the adapter: it holds the target/draft caches, the current bonus token and
/// captured target hidden states across requests, and exposes exactly two
/// operations (`begin`, `generateBlock`).
///
/// Everything numeric is delegated to `runDFlashGreedyRound`, which is the same
/// round the validated `mlx-bench dflash` path runs, so the ranked worker and
/// the local bench cannot drift apart.
public final class LagunaDFlashBlockSession {
    private let target: any DFlashTargetModel
    private let drafter: DFlashDraftModel
    private var targetCache: [KVCache]
    private let draftCache: [KVCache]

    private var bonus: Int?
    private var targetHidden: MLXArray?

    /// Seed prompt length; the physical KV offset must stay `seedTokenCount +
    /// decodedTokenCount` for the whole session.
    public private(set) var seedTokenCount = 0
    public private(set) var decodedTokenCount = 0
    public private(set) var rollbackRoundCount = 0
    public private(set) var acceptedDraftTotal = 0
    public private(set) var rejectedDraftTotal = 0
    public private(set) var roundCount = 0

    /// Every round emits exactly one target-produced tail token (the accepted
    /// draft prefix plus one). The L3 ledger the box wrapper checks is
    /// `accepted + rejected + target_tail == declared_rows`, so the tail total
    /// is the round count by construction.
    public var targetTailTotal: Int { roundCount }

    /// Reference-side accessors, used ONLY by the reference worker built from
    /// the pinned baseline tree when it serves `dflash_reference_rows`. The
    /// candidate never reaches this kind.
    public var referenceTarget: any DFlashTargetModel { target }
    public var referenceTargetLayerIds: [Int] { drafter.config.targetLayerIds }

    /// When true every round also produces the Criterion E work-binding
    /// readouts. This costs a full-logits verify forward instead of the greedy
    /// fast path, which is intentional: the ranked contract requires the per-row
    /// lm_head to have actually run.
    private let collectWorkBinding: Bool

    public init(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        collectWorkBinding: Bool = true
    ) throws {
        self.target = target
        self.drafter = drafter
        self.collectWorkBinding = collectWorkBinding
        self.targetCache = target.newCache(parameters: nil)
        self.draftCache = try drafter.makeCache()
        // The round trims the draft cache back to the committed offset every
        // time, so an untrimmable draft cache would silently desynchronize the
        // drafter from the target. Fail at construction instead.
        guard canTrimPromptCache(self.draftCache) else {
            throw MLXFastError.invalidInput(
                "DFlash draft cache is not trimmable; block decode cannot "
                    + "keep the drafter aligned with the target"
            )
        }
    }

    /// Untimed, input-independent warm of the block-decode graph shapes.
    ///
    /// Called after the trusted allocator clear and before the first timed
    /// round so the scored window does not pay a one-time first-touch spike.
    /// Runs on throwaway cache state -- never on the scored seed.
    public func warmWorkingSetAfterAllocatorReset() throws {
        let warmupSession = try LagunaDFlashBlockSession(
            target: target,
            drafter: drafter,
            collectWorkBinding: collectWorkBinding
        )
        // Deliberately SHORT. Warming compiles kernel shapes and must not
        // straddle the rotating sliding-window cache's wrap seam: a seed at the
        // window size plus a widest-block verify pushes rejected rows past the
        // ring, after which rollback cannot trim them and the round throws
        // `untrimmableCache`. The seam is a real hazard for the scored window
        // (contract layer L4); a warmup must not be what discovers it.
        let seed = try warmupSession.begin(
            seedTokens: Array(
                repeating: 2,
                count: MLXFastConstants.experimentalDFlashWarmupSeedTokens
            )
        )
        // Warm the widest legal block so a later narrow round cannot be the
        // first to compile its shape.
        _ = try warmupSession.generateBlock(
            previousCommittedToken: seed,
            maxBlockSize: MLXFastConstants.experimentalDFlashMaxBlockSize
        )
    }

    /// Seed prefill. Returns the post-prefill argmax ("bonus") token, which is
    /// the first emitted token of the run and the `previousCommittedToken` the
    /// first block request must echo back.
    public func begin(seedTokens: [Int]) throws -> Int {
        guard bonus == nil else {
            throw MLXFastError.invalidInput(
                "DFlash block session was already begun"
            )
        }
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "DFlash block session requires a non-empty seed"
            )
        }
        guard seedTokens.allSatisfy({ $0 >= 0 && $0 < MLXFastConstants.vocabSize })
        else {
            throw MLXFastError.invalidInput(
                "DFlash seed contains an out-of-vocabulary token"
            )
        }

        let promptTokens = MLXArray(seedTokens.map { Int32($0) })[.newAxis, .ellipsis]
        let prefill = try target.forwardGreedyTokensForDFlash(
            promptTokens,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let bonusArray = prefill.tokens[0..., -1]
        eval(bonusArray, prefill.targetHidden)
        let seedToken = Int(bonusArray.item(Int32.self))
        guard seedToken >= 0, seedToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "DFlash seed prefill produced an out-of-vocabulary token"
            )
        }

        bonus = seedToken
        targetHidden = prefill.targetHidden
        seedTokenCount = seedTokens.count
        decodedTokenCount = 0
        return seedToken
    }

    /// One target-verified block. `previousCommittedToken` must be the last
    /// token this session emitted -- the parent echoing it back is what proves
    /// the two sides agree on the committed prefix.
    public func generateBlock(
        previousCommittedToken: Int,
        maxBlockSize: Int
    ) throws -> LagunaDFlashBlockResult {
        guard let currentBonus = bonus, let currentHidden = targetHidden else {
            throw MLXFastError.invalidInput(
                "DFlash block requested before begin"
            )
        }
        guard previousCommittedToken == currentBonus else {
            throw MLXFastError.invalidInput(
                "DFlash block request echoed token \(previousCommittedToken) "
                    + "but the session's committed token is \(currentBonus)"
            )
        }
        guard maxBlockSize >= 1,
              maxBlockSize <= MLXFastConstants.experimentalDFlashMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "DFlash block size \(maxBlockSize) is outside 1..."
                    + "\(MLXFastConstants.experimentalDFlashMaxBlockSize)"
            )
        }

        // K=1 is the SERIAL CONTROL the paired score divides by. It runs the
        // same worker, the same protocol and the same target forward as a real
        // block round -- just one row and zero drafts -- so the denominator
        // measures this implementation at width 1 rather than some other
        // code path. `runDFlashGreedyRound` needs at least one draft, so the
        // single-row case is handled here.
        if maxBlockSize == 1 {
            return try generateSerialRow(previousCommittedToken: currentBonus)
        }

        let result = try runDFlashGreedyRound(
            target: target,
            drafter: drafter,
            targetCache: &targetCache,
            draftCache: draftCache,
            bonus: currentBonus,
            targetHidden: currentHidden,
            promptTokenCount: seedTokenCount,
            generatedTokenCount: decodedTokenCount,
            blockSize: maxBlockSize,
            maxEmitCount: maxBlockSize,
            workBinding: collectWorkBinding
        )

        guard !result.tokens.isEmpty, result.tokens.count <= maxBlockSize else {
            throw MLXFastError.invalidInput(
                "DFlash round emitted \(result.tokens.count) tokens for a "
                    + "block of \(maxBlockSize)"
            )
        }
        guard result.tokens.allSatisfy({
            $0 >= 0 && $0 < MLXFastConstants.vocabSize
        }) else {
            throw MLXFastError.invalidInput(
                "DFlash round emitted an out-of-vocabulary token"
            )
        }

        bonus = result.bonus
        targetHidden = result.targetHidden
        decodedTokenCount += result.tokens.count

        // Ledger: physical KV position must equal the logical one. A submission
        // that leaves rejected rows resident (skipping rollback to save time)
        // trips here before its tokens are ever scored.
        let observedOffset = targetCache.first?.offset ?? -1
        let expectedOffset = seedTokenCount + decodedTokenCount
        guard observedOffset == expectedOffset else {
            throw MLXFastError.invalidInput(
                "DFlash target cache offset \(observedOffset) diverged from the "
                    + "logical position \(expectedOffset)"
            )
        }

        let rejected = Swift.max(0, (maxBlockSize - 1) - result.accepted)
        if rejected > 0 {
            rollbackRoundCount += 1
        }
        acceptedDraftTotal += result.accepted
        rejectedDraftTotal += rejected
        roundCount += 1

        let binding = result.workBinding
        return LagunaDFlashBlockResult(
            tokens: result.tokens,
            targetCacheOffset: observedOffset,
            acceptedDraftCount: result.accepted,
            rejectedDraftCount: rejected,
            declaredRows: binding?.declaredRows ?? 0,
            perRowHiddenDigest: binding?.hiddenDigest ?? [],
            perRowTop2Tokens: binding?.top2Tokens ?? [],
            perRowTop2Logits: binding?.top2Logits ?? []
        )
    }

    /// One width-1 target row: the serial control. Zero drafts, one declared
    /// row, nothing to roll back.
    ///
    /// The full-logits forward is deliberate even though only the argmax is
    /// needed: the control has to produce the same per-row top-2 readouts the
    /// block path produces, or the parent could not compare the two sides
    /// against the same reference.
    private func generateSerialRow(
        previousCommittedToken: Int
    ) throws -> LagunaDFlashBlockResult {
        let input = MLXArray([Int32(previousCommittedToken)])[.newAxis, .ellipsis]
        let out = try target.forwardForDFlash(
            input,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let logitRow = out.logits[0, -1, 0...]
        let (top2Tokens, top2Logits) = LagunaDFlashReference.topTwo(of: logitRow)
        eval(out.targetHidden)
        guard let next = top2Tokens.first,
              next >= 0,
              next < MLXFastConstants.vocabSize
        else {
            throw MLXFastError.invalidInput(
                "DFlash serial row produced no in-vocabulary token"
            )
        }

        bonus = next
        targetHidden = out.targetHidden
        decodedTokenCount += 1
        roundCount += 1

        let observedOffset = targetCache.first?.offset ?? -1
        let expectedOffset = seedTokenCount + decodedTokenCount
        guard observedOffset == expectedOffset else {
            throw MLXFastError.invalidInput(
                "DFlash serial row target cache offset \(observedOffset) "
                    + "diverged from the logical position \(expectedOffset)"
            )
        }

        return LagunaDFlashBlockResult(
            tokens: [next],
            targetCacheOffset: observedOffset,
            acceptedDraftCount: 0,
            rejectedDraftCount: 0,
            declaredRows: 1,
            // Amendment 1: hidden digests are a SELF-consistency instrument
            // only, never compared across builds, so the control does not pay
            // to materialize one.
            perRowHiddenDigest: [""],
            perRowTop2Tokens: [top2Tokens],
            perRowTop2Logits: [top2Logits]
        )
    }
}
