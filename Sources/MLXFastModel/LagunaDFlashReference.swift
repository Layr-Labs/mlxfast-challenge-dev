import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative

/// One reference verdict row, as computed by the REFERENCE worker.
///
/// Contract layer L1: these values come from a worker built out of the pinned
/// baseline tree, loading organizer-transformed weights, running strictly after
/// the timed window. The candidate never computes them and never sees them.
public struct LagunaDFlashReferenceRow: Sendable {
    /// Reference argmax computed in a genuine width-1 decode frame.
    public let sequentialArgmax: Int
    /// Reference argmax for the same position computed in a block-shaped frame
    /// whose width is the number of rows in this request.
    public let blockArgmax: Int
    /// Top-2 token ids at this position, highest logit first (width-1 frame).
    public let top2Tokens: [Int]
    /// Top-2 logit VALUES aligned with `top2Tokens`. Amendment 1 of the
    /// contract: these -- not the hidden-state digest -- are the cross-build
    /// work binder, compared with a tolerance.
    public let top2Logits: [Double]
}

/// One reference-rows answer: the width-1 rows plus every block frame that was
/// replayed for the same positions.
public struct LagunaDFlashReferenceRows: Sendable {
    public let rows: [LagunaDFlashReferenceRow]
    /// Replayed block widths, ascending. Always contains `rows.count`.
    public let frameWidths: [Int]
    /// `frameArgmax[i]` is the width-`frameWidths[i]` argmax for the SAME
    /// `rows.count` positions, so every entry has `rows.count` elements.
    public let frameArgmax: [[Int]]
    /// True when this answer was served by walking the live continuous cache
    /// forward, false when the frame had to be rebuilt from scratch first.
    public let continuedLiveFrame: Bool
}

/// Reference-row computation for the DFlash track.
///
/// STATEFUL, and that is the whole point. The candidate builds its state one
/// way and only one way: `LagunaDFlashBlockSession.begin` runs a single bulk
/// forward over the SEED, and every position after the seed is a separate
/// one-token forward on a cache that is never rebuilt. A reference that instead
/// bulk-forwards `tokens[0 ..< startOffset]` before single-stepping is running a
/// DIFFERENT computation the moment `startOffset` passes the seed: the
/// accumulation order differs, and for the sliding-window `RotatingKVCache` it
/// is a different code path entirely (`updateConcat` for a multi-row write
/// versus `updateInPlace` for a single row). At a near tie the two disagree, so
/// the "sequential" argmax stops being what a sequential decoder produces --
/// which is how goldens came to contain rows whose `sequential_argmax`
/// contradicted the very chain they were generated from, and how the honest K=1
/// serial control came to be rejected on its own reference.
///
/// So this type keeps ONE continuously-advanced cache and walks it forward token
/// by token, exactly as the candidate does. Requests arrive in order, so the
/// common case is a pure continuation, which is also O(n) over a run instead of
/// the O(n^2) a per-round bulk re-prefill costs. An out-of-order request (a
/// deliberate replay, a new chain) falls back to rebuilding from scratch --
/// using the SAME seed-bulk-then-single-step construction, because a fallback
/// that re-prefilled the whole prefix would reintroduce the defect it exists to
/// tolerate.
public final class LagunaDFlashReference {
    private let target: any DFlashTargetModel
    private let targetLayerIds: [Int]

    /// The one cache every width-1 row is produced on.
    private var liveCache: [KVCache]?
    /// Length of the seed prefix `liveCache` was BULK-forwarded over. Every
    /// position after it was applied one row at a time.
    private var seedPrefixLength = -1
    /// The tokens already applied to `liveCache`, i.e. `tokens[0 ..< offset]`.
    /// Kept in full so a request can be checked against the exact prefix the
    /// cache holds rather than against a position counter alone.
    private var walkedTokens = [Int]()

    /// How many times the frame had to be rebuilt from scratch. Zero-cost
    /// audit signal: a run whose row requests arrive in order rebuilds once.
    public private(set) var rebuildCount = 0

    public init(target: any DFlashTargetModel, targetLayerIds: [Int]) {
        self.target = target
        self.targetLayerIds = targetLayerIds
    }

    /// Seed prefill, in the candidate's frame.
    ///
    /// Returns the post-prefill argmax -- the run's seed token -- computed the
    /// way `LagunaDFlashBlockSession.begin` computes it: ONE bulk forward over
    /// the whole seed, argmax of the last row. Bulk-forwarding all but the last
    /// seed token and then single-stepping it is a different computation and can
    /// disagree at a near tie, which would fail the seed check for a correct
    /// candidate.
    ///
    /// Leaves the continuous frame positioned at `seedTokens.count`, ready for
    /// the first row request.
    @discardableResult
    public func prefillSeed(_ seedTokens: [Int]) throws -> Int {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "DFlash reference seed prefill requires a non-empty seed"
            )
        }
        guard seedTokens.allSatisfy({
            $0 >= 0 && $0 < MLXFastConstants.vocabSize
        }) else {
            throw MLXFastError.invalidInput(
                "DFlash reference seed contains an out-of-vocabulary token"
            )
        }

        let cache = target.newCache(parameters: nil)
        let prompt = MLXArray(seedTokens.map { Int32($0) })[.newAxis, .ellipsis]
        let out = try target.forwardForDFlash(
            prompt,
            cache: cache,
            targetLayerIds: targetLayerIds
        )
        let lastRow = out.logits.argMax(axis: -1)[0..., -1]
        eval(lastRow, out.targetHidden)
        let seedToken = Int(lastRow.item(Int32.self))
        guard seedToken >= 0, seedToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "DFlash reference seed prefill produced an out-of-vocabulary "
                    + "token"
            )
        }

        liveCache = cache
        seedPrefixLength = seedTokens.count
        walkedTokens = seedTokens
        return seedToken
    }

    /// Reference rows for `count` positions starting at `startOffset`.
    ///
    /// `tokens` is the FULL token sequence (seed prompt, the post-prefill seed
    /// token, then every emitted token) and `seedTokenCount` says how much of
    /// its head is the SEED -- the only span the candidate ever bulk-forwards.
    /// `startOffset` is the absolute index in `tokens` of the INPUT token for
    /// the first requested row, so row `j`:
    ///   * is fed `tokens[startOffset + j]`,
    ///   * sees context `tokens[0 ... startOffset + j]`,
    ///   * predicts the token the candidate emitted at `startOffset + j + 1`.
    ///
    /// Two kinds of frame come back for the same positions:
    ///   * width-1: `count` genuine one-token decode steps on the continuous
    ///     cache. This is the frame the serial control measures.
    ///   * width-`w` for every `w` in `count ... widestFrame`: one block-shaped
    ///     forward per width, each on its own `copy()` of the continuous cache
    ///     taken at the round boundary. Branching rather than re-prefilling is
    ///     what keeps the block frame a sibling of the width-1 frame instead of
    ///     a third, unrelated construction.
    ///
    /// Known approximations, stated because they bound what this can prove.
    /// The candidate's round may have verified a WIDER block (its rejected
    /// drafts occupied the tail rows); those draft ids are not emitted, so the
    /// reference cannot reconstruct that exact width, and it replays the widths
    /// the parent's schedule could have asked for instead. And the branch is
    /// taken off a cache written by single rows, whereas the candidate's block
    /// cache was written by earlier block forwards. Anything the frames do not
    /// cover falls into the capped residual bucket and must still satisfy the
    /// top-2 binding.
    public func rows(
        tokens: [Int],
        seedTokenCount: Int,
        startOffset: Int,
        count: Int,
        widestFrame: Int
    ) throws -> LagunaDFlashReferenceRows {
        guard count > 0 else {
            throw MLXFastError.invalidInput(
                "DFlash reference request needs at least one row"
            )
        }
        guard widestFrame >= count else {
            throw MLXFastError.invalidInput(
                "DFlash reference widest frame \(widestFrame) is narrower than "
                    + "the \(count) requested rows"
            )
        }
        guard seedTokenCount > 0 else {
            throw MLXFastError.invalidInput(
                "DFlash reference request needs a positive seed length"
            )
        }
        guard startOffset >= seedTokenCount else {
            throw MLXFastError.invalidInput(
                "DFlash reference request starts at \(startOffset), inside the "
                    + "\(seedTokenCount)-token seed the candidate bulk-prefills"
            )
        }
        guard startOffset + widestFrame <= tokens.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference request needs tokens[\(startOffset)..<"
                    + "\(startOffset + widestFrame)] but only \(tokens.count) "
                    + "were supplied"
            )
        }
        guard tokens.allSatisfy({ $0 >= 0 && $0 < MLXFastConstants.vocabSize })
        else {
            throw MLXFastError.invalidInput(
                "DFlash reference context contains an out-of-vocabulary token"
            )
        }

        let (cache, continued) = try continuousFrame(
            tokens: tokens,
            seedTokenCount: seedTokenCount,
            position: startOffset
        )

        // --- block frames: branch off the continuous cache -------------------
        // Taken BEFORE the width-1 walk advances anything, so every copy is the
        // cache exactly as it stood at the round boundary.
        var frameWidths = [Int]()
        var frameArgmax = [[Int]]()
        for width in count ... widestFrame where width > 1 {
            let branch = cache.map { $0.copy() }
            let inputs = MLXArray(
                tokens[startOffset ..< (startOffset + width)].map { Int32($0) }
            )[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                inputs,
                cache: branch,
                targetLayerIds: targetLayerIds
            )
            let ids = out.logits.argMax(axis: -1)
            eval(ids)
            let flattened = ids.reshaped([-1]).asArray(Int32.self).map { Int($0) }
            guard flattened.count >= count else {
                throw MLXFastError.invalidInput(
                    "DFlash reference width-\(width) frame produced "
                        + "\(flattened.count) rows for \(count) requested"
                )
            }
            frameWidths.append(width)
            frameArgmax.append(Array(flattened.prefix(count)))
        }

        // --- width-1 frame: walk the continuous cache forward ---------------
        var sequentialArgmax = [Int]()
        var top2Tokens = [[Int]]()
        var top2Logits = [[Double]]()
        sequentialArgmax.reserveCapacity(count)
        top2Tokens.reserveCapacity(count)
        top2Logits.reserveCapacity(count)
        for index in 0 ..< count {
            let token = tokens[startOffset + index]
            let step = MLXArray([Int32(token)])[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                step,
                cache: cache,
                targetLayerIds: targetLayerIds
            )
            let logitRow = out.logits[0, -1, 0...]
            let (ids, values) = Self.topTwo(of: logitRow)
            sequentialArgmax.append(ids.first ?? 0)
            top2Tokens.append(ids)
            top2Logits.append(values)
            walkedTokens.append(token)
        }

        if count == 1 {
            // A one-row block frame IS the width-1 frame -- same input, same
            // cache state, same kernel -- so it is recorded rather than
            // recomputed on a copy.
            frameWidths.insert(1, at: 0)
            frameArgmax.insert(sequentialArgmax, at: 0)
        }
        guard let blockIndex = frameWidths.firstIndex(of: count) else {
            throw MLXFastError.invalidInput(
                "DFlash reference did not replay the width-\(count) frame it "
                    + "was asked for"
            )
        }
        let blockArgmax = frameArgmax[blockIndex]

        let rows = (0 ..< count).map { index in
            LagunaDFlashReferenceRow(
                sequentialArgmax: sequentialArgmax[index],
                blockArgmax: blockArgmax[index],
                top2Tokens: top2Tokens[index],
                top2Logits: top2Logits[index]
            )
        }
        return LagunaDFlashReferenceRows(
            rows: rows,
            frameWidths: frameWidths,
            frameArgmax: frameArgmax,
            continuedLiveFrame: continued
        )
    }

    /// The continuous width-1 cache, positioned at `position`.
    ///
    /// Requests arrive in order, so the assertion this makes is that the live
    /// cache already holds exactly `tokens[0 ..< position]`. When it does the
    /// cache is handed back untouched -- no re-prefill, no recomputation. When
    /// it does not (an out-of-order replay, a different chain, a different seed
    /// length) the frame is rebuilt the ONLY admissible way: one bulk forward
    /// over the seed, then one single-token forward per position after it.
    private func continuousFrame(
        tokens: [Int],
        seedTokenCount: Int,
        position: Int
    ) throws -> ([KVCache], Bool) {
        if let liveCache,
           seedPrefixLength == seedTokenCount,
           walkedTokens.count == position,
           walkedTokens.elementsEqual(tokens[0 ..< position])
        {
            return (liveCache, true)
        }

        rebuildCount += 1
        fputs(
            "dflash-reference: rebuilding the width-1 frame at position "
                + "\(position) (seed \(seedTokenCount), live position "
                + "\(walkedTokens.count), rebuild \(rebuildCount))\n",
            stderr
        )

        let cache = target.newCache(parameters: nil)
        let seed = MLXArray(
            tokens[0 ..< seedTokenCount].map { Int32($0) }
        )[.newAxis, .ellipsis]
        let prefill = try target.forwardForDFlash(
            seed,
            cache: cache,
            targetLayerIds: targetLayerIds
        )
        eval(prefill.logits)
        var walked = Array(tokens[0 ..< seedTokenCount])
        for index in seedTokenCount ..< position {
            let step = MLXArray([Int32(tokens[index])])[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                step,
                cache: cache,
                targetLayerIds: targetLayerIds
            )
            eval(out.logits)
            walked.append(tokens[index])
        }

        liveCache = cache
        seedPrefixLength = seedTokenCount
        walkedTokens = walked
        return (cache, false)
    }

    /// Top-2 ids and values for one logit row, using the same `argPartition`
    /// extraction the vendored DFlash parity check and the candidate-side work
    /// binding use, so the two sides are compared like for like.
    ///
    /// Shared with the serial K=1 control path in `LagunaDFlashBlockSession`,
    /// which must produce readouts the reference can be compared against.
    public static func topTwo(of logitRow: MLXArray) -> ([Int], [Double]) {
        let limit = Swift.max(1, Swift.min(2, logitRow.dim(-1)))
        let indices = argPartition(-logitRow, kth: limit - 1, axis: -1)[0 ..< limit]
        let scores = logitRow[indices]
        eval(indices, scores)
        let ids = indices.asArray(Int32.self).map { Int($0) }
        let values = scores.asArray(Float.self).map { Double($0) }
        let ordered = zip(ids, values).sorted { $0.1 > $1.1 }
        return (ordered.map(\.0), ordered.map(\.1))
    }
}
