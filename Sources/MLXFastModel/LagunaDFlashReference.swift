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

/// Stateless reference-row computation for the DFlash track.
///
/// Deliberately stateless: every request rebuilds its own KV cache from the
/// supplied context. That is what makes the L1 self-consistency replay
/// meaningful -- two identical requests must return byte-identical rows with no
/// carried state to explain a difference away.
public enum LagunaDFlashReference {
    /// Reference rows for `count` positions.
    ///
    /// `tokens` is the FULL token sequence (seed prompt, the post-prefill seed
    /// token, then every emitted token). `startOffset` is the absolute index in
    /// `tokens` of the INPUT token for the first requested row, so row `j`:
    ///   * is fed `tokens[startOffset + j]`,
    ///   * sees context `tokens[0 ... startOffset + j]`,
    ///   * predicts the token the candidate emitted at `startOffset + j + 1`.
    ///
    /// Two frames are computed for the same positions:
    ///   * width-1: `count` genuine one-token decode steps. This is the frame the
    ///     serial control measures, and it is NOT the same computation as one
    ///     wide teacher-forced forward -- the accumulation order differs, which
    ///     is exactly the divergence Criterion E absorbs.
    ///   * width-`count`: one block-shaped forward over the same input rows,
    ///     reproducing the block frame the candidate declared.
    ///
    /// Known approximation, stated because it bounds what this can prove: the
    /// candidate's round may have verified a WIDER block (its rejected drafts
    /// occupied the tail rows). Those draft ids are not emitted, so the reference
    /// cannot reconstruct that exact width. Replaying at the emitted width
    /// reproduces the block-shaped (width > 1) computation for exactly the rows
    /// being scored, which is the divergence class that matters; anything the
    /// two frames do not cover falls into the capped residual bucket and must
    /// still satisfy the top-2 binding.
    public static func rows(
        target: any DFlashTargetModel,
        targetLayerIds: [Int],
        tokens: [Int],
        startOffset: Int,
        count: Int
    ) throws -> [LagunaDFlashReferenceRow] {
        guard count > 0 else { return [] }
        guard startOffset >= 0 else {
            throw MLXFastError.invalidInput(
                "DFlash reference request has a negative start offset"
            )
        }
        guard startOffset + count <= tokens.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference request needs tokens[\(startOffset)..<"
                    + "\(startOffset + count)] but only \(tokens.count) were "
                    + "supplied"
            )
        }
        guard tokens.allSatisfy({ $0 >= 0 && $0 < MLXFastConstants.vocabSize })
        else {
            throw MLXFastError.invalidInput(
                "DFlash reference context contains an out-of-vocabulary token"
            )
        }

        let inputRows = Array(tokens[startOffset ..< (startOffset + count)])
        let context = Array(tokens[0 ..< startOffset])

        // --- width-1 frame -------------------------------------------------
        var sequentialArgmax = [Int]()
        var top2Tokens = [[Int]]()
        var top2Logits = [[Double]]()
        sequentialArgmax.reserveCapacity(count)
        top2Tokens.reserveCapacity(count)
        top2Logits.reserveCapacity(count)

        var stepCache = target.newCache(parameters: nil)
        if !context.isEmpty {
            let prefill = MLXArray(context.map { Int32($0) })[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                prefill,
                cache: stepCache,
                targetLayerIds: targetLayerIds
            )
            eval(out.logits)
        }
        for token in inputRows {
            let step = MLXArray([Int32(token)])[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                step,
                cache: stepCache,
                targetLayerIds: targetLayerIds
            )
            let logitRow = out.logits[0, -1, 0...]
            let (ids, values) = topTwo(of: logitRow)
            sequentialArgmax.append(ids.first ?? 0)
            top2Tokens.append(ids)
            top2Logits.append(values)
        }

        // --- width-`count` block frame -------------------------------------
        var blockArgmax = [Int]()
        blockArgmax.reserveCapacity(count)
        var blockCache = target.newCache(parameters: nil)
        if !context.isEmpty {
            let prefill = MLXArray(context.map { Int32($0) })[.newAxis, .ellipsis]
            let out = try target.forwardForDFlash(
                prefill,
                cache: blockCache,
                targetLayerIds: targetLayerIds
            )
            eval(out.logits)
        }
        let blockInput = MLXArray(inputRows.map { Int32($0) })[.newAxis, .ellipsis]
        let blockOut = try target.forwardForDFlash(
            blockInput,
            cache: blockCache,
            targetLayerIds: targetLayerIds
        )
        let blockIds = blockOut.logits.argMax(axis: -1)
        eval(blockIds)
        let flattened = blockIds.reshaped([-1]).asArray(Int32.self).map { Int($0) }
        guard flattened.count >= count else {
            throw MLXFastError.invalidInput(
                "DFlash reference block frame produced \(flattened.count) rows "
                    + "for \(count) requested"
            )
        }
        blockArgmax = Array(flattened.prefix(count))

        return (0 ..< count).map { index in
            LagunaDFlashReferenceRow(
                sequentialArgmax: sequentialArgmax[index],
                blockArgmax: blockArgmax[index],
                top2Tokens: top2Tokens[index],
                top2Logits: top2Logits[index]
            )
        }
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
