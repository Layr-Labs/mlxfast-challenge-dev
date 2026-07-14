import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

private let gemma4MTPFastTargetEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_MTP_FAST_TARGET"
    ] else {
        return true
    }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// Make the challenge's weighted runtime tower usable by the exact
/// `Gemma4MTPTarget` API pinned in Package.resolved.
///
/// The ordinary serial `callAsFunction` remains unchanged. MTP uses the
/// library trunk's capture hook so a multi-position target verification also
/// returns the pre-norm hidden state and the last full/sliding K/V pairs that
/// Google's trained assistant consumes.
extension Gemma4RuntimeModel: Gemma4MTPTarget {
    public var mtpConfiguration: Gemma4TextConfiguration {
        configuration
    }

    public func mtpNewCache(
        parameters: GenerateParameters?
    ) -> [any KVCache] {
        newCache(parameters: parameters)
    }

    public func embedTokensForDrafter(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens) * Float(configuration.hiddenSize).squareRoot()
    }

    public func forwardForMTP(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) -> Gemma4MTPForward {
        if gemma4MTPFastTargetEnabled,
           let fast = fastMTPForward(tokens, cache: cache)
        {
            return fast
        }
        let firstSharedLayer =
            configuration.numHiddenLayers - configuration.numKvSharedLayers
        var lastFull = -1
        var lastSliding = -1
        for layerIndex in 0..<firstSharedLayer {
            switch configuration.layerTypes[layerIndex] {
            case Gemma4LayerType.full.rawValue:
                lastFull = layerIndex
            case Gemma4LayerType.sliding.rawValue:
                lastSliding = layerIndex
            default:
                break
            }
        }

        let capture = Gemma4SharedKVCapture()
        let forward = model.callCapturingPreNorm(
            tokens,
            cache: cache
        ) { layerIndex, keyValue in
            if layerIndex == lastFull {
                capture.fullAttention = keyValue
            } else if layerIndex == lastSliding {
                capture.slidingAttention = keyValue
            }
        }
        let projected = model.embedTokens.asLinear(forward.postNorm)
        let cap = MLXArray(configuration.finalLogitSoftcapping)
        let logits = tanh(projected / cap) * cap
        return Gemma4MTPForward(
            logits: logits,
            lastHidden: forward.preNorm,
            capturedSharedKV: capture.snapshot()
        )
    }

    public func rollbackSpeculativeCache(
        _ caches: [KVCache],
        accepted: Gemma4AcceptCount,
        blockSize: Int
    ) {
        let maxAccepted = accepted.maxAccepted()
        let rejectedTail = max(0, blockSize - maxAccepted - 1)
        for cache in caches where cache.isTrimmable && rejectedTail > 0 {
            _ = cache.trim(rejectedTail)
        }
    }
}

public struct Gemma4MTPBlockResult: Equatable, Sendable {
    public let tokens: [Int]
    public let acceptedDraftTokenCount: Int
    public let targetCacheOffset: Int
    public let usedDrafter: Bool

    public init(
        tokens: [Int],
        acceptedDraftTokenCount: Int,
        targetCacheOffset: Int,
        usedDrafter: Bool
    ) {
        self.tokens = tokens
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetCacheOffset = targetCacheOffset
        self.usedDrafter = usedDrafter
    }
}

/// Stateful, greedy Gemma 4 MTP block decoder.
///
/// One instance owns one request. Target cache, target hidden state, shared
/// full/sliding K/V and the current bonus token persist across block calls.
/// Every returned token came from a target argmax. Drafts are never returned
/// directly: a draft is emitted only when it equals the corresponding
/// multi-position target verification result.
public final class Gemma4TrainedMTPBlockSession: @unchecked Sendable {
    public let implementationName = "google_gemma4_trained_mtp"

    private let target: Gemma4RuntimeModel
    private let drafter: Gemma4AssistantDraftModel
    private var targetCache: [any KVCache] = []
    private var hidden: MLXArray?
    private var sharedKV: Gemma4SharedKV?
    private var bonusToken: Int?
    private var hostCacheOffset = 0

    public private(set) var decodedTokenCount = 0
    public private(set) var roundCount = 0
    public private(set) var acceptedDraftTokenCount = 0
    public private(set) var targetOnlyTailTokenCount = 0

    public init(
        target: Gemma4RuntimeModel,
        drafter: Gemma4AssistantDraftModel
    ) throws {
        self.target = target
        self.drafter = drafter
        try drafter.bind(target: target)
    }

    public func begin(seedTokens: [Int]) throws -> Int {
        guard !seedTokens.isEmpty,
              seedTokens.allSatisfy({
                  $0 >= 0 && $0 < MLXFastConstants.vocabSize
              })
        else {
            throw MLXFastError.invalidInput(
                "trained MTP seed tokens must be a nonempty in-vocabulary sequence"
            )
        }

        targetCache = target.mtpNewCache(parameters: nil)
        let input = MLXArray(seedTokens.map(Int32.init), [1, seedTokens.count])
        let prefill = target.forwardForMTP(input, cache: targetCache)
        let seedArray = prefill.logits[
            0..., -1, 0...
        ].asType(.float32).argMax(axis: -1)
        let nextHidden = prefill.lastHidden[
            0..., (prefill.lastHidden.dim(1) - 1)..<prefill.lastHidden.dim(1), 0...
        ]
        let nextSharedKV = prefill.capturedSharedKV
        eval(
            seedArray,
            nextHidden,
            nextSharedKV.fullAttention.0,
            nextSharedKV.fullAttention.1,
            nextSharedKV.slidingAttention.0,
            nextSharedKV.slidingAttention.1
        )

        let seedToken = Int(seedArray.item(Int32.self))
        hostCacheOffset = seedTokens.count
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP prefill")
        hidden = nextHidden
        sharedKV = nextSharedKV
        bonusToken = seedToken
        decodedTokenCount = 0
        roundCount = 0
        acceptedDraftTokenCount = 0
        targetOnlyTailTokenCount = 0
        return seedToken
    }

    public func generateBlock(
        previousCommittedToken: Int,
        maxBlockSize: Int
    ) throws -> Gemma4MTPBlockResult {
        guard maxBlockSize > 0,
              maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP block size must be in "
                    + "1...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard decodedTokenCount <=
            MLXFastConstants.experimentalMTPMaxTotalTokens - maxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP block would exceed the fixed decode window"
            )
        }
        guard let bonusToken, previousCommittedToken == bonusToken else {
            throw MLXFastError.invalidInput(
                "trained MTP block input is not the session's last committed token"
            )
        }
        guard hidden != nil, sharedKV != nil, !targetCache.isEmpty else {
            throw MLXFastError.invalidInput(
                "trained MTP block requested before prefill"
            )
        }
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP block start")

        let result: Gemma4MTPBlockResult
        if maxBlockSize == 1 {
            result = try generateTargetOnlyTail(previousToken: bonusToken)
        } else {
            result = try generateSpeculativeRound(
                previousToken: bonusToken,
                blockSize: maxBlockSize
            )
        }
        decodedTokenCount += result.tokens.count
        return result
    }

    private func generateTargetOnlyTail(
        previousToken: Int
    ) throws -> Gemma4MTPBlockResult {
        let input = MLXArray([Int32(previousToken)], [1, 1])
        let output = target.forwardForMTP(input, cache: targetCache)
        let tokenArray = output.logits[
            0..., -1, 0...
        ].asType(.float32).argMax(axis: -1)
        let nextHidden = output.lastHidden[
            0..., (output.lastHidden.dim(1) - 1)..<output.lastHidden.dim(1), 0...
        ]
        let nextSharedKV = output.capturedSharedKV
        eval(tokenArray, nextHidden)

        let token = Int(tokenArray.item(Int32.self))
        hidden = nextHidden
        sharedKV = nextSharedKV
        bonusToken = token
        hostCacheOffset += 1
        targetOnlyTailTokenCount += 1
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP target-only tail")
        return Gemma4MTPBlockResult(
            tokens: [token],
            acceptedDraftTokenCount: 0,
            targetCacheOffset: hostCacheOffset,
            usedDrafter: false
        )
    }

    private func generateSpeculativeRound(
        previousToken: Int,
        blockSize: Int
    ) throws -> Gemma4MTPBlockResult {
        guard var draftHidden = hidden, let currentSharedKV = sharedKV else {
            throw MLXFastError.invalidInput("trained MTP state is incomplete")
        }

        let draftCount = blockSize - 1
        let positionOffset = Gemma4.PositionOffset.scalar(hostCacheOffset)
        var draftInput = MLXArray([Int32(previousToken)], [1, 1])
        var draftTokens: [MLXArray] = []
        draftTokens.reserveCapacity(draftCount)

        for _ in 0..<draftCount {
            let targetEmbedding = target.embedTokensForDrafter(draftInput)
            let assistantInput = concatenated(
                [targetEmbedding, draftHidden],
                axis: -1
            )
            let assistantOutput = drafter(
                inputsEmbeds: assistantInput,
                sharedKV: currentSharedKV,
                positionOffset: positionOffset
            )
            let draft = assistantOutput.logits[
                0..., -1, 0...
            ].argMax(axis: -1)
            let draft2D = draft.reshaped([1, 1])
            draftTokens.append(draft2D)
            draftInput = draft2D
            draftHidden = assistantOutput.lastHidden
        }

        let previousColumn = MLXArray([Int32(previousToken)], [1, 1])
        let verifyInput = concatenated([previousColumn] + draftTokens, axis: 1)
        let verifyOutput = target.forwardForMTP(
            verifyInput,
            cache: targetCache
        )
        let targetTokens = verifyOutput.logits
            .asType(.float32)
            .argMax(axis: -1)
        let draftConcat = concatenated(draftTokens, axis: 1)
        eval(targetTokens, draftConcat)

        let targetValues = targetTokens
            .squeezed(axis: 0)
            .asArray(Int32.self)
            .map(Int.init)
        let draftValues = draftConcat
            .squeezed(axis: 0)
            .asArray(Int32.self)
            .map(Int.init)
        guard targetValues.count == blockSize,
              draftValues.count == draftCount
        else {
            throw MLXFastError.invalidInput(
                "trained MTP verification returned an unexpected tensor shape"
            )
        }

        var accepted = 0
        while accepted < draftCount,
              draftValues[accepted] == targetValues[accepted]
        {
            accepted += 1
        }
        let committed = Array(targetValues.prefix(accepted + 1))
        if accepted < draftCount {
            target.rollbackSpeculativeCache(
                targetCache,
                accepted: .scalar(accepted),
                blockSize: blockSize
            )
        }

        let rejected = draftCount - accepted
        let nextHidden = verifyOutput.lastHidden[
            0..., accepted..<(accepted + 1), 0...
        ]
        let nextSharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOutput.capturedSharedKV,
            rejected: rejected
        )
        eval(
            nextHidden,
            nextSharedKV.fullAttention.0,
            nextSharedKV.fullAttention.1,
            nextSharedKV.slidingAttention.0,
            nextSharedKV.slidingAttention.1
        )

        hidden = nextHidden
        sharedKV = nextSharedKV
        bonusToken = committed[committed.count - 1]
        hostCacheOffset += committed.count
        roundCount += 1
        acceptedDraftTokenCount += accepted
        try requireCacheOffsets(hostCacheOffset, context: "trained MTP commit/rollback")
        return Gemma4MTPBlockResult(
            tokens: committed,
            acceptedDraftTokenCount: accepted,
            targetCacheOffset: hostCacheOffset,
            usedDrafter: true
        )
    }

    private func requireCacheOffsets(
        _ expected: Int,
        context: String
    ) throws {
        guard expected >= 0,
              targetCache.allSatisfy({ $0.offset == expected })
        else {
            let offsets = targetCache.map(\.offset)
            throw MLXFastError.invalidInput(
                "\(context) target cache offsets \(offsets) do not match host offset \(expected)"
            )
        }
    }
}
