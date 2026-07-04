import Foundation
import MLX
import MLXFastCore

public enum DeepSeekGateScoring: String, Equatable {
    case softmax
    case sigmoid
    case sqrtsoftplus
}

public struct DeepSeekMoEGateResult {
    public let indices: MLXArray
    public let weights: MLXArray
}

public enum DeepSeekMoEGate {
    public static func route(
        hidden: MLXArray,
        inputIDs: MLXArray? = nil,
        weight: MLXArray,
        weightTransposed: MLXArray? = nil,
        correctionBias: MLXArray? = nil,
        tokenToExpert: MLXArray? = nil,
        fixedTokenToExpertIndices: MLXArray? = nil,
        topK: Int,
        routedScalingFactor: Double,
        normTopKProb: Bool,
        scoring: DeepSeekGateScoring,
        asyncEvalIndices: Bool = false
    ) throws -> DeepSeekMoEGateResult {
        let logits = DeepSeekOps.cast(
            matmul(hidden, weightTransposed ?? weight.T),
            to: .float32
        )
        let scores = score(logits, scoring: scoring)

        let indices: MLXArray
        if let tokenToExpert {
            if let fixed = fixedTokenToExpertIndices,
               fixed.shape.count == 3,
               fixed.shape[0] == hidden.shape[0],
               fixed.shape[1] == hidden.shape[1],
               fixed.shape[2] == topK
            {
                indices = DeepSeekOps.cast(fixed, to: .int32)
            } else {
                guard let inputIDs else {
                    throw MLXFastError.invalidInput("hash routing requires input ids")
                }
                indices = tokenToExpert[inputIDs].asType(.int32)
            }
        } else {
            let biased = correctionBias.map { scores + $0 } ?? scores
            indices = argPartition(-biased, kth: topK - 1, axis: -1)[
                .ellipsis,
                0..<topK
            ].asType(.int32)
        }

        if asyncEvalIndices {
            asyncEval(indices)
        }

        var selectedWeights = takeAlong(scores, indices, axis: -1)
        if scoring != .softmax && normTopKProb {
            selectedWeights = selectedWeights / (selectedWeights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selectedWeights = selectedWeights * Float(routedScalingFactor)

        return DeepSeekMoEGateResult(indices: indices, weights: selectedWeights)
    }

    public static func score(_ logits: MLXArray, scoring: DeepSeekGateScoring) -> MLXArray {
        switch scoring {
        case .softmax:
            return softmax(logits, axis: -1, precise: true)
        case .sigmoid:
            return sigmoid(logits)
        case .sqrtsoftplus:
            return sqrt(logAddExp(logits, 0))
        }
    }
}
