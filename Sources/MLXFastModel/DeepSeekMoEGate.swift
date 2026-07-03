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
        topK: Int,
        routedScalingFactor: Double,
        normTopKProb: Bool,
        scoring: DeepSeekGateScoring
    ) throws -> DeepSeekMoEGateResult {
        let indices: MLXArray
        if let tokenToExpert {
            guard let inputIDs else {
                throw MLXFastError.invalidInput("hash routing requires input ids")
            }
            indices = tokenToExpert[inputIDs].asType(.int32)
            if let selected = decodeHashRouteSelectedScores(
                hidden: hidden,
                weight: weight,
                indices: indices,
                topK: topK,
                routedScalingFactor: routedScalingFactor,
                normTopKProb: normTopKProb,
                scoring: scoring
            ) {
                return selected
            }
        } else {
            let logits = DeepSeekOps.cast(
                matmul(hidden, weightTransposed ?? weight.T),
                to: .float32
            )
            let scores = score(logits, scoring: scoring)
            let biased = correctionBias.map { scores + $0 } ?? scores
            indices = argPartition(-biased, kth: topK - 1, axis: -1)[
                .ellipsis,
                0..<topK
            ].asType(.int32)
            var selectedWeights = takeAlong(scores, indices, axis: -1)
            if scoring != .softmax && normTopKProb {
                selectedWeights = selectedWeights / (selectedWeights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            selectedWeights = selectedWeights * Float(routedScalingFactor)

            return DeepSeekMoEGateResult(indices: indices, weights: selectedWeights)
        }

        let logits = DeepSeekOps.cast(
            matmul(hidden, weightTransposed ?? weight.T),
            to: .float32
        )
        let scores = score(logits, scoring: scoring)
        var selectedWeights = takeAlong(scores, indices, axis: -1)
        if scoring != .softmax && normTopKProb {
            selectedWeights = selectedWeights / (selectedWeights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selectedWeights = selectedWeights * Float(routedScalingFactor)

        return DeepSeekMoEGateResult(indices: indices, weights: selectedWeights)
    }

    private static func decodeHashRouteSelectedScores(
        hidden: MLXArray,
        weight: MLXArray,
        indices: MLXArray,
        topK: Int,
        routedScalingFactor: Double,
        normTopKProb: Bool,
        scoring: DeepSeekGateScoring
    ) -> DeepSeekMoEGateResult? {
        guard scoring != .softmax,
              hidden.shape.count == 3,
              hidden.shape[0] == 1,
              hidden.shape[1] == 1,
              weight.shape.count == 2,
              hidden.shape[2] == weight.shape[1],
              indices.shape == [1, 1, topK]
        else {
            return nil
        }

        let selectedWeight = weight.take(indices.reshaped([topK]), axis: 0)
        let logits = DeepSeekOps.cast(
            matmul(
                hidden.reshaped([1, hidden.shape[2]]),
                selectedWeight.T
            ).reshaped([1, 1, topK]),
            to: .float32
        )
        var selectedWeights = score(logits, scoring: scoring)
        if normTopKProb {
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
