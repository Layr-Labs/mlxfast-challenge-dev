import Foundation
import MLX
import MLXFastCore

public struct DeepSeekMLPWeights {
    public let gate: DeepSeekLinearWeight
    public let up: DeepSeekLinearWeight
    public let down: DeepSeekLinearWeight

    public init(gate: DeepSeekLinearWeight, up: DeepSeekLinearWeight, down: DeepSeekLinearWeight) {
        self.gate = gate
        self.up = up
        self.down = down
    }

    public init(gate: MLXArray, up: MLXArray, down: MLXArray) {
        self.init(
            gate: DeepSeekLinearWeight(gate),
            up: DeepSeekLinearWeight(up),
            down: DeepSeekLinearWeight(down)
        )
    }
}

public struct DeepSeekMoEWeights {
    public let gate: MLXArray
    public let correctionBias: MLXArray?
    public let tokenToExpert: MLXArray?
    public let sharedExperts: DeepSeekMLPWeights

    public init(
        gate: MLXArray,
        correctionBias: MLXArray?,
        tokenToExpert: MLXArray?,
        sharedExperts: DeepSeekMLPWeights
    ) {
        self.gate = gate
        self.correctionBias = correctionBias
        self.tokenToExpert = tokenToExpert
        self.sharedExperts = sharedExperts
    }
}

public struct DeepSeekMoESpec: Equatable {
    public let routedExperts: DeepSeekRoutedExpertSpec
    public let expertsPerToken: Int
    public let routedScalingFactor: Double
    public let normTopKProb: Bool
    public let scoring: DeepSeekGateScoring

    public init(
        routedExperts: DeepSeekRoutedExpertSpec,
        expertsPerToken: Int,
        routedScalingFactor: Double,
        normTopKProb: Bool,
        scoring: DeepSeekGateScoring
    ) {
        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.routedScalingFactor = routedScalingFactor
        self.normTopKProb = normTopKProb
        self.scoring = scoring
    }

    public init(layerIndex: Int, config: DeepSeekConfig) throws {
        guard let scoring = DeepSeekGateScoring(rawValue: config.scoringFunc) else {
            throw MLXFastError.invalidInput(
                "unsupported DeepSeek MoE gate scoring function \(config.scoringFunc)"
            )
        }
        self.init(
            routedExperts: DeepSeekRoutedExpertSpec(layerIndex: layerIndex, config: config),
            expertsPerToken: config.expertsPerToken,
            routedScalingFactor: config.routedScalingFactor,
            normTopKProb: config.normTopkProb,
            scoring: scoring
        )
    }
}

public enum DeepSeekMLP {
    public static func forward(
        _ x: MLXArray,
        weights: DeepSeekMLPWeights,
        swigluLimit: Double
    ) -> MLXArray {
        let gate = DeepSeekOps.linear(input: x, weight: weights.gate)
        let up = DeepSeekOps.linear(input: x, weight: weights.up)
        let hidden = DeepSeekOps.limitedSwiGLU(gate: gate, up: up, limit: swigluLimit)
        return DeepSeekOps.linear(input: hidden, weight: weights.down)
    }

    /// Single-token (M == 1) MLP that fuses the gate and up projections into one
    /// quantized matmul. gate and up share the input `x` and have independent
    /// output rows with matching quantization, so concatenating their rows and
    /// running one `quantizedMM` then splitting the halves is bit-identical to two
    /// separate calls — but ONLY at M == 1. For M > 1, MLX's per-row batch limit
    /// differs for output dims in (2048, 4096], which can switch a straddle-band
    /// matmul from the qmv to the qmm_t kernel (a different reduction order that is
    /// not bit-identical), so this guards on M == 1 and otherwise falls back.
    public static func forwardFusedGateUp(
        _ x: MLXArray,
        weights: DeepSeekMLPWeights,
        swigluLimit: Double
    ) -> MLXArray {
        let rank = x.shape.count
        guard
            rank >= 2,
            x.shape[rank - 2] == 1,
            let gateScales = weights.gate.scales,
            let upScales = weights.up.scales,
            weights.gate.biases == nil,
            weights.up.biases == nil,
            weights.gate.groupSize == weights.up.groupSize,
            weights.gate.bits == weights.up.bits,
            weights.gate.mode == weights.up.mode,
            weights.gate.logicalShape.count == 2,
            weights.up.logicalShape.count == 2,
            weights.gate.logicalShape[0] == weights.up.logicalShape[0],
            weights.gate.logicalShape[1] == weights.up.logicalShape[1]
        else {
            return forward(x, weights: weights, swigluLimit: swigluLimit)
        }

        let intermediate = weights.gate.logicalShape[0]
        let hiddenSize = weights.gate.logicalShape[1]
        let fusedGateUp = DeepSeekLinearWeight(
            weight: concatenated([weights.gate.weight, weights.up.weight], axis: 0),
            scales: concatenated([gateScales, upScales], axis: 0),
            biases: nil,
            logicalShape: [2 * intermediate, hiddenSize],
            groupSize: weights.gate.groupSize,
            bits: weights.gate.bits,
            mode: weights.gate.mode
        )
        let projected = DeepSeekOps.linear(input: x, weight: fusedGateUp)
        let gate = projected[.ellipsis, 0 ..< intermediate]
        let up = projected[.ellipsis, intermediate ..< (2 * intermediate)]
        let hidden = DeepSeekOps.limitedSwiGLU(gate: gate, up: up, limit: swigluLimit)
        return DeepSeekOps.linear(input: hidden, weight: weights.down)
    }
}

public enum DeepSeekMoE {
    public static func forward(
        _ x: MLXArray,
        inputIDs: MLXArray?,
        weights: DeepSeekMoEWeights,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekMoESpec
    ) throws -> MLXArray {
        let routing = try DeepSeekMoEGate.route(
            hidden: x,
            inputIDs: inputIDs,
            weight: weights.gate,
            correctionBias: weights.correctionBias,
            tokenToExpert: weights.tokenToExpert,
            topK: spec.expertsPerToken,
            routedScalingFactor: spec.routedScalingFactor,
            normTopKProb: spec.normTopKProb,
            scoring: spec.scoring
        )
        let routed = try DeepSeekRoutedExperts.forward(
            x,
            expertIndices: routing.indices,
            loader: loader,
            spec: spec.routedExperts
        )
        let shared = DeepSeekMLP.forward(
            x,
            weights: weights.sharedExperts,
            swigluLimit: spec.routedExperts.swigluLimit
        )
        return combine(
            routedExpertOutput: routed,
            routeWeights: routing.weights,
            sharedExpertOutput: shared
        )
    }

    public static func combine(
        routedExpertOutput: MLXArray,
        routeWeights: MLXArray,
        sharedExpertOutput: MLXArray
    ) -> MLXArray {
        let weightedRouted = (
            routedExpertOutput * routeWeights.expandedDimensions(axis: -1).asType(routedExpertOutput.dtype)
        ).sum(axis: -2)
        return weightedRouted + sharedExpertOutput
    }

    public static func forward(
        _ x: MLXArray,
        routeWeights: MLXArray,
        sharedWeights: DeepSeekMLPWeights,
        swigluLimit: Double,
        routedExpertOutput: (_ x: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        let routed = try routedExpertOutput(x)
        let shared = DeepSeekMLP.forward(x, weights: sharedWeights, swigluLimit: swigluLimit)
        return combine(
            routedExpertOutput: routed,
            routeWeights: routeWeights,
            sharedExpertOutput: shared
        )
    }
}
