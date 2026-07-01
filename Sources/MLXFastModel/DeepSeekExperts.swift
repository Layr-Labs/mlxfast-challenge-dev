import Foundation
import MLX
import MLXFastCore

public struct DeepSeekRoutedExpertSpec: Equatable {
    public let layerIndex: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let swigluLimit: Double

    public init(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        swigluLimit: Double
    ) {
        self.layerIndex = layerIndex
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.swigluLimit = swigluLimit
    }

    public init(layerIndex: Int, config: DeepSeekConfig) {
        self.init(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swigluLimit: config.swigluLimit
        )
    }
}

public enum DeepSeekRoutedExperts {
    public static func forward(
        _ x: MLXArray,
        expertIndices: MLXArray,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) throws -> MLXArray {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("routed expert input must have shape [batch, length, hidden]")
        }
        guard expertIndices.shape.count == 3 else {
            throw MLXFastError.invalidInput("expert indices must have shape [batch, length, topK]")
        }
        guard x.shape[0] == expertIndices.shape[0], x.shape[1] == expertIndices.shape[1] else {
            throw MLXFastError.invalidInput("expert indices batch/length must match routed expert input")
        }
        guard x.shape[2] == spec.hiddenSize else {
            throw MLXFastError.invalidInput(
                "routed expert input hidden size \(x.shape[2]) expected \(spec.hiddenSize)"
            )
        }

        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let topK = expertIndices.shape[2]
        let hiddenSize = spec.hiddenSize
        let selectedExperts = expertIndices.asArray(Int32.self).map(Int.init)

        let tokenCount = batchSize * sequenceLength
        let outputCount = tokenCount * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, hiddenSize], dtype: x.dtype)
        }

        let xFlat = x.reshaped([tokenCount, hiddenSize])

        if tokenCount == 1 {
            var allDistinct = true
            for i in 0..<selectedExperts.count where allDistinct {
                for j in (i + 1)..<selectedExperts.count where selectedExperts[i] == selectedExperts[j] {
                    allDistinct = false
                    break
                }
            }
            if allDistinct {
                var expertOutputs: [MLXArray] = []
                expertOutputs.reserveCapacity(topK)
                for expertIndex in selectedExperts {
                    let expertWeights = try weights(
                        forExpert: expertIndex,
                        loader: loader,
                        spec: spec
                    )
                    expertOutputs.append(
                        DeepSeekMLP.forward(
                            xFlat,
                            weights: expertWeights,
                            swigluLimit: spec.swigluLimit
                        )
                    )
                }
                return concatenated(expertOutputs, axis: 0)
                    .reshaped([batchSize, sequenceLength, topK, hiddenSize])
            }
        }

        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(flatIndicesByExpert.count)
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)

        for (expertIndex, flatIndices) in flatIndicesByExpert {
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec
            )
            let tokenRows = flatIndices.map { Int32($0 / topK) }
            let tokens = xFlat.take(MLXArray(tokenRows), axis: 0)
            let expertOutput = DeepSeekMLP.forward(
                tokens,
                weights: expertWeights,
                swigluLimit: spec.swigluLimit
            )
            expertOutputs.append(expertOutput)
            scatterOrder.append(contentsOf: flatIndices)
        }

        let combined = concatenated(expertOutputs, axis: 0)

        // scatterOrder[row] is the activation flat index that produced combined
        // row `row`. Invert it so a single gather places every output back into
        // activation order, replacing the previous per-row scatter loop.
        var inverse = [Int32](repeating: 0, count: outputCount)
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
        }
        let ordered = combined.take(MLXArray(inverse), axis: 0)

        return ordered.reshaped([batchSize, sequenceLength, topK, hiddenSize])
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .gate
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex
            ),
            up: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .up
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex
            ),
            down: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .down
                ),
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex
            )
        )
    }
}
