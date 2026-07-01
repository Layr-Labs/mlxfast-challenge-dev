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

        // Decode fast path: a single token routed to topK experts (M == 1). The
        // per-expert gather and the inverse-permutation reorder of the general
        // path are identities here, so run each selected expert directly on the
        // one token row and concatenate outputs in routing order. Each expert
        // fuses gate+up into one quantizedMM. Both are bit-identical ONLY at
        // M == 1 and only when the selected experts are all distinct: the general
        // path batches a duplicated expert (hash-routing layers) into one M == 2
        // matmul, which uses a different kernel than two M == 1 calls, so fall
        // through to the general path if any expert repeats.
        if tokenCount == 1 {
            var seen = Set<Int>()
            seen.reserveCapacity(topK)
            var allDistinct = true
            for expertIndex in selectedExperts where !seen.insert(expertIndex).inserted {
                allDistinct = false
                break
            }
            if allDistinct {
                let token = x.reshaped([1, hiddenSize])
                var singleTokenOutputs: [MLXArray] = []
                singleTokenOutputs.reserveCapacity(topK)
                for expertIndex in selectedExperts {
                    let expertWeights = try weights(
                        forExpert: expertIndex,
                        loader: loader,
                        spec: spec
                    )
                    singleTokenOutputs.append(
                        DeepSeekMLP.forwardFusedGateUp(
                            token,
                            weights: expertWeights,
                            swigluLimit: spec.swigluLimit
                        )
                    )
                }
                return concatenated(singleTokenOutputs, axis: 0)
                    .reshaped([batchSize, sequenceLength, topK, hiddenSize])
            }
        }

        // Group activation flat-indices by expert so each expert runs one batched
        // matmul over all of its tokens instead of one matmul per token.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        // Flatten the token axis once. Row (batch * sequenceLength + position)
        // equals x[batch, position], so an activation flat index maps to token row
        // flatIndex / topK. Gathering rows with a single `take` replaces the
        // per-token slice+concat that built each expert batch previously.
        let xFlat = x.reshaped([tokenCount, hiddenSize])

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
