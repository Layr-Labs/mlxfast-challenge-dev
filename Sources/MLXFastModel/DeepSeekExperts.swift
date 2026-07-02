import Foundation
import MLX
import MLXFastCore

public struct DeepSeekRoutedExpertSpec: Equatable {
    public let layerIndex: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let swigluLimit: Double
    public let expertCount: Int

    public init(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        swigluLimit: Double,
        expertCount: Int = MLXFastConstants.routedExperts
    ) {
        self.layerIndex = layerIndex
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.swigluLimit = swigluLimit
        self.expertCount = expertCount
    }

    public init(layerIndex: Int, config: DeepSeekConfig) {
        self.init(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swigluLimit: config.swigluLimit,
            expertCount: config.routedExperts
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
        guard spec.expertCount > 0 else {
            throw MLXFastError.invalidInput("routed expert count must be positive")
        }

        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let topK = expertIndices.shape[2]
        let hiddenSize = spec.hiddenSize

        let tokenCount = batchSize * sequenceLength
        let outputCount = tokenCount * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, hiddenSize], dtype: x.dtype)
        }
        let selectedExperts = expertIndices.asArray(Int32.self).map(Int.init)

        // Group activation flat-indices by expert so each expert runs one batched
        // matmul over all of its tokens instead of one matmul per token.
        var countsByExpert = Array(repeating: 0, count: spec.expertCount)
        var activeSlotByExpert = Array(repeating: -1, count: spec.expertCount)
        var activeExperts: [Int] = []
        activeExperts.reserveCapacity(min(outputCount, spec.expertCount))
        for expertIndex in selectedExperts {
            guard expertIndex >= 0, expertIndex < spec.expertCount else {
                throw MLXFastError.invalidInput(
                    "routed expert index \(expertIndex) outside 0..<\(spec.expertCount)"
                )
            }
            if countsByExpert[expertIndex] == 0 {
                activeSlotByExpert[expertIndex] = activeExperts.count
                activeExperts.append(expertIndex)
            }
            countsByExpert[expertIndex] += 1
        }

        var flatIndicesByActiveExpert = activeExperts.map { expertIndex -> [Int] in
            var bucket: [Int] = []
            bucket.reserveCapacity(countsByExpert[expertIndex])
            return bucket
        }
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByActiveExpert[activeSlotByExpert[expertIndex]].append(flatIndex)
        }

        // Flatten the token axis once. Row (batch * sequenceLength + position)
        // equals x[batch, position], so an activation flat index maps to token row
        // flatIndex / topK. Gathering rows with a single `take` replaces the
        // per-token slice+concat that built each expert batch previously.
        let xFlat = x.reshaped([tokenCount, hiddenSize])

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(activeExperts.count)
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)

        for (slot, expertIndex) in activeExperts.enumerated() {
            let flatIndices = flatIndicesByActiveExpert[slot]
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec
            )
            var tokenRows: [Int32] = []
            tokenRows.reserveCapacity(flatIndices.count)
            for flatIndex in flatIndices {
                tokenRows.append(Int32(flatIndex / topK))
            }
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
        try loader.routedExpertMLPWeights(
            layerIndex: spec.layerIndex,
            expertIndex: expertIndex,
            hiddenSize: spec.hiddenSize,
            intermediateSize: spec.intermediateSize
        )
    }
}
