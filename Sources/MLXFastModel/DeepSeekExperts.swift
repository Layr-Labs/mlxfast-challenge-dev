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
        let tokenCount = batchSize * sequenceLength

        // Prefill-shaped calls touch essentially every expert, so schedule
        // whole-stacked-tensor staging for this layer (and the next) BEFORE
        // the routing sync below: the sequential ~1 GiB reads then overlap
        // the GPU drain instead of following it. Staging needs no routing
        // indices, and a failed stage falls back to the per-slice path.
        var stagingScheduled = false
        if tokenCount >= stagingMinimumTokenCount,
           let stager = loader.expertLayerStager,
           let plan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex)
        {
            stager.schedule(plan)
            if let nextPlan = loader.stagedExpertLayerPlan(layerIndex: spec.layerIndex + 1) {
                stager.schedule(nextPlan)
            }
            stagingScheduled = true
        }

        let selectedExperts = expertIndices.asArray(Int32.self).map(Int.init)
        let useStaged = stagingScheduled
            && loader.expertLayerStager?.waitForLayer(spec.layerIndex) == true
        defer {
            if useStaged {
                loader.expertLayerStager?.releaseLayer(spec.layerIndex)
            }
        }
        // On the streaming path, read every selected expert's tensors through
        // the concurrent pool before the per-expert loop: the SSD then serves
        // the layer's byte ranges at high queue depth instead of one blocking
        // pread at a time on this thread. Weights built below consult this
        // dictionary first; anything missing falls back to the original
        // single-threaded bank read.
        var prefetchedTensors: [String: MaterializedTensor] = [:]
        if !useStaged {
            prefetchedTensors = loader.parallelExpertTensors(
                layerIndex: spec.layerIndex,
                expertIndices: selectedExperts
            )
            if prefetchedTensors.isEmpty {
                // Pool unavailable: keep the kernel read-ahead advisories so
                // SSD I/O still overlaps the per-expert GPU compute below.
                loader.expertPrefetcher.prefetch(
                    layerIndex: spec.layerIndex,
                    expertIndices: selectedExperts
                )
            }
        }

        let outputCount = tokenCount * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, hiddenSize], dtype: x.dtype)
        }

        // Decode fast path: one token, topK activations. Stack each
        // projection's per-activation expert weights and run ONE gather-
        // quantized matmul per projection instead of topK separate quantized
        // matmuls plus per-expert gather/scatter. gatherQuantizedMM computes
        // the identical per-row quantized dot products as quantizedMM, so
        // this only removes kernel-launch and graph overhead. Any shape or
        // quantization irregularity falls back to the general path below.
        if tokenCount == 1, !useStaged,
           let output = try batchedSingleTokenForward(
               x: x,
               selectedExperts: selectedExperts,
               loader: loader,
               spec: spec,
               prefetched: prefetchedTensors
           )
        {
            return output.reshaped([batchSize, sequenceLength, topK, hiddenSize])
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
                spec: spec,
                preferStaged: useStaged,
                prefetched: prefetchedTensors
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

    /// Single-token routed MoE with one gatherQuantizedMM per projection.
    /// Returns nil when the per-activation weights are not uniformly
    /// quantized (mode/groupSize/bits/shape), in which case the caller keeps
    /// the per-expert loop.
    private static func batchedSingleTokenForward(
        x: MLXArray,
        selectedExperts: [Int],
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec,
        prefetched: [String: MaterializedTensor]
    ) throws -> MLXArray? {
        var gateWeights: [DeepSeekLinearWeight] = []
        var upWeights: [DeepSeekLinearWeight] = []
        var downWeights: [DeepSeekLinearWeight] = []
        gateWeights.reserveCapacity(selectedExperts.count)
        upWeights.reserveCapacity(selectedExperts.count)
        downWeights.reserveCapacity(selectedExperts.count)
        for expertIndex in selectedExperts {
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec,
                prefetched: prefetched
            )
            gateWeights.append(expertWeights.gate)
            upWeights.append(expertWeights.up)
            downWeights.append(expertWeights.down)
        }

        guard
            let gate = stackedQuantizedWeight(gateWeights),
            let up = stackedQuantizedWeight(upWeights),
            let down = stackedQuantizedWeight(downWeights)
        else {
            return nil
        }

        // x is [1, 1, hidden]; per-activation rows come out as
        // [topK, 1, intermediate].
        let xIn = x.reshaped([1, 1, spec.hiddenSize])
        let rhsIndices = MLXArray((0..<Int32(selectedExperts.count)).map { $0 })
        let gateOut = gatherQuantizedMM(
            xIn, gate.weight, scales: gate.scales, biases: gate.biases,
            rhsIndices: rhsIndices,
            transpose: true, groupSize: gate.groupSize, bits: gate.bits, mode: gate.mode,
            sortedIndices: true
        )
        let upOut = gatherQuantizedMM(
            xIn, up.weight, scales: up.scales, biases: up.biases,
            rhsIndices: rhsIndices,
            transpose: true, groupSize: up.groupSize, bits: up.bits, mode: up.mode,
            sortedIndices: true
        )
        let hidden = DeepSeekOps.limitedSwiGLU(
            gate: gateOut,
            up: upOut,
            limit: spec.swigluLimit
        )
        // Row i of `hidden` multiplies activation i's own down matrix.
        let downOut = gatherQuantizedMM(
            hidden, down.weight, scales: down.scales, biases: down.biases,
            rhsIndices: rhsIndices,
            transpose: true, groupSize: down.groupSize, bits: down.bits, mode: down.mode,
            sortedIndices: true
        )
        return downOut
    }

    private struct StackedQuantizedWeight {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray?
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode
    }

    private static func stackedQuantizedWeight(
        _ weights: [DeepSeekLinearWeight]
    ) -> StackedQuantizedWeight? {
        guard let first = weights.first, first.isQuantized, let firstScales = first.scales else {
            return nil
        }
        let hasBiases = first.biases != nil
        for weight in weights {
            guard
                weight.isQuantized,
                weight.groupSize == first.groupSize,
                weight.bits == first.bits,
                weight.mode == first.mode,
                weight.weight.shape == first.weight.shape,
                weight.scales?.shape == firstScales.shape,
                (weight.biases != nil) == hasBiases
            else {
                return nil
            }
        }
        return StackedQuantizedWeight(
            weight: stacked(weights.map { $0.weight }, axis: 0),
            scales: stacked(weights.compactMap { $0.scales }, axis: 0),
            biases: hasBiases ? stacked(weights.compactMap { $0.biases }, axis: 0) : nil,
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode
        )
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec,
        preferStaged: Bool = false,
        prefetched: [String: MaterializedTensor] = [:]
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .gate
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                prefetched: prefetched
            ),
            up: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .up
                ),
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                prefetched: prefetched
            ),
            down: loader.expertLinearWeight(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: spec.layerIndex,
                    expertIndex: expertIndex,
                    projection: .down
                ),
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                prefetched: prefetched
            )
        )
    }
}

/// Below this many tokens the unique-expert count is small enough that
/// per-slice streaming reads less than a whole stacked tensor; decode and the
/// hidden one-token gates stay on the existing path. At 64 tokens (384
/// activations) the expected unique-expert coverage already exceeds 3/4 of
/// the stack, and the scored 512-token prefills touch essentially all of it.
private let stagingMinimumTokenCount = 64
