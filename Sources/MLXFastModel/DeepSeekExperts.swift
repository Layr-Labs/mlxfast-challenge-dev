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
        spec: DeepSeekRoutedExpertSpec,
        onRoutingSynced: (() -> Void)? = nil
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
        if !useStaged {
            // Kernel read-ahead for every byte range this layer is about to
            // pread, so SSD I/O overlaps the per-expert GPU compute below.
            loader.expertPrefetcher.prefetch(layerIndex: spec.layerIndex, expertIndices: selectedExperts)
        }

        let outputCount = tokenCount * topK
        guard outputCount > 0 else {
            return zeros([batchSize, sequenceLength, topK, hiddenSize], dtype: x.dtype)
        }

        // Prefill-shaped batched path: when the layer's stacked code tensors
        // are already in RAM (staged whole-layer buffer or pinned hash-layer
        // store) run all experts through three sorted gatherQuantizedMM calls
        // instead of ~256 per-expert QMM dispatches plus per-expert
        // SwiGLU/gather/concat chains. Row values are the same per-row
        // quantized dot products over the same bytes; the correctness gate
        // guards the math. Any missing piece falls through to the existing
        // per-expert loop unchanged.
        if tokenCount >= stagingMinimumTokenCount,
           useStaged || loader.pinnedLayerIsFullyResident(layerIndex: spec.layerIndex),
           let batched = batchedResidentForward(
               x,
               selectedExperts: selectedExperts,
               batchSize: batchSize,
               sequenceLength: sequenceLength,
               topK: topK,
               loader: loader,
               spec: spec
           )
        {
            return batched
        }

        // Group activation flat-indices by expert so each expert runs one batched
        // matmul over all of its tokens instead of one matmul per token.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        var expertOrder: [Int] = []
        expertOrder.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            if flatIndicesByExpert[expertIndex] == nil {
                flatIndicesByExpert[expertIndex] = []
                expertOrder.append(expertIndex)
            }
            flatIndicesByExpert[expertIndex]?.append(flatIndex)
        }

        // Flatten the token axis once. Row (batch * sequenceLength + position)
        // equals x[batch, position], so an activation flat index maps to token row
        // flatIndex / topK. Gathering rows with a single `take` replaces the
        // per-token slice+concat that built each expert batch previously.
        let xFlat = x.reshaped([tokenCount, hiddenSize])

        // The shared expert depends only on x (RAM-resident weights, no SSD
        // read), so it is the one piece of GPU work that can run during the
        // routed experts' blocking SSD reads below. Fire the overlap hook
        // (which evals the already-built shared MLP graph) right before the
        // concurrentPerform barrier, filling the GPU-idle read window. Only
        // on the decode path where that idle window exists.
        if tokenCount == 1, !useStaged {
            onRoutingSynced?()
        }

        // Decode/1-token path: the per-expert code slices are otherwise read
        // one blocking pread at a time on the compute thread. Read them
        // concurrently up front through a capacity-0 side bank (byte-identical
        // ranges), so the per-expert loop below builds its MLXArrays from
        // already-fetched bytes instead of serializing on each pread. Anything
        // not prefetched falls back to the normal per-expert bank read.
        var decodePrefetch: [String: StagedExpertCode]?
        if tokenCount == 1, !useStaged {
            decodePrefetch = loader.consumeScheduledPinnedDecodeExpertCodes(
                layerIndex: spec.layerIndex
            ) ?? loader.prefetchDecodeExpertCodes(
                layerIndex: spec.layerIndex,
                expertIndices: expertOrder,
                hiddenSize: spec.hiddenSize,
                intermediateSize: spec.intermediateSize
            )
        } else if useStaged {
            // Prefill/warmup staged path: build the active experts' base
            // MLXArrays from the staged layer buffer concurrently so the loop
            // below skips the serial Data->Metal copy (same win as decode).
            decodePrefetch = loader.prefetchStagedExpertCodes(
                layerIndex: spec.layerIndex,
                expertIndices: expertOrder,
                hiddenSize: spec.hiddenSize,
                intermediateSize: spec.intermediateSize
            )
        }

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(flatIndicesByExpert.count)
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)

        for expertIndex in expertOrder {
            guard let flatIndices = flatIndicesByExpert[expertIndex] else {
                continue
            }
            let expertWeights = try weights(
                forExpert: expertIndex,
                loader: loader,
                spec: spec,
                preferStaged: useStaged,
                decodePrefetch: decodePrefetch
            )
            let tokens: MLXArray
            if tokenCount == 1, flatIndices.count == 1 {
                tokens = xFlat
            } else {
                let tokenRows = flatIndices.map { Int32($0 / topK) }
                tokens = xFlat.take(MLXArray(tokenRows), axis: 0)
            }
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
        // activation order, replacing the previous per-row scatter loop. When
        // the scatter order is already the identity (decode's first-seen
        // expert order visits slots in order), the gather is a no-op node --
        // skip it; the tensor values are identical by definition.
        var inverse = [Int32](repeating: 0, count: outputCount)
        var isIdentity = true
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
            if flatIndex != row {
                isIdentity = false
            }
        }
        let ordered = isIdentity ? combined : combined.take(MLXArray(inverse), axis: 0)

        let result = ordered.reshaped([batchSize, sequenceLength, topK, hiddenSize])
        // Decode path: dispatch this layer's routed-expert graph now so the
        // GPU runs the QMMs while the CPU builds the next block's graph,
        // matching the batched prefill path's layer-exit dispatch. Scheduling
        // only; identical nodes evaluate either way.
        if tokenCount == 1 {
            asyncEval(result)
        }
        return result
    }

    /// Whole-layer routed-expert forward over RAM-resident stacked tensors.
    ///
    /// Bit-identity contract: every expert still runs the SAME per-expert
    /// quantizedMM kernel over the SAME token batch (an [n_e, hidden] GEMM)
    /// and the same weight bytes — the weights are just zero-copy first-axis
    /// views into ONE stacked MLXArray per projection instead of 256
    /// separately copied slice buffers. The limitedSwiGLU chain is elementwise
    /// and batch-size invariant, so evaluating it once over the concatenated
    /// [N, intermediate] activations produces the same elements as 256
    /// per-expert evaluations. gatherQuantizedMM is deliberately NOT used: its
    /// per-row matvec accumulation order differs from the per-expert GEMM and
    /// shifts bf16 argmax ties.
    private static func batchedResidentForward(
        _ x: MLXArray,
        selectedExperts: [Int],
        batchSize: Int,
        sequenceLength: Int,
        topK: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) -> MLXArray? {
        // Staged layers whose gate+up bytes were fused off-thread run ONE
        // first-projection QMM per expert instead of two. Pinned hash layers
        // never have a fused entry and keep the three-projection path below.
        if let fused = fusedBatchedResidentForward(
            x,
            selectedExperts: selectedExperts,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            topK: topK,
            loader: loader,
            spec: spec
        ) {
            return fused
        }
        // Build the three stacked projections concurrently: each is one big
        // Data->Metal copy (vs 256 slice copies on the per-expert path).
        var projections = [DeepSeekWeightLoader.StackedExpertProjection?](repeating: nil, count: 3)
        let expectedShapes: [[Int]] = [
            [spec.intermediateSize, spec.hiddenSize],
            [spec.intermediateSize, spec.hiddenSize],
            [spec.hiddenSize, spec.intermediateSize],
        ]
        let kinds: [DeepSeekExpertProjection] = [.gate, .up, .down]
        projections.withUnsafeMutableBufferPointer { buffer in
            let sink = StackedProjectionSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: 3) { index in
                sink.buffer[index] = loader.stackedExpertProjection(
                    layerIndex: spec.layerIndex,
                    projection: kinds[index],
                    expectedShape: expectedShapes[index]
                )
            }
        }
        guard let gate = projections[0], let up = projections[1], let down = projections[2] else {
            return nil
        }

        let outputCount = selectedExperts.count
        // Group activation flat-indices by expert exactly like the per-expert
        // path; iteration order only affects concat order, which the inverse
        // permutation undoes.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        let xFlat = x.reshaped([batchSize * sequenceLength, spec.hiddenSize])
        let gateLogical = [spec.intermediateSize, spec.hiddenSize]
        let downLogical = [spec.hiddenSize, spec.intermediateSize]

        // One gather for every expert's token batch. Each expert's segment
        // view of xSorted holds the same rows in the same within-expert order
        // as the per-expert take, so the per-expert GEMM inputs are identical.
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)
        var segmentLengths: [Int] = []
        segmentLengths.reserveCapacity(flatIndicesByExpert.count)
        var expertOrder: [Int] = []
        expertOrder.reserveCapacity(flatIndicesByExpert.count)
        for (expertIndex, flatIndices) in flatIndicesByExpert {
            expertOrder.append(expertIndex)
            segmentLengths.append(flatIndices.count)
            scatterOrder.append(contentsOf: flatIndices)
        }
        let sortedRows = scatterOrder.map { Int32($0 / topK) }
        let xSorted = xFlat.take(MLXArray(sortedRows), axis: 0)

        var gateOutputs: [MLXArray] = []
        var upOutputs: [MLXArray] = []
        var downWeights: [DeepSeekLinearWeight] = []
        gateOutputs.reserveCapacity(expertOrder.count)
        upOutputs.reserveCapacity(expertOrder.count)
        downWeights.reserveCapacity(expertOrder.count)

        var tokenStart = 0
        for (position, expertIndex) in expertOrder.enumerated() {
            let gateWeight = DeepSeekLinearWeight(
                weight: gate.weight[expertIndex],
                scales: gate.scales[expertIndex],
                biases: gate.biases.map { $0[expertIndex] },
                logicalShape: gateLogical,
                groupSize: gate.groupSize,
                bits: gate.bits,
                mode: gate.mode
            )
            let upWeight = DeepSeekLinearWeight(
                weight: up.weight[expertIndex],
                scales: up.scales[expertIndex],
                biases: up.biases.map { $0[expertIndex] },
                logicalShape: gateLogical,
                groupSize: up.groupSize,
                bits: up.bits,
                mode: up.mode
            )
            downWeights.append(
                DeepSeekLinearWeight(
                    weight: down.weight[expertIndex],
                    scales: down.scales[expertIndex],
                    biases: down.biases.map { $0[expertIndex] },
                    logicalShape: downLogical,
                    groupSize: down.groupSize,
                    bits: down.bits,
                    mode: down.mode
                )
            )
            let tokens = xSorted[tokenStart..<(tokenStart + segmentLengths[position])]
            tokenStart += segmentLengths[position]
            gateOutputs.append(DeepSeekOps.linear(input: tokens, weight: gateWeight))
            upOutputs.append(DeepSeekOps.linear(input: tokens, weight: upWeight))
        }

        // One elementwise SwiGLU over all experts' activations. Elementwise
        // ops are batch-size invariant, so each row matches the per-expert
        // evaluation bit for bit.
        let hiddenAll = DeepSeekOps.limitedSwiGLU(
            gate: concatenated(gateOutputs, axis: 0),
            up: concatenated(upOutputs, axis: 0),
            limit: spec.swigluLimit
        )

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(downWeights.count)
        var segmentStart = 0
        for (index, weight) in downWeights.enumerated() {
            let segment = hiddenAll[segmentStart..<(segmentStart + segmentLengths[index])]
            expertOutputs.append(DeepSeekOps.linear(input: segment, weight: weight))
            segmentStart += segmentLengths[index]
        }

        let combined = concatenated(expertOutputs, axis: 0)
        var inverse = [Int32](repeating: 0, count: outputCount)
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
        }
        let ordered = combined.take(MLXArray(inverse), axis: 0)
        return ordered.reshaped([batchSize, sequenceLength, topK, spec.hiddenSize])
    }

    /// Fused-gate+up variant of `batchedResidentForward` for staged layers.
    ///
    /// Bit-identity contract: each expert runs the SAME per-expert quantizedMM
    /// kernel over the SAME token batch; the weight's output rows are gate_e's
    /// rows followed by up_e's rows (host-interleaved from the identical file
    /// bytes), so every output element is the same quantized dot product with
    /// the same reduction order as the separate gate/up QMMs — output-row
    /// concatenation touches no row's bytes, scales, or accumulation. The
    /// concatenated [N, 2I] activations are split back by columns before the
    /// unchanged elementwise limitedSwiGLU: concatenated(fused)[.., 0..<I]
    /// holds exactly the elements concatenated(gateOutputs) held, position for
    /// position. Down projection, segment slicing, inverse permutation, and
    /// reshape are copied verbatim from the three-projection path.
    private static func fusedBatchedResidentForward(
        _ x: MLXArray,
        selectedExperts: [Int],
        batchSize: Int,
        sequenceLength: Int,
        topK: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec
    ) -> MLXArray? {
        // Build the fused first projection and the down projection
        // concurrently: each is one big Data->Metal copy.
        var projections = [DeepSeekWeightLoader.StackedExpertProjection?](repeating: nil, count: 2)
        projections.withUnsafeMutableBufferPointer { buffer in
            let sink = StackedProjectionSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: 2) { index in
                if index == 0 {
                    sink.buffer[0] = loader.fusedFirstStackedProjection(
                        layerIndex: spec.layerIndex,
                        hiddenSize: spec.hiddenSize,
                        intermediateSize: spec.intermediateSize
                    )
                } else {
                    sink.buffer[1] = loader.stackedExpertProjection(
                        layerIndex: spec.layerIndex,
                        projection: .down,
                        expectedShape: [spec.hiddenSize, spec.intermediateSize]
                    )
                }
            }
        }
        guard let fused = projections[0], let down = projections[1] else {
            return nil
        }

        let outputCount = selectedExperts.count
        // Group activation flat-indices by expert exactly like the per-expert
        // path; iteration order only affects concat order, which the inverse
        // permutation undoes.
        var flatIndicesByExpert: [Int: [Int]] = [:]
        flatIndicesByExpert.reserveCapacity(min(outputCount, 256))
        for (flatIndex, expertIndex) in selectedExperts.enumerated() {
            flatIndicesByExpert[expertIndex, default: []].append(flatIndex)
        }

        let xFlat = x.reshaped([batchSize * sequenceLength, spec.hiddenSize])
        let fusedLogical = [2 * spec.intermediateSize, spec.hiddenSize]
        let downLogical = [spec.hiddenSize, spec.intermediateSize]

        // One gather for every expert's token batch. Each expert's segment
        // view of xSorted holds the same rows in the same within-expert order
        // as the per-expert take, so the per-expert GEMM inputs are identical.
        var scatterOrder: [Int] = []
        scatterOrder.reserveCapacity(outputCount)
        var segmentLengths: [Int] = []
        segmentLengths.reserveCapacity(flatIndicesByExpert.count)
        var expertOrder: [Int] = []
        expertOrder.reserveCapacity(flatIndicesByExpert.count)
        for (expertIndex, flatIndices) in flatIndicesByExpert {
            expertOrder.append(expertIndex)
            segmentLengths.append(flatIndices.count)
            scatterOrder.append(contentsOf: flatIndices)
        }
        let sortedRows = scatterOrder.map { Int32($0 / topK) }
        let xSorted = xFlat.take(MLXArray(sortedRows), axis: 0)

        var fusedOutputs: [MLXArray] = []
        var downWeights: [DeepSeekLinearWeight] = []
        fusedOutputs.reserveCapacity(expertOrder.count)
        downWeights.reserveCapacity(expertOrder.count)

        var tokenStart = 0
        for (position, expertIndex) in expertOrder.enumerated() {
            let fusedWeight = DeepSeekLinearWeight(
                weight: fused.weight[expertIndex],
                scales: fused.scales[expertIndex],
                biases: nil,
                logicalShape: fusedLogical,
                groupSize: fused.groupSize,
                bits: fused.bits,
                mode: fused.mode
            )
            downWeights.append(
                DeepSeekLinearWeight(
                    weight: down.weight[expertIndex],
                    scales: down.scales[expertIndex],
                    biases: down.biases.map { $0[expertIndex] },
                    logicalShape: downLogical,
                    groupSize: down.groupSize,
                    bits: down.bits,
                    mode: down.mode
                )
            )
            let tokens = xSorted[tokenStart..<(tokenStart + segmentLengths[position])]
            tokenStart += segmentLengths[position]
            fusedOutputs.append(DeepSeekOps.linear(input: tokens, weight: fusedWeight))
        }

        // Split the fused activations back into gate/up columns, then run the
        // SAME single elementwise SwiGLU over all experts' activations.
        let allFused = concatenated(fusedOutputs, axis: 0)
        let intermediate = spec.intermediateSize
        let hiddenAll = DeepSeekOps.limitedSwiGLU(
            gate: allFused[0..., 0..<intermediate],
            up: allFused[0..., intermediate..<(2 * intermediate)],
            limit: spec.swigluLimit
        )

        var expertOutputs: [MLXArray] = []
        expertOutputs.reserveCapacity(downWeights.count)
        var segmentStart = 0
        for (index, weight) in downWeights.enumerated() {
            let segment = hiddenAll[segmentStart..<(segmentStart + segmentLengths[index])]
            expertOutputs.append(DeepSeekOps.linear(input: segment, weight: weight))
            segmentStart += segmentLengths[index]
        }

        let combined = concatenated(expertOutputs, axis: 0)
        var inverse = [Int32](repeating: 0, count: outputCount)
        for (row, flatIndex) in scatterOrder.enumerated() {
            inverse[flatIndex] = Int32(row)
        }
        let ordered = combined.take(MLXArray(inverse), axis: 0)
        return ordered.reshaped([batchSize, sequenceLength, topK, spec.hiddenSize])
    }

    public static func weights(
        forExpert expertIndex: Int,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekRoutedExpertSpec,
        preferStaged: Bool = false,
        decodePrefetch: [String: StagedExpertCode]? = nil
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .gate,
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            ),
            up: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .up,
                expectedShape: [spec.intermediateSize, spec.hiddenSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            ),
            down: loader.routedExpertLinearWeight(
                layerIndex: spec.layerIndex,
                projection: .down,
                expectedShape: [spec.hiddenSize, spec.intermediateSize],
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch
            )
        )
    }
}

// Lets concurrentPerform write the three stacked projections into distinct
// slots from worker threads; each index is written by exactly one iteration.
private struct StackedProjectionSink: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<DeepSeekWeightLoader.StackedExpertProjection?>
}

/// Below this many tokens the unique-expert count is small enough that
/// per-slice streaming reads less than a whole stacked tensor; decode and the
/// hidden one-token gates stay on the existing path. At 64 tokens (384
/// activations) the expected unique-expert coverage already exceeds 3/4 of
/// the stack, and the scored 512-token prefills touch essentially all of it.
private let stagingMinimumTokenCount = 64
