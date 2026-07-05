import Foundation
import MLX
import MLXFastCore

public enum DeepSeekWeightNames {
    public static func model(_ suffix: String) -> [String] {
        [
            "model.\(suffix)",
            "language_model.model.\(suffix)",
        ]
    }

    public static func layer(_ layerIndex: Int, _ suffix: String) -> [String] {
        model("layers.\(layerIndex).\(suffix)")
    }

    public static let embedTokens = model("embed_tokens.weight")
    public static let finalNorm = model("norm.weight")
    public static let hcHeadFn = model("hc_head.fn")
    public static let hcHeadBase = model("hc_head.base")
    public static let hcHeadScale = model("hc_head.scale")

    public static let lmHead = [
        "lm_head.weight",
        "language_model.lm_head.weight",
    ]

    public static func attention(_ layerIndex: Int, _ suffix: String) -> [String] {
        layer(layerIndex, "attn.\(suffix)")
    }

    public static func feedForward(_ layerIndex: Int, _ suffix: String) -> [String] {
        layer(layerIndex, "ffn.\(suffix)")
    }

    public static func routedExpert(
        layerIndex: Int,
        expertIndex: Int,
        projection: DeepSeekExpertProjection
    ) -> [String] {
        let sanitized = projection.rawValue
        var candidates = [
            "ffn.switch_mlp.\(sanitized).weight",
            "ffn.switch_mlp.\(expertIndex).\(sanitized).weight",
            "ffn.experts.\(expertIndex).\(sanitized).weight",
        ].flatMap { layer(layerIndex, $0) }

        if let legacy = projection.legacyName {
            candidates += layer(layerIndex, "ffn.experts.\(expertIndex).\(legacy).weight")
        }
        return candidates
    }

    public static func attentionNorm(_ layerIndex: Int) -> [String] {
        layer(layerIndex, "attn_norm.weight")
    }

    public static func feedForwardNorm(_ layerIndex: Int) -> [String] {
        layer(layerIndex, "ffn_norm.weight")
    }

    public static func hyperConnection(
        layerIndex: Int,
        block: DeepSeekHyperConnectionBlock,
        component: DeepSeekHyperConnectionComponent
    ) -> [String] {
        layer(layerIndex, "\(block.rawValue).\(component.rawValue)")
    }
}

public enum DeepSeekHyperConnectionBlock: String, Equatable {
    case attention = "attn_hc"
    case feedForward = "ffn_hc"
}

public enum DeepSeekHyperConnectionComponent: String, Equatable {
    case fn
    case base
    case scale
}

public enum DeepSeekExpertProjection: String, Equatable, Hashable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"

    public var legacyName: String? {
        switch self {
        case .gate:
            return "w1"
        case .down:
            return "w2"
        case .up:
            return "w3"
        }
    }
}

private struct RoutedExpertProjectionKey: Hashable {
    let layerIndex: Int
    let projection: DeepSeekExpertProjection
}

private struct RoutedExpertProjectionPlan {
    let name: String
    let record: ExpertTensorRecord
    let scalesName: String
    let scalesRecord: ExpertTensorRecord?
    let biasesName: String
    let biasesRecord: ExpertTensorRecord?
    let decodePrefetchKeys: [String]

    func decodePrefetchKey(for expertIndex: Int) -> String {
        guard expertIndex >= 0, expertIndex < decodePrefetchKeys.count else {
            return DeepSeekWeightLoader.decodePrefetchKey(name, expertIndex)
        }
        return decodePrefetchKeys[expertIndex]
    }
}

public struct DeepSeekWeightLoader {
    public let denseStore: DenseTensorStore
    public let expertBank: ExpertSlotBank
    public let expertStreamingConfig: ExpertStreamingConfig
    public let expertStreamingMetrics: ExpertStreamingMetrics?
    public let expertPrefetcher: ExpertPrefetcher
    public let residentExpertScales: ResidentExpertTensors?
    public let pinnedExpertCodes: ResidentExpertTensors?
    public let expertLayerStager: ExpertLayerStager?
    // Dedicated capacity-0 side bank for concurrent decode-step slice reads.
    // Capacity 0 => no cache/LRU mutation, so concurrent preads are race-free
    // and read byte-identical ranges through the trusted metered path.
    private let decodeSideBank: ExpertSlotBank?
    private let bridge: MLXArrayTensorBridge
    private let scheduledPinnedDecodePrefetches = ScheduledDecodePrefetches()
    private let decodeExpertHistory = DecodeExpertHistory()
    private let routedExpertProjectionPlans: [RoutedExpertProjectionKey: RoutedExpertProjectionPlan]

    /// Sized for the official 48 GB runner: pin only where at least that
    /// budget exists, and never more than two layers of codes (~6.4 GiB).
    private static let pinningMinimumPhysicalMemoryBytes: UInt64 = 40 << 30
    // Pinning trades 3.2 GiB of RAM per hash layer for guaranteed hits on
    // ~2/43 layers. With init page-cache warming and predictive decode
    // prefetch in place, those same gigabytes serve ALL layers better as
    // kernel page cache, and the hash layers' exact token-derived experts are
    // side-bank prefetched at step entry instead (same overlap, no residency).
    private static let pinnedHashLayerCap = 0

    public init(
        weightsPath: String,
        expertStreamingConfig: ExpertStreamingConfig = .fromEnvironment(),
        expertStreamingMetrics: ExpertStreamingMetrics? = nil,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) throws {
        let metrics = expertStreamingMetrics
            ?? (expertStreamingConfig.recordsMetrics ? ExpertStreamingMetrics() : nil)
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
        self.expertStreamingConfig = expertStreamingConfig
        self.expertStreamingMetrics = metrics
        let expertBank = try ExpertSlotBank(
            manifestPath: "\(weightsPath)/experts/manifest.json",
            capacity: expertStreamingConfig.tensorCacheCapacity,
            metrics: metrics
        )
        self.expertBank = expertBank
        self.routedExpertProjectionPlans = Self.buildRoutedExpertProjectionPlans(expertBank: expertBank)
        self.expertPrefetcher = ExpertPrefetcher(expertBank: expertBank)
        // Resident stores come from a process-wide registry: the trusted
        // benchmark harness keeps two loaders alive at once, and duplicating
        // ~14 GiB of resident data per loader would threaten the 48 GB
        // runner budget.
        let manifestPath = "\(weightsPath)/experts/manifest.json"
        self.residentExpertScales = ResidentExpertStoreRegistry.scales(
            manifestPath: manifestPath,
            metrics: metrics
        )
        // Pinning trades RAM for guaranteed hits on the token-id-routed
        // layers; only worthwhile at the official 48 GB budget or above,
        // and capped so the pinned codes (~3.2 GiB per layer) leave headroom
        // for the resident scales, staging buffers, and page cache inside
        // that budget. Both constants encode the OFFICIAL runner's memory
        // math — do not raise them because a larger local machine has room.
        let hashLayerCount = (try? DeepSeekConfig.load(from: weightsPath))?.numHashLayers ?? 0
        self.pinnedExpertCodes = ProcessInfo.processInfo.physicalMemory >= Self.pinningMinimumPhysicalMemoryBytes
            ? ResidentExpertStoreRegistry.pinnedHashLayerCodes(
                manifestPath: manifestPath,
                hashLayerCount: min(hashLayerCount, Self.pinnedHashLayerCap),
                metrics: metrics
            )
            : nil
        self.expertLayerStager = ExpertLayerStager(
            manifestPath: manifestPath,
            metrics: metrics
        )
        self.decodeSideBank = try? ExpertSlotBank(
            manifestPath: manifestPath,
            capacity: 0,
            metrics: metrics
        )
        self.bridge = bridge
    }

    public init(
        denseStore: DenseTensorStore,
        expertBank: ExpertSlotBank,
        expertStreamingConfig: ExpertStreamingConfig = ExpertStreamingConfig(),
        expertStreamingMetrics: ExpertStreamingMetrics? = nil,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) {
        self.denseStore = denseStore
        self.expertBank = expertBank
        self.routedExpertProjectionPlans = Self.buildRoutedExpertProjectionPlans(expertBank: expertBank)
        self.expertStreamingConfig = expertStreamingConfig
        self.expertStreamingMetrics = expertStreamingMetrics ?? expertBank.metrics
        self.expertPrefetcher = ExpertPrefetcher(expertBank: expertBank)
        self.residentExpertScales = nil
        self.pinnedExpertCodes = nil
        self.expertLayerStager = nil
        self.decodeSideBank = nil
        self.bridge = bridge
    }

    public func resolveDenseName(_ candidates: [String]) throws -> String {
        for candidate in candidates where denseStore.record(named: candidate) != nil {
            return candidate
        }
        throw MLXFastError.invalidInput(
            "dense tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func materializedDenseTensor(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let name = try resolveDenseName(candidates)
        let tensor = try denseStore.materializedTensor(named: name)
        try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func denseArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedDenseTensor(
                candidates: candidates,
                expectedShape: expectedShape
            )
        )
    }

    public func optionalDenseArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray? {
        for candidate in candidates where denseStore.record(named: candidate) != nil {
            return try denseArray(candidates: [candidate], expectedShape: expectedShape)
        }
        return nil
    }

    public func denseLinearWeight(
        candidates: [String],
        expectedShape: [Int]
    ) throws -> DeepSeekLinearWeight {
        let name = try resolveDenseName(candidates)
        return try linearWeight(
            baseName: name,
            expectedShape: expectedShape,
            tensor: denseStore.materializedTensor(named: name),
            companionTensor: { companionName, _ in
                guard denseStore.record(named: companionName) != nil else {
                    return nil
                }
                return try denseStore.materializedTensor(named: companionName)
            }
        )
    }

    public func materializedExpertTensor(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let tensor = try expertBank.materializedTensor(named: name)
        try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func expertArray(named name: String, expectedShape: [Int]? = nil) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedExpertTensor(
                named: name,
                expectedShape: expectedShape
            )
        )
    }

    public func resolveExpertName(_ candidates: [String]) throws -> String {
        for candidate in candidates where expertBank.record(named: candidate) != nil {
            return candidate
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func expertArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try expertArray(
            named: resolveExpertName(candidates),
            expectedShape: expectedShape
        )
    }

    static func decodePrefetchKey(_ name: String, _ index: Int) -> String {
        "\(name)#\(index)"
    }

    private static func buildRoutedExpertProjectionPlans(
        expertBank: ExpertSlotBank
    ) -> [RoutedExpertProjectionKey: RoutedExpertProjectionPlan] {
        let maxLayerIndex = expertBank.manifest.expertTensors.compactMap(\.layerIndex).max() ?? -1
        guard maxLayerIndex >= 0 else {
            return [:]
        }
        var plans: [RoutedExpertProjectionKey: RoutedExpertProjectionPlan] = [:]
        plans.reserveCapacity((maxLayerIndex + 1) * 3)
        for layerIndex in 0...maxLayerIndex {
            for projection in [DeepSeekExpertProjection.gate, .up, .down] {
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: 0,
                    projection: projection
                )
                guard
                    let candidate = candidates.first(where: { expertBank.record(named: $0) != nil }),
                    let record = expertBank.record(named: candidate),
                    record.shape.count >= 3,
                    let expertCount = record.shape.first,
                    expertCount > 0
                else {
                    continue
                }
                let scalesName = Self.companionName(for: candidate, suffix: "scales")
                let biasesName = Self.companionName(for: candidate, suffix: "biases")
                plans[RoutedExpertProjectionKey(layerIndex: layerIndex, projection: projection)] =
                    RoutedExpertProjectionPlan(
                        name: candidate,
                        record: record,
                        scalesName: scalesName,
                        scalesRecord: expertBank.record(named: scalesName),
                        biasesName: biasesName,
                        biasesRecord: expertBank.record(named: biasesName),
                        decodePrefetchKeys: (0..<expertCount).map { Self.decodePrefetchKey(candidate, $0) }
                    )
            }
        }
        return plans
    }

    private func routedExpertProjectionPlan(
        layerIndex: Int,
        projection: DeepSeekExpertProjection
    ) -> RoutedExpertProjectionPlan? {
        routedExpertProjectionPlans[RoutedExpertProjectionKey(layerIndex: layerIndex, projection: projection)]
    }

    /// Resolves a code tensor's RAM-resident scales slice as a
    /// MaterializedTensor (raw bytes + shape/dtype), off the compute thread.
    /// Thread-safe: the resident store is immutable and every decode returns
    /// a fresh buffer. Returns nil when the scales are not resident.
    static func residentScalesTensor(
        residentScales: ResidentExpertTensors?,
        codeName: String,
        expertIndex: Int
    ) -> MaterializedTensor? {
        guard let residentScales else {
            return nil
        }
        let scalesName = codeName.hasSuffix(".weight")
            ? String(codeName.dropLast(".weight".count)) + ".scales"
            : codeName + ".scales"
        return residentScales.materializedTensor(
            named: scalesName,
            firstAxisIndex: expertIndex
        )
    }

    /// Builds the base scales MLXArray for a code tensor off the compute
    /// thread, but only when the scales are RAM-resident (the common frontier
    /// path). Thread-safe: the resident store is immutable and makeArray is a
    /// per-buffer copy. Returns nil to leave scales construction on the compute
    /// thread (byte-identical either way).
    static func residentScalesArray(
        residentScales: ResidentExpertTensors?,
        bridge: MLXArrayTensorBridge,
        codeName: String,
        expertIndex: Int
    ) -> MLXArray? {
        guard let scalesTensor = residentScalesTensor(
            residentScales: residentScales,
            codeName: codeName,
            expertIndex: expertIndex
        ) else {
            return nil
        }
        return try? bridge.makeArray(from: scalesTensor)
    }

    /// Concurrently pre-reads the routed-expert code slices a 1-token decode
    /// step is about to consume, through the capacity-0 side bank, returning a
    /// map keyed by `decodePrefetchKey`. Returns nil when the fast path does
    /// not apply (no side bank, main byte-cache disabled, or nothing to fetch);
    /// callers then use the normal serial per-expert bank reads. Non-pinned
    /// slices read through the trusted side bank; pinned hash-layer code slices
    /// are built from their resident bytes off-thread too. Byte-identical to
    /// the serial path.
    public func prefetchDecodeExpertCodes(
        layerIndex: Int,
        expertIndices: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) -> [String: StagedExpertCode]? {
        guard let sideBank = decodeSideBank, expertBank.capacity > 0, !expertIndices.isEmpty else {
            return nil
        }
        let projections: [(DeepSeekExpertProjection, [Int])] = [
            (.gate, [intermediateSize, hiddenSize]),
            (.up, [intermediateSize, hiddenSize]),
            (.down, [hiddenSize, intermediateSize]),
        ]
        var keys: [String] = []
        var names: [String] = []
        var indices: [Int] = []
        var seen = Set<String>()
        for expertIndex in expertIndices {
            for (projection, expectedShape) in projections {
                if let plan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: projection),
                   plan.record.shape.count == expectedShape.count + 1,
                   plan.record.shape.first.map({ expertIndex < $0 }) == true {
                    let key = plan.decodePrefetchKey(for: expertIndex)
                    if seen.insert(key).inserted {
                        keys.append(key)
                        names.append(plan.name)
                        indices.append(expertIndex)
                    }
                    continue
                }
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection
                )
                for candidate in candidates {
                    guard let record = expertBank.record(named: candidate) else {
                        continue
                    }
                    let isStacked = record.shape.count == expectedShape.count + 1
                        && record.shape.first.map { expertIndex < $0 } == true
                    // Match expertLinearWeight: first matching candidate wins.
                    guard isStacked else {
                        break
                    }
                    let key = Self.decodePrefetchKey(candidate, expertIndex)
                    if seen.insert(key).inserted {
                        keys.append(key)
                        names.append(candidate)
                        indices.append(expertIndex)
                    }
                    break
                }
            }
        }
        guard !keys.isEmpty else {
            return nil
        }
        let bridge = self.bridge
        let residentScales = self.residentExpertScales
        let pinnedCodes = self.pinnedExpertCodes
        var results = [StagedExpertCode?](repeating: nil, count: keys.count)
        results.withUnsafeMutableBufferPointer { buffer in
            let sink = DecodePrefetchSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: keys.count) { index in
                let name = names[index]
                let expertIndex = indices[index]
                // Read the slice AND build its base MLXArray (the eager
                // Data->Metal copy) here on the worker thread. The compute
                // thread's per-expert loop then skips that memcpy and only
                // wires the lazy reshape/quant assembly into the graph.
                // Byte-identical: same bytes, same array constructor.
                guard
                    let tensor = pinnedCodes?.materializedTensor(
                        named: name,
                        firstAxisIndex: expertIndex
                    ) ?? (try? sideBank.materializedTensor(
                        named: name,
                        firstAxisIndex: expertIndex
                    )),
                    let array = try? bridge.makeArray(from: tensor)
                else {
                    return
                }
                // Also build the resident scales' base array off-thread.
                let scalesArray = Self.residentScalesArray(
                    residentScales: residentScales,
                    bridge: bridge,
                    codeName: name,
                    expertIndex: expertIndex
                )
                sink.buffer[index] = StagedExpertCode(
                    tensor: tensor,
                    array: array,
                    scalesArray: scalesArray
                )
            }
        }
        var map: [String: StagedExpertCode] = [:]
        map.reserveCapacity(keys.count)
        for (index, key) in keys.enumerated() {
            if let staged = results[index] {
                map[key] = staged
            }
        }
        return map.isEmpty ? nil : map
    }

    /// Starts building resident pinned-code decode slices for a hash-routed
    /// layer before that layer reaches MoE. Only the first two hash layers are
    /// pinned on the official runner; unpinned layers return false so callers
    /// still issue the normal disk read-ahead.
    @discardableResult
    public func schedulePinnedDecodeExpertCodes(
        layerIndex: Int,
        expertIndices: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) -> Bool {
        guard let pinnedCodes = self.pinnedExpertCodes, !expertIndices.isEmpty else {
            scheduledPinnedDecodePrefetches.remove(layerIndex: layerIndex)
            return false
        }
        let plan = pinnedDecodeExpertCodePlan(
            pinnedCodes: pinnedCodes,
            layerIndex: layerIndex,
            expertIndices: expertIndices,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize
        )
        guard !plan.keys.isEmpty else {
            scheduledPinnedDecodePrefetches.remove(layerIndex: layerIndex)
            return false
        }
        let fullyPinned = plan.keys.count == Set(expertIndices).count * 3
        guard fullyPinned else {
            scheduledPinnedDecodePrefetches.remove(layerIndex: layerIndex)
            return false
        }
        let bridge = self.bridge
        let residentScales = self.residentExpertScales
        scheduledPinnedDecodePrefetches.schedule(layerIndex: layerIndex) {
            Self.buildPinnedDecodeExpertCodes(
                keys: plan.keys,
                names: plan.names,
                indices: plan.indices,
                pinnedCodes: pinnedCodes,
                residentScales: residentScales,
                bridge: bridge
            )
        }
        return true
    }

    public func consumeScheduledPinnedDecodeExpertCodes(layerIndex: Int) -> [String: StagedExpertCode]? {
        scheduledPinnedDecodePrefetches.consume(layerIndex: layerIndex)
    }

    /// Records the experts a one-token decode step routed at a layer. Used
    /// only as a prefetch hint for the NEXT step (decode expert selection has
    /// strong step-to-step locality); never affects routing or outputs.
    public func recordDecodeExperts(layerIndex: Int, experts: [Int]) {
        decodeExpertHistory.record(layerIndex: layerIndex, experts: experts)
    }

    /// Predictive decode prefetch (sanctioned: latency-only, prompt-agnostic
    /// mechanism — predictions come from the CURRENT request's own routing
    /// history, like KV reuse). Schedules a background side-bank read + base
    /// MLXArray build for the experts the layer routed on the PREVIOUS decode
    /// step, so when the layer's actual routing resolves, correctly predicted
    /// slices are already read and built. Mispredicted experts are simply not
    /// consumed; missing ones are fetched on demand exactly as before, so
    /// outputs are unchanged by construction.
    public func schedulePredictedDecodeExpertCodes(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int
    ) {
        schedulePredictedDecodeExpertCodes(
            layerIndex: layerIndex,
            experts: decodeExpertHistory.predicted(layerIndex: layerIndex) ?? [],
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize
        )
    }

    /// Variant with an explicit expert set — used with the hash layers' exact
    /// token-derived routing at decode entry, where the experts are known
    /// before the forward begins.
    public func schedulePredictedDecodeExpertCodes(
        layerIndex: Int,
        experts: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) {
        guard
            !scheduledPinnedDecodePrefetches.hasEntry(layerIndex: layerIndex),
            !experts.isEmpty
        else {
            return
        }
        let loader = self
        scheduledPinnedDecodePrefetches.schedule(layerIndex: layerIndex) {
            loader.prefetchDecodeExpertCodes(
                layerIndex: layerIndex,
                expertIndices: experts,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize
            )
        }
    }

    /// Fills in any experts the consumed (possibly predicted) prefetch map is
    /// missing, using the same concurrent side-bank reads as the full
    /// prefetch. Returns the merged map (or the fresh map when nothing was
    /// consumed).
    public func mergeMissingDecodeExpertCodes(
        _ consumed: [String: StagedExpertCode]?,
        layerIndex: Int,
        experts: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) -> [String: StagedExpertCode]? {
        guard var map = consumed else {
            return prefetchDecodeExpertCodes(
                layerIndex: layerIndex,
                expertIndices: experts,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize
            )
        }
        // An expert is covered when its gate-projection key is present (all
        // three projections are always built together).
        var missing: [Int] = []
        for expertIndex in experts {
            let candidates = DeepSeekWeightNames.routedExpert(
                layerIndex: layerIndex,
                expertIndex: expertIndex,
                projection: .gate
            )
            let covered = candidates.contains { candidate in
                map[Self.decodePrefetchKey(candidate, expertIndex)] != nil
            }
            if !covered {
                missing.append(expertIndex)
            }
        }
        if !missing.isEmpty,
           let fetched = prefetchDecodeExpertCodes(
               layerIndex: layerIndex,
               expertIndices: missing,
               hiddenSize: hiddenSize,
               intermediateSize: intermediateSize
           ) {
            map.merge(fetched) { current, _ in current }
        }
        return map
    }

    /// Per-expert-completion variant of the decode-step slice prefetch:
    /// schedules the SAME side-bank reads + base MLXArray builds as
    /// prefetchDecodeExpertCodes (byte-identical worker bodies, same
    /// decodePrefetchKey layout), but signals readiness PER EXPERT instead of
    /// after the whole batch, so the decode compute thread can start building
    /// an expert's subgraph as soon as that expert's OWN gate/up/down slices
    /// are in RAM — overlapping graph-build CPU work with the SSD tail of the
    /// remaining experts' reads. Slices already present in `consumed`
    /// (scheduled pinned/predicted prebuilds, built by the same worker body)
    /// are pre-populated into the stream and NOT re-read. Returns nil when the
    /// fast path does not apply — same guards as prefetchDecodeExpertCodes,
    /// plus nothing consumed to serve — and callers then fall back to
    /// mergeMissingDecodeExpertCodes exactly as before. Correctness never
    /// depends on the stream: a failed or slow slice is simply absent and the
    /// consumer's normal bank read supplies the same bytes.
    public func streamDecodeExpertCodes(
        layerIndex: Int,
        expertIndices: [Int],
        hiddenSize: Int,
        intermediateSize: Int,
        consumed: [String: StagedExpertCode]? = nil
    ) -> DecodeExpertCodeStream? {
        guard let sideBank = decodeSideBank, expertBank.capacity > 0, !expertIndices.isEmpty else {
            return nil
        }
        let projections: [(DeepSeekExpertProjection, [Int])] = [
            (.gate, [intermediateSize, hiddenSize]),
            (.up, [intermediateSize, hiddenSize]),
            (.down, [hiddenSize, intermediateSize]),
        ]
        let prepopulated = consumed ?? [:]
        var keys: [String] = []
        var names: [String] = []
        var indices: [Int] = []
        var roles: [DeepSeekExpertProjection] = []
        var seen = Set<String>()
        // Fused gate+up decode weight: eligibility is a per-LAYER record
        // contract check (identical stacked shapes/dtypes, no biases). An
        // expert additionally participates only when BOTH its gate and up
        // slices are scheduled on THIS stream via the plan path, so
        // prepopulated entries (scheduled pinned/predicted prebuilds) and the
        // candidates fallback keep the unfused path exactly as before.
        let fusionEligible = decodeGateUpFusionEligible(
            layerIndex: layerIndex,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize
        )
        var scheduledFusionHalves: [Int: (gate: Bool, up: Bool)] = [:]
        // Same (expert, projection) -> (key, name) resolution as
        // prefetchDecodeExpertCodes; the only addition is skipping slices the
        // consumed map already holds.
        for expertIndex in expertIndices {
            for (projection, expectedShape) in projections {
                if let plan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: projection),
                   plan.record.shape.count == expectedShape.count + 1,
                   plan.record.shape.first.map({ expertIndex < $0 }) == true {
                    let key = plan.decodePrefetchKey(for: expertIndex)
                    if seen.insert(key).inserted, prepopulated[key] == nil {
                        keys.append(key)
                        names.append(plan.name)
                        indices.append(expertIndex)
                        roles.append(projection)
                        if fusionEligible, projection != .down {
                            var halves = scheduledFusionHalves[expertIndex] ?? (gate: false, up: false)
                            if projection == .gate {
                                halves.gate = true
                            } else {
                                halves.up = true
                            }
                            scheduledFusionHalves[expertIndex] = halves
                        }
                    }
                    continue
                }
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection
                )
                for candidate in candidates {
                    guard let record = expertBank.record(named: candidate) else {
                        continue
                    }
                    let isStacked = record.shape.count == expectedShape.count + 1
                        && record.shape.first.map { expertIndex < $0 } == true
                    // Match expertLinearWeight: first matching candidate wins.
                    guard isStacked else {
                        break
                    }
                    let key = Self.decodePrefetchKey(candidate, expertIndex)
                    if seen.insert(key).inserted, prepopulated[key] == nil {
                        keys.append(key)
                        names.append(candidate)
                        indices.append(expertIndex)
                        roles.append(projection)
                    }
                    break
                }
            }
        }
        guard !keys.isEmpty else {
            // Nothing left to read: either every resolvable slice was already
            // consumed (serve those prebuilt entries through a no-wait
            // stream), or the layer resolves no stacked slices at all (nil ->
            // caller fallback, which also resolves nothing, exactly as today).
            return prepopulated.isEmpty
                ? nil
                : DecodeExpertCodeStream(results: prepopulated, expertGroups: [:])
        }
        var expertGroups: [Int: DispatchGroup] = [:]
        for expertIndex in indices where expertGroups[expertIndex] == nil {
            expertGroups[expertIndex] = DispatchGroup()
        }
        let stream = DecodeExpertCodeStream(results: prepopulated, expertGroups: expertGroups)
        let fusionExperts = Set(
            scheduledFusionHalves.filter { $0.value.gate && $0.value.up }.map(\.key)
        )
        let fusion: DecodeGateUpFusionState? = fusionExperts.isEmpty
            ? nil
            : DecodeGateUpFusionState(experts: fusionExperts)
        let bridge = self.bridge
        let residentScales = self.residentExpertScales
        let pinnedCodes = self.pinnedExpertCodes
        // Every enter() happens before the stream is handed out (and thus
        // before any waitForExpert can run); each worker leaves exactly once
        // via defer, so an expert's group balances when its last slice lands.
        for index in keys.indices {
            let key = keys[index]
            let name = names[index]
            let expertIndex = indices[index]
            let role = roles[index]
            guard let group = expertGroups[expertIndex] else {
                continue
            }
            group.enter()
            decodeExpertStreamQueue.async {
                defer { group.leave() }
                // Identical worker body to prefetchDecodeExpertCodes: read
                // the slice AND build its base MLXArray (the eager
                // Data->Metal copy) off the compute thread. Byte-identical:
                // same bytes, same array constructor.
                guard
                    let tensor = pinnedCodes?.materializedTensor(
                        named: name,
                        firstAxisIndex: expertIndex
                    ) ?? (try? sideBank.materializedTensor(
                        named: name,
                        firstAxisIndex: expertIndex
                    )),
                    let array = try? bridge.makeArray(from: tensor)
                else {
                    return
                }
                // Also build the resident scales' base array off-thread. The
                // slice is resolved tensor-first (the exact two calls
                // residentScalesArray makes) so the fusion step below can
                // reuse the raw scales bytes without a second expansion.
                let scalesTensor = Self.residentScalesTensor(
                    residentScales: residentScales,
                    codeName: name,
                    expertIndex: expertIndex
                )
                let scalesArray = scalesTensor.flatMap { try? bridge.makeArray(from: $0) }
                stream.store(
                    key: key,
                    code: StagedExpertCode(tensor: tensor, array: array, scalesArray: scalesArray)
                )
                // Fused gate+up build: deposit this half; whichever of the
                // expert's gate/up workers lands SECOND receives the pair and
                // builds the fused weight here, before its group.leave, so a
                // successful waitForExpert always observes the fused entry.
                // Failure at any step (non-resident scales, guard mismatch)
                // simply leaves the fused slot empty and the consumer keeps
                // the unfused per-projection path over the individual entries
                // stored above — correctness never depends on the fusion.
                guard
                    let fusion,
                    role != .down,
                    let scalesTensor,
                    let pair = fusion.deposit(
                        expertIndex: expertIndex,
                        projection: role,
                        half: DecodeGateUpFusionState.Half(codes: tensor, scales: scalesTensor)
                    ),
                    let fused = Self.buildFusedGateUpWeight(
                        gate: pair.gate,
                        up: pair.up,
                        hiddenSize: hiddenSize,
                        intermediateSize: intermediateSize,
                        bridge: bridge
                    )
                else {
                    return
                }
                stream.storeFusedGateUp(expertIndex: expertIndex, weight: fused)
            }
        }
        return stream
    }

    /// Layer-level record contract for the decode fused gate+up weight: both
    /// projections must resolve through the precomputed plan to stacked U32
    /// code tensors of IDENTICAL shape sized for [intermediateSize,
    /// hiddenSize], with scales companions of identical shape/dtype, and
    /// neither may carry a biases companion (the fused weight is built
    /// biases-free, so mode/groupSize/bits derivation matches the
    /// per-projection linearWeight exactly). Any mismatch keeps the whole
    /// layer on the unfused path.
    private func decodeGateUpFusionEligible(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int
    ) -> Bool {
        guard
            let gatePlan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: .gate),
            let upPlan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: .up),
            gatePlan.record.dtype == "U32",
            upPlan.record.dtype == "U32",
            gatePlan.record.shape.count == 3,
            gatePlan.record.shape == upPlan.record.shape,
            gatePlan.record.shape[1] == intermediateSize,
            let packedInput = gatePlan.record.shape.last,
            packedInput > 0,
            (packedInput * 32) % hiddenSize == 0,
            [2, 4, 8].contains(packedInput * 32 / hiddenSize),
            let gateScales = gatePlan.scalesRecord,
            let upScales = upPlan.scalesRecord,
            gateScales.dtype == upScales.dtype,
            gateScales.shape == upScales.shape,
            gateScales.shape.count == 3,
            gateScales.shape[0] == gatePlan.record.shape[0],
            gateScales.shape[1] == intermediateSize,
            let scaleGroups = gateScales.shape.last,
            scaleGroups > 0,
            hiddenSize % scaleGroups == 0,
            gatePlan.biasesRecord == nil,
            upPlan.biasesRecord == nil
        else {
            return false
        }
        return true
    }

    /// Builds ONE fused quantized weight whose output rows are the gate
    /// projection's rows followed by the up projection's rows, for the decode
    /// streamed path. Bit-exactness: quantizedMM computes each OUTPUT row as
    /// an independent per-group dot product over that row's code/scale bytes
    /// and the shared input, so row-concatenating two weight matrices that
    /// share the same input leaves every row's accumulation order — and
    /// therefore its result bits — unchanged; rows [0..<intermediate] of the
    /// fused output are exactly the gate output and rows [intermediate...]
    /// exactly the up output (the same argument the batched prefill path
    /// documents for slicing experts out of one stacked tensor). The byte
    /// concatenation IS axis-0 concat for row-major [rows, width] tensors,
    /// and groupSize/bits/mode are derived from the same inputs linearWeight
    /// uses for each half (biases were guarded absent at the layer level), so
    /// they match the individual weights by construction. Returns nil when
    /// any shape/dtype guard fails; callers then keep the unfused path.
    private static func buildFusedGateUpWeight(
        gate: DecodeGateUpFusionState.Half,
        up: DecodeGateUpFusionState.Half,
        hiddenSize: Int,
        intermediateSize: Int,
        bridge: MLXArrayTensorBridge
    ) -> DeepSeekLinearWeight? {
        guard
            gate.codes.dtype == .u32,
            up.codes.dtype == .u32,
            gate.codes.shape.count == 2,
            gate.codes.shape == up.codes.shape,
            gate.scales.dtype == up.scales.dtype,
            gate.scales.shape.count == 2,
            gate.scales.shape == up.scales.shape,
            let rows = gate.codes.shape.first,
            rows == intermediateSize,
            gate.scales.shape.first == rows,
            let packedInput = gate.codes.shape.last,
            packedInput > 0,
            (packedInput * 32) % hiddenSize == 0,
            let scaleGroups = gate.scales.shape.last,
            scaleGroups > 0,
            hiddenSize % scaleGroups == 0
        else {
            return nil
        }
        let bits = packedInput * 32 / hiddenSize
        guard [2, 4, 8].contains(bits) else {
            return nil
        }
        var fusedCodeBytes = Data(capacity: gate.codes.bytes.count + up.codes.bytes.count)
        fusedCodeBytes.append(gate.codes.bytes)
        fusedCodeBytes.append(up.codes.bytes)
        var fusedScalesBytes = Data(capacity: gate.scales.bytes.count + up.scales.bytes.count)
        fusedScalesBytes.append(gate.scales.bytes)
        fusedScalesBytes.append(up.scales.bytes)
        guard
            let fusedCodes = try? MaterializedTensor(
                name: "\(gate.codes.name)#fused-gate-up",
                dtype: .u32,
                shape: [2 * rows, packedInput],
                bytes: fusedCodeBytes
            ),
            let fusedScales = try? MaterializedTensor(
                name: "\(gate.scales.name)#fused-gate-up",
                dtype: gate.scales.dtype,
                shape: [2 * rows, scaleGroups],
                bytes: fusedScalesBytes
            ),
            let weightArray = try? bridge.makeArray(from: fusedCodes),
            let scalesArray = try? bridge.makeArray(from: fusedScales)
        else {
            return nil
        }
        // Same mode predicate as linearWeight with the biases companion
        // guarded absent: u8 scales => mxfp4, otherwise affine.
        let mode: QuantizationMode = gate.scales.dtype == .u8 ? .mxfp4 : .affine
        return DeepSeekLinearWeight(
            weight: weightArray,
            scales: scalesArray,
            biases: nil,
            logicalShape: [2 * rows, hiddenSize],
            groupSize: hiddenSize / scaleGroups,
            bits: bits,
            mode: mode
        )
    }

    /// Load-time row-concatenation of two quantized dense weights that
    /// consume the SAME input vector, into ONE quantized weight whose output
    /// rows are `first`'s rows followed by `second`'s rows.
    ///
    /// Bit-exactness: quantizedMM computes each OUTPUT row as an independent
    /// per-group dot product over that row's code/scale(/bias) bytes and the
    /// shared input, so row-concatenation changes neither any row's
    /// accumulation order nor its result bits — columns [0..<firstRows] of
    /// the fused linear output are exactly `first`'s output and
    /// [firstRows..<firstRows+secondRows] exactly `second`'s (the same
    /// argument the decode streamed gate+up fusion above and the batched
    /// prefill path document). `concatenated(axis: 0)` on the row-major 2-D
    /// packed codes/scales/biases IS the byte-level row concat, and the fused
    /// weight reuses the halves' identical groupSize/bits/mode verbatim.
    /// Guards: both quantized with identical groupSize/bits/mode and scales
    /// dtype, 2-D packed layouts with matching widths and logical input, and
    /// biases either both absent or both present with matching shapes/dtypes;
    /// any mismatch returns nil and callers keep the unfused per-projection
    /// path. Call at LOAD time only — the concat allocates real buffers that
    /// warmup then materializes outside every scored window.
    static func fusedRowConcatWeight(
        _ first: DeepSeekLinearWeight,
        _ second: DeepSeekLinearWeight
    ) -> DeepSeekLinearWeight? {
        guard
            let firstScales = first.scales,
            let secondScales = second.scales,
            first.groupSize == second.groupSize,
            first.bits == second.bits,
            first.mode == second.mode,
            first.weight.dtype == second.weight.dtype,
            firstScales.dtype == secondScales.dtype,
            first.logicalShape.count == 2,
            second.logicalShape.count == 2,
            let firstRows = first.logicalShape.first,
            let secondRows = second.logicalShape.first,
            firstRows > 0,
            secondRows > 0,
            first.logicalShape.last == second.logicalShape.last,
            first.weight.shape.count == 2,
            second.weight.shape.count == 2,
            first.weight.shape[0] == firstRows,
            second.weight.shape[0] == secondRows,
            first.weight.shape[1] == second.weight.shape[1],
            firstScales.shape.count == 2,
            secondScales.shape.count == 2,
            firstScales.shape[0] == firstRows,
            secondScales.shape[0] == secondRows,
            firstScales.shape[1] == secondScales.shape[1]
        else {
            return nil
        }
        let fusedBiases: MLXArray?
        switch (first.biases, second.biases) {
        case (nil, nil):
            fusedBiases = nil
        case (let firstBiases?, let secondBiases?):
            guard
                firstBiases.dtype == secondBiases.dtype,
                firstBiases.shape.count == 2,
                secondBiases.shape.count == 2,
                firstBiases.shape[0] == firstRows,
                secondBiases.shape[0] == secondRows,
                firstBiases.shape[1] == firstScales.shape[1],
                secondBiases.shape[1] == secondScales.shape[1]
            else {
                return nil
            }
            fusedBiases = concatenated([firstBiases, secondBiases], axis: 0)
        default:
            return nil
        }
        return DeepSeekLinearWeight(
            weight: concatenated([first.weight, second.weight], axis: 0),
            scales: concatenated([firstScales, secondScales], axis: 0),
            biases: fusedBiases,
            logicalShape: [firstRows + secondRows, first.logicalShape[1]],
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode
        )
    }

    private func pinnedDecodeExpertCodePlan(
        pinnedCodes: ResidentExpertTensors,
        layerIndex: Int,
        expertIndices: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) -> (keys: [String], names: [String], indices: [Int]) {
        let projections: [(DeepSeekExpertProjection, [Int])] = [
            (.gate, [intermediateSize, hiddenSize]),
            (.up, [intermediateSize, hiddenSize]),
            (.down, [hiddenSize, intermediateSize]),
        ]
        var keys: [String] = []
        var names: [String] = []
        var indices: [Int] = []
        var seen = Set<String>()
        for expertIndex in expertIndices {
            for (projection, expectedShape) in projections {
                if let plan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: projection),
                   plan.record.shape.count == expectedShape.count + 1,
                   plan.record.shape.first.map({ expertIndex < $0 }) == true {
                    guard pinnedCodes.isResident(name: plan.name) else {
                        continue
                    }
                    let key = plan.decodePrefetchKey(for: expertIndex)
                    if seen.insert(key).inserted {
                        keys.append(key)
                        names.append(plan.name)
                        indices.append(expertIndex)
                    }
                    continue
                }
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection
                )
                for candidate in candidates {
                    guard let record = expertBank.record(named: candidate) else {
                        continue
                    }
                    let isStacked = record.shape.count == expectedShape.count + 1
                        && record.shape.first.map { expertIndex < $0 } == true
                    guard isStacked else {
                        break
                    }
                    guard pinnedCodes.isResident(name: candidate) else {
                        break
                    }
                    let key = Self.decodePrefetchKey(candidate, expertIndex)
                    if seen.insert(key).inserted {
                        keys.append(key)
                        names.append(candidate)
                        indices.append(expertIndex)
                    }
                    break
                }
            }
        }
        return (keys, names, indices)
    }

    private static func buildPinnedDecodeExpertCodes(
        keys: [String],
        names: [String],
        indices: [Int],
        pinnedCodes: ResidentExpertTensors,
        residentScales: ResidentExpertTensors?,
        bridge: MLXArrayTensorBridge
    ) -> [String: StagedExpertCode]? {
        var results = [StagedExpertCode?](repeating: nil, count: keys.count)
        results.withUnsafeMutableBufferPointer { buffer in
            let sink = DecodePrefetchSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: keys.count) { index in
                let name = names[index]
                let expertIndex = indices[index]
                guard
                    let tensor = pinnedCodes.materializedTensor(
                        named: name,
                        firstAxisIndex: expertIndex
                    ),
                    let array = try? bridge.makeArray(from: tensor)
                else {
                    return
                }
                let scalesArray = Self.residentScalesArray(
                    residentScales: residentScales,
                    bridge: bridge,
                    codeName: name,
                    expertIndex: expertIndex
                )
                sink.buffer[index] = StagedExpertCode(
                    tensor: tensor,
                    array: array,
                    scalesArray: scalesArray
                )
            }
        }
        var map: [String: StagedExpertCode] = [:]
        map.reserveCapacity(keys.count)
        for (index, key) in keys.enumerated() {
            if let staged = results[index] {
                map[key] = staged
            }
        }
        return map.isEmpty ? nil : map
    }

    /// Prefill/warmup analogue of prefetchDecodeExpertCodes for the staged
    /// (whole-layer) path: builds each active expert's base MLXArray from the
    /// already-staged layer buffer concurrently, so the per-expert compute
    /// loop skips the serial Data->Metal copy. Byte-identical to the serial
    /// stagedSliceTensor + bridge.makeArray it replaces. Returns nil when the
    /// layer is not staged; callers then use the serial staged path.
    public func prefetchStagedExpertCodes(
        layerIndex: Int,
        expertIndices: [Int],
        hiddenSize: Int,
        intermediateSize: Int
    ) -> [String: StagedExpertCode]? {
        guard expertLayerStager != nil, !expertIndices.isEmpty else {
            return nil
        }
        let projections: [(DeepSeekExpertProjection, [Int])] = [
            (.gate, [intermediateSize, hiddenSize]),
            (.up, [intermediateSize, hiddenSize]),
            (.down, [hiddenSize, intermediateSize]),
        ]
        var keys: [String] = []
        var names: [String] = []
        var indices: [Int] = []
        var seen = Set<String>()
        for expertIndex in expertIndices {
            for (projection, expectedShape) in projections {
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection
                )
                for candidate in candidates {
                    guard let record = expertBank.record(named: candidate) else {
                        continue
                    }
                    let isStacked = record.shape.count == expectedShape.count + 1
                        && record.shape.first.map { expertIndex < $0 } == true
                    guard isStacked else {
                        break
                    }
                    if pinnedExpertCodes?.isResident(name: candidate) == true {
                        break
                    }
                    let key = Self.decodePrefetchKey(candidate, expertIndex)
                    if seen.insert(key).inserted {
                        keys.append(key)
                        names.append(candidate)
                        indices.append(expertIndex)
                    }
                    break
                }
            }
        }
        guard !keys.isEmpty else {
            return nil
        }
        let bridge = self.bridge
        let residentScales = self.residentExpertScales
        var results = [StagedExpertCode?](repeating: nil, count: keys.count)
        results.withUnsafeMutableBufferPointer { buffer in
            let sink = DecodePrefetchSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: keys.count) { index in
                // Cut the slice from the staged layer buffer AND build its base
                // MLXArray concurrently; the compute thread then only wires the
                // lazy reshape/quant assembly. Byte-identical to the serial path.
                guard
                    let tensor = self.stagedSliceTensor(
                        recordName: names[index],
                        expertIndex: indices[index]
                    ),
                    let array = try? bridge.makeArray(from: tensor)
                else {
                    return
                }
                let scalesArray = Self.residentScalesArray(
                    residentScales: residentScales,
                    bridge: bridge,
                    codeName: names[index],
                    expertIndex: indices[index]
                )
                sink.buffer[index] = StagedExpertCode(
                    tensor: tensor,
                    array: array,
                    scalesArray: scalesArray
                )
            }
        }
        var map: [String: StagedExpertCode] = [:]
        map.reserveCapacity(keys.count)
        for (index, key) in keys.enumerated() {
            if let staged = results[index] {
                map[key] = staged
            }
        }
        return map.isEmpty ? nil : map
    }

    /// One whole stacked routed-expert projection ([experts, rows, packed]
    /// codes plus stacked scales/biases companions) built for the batched
    /// prefill path. Byte sources are, in order: the staged whole-layer
    /// buffer, the RAM-pinned hash-layer store, or (scales) the resident
    /// scales store — all byte-identical stand-ins for the bank's stacked
    /// tensor by the same slice arithmetic the per-expert path uses.
    public struct StackedExpertProjection {
        public let weight: MLXArray
        public let scales: MLXArray
        public let biases: MLXArray?
        public let groupSize: Int
        public let bits: Int
        public let mode: QuantizationMode
    }

    /// Builds the whole stacked projection for a layer when every byte is
    /// already in RAM (staged buffer, pinned codes, resident scales). Returns
    /// nil when any piece is missing so callers fall back to the per-expert
    /// path, which reproduces existing behavior exactly.
    public func stackedExpertProjection(
        layerIndex: Int,
        projection: DeepSeekExpertProjection,
        expectedShape: [Int]
    ) -> StackedExpertProjection? {
        let candidates = DeepSeekWeightNames.routedExpert(
            layerIndex: layerIndex,
            expertIndex: 0,
            projection: projection
        )
        guard
            let candidate = candidates.first(where: { expertBank.record(named: $0) != nil }),
            let record = expertBank.record(named: candidate),
            record.shape.count == expectedShape.count + 1,
            record.dtype == "U32",
            let logicalInput = expectedShape.last,
            let packedInput = record.shape.last,
            logicalInput > 0,
            packedInput > 0,
            (packedInput * 32) % logicalInput == 0
        else {
            return nil
        }
        let bits = packedInput * 32 / logicalInput
        guard [2, 4, 8].contains(bits) else {
            return nil
        }

        // Whole-tensor code bytes: staged layer buffer first, then the pinned
        // hash-layer store. Both hold the exact stacked tensor bytes.
        let codesTensor: MaterializedTensor?
        if let staged = expertLayerStager?.stagedBytes(recordName: candidate),
           staged.count == record.byteLength,
           let dtype = try? TensorDType.parse(record.dtype) {
            codesTensor = try? MaterializedTensor(
                name: candidate,
                dtype: dtype,
                shape: record.shape,
                bytes: staged
            )
        } else {
            codesTensor = pinnedExpertCodes?.materializedTensor(named: candidate, firstAxisIndex: nil)
        }
        guard let codesTensor, let weightArray = try? bridge.makeArray(from: codesTensor) else {
            return nil
        }

        let scalesName = Self.companionName(for: candidate, suffix: "scales")
        guard let scalesRecord = expertBank.record(named: scalesName),
              scalesRecord.shape.count == expectedShape.count + 1,
              let scaleGroups = scalesRecord.shape.last,
              scaleGroups > 0,
              logicalInput % scaleGroups == 0
        else {
            return nil
        }
        // Staged whole-tensor scale bytes first: packed-resident scales are
        // deliberately kept in the staging plan so this no-decode byte path
        // serves prefill; the packed store then only ever expands small
        // per-expert slices on worker threads (decode steps), never whole
        // tensors on a scored path. Raw-resident scales (excluded from the
        // plan) fall through to the zero-copy resident branch as before.
        let scalesTensor: MaterializedTensor?
        if let staged = expertLayerStager?.stagedBytes(recordName: scalesName),
           staged.count == scalesRecord.byteLength,
           let dtype = try? TensorDType.parse(scalesRecord.dtype) {
            scalesTensor = try? MaterializedTensor(
                name: scalesName,
                dtype: dtype,
                shape: scalesRecord.shape,
                bytes: staged
            )
        } else if let resident = residentExpertScales?.materializedTensor(named: scalesName, firstAxisIndex: nil) {
            scalesTensor = resident
        } else {
            scalesTensor = nil
        }
        guard let scalesTensor, let scalesArray = try? bridge.makeArray(from: scalesTensor) else {
            return nil
        }

        let biasesName = Self.companionName(for: candidate, suffix: "biases")
        var biasesArray: MLXArray?
        if let biasesRecord = expertBank.record(named: biasesName) {
            // Biases exist but are only in RAM when staged; without them the
            // batched path cannot reproduce the affine math — bail out.
            guard let staged = expertLayerStager?.stagedBytes(recordName: biasesName),
                  staged.count == biasesRecord.byteLength,
                  let dtype = try? TensorDType.parse(biasesRecord.dtype),
                  let tensor = try? MaterializedTensor(
                      name: biasesName,
                      dtype: dtype,
                      shape: biasesRecord.shape,
                      bytes: staged
                  ),
                  let array = try? bridge.makeArray(from: tensor)
            else {
                return nil
            }
            biasesArray = array
        }

        let mode: QuantizationMode = biasesArray == nil && scalesTensor.dtype == .u8 ? .mxfp4 : .affine
        return StackedExpertProjection(
            weight: weightArray,
            scales: scalesArray,
            biases: biasesArray,
            groupSize: logicalInput / scaleGroups,
            bits: bits,
            mode: mode
        )
    }

    /// True when every projection's stacked code tensor for the layer is
    /// RAM-pinned, so the batched prefill path can serve the layer without
    /// staging.
    public func pinnedLayerIsFullyResident(layerIndex: Int) -> Bool {
        guard let pinnedCodes = pinnedExpertCodes else {
            return false
        }
        for projection in [DeepSeekExpertProjection.gate, .up, .down] {
            let candidates = DeepSeekWeightNames.routedExpert(
                layerIndex: layerIndex,
                expertIndex: 0,
                projection: projection
            )
            guard let candidate = candidates.first(where: { expertBank.record(named: $0) != nil }),
                  pinnedCodes.isResident(name: candidate)
            else {
                return false
            }
        }
        return true
    }

    public func expertLinearWeight(
        candidates: [String],
        expectedShape: [Int],
        expertIndex: Int,
        preferStaged: Bool = false,
        decodePrefetch: [String: StagedExpertCode]? = nil
    ) throws -> DeepSeekLinearWeight {
        for candidate in candidates {
            guard let record = expertBank.record(named: candidate) else {
                continue
            }
            return try buildExpertLinearWeight(
                candidate: candidate,
                record: record,
                expectedShape: expectedShape,
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch,
                plan: nil
            )
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func routedExpertLinearWeight(
        layerIndex: Int,
        projection: DeepSeekExpertProjection,
        expectedShape: [Int],
        expertIndex: Int,
        preferStaged: Bool = false,
        decodePrefetch: [String: StagedExpertCode]? = nil
    ) throws -> DeepSeekLinearWeight {
        if let plan = routedExpertProjectionPlan(layerIndex: layerIndex, projection: projection),
           plan.record.shape.count == expectedShape.count + 1,
           plan.record.shape.first.map({ expertIndex < $0 }) == true {
            return try buildExpertLinearWeight(
                candidate: plan.name,
                record: plan.record,
                expectedShape: expectedShape,
                expertIndex: expertIndex,
                preferStaged: preferStaged,
                decodePrefetch: decodePrefetch,
                plan: plan
            )
        }
        return try expertLinearWeight(
            candidates: DeepSeekWeightNames.routedExpert(
                layerIndex: layerIndex,
                expertIndex: expertIndex,
                projection: projection
            ),
            expectedShape: expectedShape,
            expertIndex: expertIndex,
            preferStaged: preferStaged,
            decodePrefetch: decodePrefetch
        )
    }

    private func buildExpertLinearWeight(
        candidate: String,
        record: ExpertTensorRecord,
        expectedShape: [Int],
        expertIndex: Int,
        preferStaged: Bool,
        decodePrefetch: [String: StagedExpertCode]?,
        plan: RoutedExpertProjectionPlan?
    ) throws -> DeepSeekLinearWeight {
            let isStacked = record.shape.count == expectedShape.count + 1
                && record.shape.first.map { expertIndex < $0 } == true
            let tensor: MaterializedTensor
            // When present, a base weight MLXArray already built off the compute
            // thread by prefetchDecodeExpertCodes (byte-identical to
            // bridge.makeArray(from: tensor)); using it skips the eager
            // Data->Metal copy here on the compute thread.
            var prebuiltWeightArray: MLXArray?
            var prebuiltScalesArray: MLXArray?
            let prefetchKey = plan?.decodePrefetchKey(for: expertIndex)
                ?? Self.decodePrefetchKey(candidate, expertIndex)
            if isStacked,
               let prefetched = decodePrefetch?[prefetchKey] {
                // Concurrently pre-read slice + pre-built base arrays (decode
                // side-bank, pinned resident prebuild, OR prefill staged-buffer).
                // Checked before the resident/staged fallbacks so prebuilt base
                // arrays are used when present.
                tensor = prefetched.tensor
                prebuiltWeightArray = prefetched.array
                prebuiltScalesArray = prefetched.scalesArray
            } else if isStacked,
               let pinned = pinnedExpertCodes?.materializedTensor(
                   named: candidate,
                   firstAxisIndex: expertIndex
               ) {
                tensor = pinned
            } else if preferStaged, isStacked,
               let staged = stagedSliceTensor(recordName: candidate, expertIndex: expertIndex) {
                tensor = staged
            } else if isStacked {
                tensor = try expertBank.materializedTensor(named: candidate, firstAxisIndex: expertIndex)
            } else {
                tensor = try expertBank.materializedTensor(named: candidate)
            }
            return try linearWeight(
                baseName: candidate,
                expectedShape: expectedShape,
                tensor: tensor,
                prebuiltWeightArray: prebuiltWeightArray,
                prebuiltScalesArray: prebuiltScalesArray,
                companionTensor: { companionName, shouldSlice in
                    if let resident = residentExpertScales?.materializedTensor(
                        named: companionName,
                        firstAxisIndex: shouldSlice ? expertIndex : nil
                    ) {
                        return resident
                    }
                    if preferStaged, shouldSlice,
                       let staged = stagedSliceTensor(
                           recordName: companionName,
                           expertIndex: expertIndex
                        ) {
                        return staged
                    }
                    if let plan {
                        if companionName == plan.scalesName {
                            guard plan.scalesRecord != nil else {
                                return nil
                            }
                        } else if companionName == plan.biasesName {
                            guard plan.biasesRecord != nil else {
                                return nil
                            }
                        } else {
                            guard expertBank.record(named: companionName) != nil else {
                                return nil
                            }
                        }
                    } else {
                        guard expertBank.record(named: companionName) != nil else {
                            return nil
                        }
                    }
                    return try shouldSlice
                        ? expertBank.materializedTensor(named: companionName, firstAxisIndex: expertIndex)
                        : expertBank.materializedTensor(named: companionName)
                },
                shouldSliceCompanions: isStacked
            )
    }

    /// Stacked expert record names for one layer, for whole-tensor staging:
    /// the three projection code tensors, plus scales companions when they are
    /// not RAM-resident and biases companions when the manifest has them.
    /// Returns nil when any projection does not resolve to a stacked record,
    /// in which case callers keep the per-slice streaming path.
    public func stagedExpertLayerPlan(layerIndex: Int) -> ExpertLayerStager.LayerPlan? {
        guard expertLayerStager != nil else {
            return nil
        }
        var names: [String] = []
        for projection in [DeepSeekExpertProjection.gate, .up, .down] {
            let candidates = DeepSeekWeightNames.routedExpert(
                layerIndex: layerIndex,
                expertIndex: 0,
                projection: projection
            )
            guard let candidate = candidates.first(where: { expertBank.record(named: $0) != nil }),
                  let record = expertBank.record(named: candidate),
                  record.shape.count == 3,
                  let firstDimension = record.shape.first,
                  firstDimension > 0,
                  record.byteLength % firstDimension == 0
            else {
                return nil
            }
            if pinnedExpertCodes?.isResident(name: candidate) != true {
                names.append(candidate)
            }
            if record.dtype == "U32" {
                for suffix in ["scales", "biases"] {
                    let companion = Self.companionName(for: candidate, suffix: suffix)
                    guard let companionRecord = expertBank.record(named: companion) else {
                        continue
                    }
                    // Scales skip staging only when the resident store can
                    // hand back the whole tensor as zero-copy raw bytes.
                    // Packed-resident scales stay IN the staging plan: the
                    // prefill batched path then builds them from the staged
                    // bytes (the no-decode byte path) instead of paying a
                    // whole-tensor palette expansion per forward.
                    let scalesResident = suffix == "scales"
                        && residentExpertScales?.servesZeroCopyBytes(name: companion) == true
                    if !scalesResident, companionRecord.shape.count >= 2 {
                        names.append(companion)
                    }
                }
            }
        }
        guard !names.isEmpty else {
            // Everything the plan would stage is already RAM-resident;
            // the per-slice path serves those layers from memory.
            return nil
        }
        return ExpertLayerStager.LayerPlan(layerIndex: layerIndex, recordNames: names)
    }

    /// Fabricates the same MaterializedTensor the bank's firstAxisIndex read
    /// would return, from a staged whole-tensor buffer: identical name, dtype,
    /// shape, and — by the bank's own slice arithmetic — identical bytes.
    private func stagedSliceTensor(recordName: String, expertIndex: Int) -> MaterializedTensor? {
        guard
            let stager = expertLayerStager,
            let record = expertBank.record(named: recordName),
            let bytes = stager.stagedBytes(recordName: recordName),
            let firstDimension = record.shape.first,
            record.shape.count >= 2,
            expertIndex >= 0,
            expertIndex < firstDimension,
            record.byteLength % firstDimension == 0,
            bytes.count == record.byteLength,
            let dtype = try? TensorDType.parse(record.dtype)
        else {
            return nil
        }
        let sliceByteLength = record.byteLength / firstDimension
        let start = bytes.startIndex + expertIndex * sliceByteLength
        let slice = bytes[start..<(start + sliceByteLength)]
        return try? MaterializedTensor(
            name: "\(recordName)[\(expertIndex)]",
            dtype: dtype,
            shape: Array(record.shape.dropFirst()),
            bytes: slice
        )
    }

    public func embedTokens(expectedShape: [Int]) throws -> DeepSeekLinearWeight {
        try denseLinearWeight(candidates: DeepSeekWeightNames.embedTokens, expectedShape: expectedShape)
    }

    public func lmHead(expectedShape: [Int]) throws -> DeepSeekLinearWeight {
        try denseLinearWeight(candidates: DeepSeekWeightNames.lmHead, expectedShape: expectedShape)
    }

    public func validateRequiredMetadata(config: DeepSeekConfig) throws {
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.embedTokens,
            expectedShape: [config.vocabSize, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.finalNorm,
            expectedShape: [config.hiddenSize]
        )
        try validateHeadHyperConnectionMetadata(config: config)
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.lmHead,
            expectedShape: [config.vocabSize, config.hiddenSize]
        )

        for layerIndex in 0..<config.numHiddenLayers {
            try validateBlockMetadata(layerIndex: layerIndex, config: config)
            try validateLocalAttentionMetadata(layerIndex: layerIndex, config: config)
            if config.compressRatios[layerIndex] != 0 {
                try validateCompressedAttentionMetadata(layerIndex: layerIndex, config: config)
            }
            try validateMoEMetadata(layerIndex: layerIndex, config: config)
        }
    }

    public func finalNorm(expectedShape: [Int]? = nil) throws -> MLXArray {
        try denseArray(candidates: DeepSeekWeightNames.finalNorm, expectedShape: expectedShape)
    }

    public func modelWeights(config: DeepSeekConfig) throws -> DeepSeekModelWeights {
        try DeepSeekModelWeights(
            embedTokens: embedTokens(expectedShape: [config.vocabSize, config.hiddenSize]),
            finalNorm: finalNorm(expectedShape: [config.hiddenSize]),
            headHyperConnection: headHyperConnectionWeights(config: config),
            lmHead: lmHead(expectedShape: [config.vocabSize, config.hiddenSize])
        )
    }

    public func headHyperConnectionWeights(config: DeepSeekConfig) throws -> DeepSeekHeadHyperConnectionWeights {
        try DeepSeekHeadHyperConnectionWeights(
            fn: denseArray(
                candidates: DeepSeekWeightNames.hcHeadFn,
                expectedShape: [config.hcMult, config.hcMult * config.hiddenSize]
            ),
            base: denseArray(
                candidates: DeepSeekWeightNames.hcHeadBase,
                expectedShape: [config.hcMult]
            ),
            scale: denseArray(
                candidates: DeepSeekWeightNames.hcHeadScale,
                expectedShape: [1]
            )
        )
    }

    public func localAttentionWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekLocalAttentionWeights {
        try localAttentionWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            qLoraRank: config.qLoraRank,
            outputLoraRank: config.outputLoraRank,
            spec: DeepSeekLocalAttentionSpec(config: config),
            attentionBias: config.attentionBias
        )
    }

    public func localAttentionWeights(
        layerIndex: Int,
        hiddenSize: Int,
        qLoraRank: Int,
        outputLoraRank: Int,
        spec: DeepSeekLocalAttentionSpec,
        attentionBias: Bool = false
    ) throws -> DeepSeekLocalAttentionWeights {
        let groupedInput = spec.numAttentionHeads * spec.headDim / spec.outputGroups
        let wqA = try denseLinearWeight(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"),
            expectedShape: [qLoraRank, hiddenSize]
        )
        let wkv = try denseLinearWeight(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wkv.weight"),
            expectedShape: [spec.headDim, hiddenSize]
        )
        return try DeepSeekLocalAttentionWeights(
            wqA: wqA,
            qNorm: denseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"),
                expectedShape: [qLoraRank]
            ),
            wqB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"),
                expectedShape: [spec.numAttentionHeads * spec.headDim, qLoraRank]
            ),
            wkv: wkv,
            kvNorm: denseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "kv_norm.weight"),
                expectedShape: [spec.headDim]
            ),
            woA: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wo_a.weight"),
                expectedShape: [spec.outputGroups, outputLoraRank, groupedInput]
            ),
            woB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.weight"),
                expectedShape: [hiddenSize, spec.outputGroups * outputLoraRank]
            ),
            woBBias: attentionBias
                ? optionalDenseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.bias"),
                    expectedShape: [hiddenSize]
                )
                : nil,
            attentionSink: optionalDenseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "attn_sink"),
                expectedShape: [spec.numAttentionHeads]
            ),
            // Load-time wq_a+wkv fusion: BOTH projections consume the exact
            // same post-norm hidden tensor in DeepSeekLocalAttention.forward
            // and DeepSeekCompressedAttention.forward, so one QMM whose
            // output rows are [qLoraRank of wq_a | headDim of wkv] replaces
            // two dispatches per layer forward with bit-identical rows.
            // Guard failures leave fusedWqAWkv nil and the unfused path
            // untouched.
            fusedWqAWkv: Self.fusedRowConcatWeight(wqA, wkv)
        )
    }

    public func compressedAttentionWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekCompressedAttentionWeights {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.headDim * (ratio == 4 ? 2 : 1)
        return try DeepSeekCompressedAttentionWeights(
            attention: localAttentionWeights(layerIndex: layerIndex, config: config),
            compressor: DeepSeekCompressorWeights(
                wkv: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wkv.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                wgate: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wgate.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                ape: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.ape"),
                    expectedShape: [ratio, outDim]
                ),
                norm: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.norm.weight"),
                    expectedShape: [config.headDim]
                )
            ),
            indexer: ratio == 4 ? indexerWeights(layerIndex: layerIndex, config: config) : nil
        )
    }

    public func indexerWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekIndexerWeights {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.indexHeadDim * (ratio == 4 ? 2 : 1)
        return try DeepSeekIndexerWeights(
            wqB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.wq_b.weight"),
                expectedShape: [config.indexHeads * config.indexHeadDim, config.qLoraRank]
            ),
            weightsProj: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.weights_proj.weight"),
                expectedShape: [config.indexHeads, config.hiddenSize]
            ),
            compressor: DeepSeekCompressorWeights(
                wkv: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wkv.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                wgate: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wgate.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                ape: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.ape"),
                    expectedShape: [ratio, outDim]
                ),
                norm: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.norm.weight"),
                    expectedShape: [config.indexHeadDim]
                )
            )
        )
    }

    public func blockWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekBlockWeights {
        try blockWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            spec: DeepSeekBlockSpec(config: config)
        )
    }

    public func blockWeights(
        layerIndex: Int,
        hiddenSize: Int,
        spec: DeepSeekBlockSpec
    ) throws -> DeepSeekBlockWeights {
        try DeepSeekBlockWeights(
            attentionNorm: denseArray(
                candidates: DeepSeekWeightNames.attentionNorm(layerIndex),
                expectedShape: [hiddenSize]
            ),
            feedForwardNorm: denseArray(
                candidates: DeepSeekWeightNames.feedForwardNorm(layerIndex),
                expectedShape: [hiddenSize]
            ),
            attentionHyperConnection: hyperConnectionWeights(
                layerIndex: layerIndex,
                block: .attention,
                hiddenSize: hiddenSize,
                spec: spec
            ),
            feedForwardHyperConnection: hyperConnectionWeights(
                layerIndex: layerIndex,
                block: .feedForward,
                hiddenSize: hiddenSize,
                spec: spec
            )
        )
    }

    public func sharedMLPWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekMLPWeights {
        try sharedMLPWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.sharedExperts
        )
    }

    public func moeWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekMoEWeights {
        try moeWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            routedExperts: config.routedExperts,
            vocabSize: config.vocabSize,
            expertsPerToken: config.expertsPerToken,
            sharedIntermediateSize: config.moeIntermediateSize * config.sharedExperts,
            isHashLayer: layerIndex < config.numHashLayers
        )
    }

    public func moeWeights(
        layerIndex: Int,
        hiddenSize: Int,
        routedExperts: Int,
        vocabSize: Int,
        expertsPerToken: Int,
        sharedIntermediateSize: Int,
        isHashLayer: Bool
    ) throws -> DeepSeekMoEWeights {
        try DeepSeekMoEWeights(
            gate: denseArray(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.weight"),
                expectedShape: [routedExperts, hiddenSize]
            ),
            correctionBias: isHashLayer
                ? nil
                : optionalDenseArray(
                    candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.e_score_correction_bias"),
                    expectedShape: [routedExperts]
                ),
            tokenToExpert: isHashLayer
                ? try optionalDenseArray(
                    candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.tid2eid"),
                    expectedShape: [vocabSize, expertsPerToken]
                )
                : nil,
            sharedExperts: sharedMLPWeights(
                layerIndex: layerIndex,
                hiddenSize: hiddenSize,
                intermediateSize: sharedIntermediateSize
            )
        )
    }

    public func sharedMLPWeights(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int
    ) throws -> DeepSeekMLPWeights {
        let gate = try denseLinearWeight(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"),
            expectedShape: [intermediateSize, hiddenSize]
        )
        let up = try denseLinearWeight(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"),
            expectedShape: [intermediateSize, hiddenSize]
        )
        // Load-time gate+up fusion: both halves were validated against the
        // identical [intermediateSize, hiddenSize] logical shape above and
        // consume the same MoE input, so one QMM + last-axis halves replaces
        // two dispatches per shared-expert forward (decode AND prefill) with
        // bit-identical rows. Guard failures (unquantized halves, metadata
        // mismatch) leave fusedGateUp nil and the unfused path untouched.
        return DeepSeekMLPWeights(
            gate: gate,
            up: up,
            down: try denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"),
                expectedShape: [hiddenSize, intermediateSize]
            ),
            fusedGateUp: Self.fusedRowConcatWeight(gate, up)
        )
    }

    public func hyperConnectionWeights(
        layerIndex: Int,
        block: DeepSeekHyperConnectionBlock,
        hiddenSize: Int,
        spec: DeepSeekBlockSpec
    ) throws -> DeepSeekHyperConnectionWeights {
        let mix = (2 + spec.hcMult) * spec.hcMult
        return try DeepSeekHyperConnectionWeights(
            fn: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .fn
                ),
                expectedShape: [mix, spec.hcMult * hiddenSize]
            ),
            base: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .base
                ),
                expectedShape: [mix]
            ),
            scale: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .scale
                ),
                expectedShape: [3]
            )
        )
    }

    private func validateShape(
        _ actualShape: [Int],
        expectedShape: [Int]?,
        tensorName: String
    ) throws {
        guard let expectedShape else {
            return
        }
        guard actualShape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(tensorName) shape \(actualShape) does not match expected shape \(expectedShape)"
            )
        }
    }

    private func validateHeadHyperConnectionMetadata(config: DeepSeekConfig) throws {
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.hcHeadFn,
            expectedShape: [config.hcMult, config.hcMult * config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.hcHeadBase,
            expectedShape: [config.hcMult]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.hcHeadScale,
            expectedShape: [1]
        )
    }

    private func validateBlockMetadata(layerIndex: Int, config: DeepSeekConfig) throws {
        let spec = DeepSeekBlockSpec(config: config)
        let mix = (2 + spec.hcMult) * spec.hcMult
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attentionNorm(layerIndex),
            expectedShape: [config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.feedForwardNorm(layerIndex),
            expectedShape: [config.hiddenSize]
        )
        for block in [DeepSeekHyperConnectionBlock.attention, .feedForward] {
            try validateDenseTensorMetadata(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .fn
                ),
                expectedShape: [mix, spec.hcMult * config.hiddenSize]
            )
            try validateDenseTensorMetadata(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .base
                ),
                expectedShape: [mix]
            )
            try validateDenseTensorMetadata(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .scale
                ),
                expectedShape: [3]
            )
        }
    }

    private func validateLocalAttentionMetadata(layerIndex: Int, config: DeepSeekConfig) throws {
        let spec = DeepSeekLocalAttentionSpec(config: config)
        let groupedInput = spec.numAttentionHeads * spec.headDim / spec.outputGroups
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"),
            expectedShape: [config.qLoraRank, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"),
            expectedShape: [config.qLoraRank]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"),
            expectedShape: [spec.numAttentionHeads * spec.headDim, config.qLoraRank]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wkv.weight"),
            expectedShape: [spec.headDim, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "kv_norm.weight"),
            expectedShape: [spec.headDim]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wo_a.weight"),
            expectedShape: [spec.outputGroups, config.outputLoraRank, groupedInput]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.weight"),
            expectedShape: [config.hiddenSize, spec.outputGroups * config.outputLoraRank]
        )
        if config.attentionBias {
            try validateOptionalDenseTensorMetadata(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.bias"),
                expectedShape: [config.hiddenSize]
            )
        }
        try validateOptionalDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "attn_sink"),
            expectedShape: [spec.numAttentionHeads]
        )
    }

    private func validateCompressedAttentionMetadata(layerIndex: Int, config: DeepSeekConfig) throws {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.headDim * (ratio == 4 ? 2 : 1)
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wkv.weight"),
            expectedShape: [outDim, config.hiddenSize]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wgate.weight"),
            expectedShape: [outDim, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.ape"),
            expectedShape: [ratio, outDim]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.norm.weight"),
            expectedShape: [config.headDim]
        )
        if ratio == 4 {
            try validateIndexerMetadata(layerIndex: layerIndex, config: config)
        }
    }

    private func validateIndexerMetadata(layerIndex: Int, config: DeepSeekConfig) throws {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.indexHeadDim * (ratio == 4 ? 2 : 1)
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.wq_b.weight"),
            expectedShape: [config.indexHeads * config.indexHeadDim, config.qLoraRank]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.weights_proj.weight"),
            expectedShape: [config.indexHeads, config.hiddenSize]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wkv.weight"),
            expectedShape: [outDim, config.hiddenSize]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wgate.weight"),
            expectedShape: [outDim, config.hiddenSize]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.ape"),
            expectedShape: [ratio, outDim]
        )
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.norm.weight"),
            expectedShape: [config.indexHeadDim]
        )
    }

    private func validateMoEMetadata(layerIndex: Int, config: DeepSeekConfig) throws {
        try validateDenseTensorMetadata(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.weight"),
            expectedShape: [config.routedExperts, config.hiddenSize]
        )
        if layerIndex < config.numHashLayers {
            try validateOptionalDenseTensorMetadata(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.tid2eid"),
                expectedShape: [config.vocabSize, config.expertsPerToken]
            )
        } else {
            try validateOptionalDenseTensorMetadata(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.e_score_correction_bias"),
                expectedShape: [config.routedExperts]
            )
        }

        let sharedIntermediateSize = config.moeIntermediateSize * config.sharedExperts
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"),
            expectedShape: [sharedIntermediateSize, config.hiddenSize]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"),
            expectedShape: [sharedIntermediateSize, config.hiddenSize]
        )
        try validateDenseLinearMetadata(
            candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"),
            expectedShape: [config.hiddenSize, sharedIntermediateSize]
        )

        try validateRoutedExpertLinearMetadata(
            layerIndex: layerIndex,
            projection: .gate,
            config: config,
            expectedShape: [config.moeIntermediateSize, config.hiddenSize]
        )
        try validateRoutedExpertLinearMetadata(
            layerIndex: layerIndex,
            projection: .up,
            config: config,
            expectedShape: [config.moeIntermediateSize, config.hiddenSize]
        )
        try validateRoutedExpertLinearMetadata(
            layerIndex: layerIndex,
            projection: .down,
            config: config,
            expectedShape: [config.hiddenSize, config.moeIntermediateSize]
        )
    }

    private func validateDenseTensorMetadata(candidates: [String], expectedShape: [Int]) throws {
        let name = try resolveDenseName(candidates)
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        try validateShape(record.shape, expectedShape: expectedShape, tensorName: name)
    }

    private func validateOptionalDenseTensorMetadata(candidates: [String], expectedShape: [Int]) throws {
        for candidate in candidates {
            guard let record = denseStore.record(named: candidate) else {
                continue
            }
            try validateShape(record.shape, expectedShape: expectedShape, tensorName: candidate)
            return
        }
    }

    private func validateDenseLinearMetadata(candidates: [String], expectedShape: [Int]) throws {
        let name = try resolveDenseName(candidates)
        guard let record = denseStore.record(named: name) else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        try validateLinearMetadata(
            baseName: name,
            dtype: try TensorDType.parse(record.dtype),
            shape: record.shape,
            expectedShape: expectedShape,
            companionMetadata: { companionName, _ in
                guard let companion = denseStore.record(named: companionName) else {
                    return nil
                }
                return (
                    dtype: try TensorDType.parse(companion.dtype),
                    shape: companion.shape
                )
            }
        )
    }

    private func validateRoutedExpertLinearMetadata(
        layerIndex: Int,
        projection: DeepSeekExpertProjection,
        config: DeepSeekConfig,
        expectedShape: [Int]
    ) throws {
        for candidate in DeepSeekWeightNames.routedExpert(
            layerIndex: layerIndex,
            expertIndex: 0,
            projection: projection
        ) {
            guard let record = expertBank.record(named: candidate),
                  record.shape.count == expectedShape.count + 1
            else {
                continue
            }
            guard record.shape.first == config.routedExperts else {
                throw MLXFastError.invalidInput(
                    "stacked expert tensor \(candidate) first dimension \(record.shape.first ?? -1) expected \(config.routedExperts)"
                )
            }
            try validateExpertLinearMetadata(
                record: record,
                expectedShape: expectedShape,
                expertIndex: 0,
                shouldSlice: true
            )
            return
        }

        for expertIndex in 0..<config.routedExperts {
            try validateExpertLinearMetadata(
                candidates: DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection
                ),
                expectedShape: expectedShape,
                expertIndex: expertIndex
            )
        }
    }

    private func validateExpertLinearMetadata(
        candidates: [String],
        expectedShape: [Int],
        expertIndex: Int
    ) throws {
        for candidate in candidates {
            guard let record = expertBank.record(named: candidate) else {
                continue
            }
            let shouldSlice = record.shape.count == expectedShape.count + 1
                && record.shape.first.map { expertIndex < $0 } == true
            try validateExpertLinearMetadata(
                record: record,
                expectedShape: expectedShape,
                expertIndex: expertIndex,
                shouldSlice: shouldSlice
            )
            return
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    private func validateExpertLinearMetadata(
        record: ExpertTensorRecord,
        expectedShape: [Int],
        expertIndex: Int,
        shouldSlice: Bool
    ) throws {
        let shape = shouldSlice ? Array(record.shape.dropFirst()) : record.shape
        try validateLinearMetadata(
            baseName: record.name,
            dtype: try TensorDType.parse(record.dtype),
            shape: shape,
            expectedShape: expectedShape,
            companionMetadata: { companionName, shouldSliceCompanion in
                guard let companion = expertBank.record(named: companionName) else {
                    return nil
                }
                if shouldSliceCompanion {
                    guard companion.shape.count >= 2,
                          companion.shape.first.map({ expertIndex < $0 }) == true
                    else {
                        throw MLXFastError.invalidInput(
                            "expert tensor \(companionName) cannot be sliced at expert \(expertIndex)"
                        )
                    }
                    return (
                        dtype: try TensorDType.parse(companion.dtype),
                        shape: Array(companion.shape.dropFirst())
                    )
                }
                return (
                    dtype: try TensorDType.parse(companion.dtype),
                    shape: companion.shape
                )
            },
            shouldSliceCompanions: shouldSlice
        )
    }

    private func validateLinearMetadata(
        baseName: String,
        dtype: TensorDType,
        shape: [Int],
        expectedShape: [Int],
        companionMetadata: (_ companionName: String, _ shouldSlice: Bool) throws -> (dtype: TensorDType, shape: [Int])?,
        shouldSliceCompanions: Bool = false
    ) throws {
        let scalesName = Self.companionName(for: baseName, suffix: "scales")
        guard dtype == .u32,
              let scales = try companionMetadata(scalesName, shouldSliceCompanions)
        else {
            try validateShape(shape, expectedShape: expectedShape, tensorName: baseName)
            return
        }

        let biases = try companionMetadata(
            Self.companionName(for: baseName, suffix: "biases"),
            shouldSliceCompanions
        )
        let expectedRows = expectedShape.dropLast().reduce(1, *)
        guard
            let expectedInput = expectedShape.last,
            let packedInput = shape.last,
            expectedInput > 0,
            packedInput > 0
        else {
            throw MLXFastError.invalidInput("linear tensor \(baseName) has invalid expected shape \(expectedShape)")
        }
        let actualRows = shape.dropLast().reduce(1, *)
        guard actualRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) has \(actualRows) output rows; expected \(expectedRows) from \(expectedShape)"
            )
        }
        let packedBits = packedInput * 32
        guard packedBits % expectedInput == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) packed input \(packedInput) is incompatible with logical input \(expectedInput)"
            )
        }
        let bits = packedBits / expectedInput
        guard [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput("quantized tensor \(baseName) inferred unsupported bits=\(bits)")
        }
        guard let scaleGroups = scales.shape.last, scaleGroups > 0, expectedInput % scaleGroups == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales shape \(scales.shape) is incompatible with logical input \(expectedInput)"
            )
        }
        let scaleRows = scales.shape.dropLast().reduce(1, *)
        guard scaleRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales have \(scaleRows) rows; expected \(expectedRows)"
            )
        }
        if let biases {
            let biasRows = biases.shape.dropLast().reduce(1, *)
            guard biasRows == expectedRows, biases.shape.last == scaleGroups else {
                throw MLXFastError.invalidInput(
                    "quantized tensor \(baseName) biases shape \(biases.shape) does not match scales shape \(scales.shape)"
                )
            }
        }

        let groupSize = expectedInput / scaleGroups
        if biases == nil && scales.dtype == .u8 && groupSize != 32 {
            throw MLXFastError.invalidInput(
                "mxfp4 tensor \(baseName) has group size \(groupSize); MLX requires group size 32"
            )
        }
    }

    private func linearWeight(
        baseName: String,
        expectedShape: [Int],
        tensor: MaterializedTensor,
        prebuiltWeightArray: MLXArray? = nil,
        prebuiltScalesArray: MLXArray? = nil,
        companionTensor: (_ companionName: String, _ shouldSlice: Bool) throws -> MaterializedTensor?,
        shouldSliceCompanions: Bool = false
    ) throws -> DeepSeekLinearWeight {
        let scalesName = Self.companionName(for: baseName, suffix: "scales")

        // Resolve the scales' shape/dtype and, only if we still need to build
        // its MLXArray, the scales MaterializedTensor. When the scales array
        // was already built off the compute thread (prebuiltScalesArray) we
        // read its shape/dtype directly and DO NOT call the scales companion
        // closure — for the palette-packed resident scales store that closure
        // would otherwise redundantly expand the slice again on the compute
        // thread just to discard its bytes.
        let scalesShape: [Int]
        let scalesIsU8: Bool
        let scalesTensorForArray: MaterializedTensor?
        if tensor.dtype == .u32, let prebuiltScalesArray {
            scalesShape = prebuiltScalesArray.shape
            scalesIsU8 = prebuiltScalesArray.dtype == .uint8
            scalesTensorForArray = nil
        } else if tensor.dtype == .u32, let scalesTensor = try companionTensor(scalesName, shouldSliceCompanions) {
            scalesShape = scalesTensor.shape
            scalesIsU8 = scalesTensor.dtype == .u8
            scalesTensorForArray = scalesTensor
        } else {
            try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: baseName)
            return DeepSeekLinearWeight(try prebuiltWeightArray ?? bridge.makeArray(from: tensor))
        }

        let biasesTensor = try companionTensor(
            Self.companionName(for: baseName, suffix: "biases"),
            shouldSliceCompanions
        )
        let expectedRows = expectedShape.dropLast().reduce(1, *)
        guard
            let expectedInput = expectedShape.last,
            let packedInput = tensor.shape.last,
            expectedInput > 0,
            packedInput > 0
        else {
            throw MLXFastError.invalidInput("linear tensor \(baseName) has invalid expected shape \(expectedShape)")
        }
        let actualRows = tensor.shape.dropLast().reduce(1, *)
        guard actualRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) has \(actualRows) output rows; expected \(expectedRows) from \(expectedShape)"
            )
        }
        let packedBits = packedInput * 32
        guard packedBits % expectedInput == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) packed input \(packedInput) is incompatible with logical input \(expectedInput)"
            )
        }
        let bits = packedBits / expectedInput
        guard [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput("quantized tensor \(baseName) inferred unsupported bits=\(bits)")
        }
        guard let scaleGroups = scalesShape.last, scaleGroups > 0, expectedInput % scaleGroups == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales shape \(scalesShape) is incompatible with logical input \(expectedInput)"
            )
        }
        let scaleRows = scalesShape.dropLast().reduce(1, *)
        guard scaleRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales have \(scaleRows) rows; expected \(expectedRows)"
            )
        }
        if let biasesTensor {
            let biasRows = biasesTensor.shape.dropLast().reduce(1, *)
            guard biasRows == expectedRows, biasesTensor.shape.last == scaleGroups else {
                throw MLXFastError.invalidInput(
                    "quantized tensor \(baseName) biases shape \(biasesTensor.shape) does not match scales shape \(scalesShape)"
                )
            }
        }

        let mode: QuantizationMode = biasesTensor == nil && scalesIsU8 ? .mxfp4 : .affine
        let weightArray = try (prebuiltWeightArray ?? bridge.makeArray(from: tensor))
            .reshaped([expectedRows, packedInput])
        let scalesArray = try (prebuiltScalesArray ?? bridge.makeArray(from: scalesTensorForArray!))
            .reshaped([expectedRows, scaleGroups])
        let biasesArray = try biasesTensor.map { try bridge.makeArray(from: $0).reshaped([expectedRows, scaleGroups]) }
        return DeepSeekLinearWeight(
            weight: weightArray,
            scales: scalesArray,
            biases: biasesArray,
            logicalShape: expectedShape,
            groupSize: expectedInput / scaleGroups,
            bits: bits,
            mode: mode
        )
    }

    private static func companionName(for weightName: String, suffix: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".\(suffix)"
        }
        return "\(weightName).\(suffix)"
    }
}

// A routed-expert code slice read off the compute thread: the raw tensor
// (bytes + shape/dtype metadata) plus its base MLXArray, both built on a
// prefetch worker thread. Byte-identical to what the serial bank read +
// bridge.makeArray would produce on the compute thread.
public struct StagedExpertCode {
    public let tensor: MaterializedTensor
    public let array: MLXArray
    // Base scales MLXArray built off the compute thread when the scales are
    // RAM-resident, so linearWeight skips that eager copy too. nil => build on
    // the compute thread as before.
    public let scalesArray: MLXArray?
    public init(tensor: MaterializedTensor, array: MLXArray, scalesArray: MLXArray? = nil) {
        self.tensor = tensor
        self.array = array
        self.scalesArray = scalesArray
    }
}

// Lets concurrentPerform write results into distinct buffer slots from worker
// threads. Each index is written by exactly one iteration, so the aliasing is
// disjoint and the unchecked Sendable conformance is sound.
private struct DecodePrefetchSink: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<StagedExpertCode?>
}

/// Per-expert readiness view over the decode-step slice reads scheduled by
/// streamDecodeExpertCodes. Worker threads store finished slice entries
/// (tensor + base MLXArray) under the same decodePrefetchKey the batch
/// prefetch would use; the compute thread blocks per EXPERT (not on the whole
/// batch) and snapshots the map after each wait. Entries are only ever ADDED,
/// never mutated or removed, so a snapshot taken after an expert's group wait
/// always holds that expert's finished entries; other experts' keys are
/// irrelevant to its lookup. Any slice that failed or timed out is simply
/// absent and the consumer's normal bank-read fallback supplies the same
/// bytes on the compute thread.
public final class DecodeExpertCodeStream: @unchecked Sendable {
    private let lock = NSLock()
    private var results: [String: StagedExpertCode]
    // Fused gate+up weights per expert, built by the second-finishing of the
    // expert's gate/up stream workers before that worker's group.leave; a
    // per-step transient exactly like the per-slice entries above. Absent
    // entries (guards failed, timeout, prepopulated halves) mean consumers
    // keep the unfused per-projection path.
    private var fusedGateUpWeights: [Int: DeepSeekLinearWeight] = [:]
    // Immutable after init: one group per expert with scheduled reads,
    // entered once per slice before the stream is handed to the consumer.
    private let expertGroups: [Int: DispatchGroup]

    fileprivate init(results: [String: StagedExpertCode], expertGroups: [Int: DispatchGroup]) {
        self.results = results
        self.expertGroups = expertGroups
    }

    /// Blocks until every scheduled slice for the expert has finished (read +
    /// base array build) or the safety timeout passes. Experts with no
    /// scheduled reads (fully pre-populated, or nothing resolvable) return
    /// immediately. On timeout the expert's entries may be missing from the
    /// snapshot; consumers then fall back to the normal bank reads, so
    /// correctness never depends on this wait.
    public func waitForExpert(_ expertIndex: Int) {
        guard let group = expertGroups[expertIndex] else {
            return
        }
        _ = group.wait(timeout: .now() + .seconds(5))
    }

    /// Locked copy of the results map, in the exact shape
    /// weights(forExpert:decodePrefetch:) consumes. Take it AFTER
    /// waitForExpert so the expert's own entries are present (group.leave
    /// orders each store before the wait returns).
    public func snapshotResults() -> [String: StagedExpertCode] {
        lock.lock()
        defer { lock.unlock() }
        return results
    }

    fileprivate func store(key: String, code: StagedExpertCode) {
        lock.lock()
        results[key] = code
        lock.unlock()
    }

    /// The expert's prebuilt fused gate+up weight, when both halves landed on
    /// this stream and every fusion guard passed. Read AFTER waitForExpert:
    /// the fusing worker stores before its group.leave, so a successful wait
    /// orders the store before this read. nil => use the unfused path.
    public func fusedGateUpWeight(_ expertIndex: Int) -> DeepSeekLinearWeight? {
        lock.lock()
        defer { lock.unlock() }
        return fusedGateUpWeights[expertIndex]
    }

    fileprivate func storeFusedGateUp(expertIndex: Int, weight: DeepSeekLinearWeight) {
        lock.lock()
        fusedGateUpWeights[expertIndex] = weight
        lock.unlock()
    }
}

/// Per-expert rendezvous for the decode gate+up fusion: each of the expert's
/// two participating stream workers deposits its finished half (code slice +
/// resident scales slice, both immutable), and the SECOND depositor receives
/// the completed pair to build the fused weight on its own thread. Experts
/// outside the scheduled set — or halves whose scales were not resident —
/// never complete a pair, which is safe: nothing waits on fusion, and the
/// consumer falls back to the unfused path when the fused entry is absent.
private final class DecodeGateUpFusionState: @unchecked Sendable {
    struct Half {
        let codes: MaterializedTensor
        let scales: MaterializedTensor
    }

    private let lock = NSLock()
    private let experts: Set<Int>
    private var pendingGate: [Int: Half] = [:]
    private var pendingUp: [Int: Half] = [:]

    init(experts: Set<Int>) {
        self.experts = experts
    }

    func deposit(
        expertIndex: Int,
        projection: DeepSeekExpertProjection,
        half: Half
    ) -> (gate: Half, up: Half)? {
        guard experts.contains(expertIndex) else {
            return nil
        }
        lock.lock()
        defer { lock.unlock() }
        switch projection {
        case .gate:
            if let up = pendingUp.removeValue(forKey: expertIndex) {
                return (gate: half, up: up)
            }
            pendingGate[expertIndex] = half
            return nil
        case .up:
            if let gate = pendingGate.removeValue(forKey: expertIndex) {
                return (gate: gate, up: half)
            }
            pendingUp[expertIndex] = half
            return nil
        case .down:
            return nil
        }
    }
}

// Concurrent worker queue for streamDecodeExpertCodes slice reads.
// .userInitiated, never .background: background QoS on the shared pool can
// priority-invert against the compute thread's per-expert wait on the
// 12-vCPU official runner (same rationale as the other decode queues).
private let decodeExpertStreamQueue = DispatchQueue(
    label: "mlxfast.decode.expert-stream",
    qos: .userInitiated,
    attributes: .concurrent
)

/// Last decode step's routed experts per layer — a prefetch hint store.
/// Lock-guarded; values are only ever used to pre-position weight bytes.
private final class DecodeExpertHistory {
    private let lock = NSLock()
    private var lastExperts: [Int: [Int]] = [:]

    func record(layerIndex: Int, experts: [Int]) {
        lock.lock()
        lastExperts[layerIndex] = experts
        lock.unlock()
    }

    func predicted(layerIndex: Int) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        return lastExperts[layerIndex]
    }
}

private final class ScheduledDecodePrefetches {
    private final class Entry {
        let group = DispatchGroup()
        let lock = NSLock()
        var result: [String: StagedExpertCode]?
    }

    private let lock = NSLock()
    private let queue = DispatchQueue(
        label: "mlxfast.decode.pinned-prefetch",
        qos: .userInitiated,
        attributes: .concurrent
    )
    private var entries: [Int: Entry] = [:]

    func schedule(
        layerIndex: Int,
        build: @escaping () -> [String: StagedExpertCode]?
    ) {
        let entry = Entry()
        entry.group.enter()
        lock.lock()
        entries[layerIndex] = entry
        lock.unlock()

        queue.async {
            let result = build()
            entry.lock.lock()
            entry.result = result
            entry.lock.unlock()
            entry.group.leave()
        }
    }

    func hasEntry(layerIndex: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return entries[layerIndex] != nil
    }

    func consume(layerIndex: Int) -> [String: StagedExpertCode]? {
        lock.lock()
        let entry = entries.removeValue(forKey: layerIndex)
        lock.unlock()
        guard let entry else {
            return nil
        }
        guard entry.group.wait(timeout: .now() + .milliseconds(10)) == .success else {
            return nil
        }
        entry.lock.lock()
        let result = entry.result
        entry.lock.unlock()
        return result?.isEmpty == true ? nil : result
    }

    func remove(layerIndex: Int) {
        lock.lock()
        entries.removeValue(forKey: layerIndex)
        lock.unlock()
    }
}
