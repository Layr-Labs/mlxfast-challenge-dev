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

/// Pre-loaded expert scale tensors from 2-bit packed format. During transform,
/// U8 E8M0 scales (only ~4 unique values) are packed to 2 bits per value and
/// stored in `weights/experts/packed_scales.bin`. At runtime this class loads
/// the ~2 GiB packed file and unpacks per-expert slices on demand, eliminating
/// all scale SSD reads during decode.
///
/// Reference type so it is shared across value-type WeightLoader copies.
public final class PreloadedExpertScales {
    private struct Entry {
        let shape: [Int]
        let originalByteCount: Int
        let packedData: Data       // 2-bit packed bytes for the full stacked tensor
        let perExpertOriginal: Int // bytes per expert in the original U8 format
        let perExpertPacked: Int   // bytes per expert in the packed format
    }

    private var store: [String: Entry] = [:]
    private var palette: [UInt8] = []

    public var count: Int { store.count }
    public var totalBytes: Int { store.values.reduce(0) { $0 + $1.packedData.count } }

    /// Load from the packed_scales.bin written by the transform.
    public init(packedScalesPath: String) throws {
        let url = URL(fileURLWithPath: packedScalesPath)
        guard FileManager.default.fileExists(atPath: packedScalesPath) else {
            // No packed scales file -- fall through to ExpertSlotBank reads.
            return
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try parse(data)
    }

    /// Empty init (no pre-loading).
    public init() {}

    private func parse(_ data: Data) throws {
        guard data.count >= 12 else { return }
        // Magic: "ES4B" (4-bit nibble packed) or legacy "ES2B"
        guard data[0] == 0x45, data[1] == 0x53, (data[2] == 0x34 || data[2] == 0x32), data[3] == 0x42 else {
            return
        }
        let tensorCount = Int(readUInt32(data, offset: 4))
        let paletteSize = Int(readUInt32(data, offset: 8))
        guard paletteSize <= 256 else { return }

        var offset = 12
        palette = Array(data[offset..<(offset + paletteSize)])
        offset += paletteSize

        // Read per-tensor metadata
        struct TensorMeta {
            let name: String
            let originalByteCount: Int
            let packedByteCount: Int
            let shape: [Int]
        }
        var metas: [TensorMeta] = []
        for _ in 0..<tensorCount {
            guard offset + 256 + 12 <= data.count else { return }
            let nameBytes = data[offset..<(offset + 256)]
            let name = String(bytes: nameBytes.prefix { $0 != 0 }, encoding: .utf8) ?? ""
            offset += 256
            let origCount = Int(readUInt32(data, offset: offset)); offset += 4
            let packCount = Int(readUInt32(data, offset: offset)); offset += 4
            let shapeDims = Int(readUInt32(data, offset: offset)); offset += 4
            var shape: [Int] = []
            for _ in 0..<shapeDims {
                guard offset + 4 <= data.count else { return }
                shape.append(Int(readUInt32(data, offset: offset))); offset += 4
            }
            metas.append(TensorMeta(name: name, originalByteCount: origCount, packedByteCount: packCount, shape: shape))
        }

        // Read packed data
        for meta in metas {
            guard offset + meta.packedByteCount <= data.count else { return }
            let packed = data.subdata(in: offset..<(offset + meta.packedByteCount))
            offset += meta.packedByteCount
            let firstDim = meta.shape.first ?? 1
            store[meta.name] = Entry(
                shape: meta.shape,
                originalByteCount: meta.originalByteCount,
                packedData: packed,
                perExpertOriginal: firstDim > 0 ? meta.originalByteCount / firstDim : meta.originalByteCount,
                perExpertPacked: firstDim > 0 ? meta.packedByteCount / firstDim : meta.packedByteCount
            )
        }
    }

    private func readUInt32(_ data: Data, offset: Int) -> UInt32 {
        data.withUnsafeBytes { buf in
            buf.load(fromByteOffset: offset, as: UInt32.self).littleEndian
        }
    }

    /// Unpack a per-expert slice from the 4-bit nibble-packed data back to U8.
    public func scaleTensor(named name: String, firstAxisIndex: Int) throws -> MaterializedTensor? {
        guard let entry = store[name], !palette.isEmpty else { return nil }
        guard entry.shape.count > 1, let firstDim = entry.shape.first, firstAxisIndex < firstDim else {
            return nil
        }
        let packedOffset = firstAxisIndex * entry.perExpertPacked
        let packedEnd = packedOffset + entry.perExpertPacked
        guard packedEnd <= entry.packedData.count else { return nil }

        let unpackedCount = entry.perExpertOriginal
        var unpacked = Data(count: unpackedCount)
        unpacked.withUnsafeMutableBytes { outBuf in
            entry.packedData.withUnsafeBytes { inBuf in
                let inPtr = inBuf.baseAddress!.advanced(by: packedOffset)
                    .assumingMemoryBound(to: UInt8.self)
                let outPtr = outBuf.bindMemory(to: UInt8.self).baseAddress!
                for i in 0..<unpackedCount {
                    let bytePos = i / 2
                    let idx: Int
                    if i % 2 == 0 {
                        idx = Int(inPtr[bytePos] & 0x0F)        // low nibble
                    } else {
                        idx = Int((inPtr[bytePos] >> 4) & 0x0F) // high nibble
                    }
                    outPtr[i] = idx < palette.count ? palette[idx] : 0
                }
            }
        }
        let slicedShape = Array(entry.shape.dropFirst())
        return try MaterializedTensor(name: name, dtype: .u8, shape: slicedShape, bytes: unpacked)
    }

    /// Unpack a full stacked tensor.
    public func scaleTensor(named name: String) throws -> MaterializedTensor? {
        guard let entry = store[name], !palette.isEmpty else { return nil }
        var unpacked = Data(count: entry.originalByteCount)
        unpacked.withUnsafeMutableBytes { outBuf in
            entry.packedData.withUnsafeBytes { inBuf in
                let inPtr = inBuf.bindMemory(to: UInt8.self).baseAddress!
                let outPtr = outBuf.bindMemory(to: UInt8.self).baseAddress!
                for i in 0..<entry.originalByteCount {
                    let bytePos = i / 2
                    let idx: Int
                    if i % 2 == 0 {
                        idx = Int(inPtr[bytePos] & 0x0F)
                    } else {
                        idx = Int((inPtr[bytePos] >> 4) & 0x0F)
                    }
                    outPtr[i] = idx < palette.count ? palette[idx] : 0
                }
            }
        }
        return try MaterializedTensor(name: name, dtype: .u8, shape: entry.shape, bytes: unpacked)
    }
}

public struct DeepSeekWeightLoader {
    public let denseStore: DenseTensorStore
    public let expertBank: ExpertSlotBank
    // L2: caches the fully built routed-expert weights (MLXArrays) per
    // (layer, expert) so reuse skips pread + makeArray. This is the primary
    // routed-expert cache; the ExpertSlotBank byte cache is disabled (capacity 0)
    // to avoid holding the same bytes twice.
    public let expertWeightCache: ExpertWeightCache
    // Pre-loaded expert scale tensors. Scale reads are eliminated from the
    // SSD I/O path: all U8 E8M0 scales are loaded into RAM at init time
    // (~8.6 GiB) and served from memory during decode.
    public let preloadedScales: PreloadedExpertScales
    public let expertStreamingConfig: ExpertStreamingConfig
    public let expertStreamingMetrics: ExpertStreamingMetrics?
    private let bridge: MLXArrayTensorBridge

    /// Default L2 expert cache capacity in experts. Sized conservatively for 48 GB
    /// machines: 128 experts × ~12 MiB (weights only) ≈ 1.5 GiB. The O(1)
    /// doubly-linked-list LRU and MLXArray-level caching eliminate the
    /// Data->MLXArray rebuild cost on hits. Override with MLXFAST_EXPERT_CACHE_EXPERTS.
    private static let defaultL2CacheCapacity = 128

    private static func l2CacheCapacity(from config: ExpertStreamingConfig) -> Int {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["MLXFAST_EXPERT_CACHE_EXPERTS"], let parsed = Int(raw), parsed >= 0 {
            return parsed
        }
        if let raw = env["MLXFAST_EXPERT_CACHE_TENSORS"], let parsed = Int(raw), parsed >= 0 {
            return parsed / ExpertStreamingConfig.tensorsPerExpertRecord
        }
        return defaultL2CacheCapacity
    }

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
        // Disable byte-level LRU in ExpertSlotBank (capacity 0); all caching is
        // done at the MLXArray level in expertWeightCache to avoid double-caching
        // and the ~12.75 MiB/expert Data->MLXArray copy on every reuse.
        self.expertBank = try ExpertSlotBank(
            manifestPath: "\(weightsPath)/experts/manifest.json",
            capacity: 0,
            metrics: metrics
        )
        self.expertWeightCache = ExpertWeightCache(
            capacity: Self.l2CacheCapacity(from: expertStreamingConfig)
        )
        // Load 2-bit packed expert scales from weights/experts/packed_scales.bin
        // (~2 GiB). Scale reads are eliminated from SSD during decode; each
        // expert activation unpacks its scales from the compact in-memory buffer.
        self.preloadedScales = try PreloadedExpertScales(
            packedScalesPath: "\(weightsPath)/experts/packed_scales.bin"
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
        self.expertStreamingConfig = expertStreamingConfig
        self.expertStreamingMetrics = expertStreamingMetrics ?? expertBank.metrics
        self.expertWeightCache = ExpertWeightCache(
            capacity: Self.l2CacheCapacity(from: expertStreamingConfig)
        )
        // No pre-loading in the test/harness init path.
        self.preloadedScales = PreloadedExpertScales()
        self.bridge = bridge
    }

    /// Routed-expert weights for (spec.layerIndex, expertIndex), served from the
    /// L2 cache when resident. On a miss the weights are built once (pread +
    /// makeArray via expertLinearWeight) and inserted, so subsequent reuse across
    /// decode steps is a pointer return with no host->device copy and no recorded
    /// streaming bytes.
    public func routedExpertWeights(
        expertIndex: Int,
        spec: DeepSeekRoutedExpertSpec
    ) throws -> DeepSeekMLPWeights {
        let key = ExpertWeightCache.Key(layer: spec.layerIndex, expert: expertIndex)
        if let cached = expertWeightCache.value(for: key) {
            return cached
        }
        let weights = try DeepSeekRoutedExperts.weights(
            forExpert: expertIndex,
            loader: self,
            spec: spec
        )
        expertWeightCache.insert(weights, for: key)
        return weights
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

    public func expertLinearWeight(
        candidates: [String],
        expectedShape: [Int],
        expertIndex: Int
    ) throws -> DeepSeekLinearWeight {
        for candidate in candidates {
            guard let record = expertBank.record(named: candidate) else {
                continue
            }
            let isStacked = record.shape.count == expectedShape.count + 1
                && record.shape.first.map { expertIndex < $0 } == true
            let tensor = try isStacked
                ? expertBank.materializedTensor(named: candidate, firstAxisIndex: expertIndex)
                : expertBank.materializedTensor(named: candidate)
            return try linearWeight(
                baseName: candidate,
                expectedShape: expectedShape,
                tensor: tensor,
                companionTensor: { companionName, shouldSlice in
                    // Serve pre-loaded scales from RAM (zero SSD I/O).
                    if shouldSlice, let preloaded = try preloadedScales.scaleTensor(named: companionName, firstAxisIndex: expertIndex) {
                        return preloaded
                    }
                    if !shouldSlice, let preloaded = try preloadedScales.scaleTensor(named: companionName) {
                        return preloaded
                    }
                    // Fall through to SSD for non-preloaded companions.
                    guard expertBank.record(named: companionName) != nil else {
                        return nil
                    }
                    return try shouldSlice
                        ? expertBank.materializedTensor(named: companionName, firstAxisIndex: expertIndex)
                        : expertBank.materializedTensor(named: companionName)
                },
                shouldSliceCompanions: isStacked
            )
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
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
        return try DeepSeekLocalAttentionWeights(
            wqA: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"),
                expectedShape: [qLoraRank, hiddenSize]
            ),
            qNorm: denseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"),
                expectedShape: [qLoraRank]
            ),
            wqB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"),
                expectedShape: [spec.numAttentionHeads * spec.headDim, qLoraRank]
            ),
            wkv: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wkv.weight"),
                expectedShape: [spec.headDim, hiddenSize]
            ),
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
            )
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
        try DeepSeekMLPWeights(
            gate: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"),
                expectedShape: [intermediateSize, hiddenSize]
            ),
            up: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"),
                expectedShape: [intermediateSize, hiddenSize]
            ),
            down: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"),
                expectedShape: [hiddenSize, intermediateSize]
            )
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
        let scalesName = companionName(for: baseName, suffix: "scales")
        guard dtype == .u32,
              let scales = try companionMetadata(scalesName, shouldSliceCompanions)
        else {
            try validateShape(shape, expectedShape: expectedShape, tensorName: baseName)
            return
        }

        let biases = try companionMetadata(
            companionName(for: baseName, suffix: "biases"),
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
        companionTensor: (_ companionName: String, _ shouldSlice: Bool) throws -> MaterializedTensor?,
        shouldSliceCompanions: Bool = false
    ) throws -> DeepSeekLinearWeight {
        let scalesName = companionName(for: baseName, suffix: "scales")
        guard tensor.dtype == .u32, let scalesTensor = try companionTensor(scalesName, shouldSliceCompanions) else {
            try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: baseName)
            return DeepSeekLinearWeight(try bridge.makeArray(from: tensor))
        }

        let biasesTensor = try companionTensor(
            companionName(for: baseName, suffix: "biases"),
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
        guard let scaleGroups = scalesTensor.shape.last, scaleGroups > 0, expectedInput % scaleGroups == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales shape \(scalesTensor.shape) is incompatible with logical input \(expectedInput)"
            )
        }
        let scaleRows = scalesTensor.shape.dropLast().reduce(1, *)
        guard scaleRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales have \(scaleRows) rows; expected \(expectedRows)"
            )
        }
        if let biasesTensor {
            let biasRows = biasesTensor.shape.dropLast().reduce(1, *)
            guard biasRows == expectedRows, biasesTensor.shape.last == scaleGroups else {
                throw MLXFastError.invalidInput(
                    "quantized tensor \(baseName) biases shape \(biasesTensor.shape) does not match scales shape \(scalesTensor.shape)"
                )
            }
        }

        let mode: QuantizationMode = biasesTensor == nil && scalesTensor.dtype == .u8 ? .mxfp4 : .affine
        let weightArray = try bridge.makeArray(from: tensor).reshaped([expectedRows, packedInput])
        let scalesArray = try bridge.makeArray(from: scalesTensor).reshaped([expectedRows, scaleGroups])
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

    private func companionName(for weightName: String, suffix: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".\(suffix)"
        }
        return "\(weightName).\(suffix)"
    }
}
