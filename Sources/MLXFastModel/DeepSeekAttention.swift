import Foundation
import MLX
import MLXFastCore

public struct DeepSeekLocalAttentionSpec {
    public let numAttentionHeads: Int
    public let headDim: Int
    public let attentionScale: Float
    public let outputGroups: Int
    public let qkRopeHeadDim: Int
    public let ropeTheta: Double
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Double

    public init(
        numAttentionHeads: Int,
        headDim: Int,
        outputGroups: Int,
        qkRopeHeadDim: Int,
        ropeTheta: Double,
        maxPositionEmbeddings: Int,
        rmsNormEps: Double
    ) {
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.attentionScale = Float(pow(Double(headDim), -0.5))
        self.outputGroups = outputGroups
        self.qkRopeHeadDim = qkRopeHeadDim
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
    }

    public init(config: DeepSeekConfig) {
        self.init(
            numAttentionHeads: config.numAttentionHeads,
            headDim: config.headDim,
            outputGroups: config.outputGroups,
            qkRopeHeadDim: config.qkRopeHeadDim,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps
        )
    }
}

public struct DeepSeekCompressedAttentionSpec {
    public let numAttentionHeads: Int
    public let headDim: Int
    public let attentionScale: Float
    public let outputGroups: Int
    public let qkRopeHeadDim: Int
    public let ropeTheta: Double
    public let ropeScaling: DeepSeekRopeScaling?
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Double
    public let compressRatio: Int
    public let indexHeads: Int
    public let indexHeadDim: Int
    public let indexTopK: Int
    public let compressorSpec: DeepSeekCompressorSpec
    public let indexerSpec: DeepSeekIndexerSpec

    public init(
        numAttentionHeads: Int,
        headDim: Int,
        outputGroups: Int,
        qkRopeHeadDim: Int,
        ropeTheta: Double,
        ropeScaling: DeepSeekRopeScaling?,
        maxPositionEmbeddings: Int,
        rmsNormEps: Double,
        compressRatio: Int,
        indexHeads: Int,
        indexHeadDim: Int,
        indexTopK: Int
    ) {
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.attentionScale = Float(pow(Double(headDim), -0.5))
        self.outputGroups = outputGroups
        self.qkRopeHeadDim = qkRopeHeadDim
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.compressRatio = compressRatio
        self.indexHeads = indexHeads
        self.indexHeadDim = indexHeadDim
        self.indexTopK = indexTopK
        self.compressorSpec = DeepSeekCompressorSpec(
            compressRatio: compressRatio,
            headDim: headDim,
            ropeHeadDim: qkRopeHeadDim,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps
        )
        self.indexerSpec = DeepSeekIndexerSpec(
            indexHeads: indexHeads,
            indexHeadDim: indexHeadDim,
            indexTopK: indexTopK,
            qkRopeHeadDim: qkRopeHeadDim,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            compressRatio: compressRatio
        )
    }

    public init(config: DeepSeekConfig, layerIndex: Int) {
        self.init(
            numAttentionHeads: config.numAttentionHeads,
            headDim: config.headDim,
            outputGroups: config.outputGroups,
            qkRopeHeadDim: config.qkRopeHeadDim,
            ropeTheta: config.compressRopeTheta,
            ropeScaling: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps,
            compressRatio: config.compressRatios[layerIndex],
            indexHeads: config.indexHeads,
            indexHeadDim: config.indexHeadDim,
            indexTopK: config.indexTopk
        )
    }
}

public struct DeepSeekIndexerSpec {
    public let indexHeads: Int
    public let indexHeadDim: Int
    public let indexHeadScale: Float
    public let indexHeadsScale: Float
    public let indexTopK: Int
    public let qkRopeHeadDim: Int
    public let ropeTheta: Double
    public let ropeScaling: DeepSeekRopeScaling?
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Double
    public let compressRatio: Int
    public let compressorSpec: DeepSeekCompressorSpec

    public init(
        indexHeads: Int,
        indexHeadDim: Int,
        indexTopK: Int,
        qkRopeHeadDim: Int,
        ropeTheta: Double,
        ropeScaling: DeepSeekRopeScaling?,
        maxPositionEmbeddings: Int,
        rmsNormEps: Double,
        compressRatio: Int
    ) {
        self.indexHeads = indexHeads
        self.indexHeadDim = indexHeadDim
        self.indexHeadScale = Float(pow(Double(indexHeadDim), -0.5))
        self.indexHeadsScale = Float(pow(Double(indexHeads), -0.5))
        self.indexTopK = indexTopK
        self.qkRopeHeadDim = qkRopeHeadDim
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.compressRatio = compressRatio
        self.compressorSpec = DeepSeekCompressorSpec(
            compressRatio: compressRatio,
            headDim: indexHeadDim,
            ropeHeadDim: qkRopeHeadDim,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps
        )
    }

    public init(config: DeepSeekConfig, layerIndex: Int) {
        self.init(
            indexHeads: config.indexHeads,
            indexHeadDim: config.indexHeadDim,
            indexTopK: config.indexTopk,
            qkRopeHeadDim: config.qkRopeHeadDim,
            ropeTheta: config.compressRopeTheta,
            ropeScaling: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            rmsNormEps: config.rmsNormEps,
            compressRatio: config.compressRatios[layerIndex]
        )
    }
}

public struct DeepSeekLocalAttentionWeights {
    public let wqA: DeepSeekLinearWeight
    public let qNorm: MLXArray
    public let wqB: DeepSeekLinearWeight
    public let wkv: DeepSeekLinearWeight
    public let kvNorm: MLXArray
    public let woA: DeepSeekLinearWeight
    public let woB: DeepSeekLinearWeight
    public let woBBias: MLXArray?
    public let attentionSink: MLXArray?

    public init(
        wqA: DeepSeekLinearWeight,
        qNorm: MLXArray,
        wqB: DeepSeekLinearWeight,
        wkv: DeepSeekLinearWeight,
        kvNorm: MLXArray,
        woA: DeepSeekLinearWeight,
        woB: DeepSeekLinearWeight,
        woBBias: MLXArray? = nil,
        attentionSink: MLXArray? = nil
    ) {
        self.wqA = wqA
        self.qNorm = qNorm
        self.wqB = wqB
        self.wkv = wkv
        self.kvNorm = kvNorm
        self.woA = woA
        self.woB = woB
        self.woBBias = woBBias
        self.attentionSink = attentionSink
    }

    public init(
        wqA: MLXArray,
        qNorm: MLXArray,
        wqB: MLXArray,
        wkv: MLXArray,
        kvNorm: MLXArray,
        woA: MLXArray,
        woB: MLXArray,
        woBBias: MLXArray? = nil,
        attentionSink: MLXArray? = nil
    ) {
        self.init(
            wqA: DeepSeekLinearWeight(wqA),
            qNorm: qNorm,
            wqB: DeepSeekLinearWeight(wqB),
            wkv: DeepSeekLinearWeight(wkv),
            kvNorm: kvNorm,
            woA: DeepSeekLinearWeight(woA),
            woB: DeepSeekLinearWeight(woB),
            woBBias: woBBias,
            attentionSink: attentionSink
        )
    }
}

public struct DeepSeekCompressedAttentionWeights {
    public let attention: DeepSeekLocalAttentionWeights
    public let compressor: DeepSeekCompressorWeights
    public let indexer: DeepSeekIndexerWeights?

    public init(
        attention: DeepSeekLocalAttentionWeights,
        compressor: DeepSeekCompressorWeights,
        indexer: DeepSeekIndexerWeights? = nil
    ) {
        self.attention = attention
        self.compressor = compressor
        self.indexer = indexer
    }
}

public struct DeepSeekIndexerWeights {
    public let wqB: DeepSeekLinearWeight
    public let weightsProj: DeepSeekLinearWeight
    public let compressor: DeepSeekCompressorWeights

    public init(
        wqB: DeepSeekLinearWeight,
        weightsProj: DeepSeekLinearWeight,
        compressor: DeepSeekCompressorWeights
    ) {
        self.wqB = wqB
        self.weightsProj = weightsProj
        self.compressor = compressor
    }

    public init(wqB: MLXArray, weightsProj: MLXArray, compressor: DeepSeekCompressorWeights) {
        self.init(
            wqB: DeepSeekLinearWeight(wqB),
            weightsProj: DeepSeekLinearWeight(weightsProj),
            compressor: compressor
        )
    }
}

public enum DeepSeekAttentionMask {
    private static let causalCache = CausalMaskCache()

    public static func causal(
        queryLength: Int,
        keyLength: Int? = nil,
        queryOffset: Int = 0,
        keyOffset: Int = 0,
        windowSize: Int? = nil
    ) throws -> MLXArray {
        guard queryLength > 0 else {
            throw MLXFastError.invalidInput("causal mask query length must be positive")
        }
        let keyLength = keyLength ?? queryLength
        guard keyLength > 0 else {
            throw MLXFastError.invalidInput("causal mask key length must be positive")
        }
        if let windowSize {
            guard windowSize > 0 else {
                throw MLXFastError.invalidInput("causal mask window size must be positive")
            }
        }
        let cacheKey = CausalMaskCacheKey(
            queryLength: queryLength,
            keyLength: keyLength,
            relativeOffset: queryOffset - keyOffset,
            windowSize: windowSize
        )
        if let cached = causalCache.value(for: cacheKey) {
            return cached
        }

        let queryPositions = arange(
            queryOffset,
            queryOffset + queryLength,
            dtype: .int32
        ).expandedDimensions(axis: 1)
        let keyPositions = arange(
            keyOffset,
            keyOffset + keyLength,
            dtype: .int32
        ).expandedDimensions(axis: 0)

        var allowed = queryPositions .>= keyPositions
        if let windowSize {
            allowed = allowed .&& (queryPositions .< (keyPositions + windowSize))
        }
        causalCache.insert(allowed, for: cacheKey)
        return allowed
    }
}

private struct CausalMaskCacheKey: Hashable {
    let queryLength: Int
    let keyLength: Int
    let relativeOffset: Int
    let windowSize: Int?
}

private final class CausalMaskCache: @unchecked Sendable {
    private let lock = NSLock()
    private var masks: [CausalMaskCacheKey: MLXArray] = [:]
    private var order: [CausalMaskCacheKey] = []
    private let capacity = 4096

    func value(for key: CausalMaskCacheKey) -> MLXArray? {
        lock.lock()
        defer {
            lock.unlock()
        }
        return masks[key]
    }

    func insert(_ mask: MLXArray, for key: CausalMaskCacheKey) {
        lock.lock()
        defer {
            lock.unlock()
        }
        if masks[key] == nil {
            order.append(key)
        }
        masks[key] = mask
        while order.count > capacity {
            let oldest = order.removeFirst()
            masks.removeValue(forKey: oldest)
        }
    }
}

public struct DeepSeekCompressorSpec {
    public let compressRatio: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let ropeTheta: Double
    public let ropeScaling: DeepSeekRopeScaling?
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Double

    public init(
        compressRatio: Int,
        headDim: Int,
        ropeHeadDim: Int,
        ropeTheta: Double,
        ropeScaling: DeepSeekRopeScaling?,
        maxPositionEmbeddings: Int,
        rmsNormEps: Double
    ) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.ropeHeadDim = ropeHeadDim
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
    }
}

public struct DeepSeekCompressorWeights {
    public let wkv: DeepSeekLinearWeight
    public let wgate: DeepSeekLinearWeight
    public let ape: MLXArray
    public let norm: MLXArray

    public init(wkv: DeepSeekLinearWeight, wgate: DeepSeekLinearWeight, ape: MLXArray, norm: MLXArray) {
        self.wkv = wkv
        self.wgate = wgate
        self.ape = ape
        self.norm = norm
    }

    public init(wkv: MLXArray, wgate: MLXArray, ape: MLXArray, norm: MLXArray) {
        self.init(
            wkv: DeepSeekLinearWeight(wkv),
            wgate: DeepSeekLinearWeight(wgate),
            ape: ape,
            norm: norm
        )
    }
}

public enum DeepSeekKVCompressor {
    public static func forward(
        _ x: MLXArray,
        weights: DeepSeekCompressorWeights,
        spec: DeepSeekCompressorSpec,
        poolingCache: DeepSeekPoolingCache?,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        guard let poolingCache else {
            return try forwardNoCache(
                x,
                weights: weights,
                spec: spec,
                positionOffset: positionOffset
            )
        }
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("compressor input must have shape [batch, length, hidden]")
        }
        guard spec.compressRatio > 0 else {
            throw MLXFastError.invalidInput("compressor ratio must be positive")
        }

        let kv = DeepSeekOps.linear(input: x, weight: weights.wkv)
        let gate = DeepSeekOps.linear(input: x, weight: weights.wgate)
        let ready = try poolingCache.accumulateWindows(
            kv: kv,
            gate: gate,
            offset: positionOffset
        )
        let newPooled = try compressReadyWindows(
            kv: ready.kv,
            gate: ready.gate,
            weights: weights,
            spec: spec,
            positionOffset: ready.baseOffset
        )
        return poolingCache.updateAndFetch(newPooled)
    }

    public static func forwardNoCache(
        _ x: MLXArray,
        weights: DeepSeekCompressorWeights,
        spec: DeepSeekCompressorSpec,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("compressor input must have shape [batch, length, hidden]")
        }
        guard spec.compressRatio > 0 else {
            throw MLXFastError.invalidInput("compressor ratio must be positive")
        }

        let batchSize = x.shape[0]
        let usable = (x.shape[1] / spec.compressRatio) * spec.compressRatio
        if usable == 0 {
            return zeros([batchSize, 0, spec.headDim], dtype: x.dtype)
        }

        let xUsable = x[0..., 0..<usable, 0...]
        let kv = DeepSeekOps.linear(input: xUsable, weight: weights.wkv)
        let gate = DeepSeekOps.linear(input: xUsable, weight: weights.wgate)
        return try compressReadyWindows(
            kv: kv,
            gate: gate,
            weights: weights,
            spec: spec,
            positionOffset: positionOffset
        )
    }

    private static func compressReadyWindows(
        kv: MLXArray,
        gate: MLXArray,
        weights: DeepSeekCompressorWeights,
        spec: DeepSeekCompressorSpec,
        positionOffset: Int
    ) throws -> MLXArray {
        let batchSize = kv.shape[0]
        guard kv.shape[1] > 0 else {
            return zeros([batchSize, 0, spec.headDim], dtype: kv.dtype)
        }
        guard kv.shape[1] % spec.compressRatio == 0 else {
            throw MLXFastError.invalidInput("ready compressor KV length must be a multiple of ratio")
        }
        let windows = kv.shape[1] / spec.compressRatio
        let kvWindows = kv.reshaped([batchSize, windows, spec.compressRatio, -1])
        let gateWindows = gate.reshaped([batchSize, windows, spec.compressRatio, -1])
        let pooled = spec.compressRatio == 4
            ? overlapCompress(kv: kvWindows, gate: gateWindows, ape: weights.ape)
            : simpleCompress(kv: kvWindows, gate: gateWindows, ape: weights.ape)
        let normalized = DeepSeekOps.rmsNorm(input: pooled, weight: weights.norm, eps: spec.rmsNormEps)
        let rope = try DeepSeekRoPE(
            rotaryDimensions: spec.ropeHeadDim,
            base: spec.ropeTheta,
            scaling: spec.ropeScaling,
            maxPositionEmbeddings: spec.maxPositionEmbeddings,
            freqScale: spec.compressRatio
        )
        return try rope.applied(
            to: normalized.expandedDimensions(axis: 1),
            offset: positionOffset
        ).squeezed(axis: 1)
    }

    public static func simpleCompress(kv: MLXArray, gate: MLXArray, ape: MLXArray) -> MLXArray {
        let weights = softmax(gate.asType(.float32) + ape, axis: -2, precise: true)
            .asType(kv.dtype)
        return (kv * weights).sum(axis: -2)
    }

    public static func overlapCompress(kv: MLXArray, gate: MLXArray, ape: MLXArray) -> MLXArray {
        let batchSize = kv.shape[0]
        let ratio = kv.shape[2]
        let halfDim = kv.shape[3] / 2
        let gate = gate + ape.asType(gate.dtype)

        let zeroKV = zeros([batchSize, 1, ratio, halfDim], dtype: kv.dtype)
        let kvParts = kv.split(parts: 2, axis: -1)
        let shiftedKV = concatenated([zeroKV, kvParts[0][0..., ..<(-1), 0..., 0...]], axis: 1)
        let overlappedKV = concatenated([shiftedKV, kvParts[1]], axis: 2)

        let zeroGate = full([batchSize, 1, ratio, halfDim], values: -Float.infinity)
        let gateParts = gate.split(parts: 2, axis: -1)
        let shiftedGate = concatenated([zeroGate, gateParts[0][0..., ..<(-1), 0..., 0...]], axis: 1)
        let overlappedGate = concatenated([shiftedGate, gateParts[1]], axis: 2)

        let weights = softmax(overlappedGate, axis: -2, precise: true)
        return (overlappedKV * weights.asType(overlappedKV.dtype)).sum(axis: -2)
    }
}

public enum DeepSeekLocalAttention {
    public static func forward(
        _ x: MLXArray,
        weights: DeepSeekLocalAttentionWeights,
        spec: DeepSeekLocalAttentionSpec,
        mask: MLXArray? = nil,
        cache: DeepSeekLocalKVCache? = nil,
        windowSize: Int? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try validateInput(x, spec: spec)
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let rope = try DeepSeekRoPE(
            rotaryDimensions: spec.qkRopeHeadDim,
            base: spec.ropeTheta,
            scaling: nil,
            maxPositionEmbeddings: spec.maxPositionEmbeddings
        )

        var q = DeepSeekOps.linear(input: x, weight: weights.wqA)
        q = DeepSeekOps.rmsNorm(input: q, weight: weights.qNorm, eps: spec.rmsNormEps)
        q = DeepSeekOps.linear(input: q, weight: weights.wqB)
        q = q.reshaped([batchSize, sequenceLength, spec.numAttentionHeads, spec.headDim])
        q = DeepSeekHyperConnection.weightlessRMSNorm(q, eps: spec.rmsNormEps)
        q = q.transposed(0, 2, 1, 3)
        q = try rope.applied(to: q, offset: positionOffset)

        var kv = DeepSeekOps.linear(input: x, weight: weights.wkv)
        kv = DeepSeekOps.rmsNorm(input: kv, weight: weights.kvNorm, eps: spec.rmsNormEps)
        kv = kv.reshaped([batchSize, 1, sequenceLength, spec.headDim])
        kv = try rope.applied(to: kv, offset: positionOffset)
        var attentionMask = mask
        if let cache {
            let cached = try cache.updateAndFetch(kv)
            kv = cached.kv
            attentionMask = try DeepSeekAttentionMask.causal(
                queryLength: sequenceLength,
                keyLength: kv.shape[2],
                queryOffset: positionOffset,
                keyOffset: cached.keyOffset,
                windowSize: windowSize
            )
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kv,
            values: kv,
            scale: spec.attentionScale,
            mask: attentionMask,
            sinks: weights.attentionSink.map { $0.asType(q.dtype) }
        )
        out = try rope.applied(to: out, offset: positionOffset, inverse: true)

        out = out.reshaped([batchSize, spec.outputGroups, -1, sequenceLength, spec.headDim])
        out = out.transposed(0, 1, 3, 2, 4).flattened(start: -2)
        out = try DeepSeekOps.multiLinear(input: out, weight: weights.woA)
        out = out.transposed(0, 2, 1, 3).flattened(start: -2)
        return DeepSeekOps.linear(input: out, weight: weights.woB, bias: weights.woBBias)
    }

    private static func validateInput(_ x: MLXArray, spec: DeepSeekLocalAttentionSpec) throws {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("local attention input must have shape [batch, length, hidden]")
        }
        guard spec.numAttentionHeads > 0, spec.headDim > 0, spec.outputGroups > 0 else {
            throw MLXFastError.invalidInput("local attention spec dimensions must be positive")
        }
        guard spec.numAttentionHeads % spec.outputGroups == 0 else {
            throw MLXFastError.invalidInput("attention heads must be divisible by output groups")
        }
    }
}

public enum DeepSeekCompressedAttention {
    public static func forward(
        _ x: MLXArray,
        weights: DeepSeekCompressedAttentionWeights,
        spec: DeepSeekCompressedAttentionSpec,
        mask: MLXArray? = nil,
        cache: DeepSeekLayerCache? = nil,
        windowSize: Int? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try validateInput(x, spec: spec)
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let rope = try DeepSeekRoPE(
            rotaryDimensions: spec.qkRopeHeadDim,
            base: spec.ropeTheta,
            scaling: spec.ropeScaling,
            maxPositionEmbeddings: spec.maxPositionEmbeddings
        )

        let qResidual = DeepSeekOps.rmsNorm(
            input: DeepSeekOps.linear(input: x, weight: weights.attention.wqA),
            weight: weights.attention.qNorm,
            eps: spec.rmsNormEps
        )
        var q = DeepSeekOps.linear(input: qResidual, weight: weights.attention.wqB)
        q = q.reshaped([batchSize, sequenceLength, spec.numAttentionHeads, spec.headDim])
        q = DeepSeekHyperConnection.weightlessRMSNorm(q, eps: spec.rmsNormEps)
        q = q.transposed(0, 2, 1, 3)
        q = try rope.applied(to: q, offset: positionOffset)

        var kv = DeepSeekOps.linear(input: x, weight: weights.attention.wkv)
        kv = DeepSeekOps.rmsNorm(input: kv, weight: weights.attention.kvNorm, eps: spec.rmsNormEps)
        kv = kv.reshaped([batchSize, 1, sequenceLength, spec.headDim])
        kv = try rope.applied(to: kv, offset: positionOffset)
        var localMask = mask
        if let localCache = cache?.local {
            let cached = try localCache.updateAndFetch(kv)
            kv = cached.kv
            localMask = try DeepSeekAttentionMask.causal(
                queryLength: sequenceLength,
                keyLength: kv.shape[2],
                queryOffset: positionOffset,
                keyOffset: cached.keyOffset,
                windowSize: windowSize
            )
        }

        let pooled = try DeepSeekKVCompressor.forward(
            x,
            weights: weights.compressor,
            spec: spec.compressorSpec,
            poolingCache: cache?.pooled,
            positionOffset: positionOffset
        )
        let pooledMask = cache?.pooled?.makeMask(queryLength: sequenceLength, offset: positionOffset)

        let pooledLength = pooled.shape[1]
        let sinks = weights.attention.attentionSink.map { $0.asType(q.dtype) }
        var out: MLXArray
        if spec.compressRatio == 4, pooledLength > spec.indexTopK {
            guard let indexer = weights.indexer else {
                throw MLXFastError.invalidInput(
                    "Swift DeepSeek sparse compressed attention requires indexer weights"
                )
            }
            try flushDeferredIndexInput(
                cache?.deferredIndexInput,
                into: cache?.indexPooled,
                weights: indexer.compressor,
                spec: spec.indexerSpec.compressorSpec
            )
            let topK = try DeepSeekIndexer.topKNoCache(
                x: x,
                qResidual: qResidual,
                weights: indexer,
                spec: spec.indexerSpec,
                poolingCache: cache?.indexPooled,
                positionOffset: positionOffset
            )
            if let indexPooled = cache?.indexPooled {
                guard indexPooled.pooledLength == pooledLength else {
                    throw MLXFastError.invalidInput(
                        "indexer pooled length \(indexPooled.pooledLength); expected \(pooledLength)"
                    )
                }
            }
            out = try sparsePooledAttention(
                q: q,
                localKV: kv,
                pooled: pooled,
                topK: topK,
                localMask: localMask,
                pooledMask: pooledMask,
                scale: spec.attentionScale,
                sinks: sinks
            )
        } else {
            if spec.compressRatio == 4,
               let indexer = weights.indexer,
               let indexPooled = cache?.indexPooled
            {
                if let deferredIndexInput = cache?.deferredIndexInput {
                    try deferredIndexInput.append(x, offset: positionOffset)
                } else {
                    _ = try DeepSeekKVCompressor.forward(
                        x,
                        weights: indexer.compressor,
                        spec: spec.indexerSpec.compressorSpec,
                        poolingCache: indexPooled,
                        positionOffset: positionOffset
                    )
                }
            }
            let attentionMask = try extendMask(
                localMask,
                pooledLength: pooledLength,
                pooledMask: pooledMask
            )
            if pooledLength > 0 {
                kv = concatenated([kv, pooled.expandedDimensions(axis: 1)], axis: 2)
            }

            out = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: kv,
                values: kv,
                scale: spec.attentionScale,
                mask: attentionMask,
                sinks: sinks
            )
        }
        out = try rope.applied(to: out, offset: positionOffset, inverse: true)

        out = out.reshaped([batchSize, spec.outputGroups, -1, sequenceLength, spec.headDim])
        out = out.transposed(0, 1, 3, 2, 4).flattened(start: -2)
        out = try DeepSeekOps.multiLinear(input: out, weight: weights.attention.woA)
        out = out.transposed(0, 2, 1, 3).flattened(start: -2)
        return DeepSeekOps.linear(
            input: out,
            weight: weights.attention.woB,
            bias: weights.attention.woBBias
        )
    }

    private static func flushDeferredIndexInput(
        _ deferred: DeepSeekDeferredInputCache?,
        into poolingCache: DeepSeekPoolingCache?,
        weights: DeepSeekCompressorWeights,
        spec: DeepSeekCompressorSpec
    ) throws {
        guard let deferred, !deferred.isEmpty else {
            return
        }
        guard let poolingCache else {
            throw MLXFastError.invalidInput("deferred indexer inputs require an index pooling cache")
        }
        if let chunk = deferred.drainMerged() {
            _ = try DeepSeekKVCompressor.forward(
                chunk.x,
                weights: weights,
                spec: spec,
                poolingCache: poolingCache,
                positionOffset: chunk.offset
            )
        }
    }

    private static func validateInput(_ x: MLXArray, spec: DeepSeekCompressedAttentionSpec) throws {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("compressed attention input must have shape [batch, length, hidden]")
        }
        guard spec.compressRatio == 4 || spec.compressRatio == 128 else {
            throw MLXFastError.invalidInput(
                "compressed attention ratio \(spec.compressRatio) is unsupported"
            )
        }
        guard spec.numAttentionHeads > 0, spec.headDim > 0, spec.outputGroups > 0 else {
            throw MLXFastError.invalidInput("compressed attention spec dimensions must be positive")
        }
        guard spec.numAttentionHeads % spec.outputGroups == 0 else {
            throw MLXFastError.invalidInput("attention heads must be divisible by output groups")
        }
    }

    private static func extendMask(
        _ mask: MLXArray?,
        pooledLength: Int,
        pooledMask: MLXArray? = nil
    ) throws -> MLXArray? {
        guard let mask, pooledLength > 0 else {
            return mask
        }
        guard mask.shape.count == 2 || mask.shape.count == 4 else {
            throw MLXFastError.invalidInput(
                "compressed attention mask must have shape [query, key] or [batch, heads, query, key]"
            )
        }
        let poolShape = Array(mask.shape.dropLast()) + [pooledLength]
        let poolMask = try expandedPooledMask(
            pooledMask,
            targetShape: poolShape,
            dtype: mask.dtype
        )
        return concatenated([mask, poolMask], axis: -1)
    }

    private static func expandedPooledMask(
        _ pooledMask: MLXArray?,
        targetShape: [Int],
        dtype: DType
    ) throws -> MLXArray {
        guard let pooledMask else {
            return ones(targetShape, dtype: dtype)
        }
        switch pooledMask.shape.count {
        case 2:
            return broadcast(pooledMask, to: targetShape).asType(dtype)
        case 3:
            return broadcast(pooledMask.expandedDimensions(axis: 1), to: targetShape).asType(dtype)
        case targetShape.count:
            return broadcast(pooledMask, to: targetShape).asType(dtype)
        default:
            throw MLXFastError.invalidInput("pooled attention mask has unsupported shape \(pooledMask.shape)")
        }
    }

    private static func sparsePooledAttention(
        q: MLXArray,
        localKV: MLXArray,
        pooled: MLXArray,
        topK: MLXArray,
        localMask: MLXArray?,
        pooledMask: MLXArray?,
        scale: Float,
        sinks: MLXArray?
    ) throws -> MLXArray {
        let batchSize = q.shape[0]
        let heads = q.shape[1]
        let sequenceLength = q.shape[2]
        let headDim = q.shape[3]
        let selectedCount = topK.shape[2]
        let pooledLength = pooled.shape[1]

        let batchOffsets = (arange(batchSize, dtype: .int32) * Int32(pooledLength))
            .reshaped([batchSize, 1, 1])
        let flatIndex = (topK + batchOffsets).reshaped([-1])
        let selectedPooled = pooled.reshaped([batchSize * pooledLength, headDim])
            .take(flatIndex, axis: 0)
            .reshaped([batchSize, sequenceLength, selectedCount, headDim])

        let qScaled = q * scale
        var localScores = matmul(qScaled, localKV.swappedAxes(-1, -2))
        localScores = applyScoreMask(localScores, mask: localMask)
        var normalizer = localScores.logSumExp(axis: -1, keepDims: true)

        let pooledSQ = selectedPooled
        let qBL = qScaled.transposed(0, 2, 1, 3)
        var pooledScores = matmul(qBL, pooledSQ.swappedAxes(-1, -2))
            .transposed(0, 2, 1, 3)
        let sparseMask = try sparsePooledMask(
            pooledMask,
            topK: topK,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            pooledLength: pooledLength
        )
        pooledScores = applyScoreMask(pooledScores, mask: sparseMask)
        normalizer = logAddExp(
            normalizer,
            pooledScores.logSumExp(axis: -1, keepDims: true)
        )
        if let sinks {
            normalizer = logAddExp(
                normalizer,
                sinks.reshaped([1, heads, 1, 1])
            )
        }

        let localWeights = (localScores - normalizer).exp()
        let pooledWeights = (pooledScores - normalizer).exp()
        var out = matmul(localWeights, localKV)
        let pooledOut = matmul(
            pooledWeights.transposed(0, 2, 1, 3),
            pooledSQ
        ).transposed(0, 2, 1, 3)
        out = out + pooledOut
        return out.asType(q.dtype)
    }

    private static func applyScoreMask(_ scores: MLXArray, mask: MLXArray?) -> MLXArray {
        guard let mask else {
            return scores
        }
        if mask.dtype == .bool {
            return which(mask, scores, full(scores.shape, values: -Float.greatestFiniteMagnitude))
        }
        return scores + mask.asType(scores.dtype)
    }

    private static func sparsePooledMask(
        _ pooledMask: MLXArray?,
        topK: MLXArray,
        batchSize: Int,
        sequenceLength: Int,
        pooledLength: Int
    ) throws -> MLXArray? {
        guard let pooledMask else {
            return nil
        }
        let batchedMask: MLXArray
        switch pooledMask.shape.count {
        case 2:
            batchedMask = broadcast(
                pooledMask.expandedDimensions(axis: 0),
                to: [batchSize, sequenceLength, pooledLength]
            )
        case 3:
            batchedMask = broadcast(
                pooledMask,
                to: [batchSize, sequenceLength, pooledLength]
            )
        default:
            throw MLXFastError.invalidInput("pooled sparse mask has unsupported shape \(pooledMask.shape)")
        }
        return takeAlong(batchedMask, topK, axis: -1).expandedDimensions(axis: 1)
    }
}

public enum DeepSeekIndexer {
    public static func topKNoCache(
        x: MLXArray,
        qResidual: MLXArray,
        weights: DeepSeekIndexerWeights,
        spec: DeepSeekIndexerSpec,
        poolingCache: DeepSeekPoolingCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        guard x.shape.count == 3 else {
            throw MLXFastError.invalidInput("indexer input must have shape [batch, length, hidden]")
        }
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let pooled = try DeepSeekKVCompressor.forward(
            x,
            weights: weights.compressor,
            spec: spec.compressorSpec,
            poolingCache: poolingCache,
            positionOffset: positionOffset
        )
        let pooledLength = pooled.shape[1]
        guard pooledLength > 0 else {
            throw MLXFastError.invalidInput("indexer requires at least one pooled KV token")
        }

        let rope = try DeepSeekRoPE(
            rotaryDimensions: spec.qkRopeHeadDim,
            base: spec.ropeTheta,
            scaling: spec.ropeScaling,
            maxPositionEmbeddings: spec.maxPositionEmbeddings
        )
        var q = DeepSeekOps.linear(input: qResidual, weight: weights.wqB)
        q = q.reshaped([batchSize, sequenceLength, spec.indexHeads, spec.indexHeadDim])
        q = q.transposed(0, 2, 1, 3)
        q = try rope.applied(to: q, offset: positionOffset)

        var scores = matmul(
            q.asType(.float32),
            pooled.expandedDimensions(axis: 1).swappedAxes(-1, -2).asType(.float32)
        )
        scores = maximum(scores, 0) * spec.indexHeadScale
        let weightsByHead = DeepSeekOps.linear(input: x, weight: weights.weightsProj)
            .asType(.float32)
            * spec.indexHeadsScale
        scores = (
            scores * weightsByHead.swappedAxes(-1, -2).expandedDimensions(axis: -1)
        ).sum(axis: 1)

        let k = min(spec.indexTopK, pooledLength)
        return argPartition(-scores, kth: k - 1, axis: -1)[
            .ellipsis,
            0..<k
        ].asType(.int32)
    }
}
