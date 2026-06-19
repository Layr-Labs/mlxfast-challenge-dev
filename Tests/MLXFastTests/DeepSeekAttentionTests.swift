import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekLocalAttentionRunsOneTokenDensePathWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekLocalAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let weights = DeepSeekLocalAttentionWeights(
        wqA: identity,
        qNorm: ones([2], dtype: .float32),
        wqB: identity,
        wkv: identity,
        kvNorm: ones([2], dtype: .float32),
        woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
        woB: identity
    )

    let output = try DeepSeekLocalAttention.forward(
        MLXArray([Float(3), 4], [1, 1, 2]),
        weights: weights,
        spec: spec
    )

    #expect(output.shape == [1, 1, 2])
    let values = output.asArray(Float.self)
    #expect(abs(values[0] - 0.84852815) < 1e-5)
    #expect(abs(values[1] - 1.1313709) < 1e-5)
}

@Test
func deepSeekLocalAttentionUsesKVCacheWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekLocalAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let weights = DeepSeekLocalAttentionWeights(
        wqA: identity,
        qNorm: ones([2], dtype: .float32),
        wqB: identity,
        wkv: identity,
        kvNorm: ones([2], dtype: .float32),
        woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
        woB: identity
    )
    let cache = DeepSeekLocalKVCache(maxSize: 2)

    let prefill = try DeepSeekLocalAttention.forward(
        MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 2
    )
    #expect(prefill.shape == [1, 3, 2])
    #expect(cache.offset == 3)
    #expect(cache.startPosition == 1)

    let decode = try DeepSeekLocalAttention.forward(
        MLXArray([Float(7), 8], [1, 1, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 2,
        positionOffset: 3
    )
    #expect(decode.shape == [1, 1, 2])
    #expect(cache.offset == 4)
    #expect(cache.startPosition == 2)
}

@Test
func deepSeekAttentionMasksMatchPrefillAndWindowDecodeWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let prefill = try DeepSeekAttentionMask.causal(queryLength: 3)
    #expect(prefill.shape == [3, 3])
    #expect(prefill.asArray(Bool.self) == [
        true, false, false,
        true, true, false,
        true, true, true,
    ])

    let decode = try DeepSeekAttentionMask.causal(
        queryLength: 1,
        keyLength: 4,
        queryOffset: 3,
        keyOffset: 0,
        windowSize: 2
    )
    #expect(decode.shape == [1, 4])
    #expect(decode.asArray(Bool.self) == [false, false, true, true])
}

@Test
func deepSeekKVCompressorPoolsWindowsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let compressed = DeepSeekKVCompressor.simpleCompress(
        kv: MLXArray([Float(1), 3, 5, 7], [1, 1, 2, 2]),
        gate: zeros([1, 1, 2, 2], dtype: .float32),
        ape: zeros([2, 2], dtype: .float32)
    )
    #expect(compressed.shape == [1, 1, 2])
    #expect(compressed.asArray(Float.self) == [3, 5])

    let spec = DeepSeekCompressorSpec(
        compressRatio: 2,
        headDim: 2,
        ropeHeadDim: 2,
        ropeTheta: 10_000.0,
        ropeScaling: nil,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0
    )
    let weights = DeepSeekCompressorWeights(
        wkv: MLXArray([Float(1), 0, 0, 1], [2, 2]),
        wgate: zeros([2, 2], dtype: .float32),
        ape: zeros([2, 2], dtype: .float32),
        norm: ones([2], dtype: .float32)
    )
    let output = try DeepSeekKVCompressor.forwardNoCache(
        MLXArray([Float(1), 3, 5, 7], [1, 2, 2]),
        weights: weights,
        spec: spec
    )
    #expect(output.shape == [1, 1, 2])
    let values = output.asArray(Float.self)
    #expect(abs(values[0] - 0.7276069) < 1e-5)
    #expect(abs(values[1] - 1.2126781) < 1e-5)

    let overlapKV = MLXArray((1...16).map { Float($0) }, [1, 1, 4, 4])
    let unbiasedOverlap = DeepSeekKVCompressor.overlapCompress(
        kv: overlapKV,
        gate: zeros([1, 1, 4, 4], dtype: .float32),
        ape: zeros([4, 4], dtype: .float32)
    ).asArray(Float.self)
    let biasedOverlap = DeepSeekKVCompressor.overlapCompress(
        kv: overlapKV,
        gate: zeros([1, 1, 4, 4], dtype: .float32),
        ape: MLXArray(
            [
                Float(0), 0, 0, 0,
                10, 0, 0, 0,
                0, 10, 0, 0,
                0, 0, 10, 0,
            ],
            [4, 4]
        )
    ).asArray(Float.self)
    #expect(biasedOverlap.count == 2)
    #expect(zip(biasedOverlap, unbiasedOverlap).contains { abs($0 - $1) > 1e-4 })
}

@Test
func deepSeekCompressedAttentionAppendsPooledKVWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekCompressedAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        ropeScaling: nil,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0,
        compressRatio: 4,
        indexHeads: 1,
        indexHeadDim: 2,
        indexTopK: 2
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let weights = DeepSeekCompressedAttentionWeights(
        attention: DeepSeekLocalAttentionWeights(
            wqA: identity,
            qNorm: ones([2], dtype: .float32),
            wqB: identity,
            wkv: identity,
            kvNorm: ones([2], dtype: .float32),
            woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
            woB: identity
        ),
        compressor: DeepSeekCompressorWeights(
            wkv: MLXArray([Float(1), 0, 0, 1, 1, 0, 0, 1], [4, 2]),
            wgate: zeros([4, 2], dtype: .float32),
            ape: zeros([4, 4], dtype: .float32),
            norm: ones([2], dtype: .float32)
        )
    )
    let mask = try DeepSeekAttentionMask.causal(queryLength: 4)
    let output = try DeepSeekCompressedAttention.forward(
        MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [1, 4, 2]),
        weights: weights,
        spec: spec,
        mask: mask
    )

    #expect(output.shape == [1, 4, 2])
    let values = output.asArray(Float.self)
    #expect(values.allSatisfy { $0.isFinite })
}

@Test
func deepSeekCompressedAttentionUsesPoolingCacheWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekCompressedAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        ropeScaling: nil,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0,
        compressRatio: 4,
        indexHeads: 1,
        indexHeadDim: 2,
        indexTopK: 2
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let weights = DeepSeekCompressedAttentionWeights(
        attention: DeepSeekLocalAttentionWeights(
            wqA: identity,
            qNorm: ones([2], dtype: .float32),
            wqB: identity,
            wkv: identity,
            kvNorm: ones([2], dtype: .float32),
            woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
            woB: identity
        ),
        compressor: DeepSeekCompressorWeights(
            wkv: MLXArray([Float(1), 0, 0, 1, 1, 0, 0, 1], [4, 2]),
            wgate: zeros([4, 2], dtype: .float32),
            ape: zeros([4, 4], dtype: .float32),
            norm: ones([2], dtype: .float32)
        )
    )
    let cache = DeepSeekLayerCache(
        local: DeepSeekLocalKVCache(maxSize: 4),
        pooled: DeepSeekPoolingCache(ratio: 4),
        indexPooled: nil
    )
    let prefill = try DeepSeekCompressedAttention.forward(
        MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 4
    )
    #expect(prefill.shape == [1, 3, 2])
    #expect(cache.pooled?.pooledLength == 0)

    let decode = try DeepSeekCompressedAttention.forward(
        MLXArray([Float(7), 8], [1, 1, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 4,
        positionOffset: 3
    )
    #expect(decode.shape == [1, 1, 2])
    #expect(cache.pooled?.pooledLength == 1)
}

@Test
func deepSeekCompressedAttentionWarmsIndexerCacheBeforeSparseBranchWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekCompressedAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        ropeScaling: nil,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0,
        compressRatio: 4,
        indexHeads: 1,
        indexHeadDim: 2,
        indexTopK: 2
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let compressor = DeepSeekCompressorWeights(
        wkv: MLXArray([Float(1), 0, 0, 1, 1, 0, 0, 1], [4, 2]),
        wgate: zeros([4, 2], dtype: .float32),
        ape: zeros([4, 4], dtype: .float32),
        norm: ones([2], dtype: .float32)
    )
    let weights = DeepSeekCompressedAttentionWeights(
        attention: DeepSeekLocalAttentionWeights(
            wqA: identity,
            qNorm: ones([2], dtype: .float32),
            wqB: identity,
            wkv: identity,
            kvNorm: ones([2], dtype: .float32),
            woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
            woB: identity
        ),
        compressor: compressor,
        indexer: DeepSeekIndexerWeights(
            wqB: identity,
            weightsProj: MLXArray([Float(1), 1], [1, 2]),
            compressor: compressor
        )
    )
    let cache = DeepSeekLayerCache(
        local: DeepSeekLocalKVCache(maxSize: 4),
        pooled: DeepSeekPoolingCache(ratio: 4),
        indexPooled: DeepSeekPoolingCache(ratio: 4)
    )
    _ = try DeepSeekCompressedAttention.forward(
        MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 4
    )
    #expect(cache.pooled?.pooledLength == 0)
    #expect(cache.indexPooled?.pooledLength == 0)

    let decode = try DeepSeekCompressedAttention.forward(
        MLXArray([Float(7), 8], [1, 1, 2]),
        weights: weights,
        spec: spec,
        cache: cache,
        windowSize: 4,
        positionOffset: 3
    )
    #expect(decode.shape == [1, 1, 2])
    #expect(cache.pooled?.pooledLength == 1)
    #expect(cache.indexPooled?.pooledLength == 1)
}

@Test
func deepSeekCompressedAttentionRunsSparsePooledBranchWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekCompressedAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000.0,
        ropeScaling: nil,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0,
        compressRatio: 4,
        indexHeads: 1,
        indexHeadDim: 2,
        indexTopK: 1
    )
    let identity = MLXArray([Float(1), 0, 0, 1], [2, 2])
    let compressor = DeepSeekCompressorWeights(
        wkv: MLXArray([Float(1), 0, 0, 1, 1, 0, 0, 1], [4, 2]),
        wgate: zeros([4, 2], dtype: .float32),
        ape: zeros([4, 4], dtype: .float32),
        norm: ones([2], dtype: .float32)
    )
    let weights = DeepSeekCompressedAttentionWeights(
        attention: DeepSeekLocalAttentionWeights(
            wqA: identity,
            qNorm: ones([2], dtype: .float32),
            wqB: identity,
            wkv: identity,
            kvNorm: ones([2], dtype: .float32),
            woA: MLXArray([Float(1), 0, 0, 1], [1, 2, 2]),
            woB: identity
        ),
        compressor: compressor,
        indexer: DeepSeekIndexerWeights(
            wqB: identity,
            weightsProj: MLXArray([Float(1), 1], [1, 2]),
            compressor: compressor
        )
    )
    let mask = try DeepSeekAttentionMask.causal(queryLength: 8)
    let input = MLXArray((1...16).map { Float($0) }, [1, 8, 2])
    let output = try DeepSeekCompressedAttention.forward(
        input,
        weights: weights,
        spec: spec,
        mask: mask
    )

    #expect(output.shape == [1, 8, 2])
    let values = output.asArray(Float.self)
    #expect(values.allSatisfy { $0.isFinite })
}
