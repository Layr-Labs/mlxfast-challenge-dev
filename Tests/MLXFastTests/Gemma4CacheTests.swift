import Foundation
import MLX
import MLXLMCommon
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func gemma4KVCacheReturnsFullPrefillAndRetainsWindowWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let cache = Gemma4KVCache(maxSize: 3)
    let prefill = try cache.updateAndFetch(MLXArray((1...8).map { Float($0) }, [1, 1, 4, 2]))
    #expect(prefill.value.shape == [1, 1, 4, 2])
    #expect(prefill.offset == 0)
    #expect(cache.offset == 4)
    #expect(cache.startPosition == 1)

    let decode = try cache.updateAndFetch(MLXArray([Float(9), 10], [1, 1, 1, 2]))
    #expect(decode.value.shape == [1, 1, 4, 2])
    #expect(decode.offset == 1)
    #expect(cache.offset == 5)
    #expect(cache.startPosition == 2)
}

@Test
func gemma4KVCacheExposesCachedArraysForMaterialization() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let cache = Gemma4KVCache(maxSize: 4)

    #expect(cache.arraysForMaterialization().isEmpty)
    _ = try cache.updateAndFetch(MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2]))

    #expect(cache.arraysForMaterialization().map(\.shape) == [[1, 1, 2, 2]])
}

@Test
func gemma4KVCacheRejectsNonPositiveMaxSize() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let cache = Gemma4KVCache(maxSize: 0)
    #expect(throws: MLXFastError.self) {
        _ = try cache.updateAndFetch(MLXArray.zeros([1, 1, 1, 2]))
    }
}

@Test
func gemma4ModelCacheBuildsOneLayerCachePerConfiguredLayerType() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": \(MLXFastConstants.intermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "sliding_window": 4
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try Gemma4Config.load(from: root.path)
    let cache = Gemma4ModelCache(config: config)

    #expect(cache.layers.count == MLXFastConstants.numHiddenLayers)
    // Default pattern: five sliding layers, then one full-attention layer.
    #expect(cache.layers[0].keys.maxSize == 4)
    #expect(cache.layers[5].keys.maxSize == Gemma4ModelCache.unboundedCacheCap)
}

@Test
func gemma4ModelCacheMaterializesCollectedCachedStateWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    // Built via the memberwise initializer: Gemma4Config.load enforces the
    // frozen production architecture (60 layers) and correctly rejects tiny
    // fixture configs, so fixture tests must not route through it.
    let config = tinyCacheConfig(layerTypes: [.sliding, .full], slidingWindow: 4)
    let cache = Gemma4ModelCache(config: config)

    // Bespoke per-layer arrays still collect for materialization (kept for
    // the standalone Gemma4KVCache tests)...
    _ = try cache.layers[0].keys.updateAndFetch(MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2]))
    _ = try cache.layers[0].values.updateAndFetch(MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2]))
    #expect(cache.arraysForMaterialization().count == 2)

    // ...but the scored path materializes the library KV caches: a no-op
    // before any are adopted, and an eval of the adopted caches afterwards.
    cache.materializeCachedState()
    let libraryCache = KVCacheSimple()
    _ = libraryCache.update(
        keys: MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2]),
        values: MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2])
    )
    cache.adoptKVCaches([libraryCache])
    cache.materializeCachedState()
    #expect(libraryCache.offset == 2)
}

private func tinyCacheConfig(
    layerTypes: [Gemma4LayerType],
    slidingWindow: Int
) -> Gemma4Config {
    Gemma4Config(
        modelType: "gemma4_text",
        vocabSize: 128,
        hiddenSize: 64,
        intermediateSize: 128,
        numHiddenLayers: layerTypes.count,
        numAttentionHeads: 2,
        numKeyValueHeads: 2,
        numGlobalKeyValueHeads: 1,
        headDim: 32,
        globalHeadDim: 32,
        rmsNormEps: 1e-6,
        hiddenActivation: "gelu_pytorch_tanh",
        maxPositionEmbeddings: 1024,
        attentionBias: false,
        attentionDropout: 0,
        attentionKEqV: true,
        slidingWindow: slidingWindow,
        layerTypes: layerTypes,
        finalLogitSoftcapping: 30,
        tieWordEmbeddings: true,
        slidingRope: Gemma4RopeSpec(theta: 10_000, type: "default"),
        fullRope: Gemma4RopeSpec(theta: 1_000_000, type: "proportional", partialRotaryFactor: 0.25),
        quantizationGroupSize: 64,
        quantizationBits: 4
    )
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
