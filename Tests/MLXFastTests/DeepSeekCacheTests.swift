import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekLocalKVCacheReturnsFullPrefillAndRetainsWindowWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let cache = DeepSeekLocalKVCache(maxSize: 3)
    let prefill = try cache.updateAndFetch(MLXArray((1...8).map { Float($0) }, [1, 1, 4, 2]))
    #expect(prefill.kv.shape == [1, 1, 4, 2])
    #expect(prefill.keyOffset == 0)
    #expect(cache.offset == 4)
    #expect(cache.startPosition == 1)

    let decode = try cache.updateAndFetch(MLXArray([Float(9), 10], [1, 1, 1, 2]))
    #expect(decode.kv.shape == [1, 1, 4, 2])
    #expect(decode.keyOffset == 1)
    #expect(cache.offset == 5)
    #expect(cache.startPosition == 2)
}

@Test
func deepSeekPoolingCacheBuffersRemaindersAndBuildsMasksWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let cache = DeepSeekPoolingCache(ratio: 4)
    let first = try cache.accumulateWindows(
        kv: MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        gate: zeros([1, 3, 2], dtype: .float32),
        offset: 0
    )
    #expect(first.kv.shape == [1, 0, 2])
    #expect(first.baseOffset == 0)

    let second = try cache.accumulateWindows(
        kv: MLXArray([Float(7), 8], [1, 1, 2]),
        gate: zeros([1, 1, 2], dtype: .float32),
        offset: 3
    )
    #expect(second.kv.shape == [1, 4, 2])
    #expect(second.baseOffset == 0)
    _ = cache.updateAndFetch(MLXArray([Float(1), 2], [1, 1, 2]))

    let mask = try #require(cache.makeMask(queryLength: 4, offset: 0))
    #expect(mask.shape == [4, 1])
    #expect(mask.asArray(Bool.self) == [false, false, false, true])
}

@Test
func deepSeekLocalKVCacheExposesCachedArraysForMaterialization() throws {
    let cache = DeepSeekLocalKVCache(maxSize: 4)

    #expect(cache.arraysForMaterialization().isEmpty)
    _ = try cache.updateAndFetch(MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2]))

    #expect(cache.arraysForMaterialization().map(\.shape) == [[1, 1, 2, 2]])
}

@Test
func deepSeekPoolingCacheExposesBufferedAndPooledArraysForMaterialization() throws {
    let cache = DeepSeekPoolingCache(ratio: 4)
    _ = try cache.accumulateWindows(
        kv: MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        gate: zeros([1, 3, 2], dtype: .float32),
        offset: 0
    )
    _ = cache.updateAndFetch(MLXArray([Float(1), 2], [1, 1, 2]))

    #expect(cache.arraysForMaterialization().map(\.shape) == [
        [1, 3, 2],
        [1, 3, 2],
        [1, 1, 2],
    ])
}

@Test
func deepSeekModelCacheMaterializesCollectedCachedState() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "n_routed_experts": \(MLXFastConstants.routedExperts),
      "num_experts_per_tok": \(MLXFastConstants.expertsPerToken),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "sliding_window": 4
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try DeepSeekConfig.load(from: root.path)
    let cache = DeepSeekModelCache(config: config)

    _ = try cache.layers[0].local.updateAndFetch(
        MLXArray((1...4).map { Float($0) }, [1, 1, 2, 2])
    )
    _ = try cache.layers[2].pooled?.accumulateWindows(
        kv: MLXArray((1...6).map { Float($0) }, [1, 3, 2]),
        gate: zeros([1, 3, 2], dtype: .float32),
        offset: 0
    )

    #expect(cache.arraysForMaterialization().count == 3)
    cache.materializeCachedState()
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
