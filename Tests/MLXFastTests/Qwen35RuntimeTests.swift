import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastCore
@testable import MLXFastModel

@Test
func qwen35PinnedCacheTopologyIs48MambaAnd16FullAttention() throws {
    let layerTypes = Qwen35Config.expectedLayerTypes
    let topology = Qwen35CacheTopology(layerTypes: layerTypes)
    let caches: [any KVCache] = layerTypes.map {
        $0 == .linear ? MambaCache() : KVCacheSimple()
    }

    #expect(topology.mambaCount == 48)
    #expect(topology.fullAttentionCount == 16)
    try validateQwen35CacheTopology(
        caches: caches,
        layerTypes: layerTypes
    )
}

@Test
func qwen35CachePositionValidatesOnlyFullAttentionOffsets() throws {
    let firstMamba = MambaCache()
    let fullAttention = KVCacheSimple()
    let secondMamba = MambaCache()
    firstMamba.offset = 0
    fullAttention.offset = 7
    secondMamba.offset = 99
    let caches: [any KVCache] = [
        firstMamba,
        fullAttention,
        secondMamba,
    ]
    let layerTypes: [Qwen35LayerType] = [.linear, .full, .linear]

    try validateQwen35CachePosition(
        positionOffset: 7,
        caches: caches,
        layerTypes: layerTypes
    )

    fullAttention.offset = 8
    #expect(throws: MLXFastError.self) {
        try validateQwen35CachePosition(
            positionOffset: 7,
            caches: caches,
            layerTypes: layerTypes
        )
    }
}

@Test
func qwen35HostCachePositionCommitsTransactionally() throws {
    let cache = Qwen35ModelCache(layerTypes: [.linear, .full])
    var events: [String] = []

    let result: Int = try executeQwen35CachedForward(
        cache: cache,
        positionOffset: 0,
        inputLength: 8,
        validateLibraryCachePosition: {
            events.append("validate")
        },
        forward: {
            events.append("forward")
            return 42
        }
    )

    #expect(result == 42)
    #expect(events == ["validate", "forward"])
    #expect(cache.expectedPositionOffset == 8)

    #expect(throws: Qwen35RuntimeTestError.self) {
        let _: Int = try executeQwen35CachedForward(
            cache: cache,
            positionOffset: 8,
            inputLength: 1,
            validateLibraryCachePosition: {
                throw Qwen35RuntimeTestError.validation
            },
            forward: { 0 }
        )
    }
    #expect(cache.expectedPositionOffset == 8)

    #expect(throws: Qwen35RuntimeTestError.self) {
        let _: Int = try executeQwen35CachedForward(
            cache: cache,
            positionOffset: 8,
            inputLength: 1,
            validateLibraryCachePosition: {},
            forward: {
                throw Qwen35RuntimeTestError.forward
            }
        )
    }
    #expect(cache.expectedPositionOffset == 8)
}

@Test
func qwen35HostCachePositionRejectsInvalidTransitions() throws {
    #expect(
        try advancedQwen35CachePosition(
            positionOffset: 512,
            expectedPositionOffset: 512,
            inputLength: 1
        ) == 513
    )
    #expect(throws: MLXFastError.self) {
        _ = try advancedQwen35CachePosition(
            positionOffset: 511,
            expectedPositionOffset: 512,
            inputLength: 1
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try advancedQwen35CachePosition(
            positionOffset: 0,
            expectedPositionOffset: 0,
            inputLength: 0
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try advancedQwen35CachePosition(
            positionOffset: Int.max,
            expectedPositionOffset: Int.max,
            inputLength: 1
        )
    }
}

@Test
func qwen35LibraryEagerModelReturnsAllPositionsAndUsesHybridCache() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let model = Qwen35TextModel(try tinyQwen35TextConfiguration(layerCount: 4))
    let owner = Qwen35ModelCache(
        layerTypes: [.linear, .linear, .linear, .full]
    )
    let caches = try owner.cache(for: model)
    let reusedCaches = try owner.cache(for: model)
    #expect(caches.count == 4)
    #expect(caches[0] is MambaCache)
    #expect(caches[1] is MambaCache)
    #expect(caches[2] is MambaCache)
    #expect(caches[3] is KVCacheSimple)
    #expect(
        ObjectIdentifier(caches[0] as AnyObject)
            == ObjectIdentifier(reusedCaches[0] as AnyObject)
    )

    let prefill = qwen35EagerLogits(
        model: model,
        inputIDs: MLXArray([Int32(1), 2], [1, 2]),
        cache: caches
    )
    eval(prefill)
    #expect(prefill.shape == [1, 2, 32])
    #expect(caches[0].offset == 0)
    #expect(caches[3].offset == 2)

    let decode = qwen35EagerLogits(
        model: model,
        inputIDs: MLXArray([Int32(3)], [1, 1]),
        cache: caches
    )
    eval(decode)
    #expect(decode.shape == [1, 1, 32])
    #expect(caches[0].offset == 0)
    #expect(caches[3].offset == 3)
}

@Test
func qwen35LibraryTinyModelCachedDecodeMatchesUncachedFullContext() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let configuration = try tinyQwen35TextConfiguration(layerCount: 4)
    let model = Qwen35TextModel(configuration)
    seedTinyQwen35LibraryModel(model)

    let prompt = [Int32(1), 7, 3, 11]
    let continuation = Int32(5)
    let fullTokens = prompt + [continuation]
    let oneShot = model(
        MLXArray(fullTokens, [1, fullTokens.count]),
        cache: model.newCache(parameters: nil)
    )

    let decodeCaches = model.newCache(parameters: nil)
    let prefill = model(
        MLXArray(prompt, [1, prompt.count]),
        cache: decodeCaches
    )
    eval(prefill)
    let decode = model(
        MLXArray([continuation], [1, 1]),
        cache: decodeCaches
    )
    eval(oneShot, decode)

    let oneShotLast =
        oneShot
        .reshaped(-1, 32)[-1]
        .reshaped(1, 1, 32)
    let decodeDifference = qwen35MaximumAbsoluteDifference(
        oneShotLast,
        decode
    )
    #expect(
        decodeDifference <= 1e-4,
        "tiny Qwen library cached decode drifted by \(decodeDifference)"
    )

    let chunkCaches = model.newCache(parameters: nil)
    var chunkLogits: [MLXArray] = []
    for chunk in [
        Array(fullTokens[0..<2]),
        Array(fullTokens[2..<4]),
        Array(fullTokens[4..<5]),
    ] {
        let logits = model(
            MLXArray(chunk, [1, chunk.count]),
            cache: chunkCaches
        )
        eval(logits)
        chunkLogits.append(logits)
    }
    let chunked = concatenated(chunkLogits, axis: 1)
    let chunkDifference = qwen35MaximumAbsoluteDifference(oneShot, chunked)
    #expect(
        chunkDifference <= 1e-4,
        "tiny Qwen library chunked prefill drifted by \(chunkDifference)"
    )
}

@Test
func qwen35LibrarySanitizeConvertsConvAndDropsAbsentMTPHead() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let model = Qwen35TextModel(try tinyQwen35TextConfiguration(layerCount: 1))
    let sanitized = model.sanitize(weights: [
        "model.layers.0.linear_attn.conv1d.weight":
            MLXArray.zeros([384, 1, 4]),
        "model.layers.0.input_layernorm.weight":
            MLXArray.zeros([128]),
        "mtp.unused.weight":
            MLXArray.zeros([1]),
    ])

    #expect(
        sanitized["model.layers.0.linear_attn.conv1d.weight"]?.shape
            == [384, 4, 1]
    )
    let norm = try #require(
        sanitized["model.layers.0.input_layernorm.weight"]
    )
    #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 128))
    #expect(sanitized["mtp.unused.weight"] == nil)
}

private enum Qwen35RuntimeTestError: Error {
    case validation
    case forward
}

private func tinyQwen35TextConfiguration(
    layerCount: Int
) throws -> Qwen35TextConfiguration {
    let json = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 128,
          "num_hidden_layers": \(layerCount),
          "intermediate_size": 256,
          "num_attention_heads": 1,
          "num_key_value_heads": 1,
          "head_dim": 128,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 0.000001,
          "vocab_size": 32,
          "max_position_embeddings": 128,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "partial_rotary_factor": 0.25,
          "full_attention_interval": 4,
          "num_experts": 0,
          "mtp_num_hidden_layers": 0,
          "rope_parameters": {
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25
          }
        }
        """
    return try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(json.utf8)
    )
}

private func seedTinyQwen35LibraryModel(_ model: Qwen35TextModel) {
    let seeded = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            name,
            parameter -> (String, MLXArray) in
            let count = parameter.shape.reduce(1, *)
            let nameSeed = name.utf8.enumerated().reduce(0) {
                ($0 + ($1.offset + 1) * Int($1.element)) % 97
            }
            let values: [Float]
            if name.hasSuffix("norm.weight") {
                values = (0..<count).map {
                    1 + Float(($0 + nameSeed) % 7) * 0.001
                }
            } else if name.hasSuffix("A_log") {
                values = Array(repeating: 0, count: count)
            } else if name.hasSuffix("dt_bias") {
                values = (0..<count).map {
                    Float(($0 + nameSeed) % 5 - 2) * 0.01
                }
            } else {
                values = (0..<count).map {
                    Float(($0 + nameSeed) % 29 - 14) * 0.002
                }
            }
            return (
                name,
                MLXArray(values, parameter.shape).asType(parameter.dtype)
            )
        }
    )
    model.update(parameters: ModuleParameters.unflattened(seeded))
    eval(model)
}
