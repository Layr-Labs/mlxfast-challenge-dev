import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

// Fixture-based coverage for the mlx-swift-lm adapter's load chain
// (language_model. prefix strip -> sanitize -> conditional quantize ->
// update(verify: .all) -> eval), which otherwise only ever runs against the
// real 17 GB checkpoint. A tiny library model provides its own ground truth:
// we instantiate it, save its full parameter tree as a prefixed safetensors
// shard the way the transform lays files out, then require the loader to
// reconstruct a working model from disk.
@Test
func libraryAdapterLoadsPrefixStrippedShardAndServesLogits() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    // Tiny two-layer (sliding + full) text config in the transform-authored
    // flattened shape. num_kv_shared_layers must be explicit: the library
    // default (20) exceeds the layer count and would trap.
    let vocab = 128
    try """
    {
      "model_type": "gemma4_text",
      "hidden_size": 64,
      "intermediate_size": 128,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "head_dim": 32,
      "global_head_dim": 32,
      "num_key_value_heads": 2,
      "num_global_key_value_heads": 1,
      "num_kv_shared_layers": 0,
      "hidden_size_per_layer_input": 0,
      "vocab_size": \(vocab),
      "vocab_size_per_layer_input": \(vocab),
      "sliding_window": 8,
      "attention_k_eq_v": true,
      "final_logit_softcapping": 30.0,
      "use_double_wide_mlp": false,
      "tie_word_embeddings": true,
      "rms_norm_eps": 1e-6,
      "layer_types": ["sliding_attention", "full_attention"],
      "rope_parameters": {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """.write(to: root.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

    // Ground truth: the library's own tiny model. Persist its complete
    // parameter tree with the checkpoint's language_model. prefix, exactly
    // as the transform-produced shards name text-tower tensors.
    let configData = try Data(contentsOf: root.appendingPathComponent("config.json"))
    let textConfig = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configData)
    let reference = Gemma4TextModel(textConfig)
    var arrays: [String: MLXArray] = [:]
    for (path, array) in reference.parameters().flattened() {
        arrays["language_model.\(path)"] = array
    }
    eval(Array(arrays.values))
    #expect(!arrays.isEmpty)
    try save(arrays: arrays, url: root.appendingPathComponent("model.safetensors"))

    // The runtime config only supplies quantization parameters to the load
    // chain; built directly because Gemma4Config.load enforces the frozen
    // production architecture and correctly rejects tiny fixtures.
    let runtimeConfig = tinyAdapterRuntimeConfig(vocabSize: vocab)

    let loaded = try Gemma4RuntimeWeightCache.loadLibraryModel(
        weightsPath: root.path,
        config: runtimeConfig
    )

    // The fixture is unquantized (no .scales companions), so the conditional
    // quantize pass must have left every linear un-quantized and update
    // (verify: .all) accepted the full tree. A forward through both layer
    // types must produce finite logits of the right shape...
    let cache = loaded.newCache(parameters: nil)
    let prompt = MLXArray(Array(0..<8).map { Int32($0 % vocab) }, [1, 8])
    let logits = loaded(prompt, cache: cache)
    #expect(logits.shape == [1, 8, vocab])
    eval(logits)
    #expect(logits[0, 7].asArray(Float.self).allSatisfy { $0.isFinite })

    // ...and the KV cache offset must advance the way the harness's position
    // verification assumes (total tokens consumed, both layer types).
    #expect(cache.first?.offset == 8)
    let step = loaded(MLXArray([Int32(1)], [1, 1]), cache: cache)
    #expect(step.shape == [1, 1, vocab])
    #expect(cache.first?.offset == 9)
}

private func tinyAdapterRuntimeConfig(vocabSize: Int) -> Gemma4Config {
    Gemma4Config(
        modelType: "gemma4_text",
        vocabSize: vocabSize,
        hiddenSize: 64,
        intermediateSize: 128,
        numHiddenLayers: 2,
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
        slidingWindow: 8,
        layerTypes: [.sliding, .full],
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
