import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

@Test
func deepSeekConfigLoadsFrozenInvariants() throws {
    let root = try temporaryDirectory()
    try configJSON(
        layers: MLXFastConstants.numHiddenLayers,
        experts: MLXFastConstants.routedExperts,
        expertsPerToken: MLXFastConstants.expertsPerToken,
        vocab: MLXFastConstants.vocabSize
    ).write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try DeepSeekConfig.load(from: root.path)

    #expect(config.numHiddenLayers == MLXFastConstants.numHiddenLayers)
    #expect(config.routedExperts == MLXFastConstants.routedExperts)
    #expect(config.expertsPerToken == MLXFastConstants.expertsPerToken)
    #expect(config.vocabSize == MLXFastConstants.vocabSize)
    #expect(config.hiddenSize == MLXFastConstants.hiddenSize)
    #expect(config.compressRatios == DeepSeekConfig.defaultCompressRatios(layerCount: MLXFastConstants.numHiddenLayers))
    #expect(config.hcMult == 4)
    #expect(config.scoringFunc == "sqrtsoftplus")
}

@Test
func deepSeekConfigRejectsChangedArchitecture() throws {
    let root = try temporaryDirectory()
    try configJSON(
        layers: 42,
        experts: MLXFastConstants.routedExperts,
        expertsPerToken: MLXFastConstants.expertsPerToken,
        vocab: MLXFastConstants.vocabSize
    ).write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try DeepSeekConfig.load(from: root.path)
    }
}

@Test
func deepSeekConfigLoadsNestedTextConfigAndRopeScaling() throws {
    let root = try temporaryDirectory()
    try """
    {
      "text_config": {
        "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
        "n_routed_experts": \(MLXFastConstants.routedExperts),
        "num_experts_per_tok": \(MLXFastConstants.expertsPerToken),
        "vocab_size": \(MLXFastConstants.vocabSize),
        "hidden_size": \(MLXFastConstants.hiddenSize),
        "compress_ratios": \(compressRatiosJSON()),
        "rope_scaling": {
          "type": "deepseek_yarn",
          "factor": 40.0,
          "original_max_position_embeddings": 4096,
          "beta_fast": 32,
          "beta_slow": 1
        }
      }
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try DeepSeekConfig.load(from: root.path)

    #expect(config.hiddenSize == MLXFastConstants.hiddenSize)
    #expect(config.compressRatios.count == MLXFastConstants.numHiddenLayers)
    #expect(config.ropeScaling?.type == "deepseek_yarn")
    #expect(config.ropeScaling?.factor == 40.0)
    #expect(config.ropeScaling?.originalMaxPositionEmbeddings == 4096)
    #expect(config.ropeScaling?.betaFast == 32.0)
    #expect(config.ropeScaling?.betaSlow == 1.0)
}

@Test
func deepSeekConfigLoadsRealV4FlashShape() throws {
    let root = try temporaryDirectory()
    try """
    {
      "architectures": ["DeepseekV4ForCausalLM"],
      "model_type": "deepseek_v4",
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "n_routed_experts": \(MLXFastConstants.routedExperts),
      "num_experts_per_tok": \(MLXFastConstants.expertsPerToken),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "hidden_size": \(MLXFastConstants.hiddenSize),
      "intermediate_size": null,
      "moe_intermediate_size": \(MLXFastConstants.moeIntermediateSize),
      "num_attention_heads": \(MLXFastConstants.attentionHeads),
      "num_key_value_heads": \(MLXFastConstants.keyValueHeads),
      "compress_ratios": \(realV4FlashCompressRatiosJSON()),
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "model.layers.0.ffn.switch_mlp.gate_proj": {
          "group_size": 32,
          "bits": 4,
          "mode": "mxfp4"
        }
      }
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let config = try DeepSeekConfig.load(from: root.path)

    #expect(config.intermediateSize == MLXFastConstants.intermediateSize)
    #expect(config.compressRatios.count == MLXFastConstants.numHiddenLayers)
    #expect(config.compressRatios == DeepSeekConfig.defaultCompressRatios(layerCount: MLXFastConstants.numHiddenLayers))
    #expect(config.compressRatios[0] == 0)
    #expect(config.compressRatios[1] == 0)
    #expect(config.compressRatios[2] == 4)
    #expect(config.compressRatios[3] == 128)
    #expect(config.compressRatios[42] == 4)
}

@Test
func deepSeekConfigRejectsBadCompressionRatios() throws {
    let root = try temporaryDirectory()
    try """
    {
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "n_routed_experts": \(MLXFastConstants.routedExperts),
      "num_experts_per_tok": \(MLXFastConstants.expertsPerToken),
      "vocab_size": \(MLXFastConstants.vocabSize),
      "compress_ratios": [7]
    }
    """.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try DeepSeekConfig.load(from: root.path)
    }
}

private func configJSON(layers: Int, experts: Int, expertsPerToken: Int, vocab: Int) -> String {
    """
    {
      "num_hidden_layers": \(layers),
      "n_routed_experts": \(experts),
      "num_experts_per_tok": \(expertsPerToken),
      "vocab_size": \(vocab)
    }
    """
}

private func compressRatiosJSON() -> String {
    "[\(DeepSeekConfig.defaultCompressRatios(layerCount: MLXFastConstants.numHiddenLayers).map(String.init).joined(separator: ","))]"
}

private func realV4FlashCompressRatiosJSON() -> String {
    let ratios = DeepSeekConfig.defaultCompressRatios(layerCount: MLXFastConstants.numHiddenLayers) + [0]
    return "[\(ratios.map(String.init).joined(separator: ","))]"
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
