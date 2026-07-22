import Foundation
import MLXLMCommon
import MLXNN
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

// Direct unit coverage for `LagunaConfig.load` (the runtime side of the
// transformed-weights contract in docs/laguna-weight-contract.md). The
// runtime worker's pinned-configuration gate is covered separately in
// BenchmarkSupportTests; these tests pin the LagunaConfig parse/validation
// behavior itself, including the exact Poolside NVFP4 contract.

/// The pinned Poolside Laguna XS 2.1 NVFP4 runtime config.
func pinnedLagunaConfigObject() -> [String: Any] {
    let pinnedQuantization: [String: Any] = [
        "group_size": LagunaConstants.quantizationGroupSize,
        "bits": 4,
        "mode": "nvfp4",
    ]
    return [
        "model_type": "laguna",
        "vocab_size": LagunaConstants.vocabSize,
        "hidden_size": LagunaConstants.hiddenSize,
        "intermediate_size": LagunaConstants.denseIntermediateSize,
        "num_hidden_layers": LagunaConstants.numHiddenLayers,
        "num_attention_heads": LagunaConstants.fullAttentionHeads,
        "num_attention_heads_per_layer": (0..<LagunaConstants.numHiddenLayers).map {
            $0 % 4 == 0
                ? LagunaConstants.fullAttentionHeads
                : LagunaConstants.slidingAttentionHeads
        },
        "num_key_value_heads": LagunaConstants.numKeyValueHeads,
        "head_dim": LagunaConstants.headDim,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262_144,
        "attention_bias": false,
        "qkv_bias": NSNull(),
        "attention_dropout": 0.0,
        "sliding_window": LagunaConstants.slidingWindow,
        "layer_types": (0..<LagunaConstants.numHiddenLayers).map {
            $0 % 4 == 0 ? "full_attention" : "sliding_attention"
        },
        "mlp_layer_types": (0..<LagunaConstants.numHiddenLayers).map {
            $0 == 0 ? "dense" : "sparse"
        },
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "gating": "per-head",
        "gating_types": [String](
            repeating: "per_head",
            count: LagunaConstants.numHiddenLayers
        ),
        "tie_word_embeddings": false,
        "num_experts": LagunaConstants.numExperts,
        "num_experts_per_tok": LagunaConstants.numExpertsPerTok,
        "moe_intermediate_size": LagunaConstants.moeIntermediateSize,
        "shared_expert_intermediate_size": LagunaConstants.sharedExpertIntermediateSize,
        "moe_routed_scaling_factor": LagunaConstants.moeRoutedScalingFactor,
        "norm_topk_prob": true,
        "moe_apply_router_weight_on_input": false,
        "moe_router_logit_softcapping": NSNull(),
        "rope_parameters": [
            "sliding_attention": [
                "rope_theta": 10_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
            ],
            "full_attention": [
                "rope_theta": 500_000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 8_192,
                "beta_fast": 64.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
                "partial_rotary_factor": 0.5,
            ],
        ],
        "quantization": pinnedQuantization,
        "quantization_config": pinnedQuantization,
    ]
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

func loadLagunaConfig(_ object: [String: Any]) throws -> LagunaConfig {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let data = try JSONSerialization.data(withJSONObject: object)
    try data.write(to: root.appendingPathComponent("config.json"))
    return try LagunaConfig.load(from: root.path)
}

@Test
func lagunaConfigLoadsPinnedPoolsideNVFP4Geometry() throws {
    let config = try loadLagunaConfig(pinnedLagunaConfigObject())

    #expect(config.modelType == LagunaConstants.modelType)
    #expect(config.vocabSize == LagunaConstants.vocabSize)
    #expect(config.hiddenSize == LagunaConstants.hiddenSize)
    #expect(config.numHiddenLayers == LagunaConstants.numHiddenLayers)
    #expect(config.numKeyValueHeads == LagunaConstants.numKeyValueHeads)
    #expect(config.headDim == LagunaConstants.headDim)
    #expect(config.slidingWindow == LagunaConstants.slidingWindow)
    #expect(config.maxPositionEmbeddings == LagunaConstants.maxPositionEmbeddings)
    #expect(config.rmsNormEps == LagunaConstants.rmsNormEpsilon)
    #expect(!config.tieWordEmbeddings)
    #expect(config.gating == .perHead)
    #expect(config.gatingTypes == [LagunaGatingMode](
        repeating: .perHead,
        count: LagunaConstants.numHiddenLayers
    ))
    #expect(config.mlpOnlyLayers == [0])
    #expect(config.decoderSparseStep == 1)
    #expect(config.moeRoutedScalingFactor == LagunaConstants.moeRoutedScalingFactor)
    #expect(config.normTopkProb)
    #expect(!config.moeApplyRouterWeightOnInput)

    // Per-layer schedule: full attention (48 heads, YaRN partial rotary) at
    // 0, 4, 8, ..., sliding (64 heads, default RoPE) elsewhere; dense MLP
    // only at layer 0.
    for layerIndex in 0..<config.numHiddenLayers {
        let isFull = layerIndex % 4 == 0
        #expect(config.layerType(forLayer: layerIndex) == (isFull ? .full : .sliding))
        #expect(
            config.heads(forLayer: layerIndex)
                == (isFull
                    ? LagunaConstants.fullAttentionHeads
                    : LagunaConstants.slidingAttentionHeads)
        )
        #expect(config.isSparse(layer: layerIndex) == (layerIndex != 0))
    }
    #expect(config.fullRope.type == "yarn")
    #expect(config.fullRope.partialRotaryFactor == 0.5)
    #expect(config.fullRope.attentionFactor == 1.0)
    #expect(config.slidingRope.type == "default")
    #expect(config.slidingRope.partialRotaryFactor == 1.0)
    #expect(config.slidingRope.attentionFactor == nil)

    // Quantization: exact Poolside NVFP4 4-bit group-16 with no overrides.
    #expect(config.quantization.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(config.quantization.bits == LagunaConstants.quantizationBits)
    #expect(config.quantization.mode == LagunaConstants.quantizationMode)
    #expect(config.quantization.overrides.isEmpty)
}

@Test
func lagunaYaRNIgnoresSerializedAttentionFactorLikeUpstreamMLX() throws {
    let config = try loadLagunaConfig(pinnedLagunaConfigObject())
    let scaling = lagunaRopeScalingConfig(config.fullRope)

    #expect(config.fullRope.attentionFactor == 1.0)
    #expect(scaling["attention_factor"] == nil)
    #expect(scaling["mscale"] == nil)
    #expect(scaling["mscale_all_dim"] == nil)

    let upstreamDerivedMScale = 1.0 + 0.1 * log(config.fullRope.factor)
    #expect(abs(upstreamDerivedMScale - 1.3465735902799727) < 1e-12)
}

@Test
func lagunaRuntimeWiresOnlyExpertsToNVFP4WhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let model = LagunaRuntimeModel(try loadLagunaConfig(pinnedLagunaConfigObject()))
    let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())

    let denseGate = try #require(
        leaves["model.layers.0.mlp.gate_proj"] as? Linear
    )
    #expect(!(denseGate is QuantizedLinear))

    let router = try #require(leaves["model.layers.1.mlp.gate"])
    #expect(!(router is Quantized))

    let routedGate = try #require(
        leaves["model.layers.1.mlp.switch_mlp.gate_proj"] as? QuantizedSwitchLinear
    )
    #expect(routedGate.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(routedGate.bits == LagunaConstants.quantizationBits)
    #expect(routedGate.mode == .nvfp4)

    let sharedGate = try #require(
        leaves["model.layers.1.mlp.shared_expert.gate_proj"] as? QuantizedLinear
    )
    #expect(sharedGate.groupSize == LagunaConstants.quantizationGroupSize)
    #expect(sharedGate.bits == LagunaConstants.quantizationBits)
    #expect(sharedGate.mode == .nvfp4)

    let attention = try #require(
        leaves["model.layers.1.self_attn.q_proj"] as? Linear
    )
    #expect(!(attention is QuantizedLinear))
}

@Test
func lagunaConfigRejectsChangedArchitecture() throws {
    var object = pinnedLagunaConfigObject()
    object["num_hidden_layers"] = 12
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    object["tie_word_embeddings"] = true
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    object["num_experts"] = 128
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}

@Test
func lagunaConfigRejectsBehaviorChangingXSMutations() throws {
    func expectRejected(
        _ label: String,
        _ mutate: (inout [String: Any]) -> Void
    ) throws {
        var object = pinnedLagunaConfigObject()
        mutate(&object)
        #expect(throws: MLXFastError.self, Comment(rawValue: label)) {
            _ = try loadLagunaConfig(object)
        }
    }

    try expectRejected("48/64 head schedule") {
        var schedule = $0["num_attention_heads_per_layer"] as! [Int]
        schedule[1] = LagunaConstants.fullAttentionHeads
        $0["num_attention_heads_per_layer"] = schedule
    }
    try expectRejected("1-full:3-sliding attention schedule") {
        var schedule = $0["layer_types"] as! [String]
        schedule[1] = "full_attention"
        $0["layer_types"] = schedule
    }
    try expectRejected("dense layer zero only") {
        var schedule = $0["mlp_layer_types"] as! [String]
        schedule[1] = "dense"
        $0["mlp_layer_types"] = schedule
    }
    try expectRejected("mlp_only_layers") { $0["mlp_only_layers"] = [0, 2] }
    try expectRejected("decoder_sparse_step") { $0["decoder_sparse_step"] = 2 }
    try expectRejected("maximum position") { $0["max_position_embeddings"] = 131_072 }
    try expectRejected("RMS epsilon") { $0["rms_norm_eps"] = 1e-5 }
    try expectRejected("routed scaling") { $0["moe_routed_scaling_factor"] = 1.0 }
    try expectRejected("top-k normalization") { $0["norm_topk_prob"] = false }
    try expectRejected("router weight placement") {
        $0["moe_apply_router_weight_on_input"] = true
    }
    try expectRejected("per-layer gating") {
        var schedule = $0["gating_types"] as! [String]
        schedule[7] = "per_element"
        $0["gating_types"] = schedule
    }
    try expectRejected("serialized YaRN attention factor") {
        var ropes = $0["rope_parameters"] as! [String: Any]
        var full = ropes["full_attention"] as! [String: Any]
        full["attention_factor"] = 1.25
        ropes["full_attention"] = full
        $0["rope_parameters"] = ropes
    }
    try expectRejected("unexpected YaRN mscale override") {
        var ropes = $0["rope_parameters"] as! [String: Any]
        var full = ropes["full_attention"] as! [String: Any]
        full["mscale"] = 1.0
        ropes["full_attention"] = full
        $0["rope_parameters"] = ropes
    }
    try expectRejected("missing gating schedule") { $0.removeValue(forKey: "gating_types") }
}

@Test
func lagunaConfigRejectsAnyQuantizationOverride() throws {
    var object = pinnedLagunaConfigObject()
    var quantization = try #require(object["quantization"] as? [String: Any])
    quantization["model.layers.1.mlp.switch_mlp.gate_proj"] = [
        "group_size": 16,
        "bits": 4,
    ]
    object["quantization"] = quantization
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}

@Test
func lagunaConfigRequiresMatchingExplicitQuantizationBlocks() throws {
    var object = pinnedLagunaConfigObject()
    object.removeValue(forKey: "quantization_config")
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    var quantizationConfig = try #require(
        object["quantization_config"] as? [String: Any]
    )
    quantizationConfig["group_size"] = 32
    object["quantization_config"] = quantizationConfig
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }

    object = pinnedLagunaConfigObject()
    var quantization = try #require(object["quantization"] as? [String: Any])
    quantization.removeValue(forKey: "mode")
    object["quantization"] = quantization
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}

@Test
func lagunaConfigRejectsUnsupportedRopeType() throws {
    var object = pinnedLagunaConfigObject()
    var ropeParameters = try #require(object["rope_parameters"] as? [String: Any])
    ropeParameters["full_attention"] = [
        "rope_theta": 500_000.0,
        "rope_type": "llama3",
        "partial_rotary_factor": 0.5,
    ]
    object["rope_parameters"] = ropeParameters
    #expect(throws: MLXFastError.self) {
        _ = try loadLagunaConfig(object)
    }
}
