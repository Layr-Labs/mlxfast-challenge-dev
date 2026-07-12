import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

@Test
func qwen35ConstantsPinArtifactIdentityAndDimensions() {
    #expect(MLXFastConstants.referenceModelName == "Qwen3.6-27B-4bit")
    #expect(
        MLXFastConstants.defaultReferencePath
            == "reference_weights/Qwen3.6-27B-4bit"
    )
    #expect(
        MLXFastConstants.defaultReferenceCachePath
            == ".cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/"
                + "snapshots/c000ac2c2057d94be3fa931000c31723aac53282"
    )
    #expect(MLXFastConstants.vocabSize == 248_320)
    #expect(MLXFastConstants.hiddenSize == 5_120)
    #expect(MLXFastConstants.intermediateSize == 17_408)
    #expect(MLXFastConstants.numHiddenLayers == 64)
    #expect(MLXFastConstants.attentionHeads == 24)
}

@Test
func qwen35ConfigLoadsPinnedCheckpointContract() throws {
    let config = try loadQwenConfigJSON(fullQwenConfigJSON())

    #expect(config.modelType == "qwen3_5_text")
    #expect(config.vocabSize == 248_320)
    #expect(config.hiddenSize == 5_120)
    #expect(config.intermediateSize == 17_408)
    #expect(config.numHiddenLayers == 64)
    #expect(config.numAttentionHeads == 24)
    #expect(config.numKeyValueHeads == 4)
    #expect(config.headDim == 256)
    #expect(config.linearNumValueHeads == 48)
    #expect(config.linearNumKeyHeads == 16)
    #expect(config.linearValueHeadDim == 128)
    #expect(config.linearKeyHeadDim == 128)
    #expect(config.linearConvKernelDim == 4)
    #expect(config.fullAttentionInterval == 4)
    #expect(config.hiddenActivation == "silu")
    #expect(config.rmsNormEps == 1e-6)
    #expect(config.attentionOutputGate)
    #expect(config.outputGateType == "swish")
    #expect(!config.tieWordEmbeddings)
    #expect(config.rope.theta == 10_000_000)
    #expect(config.rope.partialRotaryFactor == 0.25)
    #expect(config.rope.mropeInterleaved)
    #expect(config.rope.mropeSection == [11, 11, 10])
    #expect(config.quantizationGroupSize == 64)
    #expect(config.quantizationBits == 4)
    #expect(config.quantizationMode == "affine")
    #expect(config.mtpNumHiddenLayers == 1)
    #expect(!config.mtpUseDedicatedEmbeddings)
}

@Test
func qwen35ConfigUsesThreeLinearLayersThenFullAttention() throws {
    let config = try loadQwenConfigJSON(fullQwenConfigJSON())

    #expect(config.layerTypes == Qwen35Config.expectedLayerTypes)
    #expect(config.layerTypes.count == 64)
    for (index, layerType) in config.layerTypes.enumerated() {
        let expected: Qwen35LayerType = index % 4 == 3 ? .full : .linear
        #expect(layerType == expected)
    }
}

@Test
func qwen35ConfigTreatsMTPCountAsMetadataWithoutRequiringWeights() throws {
    // The fixture directory contains config.json only. Loading the frozen
    // contract must not infer a requirement for nonexistent `mtp.*` tensors.
    let config = try loadQwenConfigJSON(fullQwenConfigJSON())
    #expect(config.mtpNumHiddenLayers == 1)
    #expect(!config.mtpUseDedicatedEmbeddings)
}

@Test
func qwen35ConfigRejectsEmptyAndMissingRequiredFields() throws {
    #expect(throws: MLXFastError.self) {
        _ = try loadQwenConfigJSON("{}")
    }

    let requiredFields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_value_head_dim",
        "linear_key_head_dim",
        "linear_conv_kernel_dim",
        "full_attention_interval",
        "layer_types",
        "rms_norm_eps",
        "hidden_act",
        "max_position_embeddings",
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "output_gate_type",
        "tie_word_embeddings",
        "mamba_ssm_dtype",
        "dtype",
        "use_cache",
        "partial_rotary_factor",
        "rope_parameters",
        "quantization",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
    ]

    for field in requiredFields {
        let invalid = try mutateQwenConfigJSON {
            $0.removeValue(forKey: field)
        }
        #expect(throws: MLXFastError.self, "missing \(field)") {
            _ = try loadQwenConfigJSON(invalid)
        }
    }
}

@Test
func qwen35ConfigRejectsNullRequiredFields() throws {
    let requiredFields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_value_head_dim",
        "linear_key_head_dim",
        "linear_conv_kernel_dim",
        "full_attention_interval",
        "layer_types",
        "rms_norm_eps",
        "hidden_act",
        "max_position_embeddings",
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "output_gate_type",
        "tie_word_embeddings",
        "mamba_ssm_dtype",
        "dtype",
        "use_cache",
        "partial_rotary_factor",
        "rope_parameters",
        "quantization",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
    ]

    for field in requiredFields {
        let invalid = try mutateQwenConfigJSON {
            $0[field] = NSNull()
        }
        #expect(throws: MLXFastError.self, "null \(field)") {
            _ = try loadQwenConfigJSON(invalid)
        }
    }
}

@Test
func qwen35ConfigRejectsInvalidNestedObjectsAndFields() throws {
    for value: Any in ["not-an-object", [1, 2, 3], NSNull()] {
        let invalidRope = try mutateQwenConfigJSON {
            $0["rope_parameters"] = value
        }
        #expect(throws: MLXFastError.self) {
            _ = try loadQwenConfigJSON(invalidRope)
        }

        let invalidQuantization = try mutateQwenConfigJSON {
            $0["quantization"] = value
        }
        #expect(throws: MLXFastError.self) {
            _ = try loadQwenConfigJSON(invalidQuantization)
        }
    }

    for field in [
        "rope_theta",
        "rope_type",
        "partial_rotary_factor",
        "mrope_interleaved",
        "mrope_section",
    ] {
        for remove in [true, false] {
            let invalid = try mutateQwenConfigJSON {
                var rope = $0["rope_parameters"] as! [String: Any]
                if remove {
                    rope.removeValue(forKey: field)
                } else {
                    rope[field] = NSNull()
                }
                $0["rope_parameters"] = rope
            }
            #expect(throws: MLXFastError.self, "\(remove ? "missing" : "null") \(field)") {
                _ = try loadQwenConfigJSON(invalid)
            }
        }
    }

    for field in ["group_size", "bits", "mode"] {
        for remove in [true, false] {
            let invalid = try mutateQwenConfigJSON {
                var quantization = $0["quantization"] as! [String: Any]
                if remove {
                    quantization.removeValue(forKey: field)
                } else {
                    quantization[field] = NSNull()
                }
                $0["quantization"] = quantization
            }
            #expect(throws: MLXFastError.self, "\(remove ? "missing" : "null") \(field)") {
                _ = try loadQwenConfigJSON(invalid)
            }
        }
    }
}

@Test
func qwen35ConfigAcceptsEitherMatchingQuantizationForm() throws {
    let quantizationConfigOnly = try mutateQwenConfigJSON {
        $0["quantization_config"] = $0.removeValue(forKey: "quantization")
    }
    #expect(try loadQwenConfigJSON(quantizationConfigOnly).quantizationBits == 4)

    let matchingForms = try mutateQwenConfigJSON {
        $0["quantization_config"] = $0["quantization"]
    }
    #expect(try loadQwenConfigJSON(matchingForms).quantizationGroupSize == 64)

    let mismatchedForms = try mutateQwenConfigJSON {
        var alternate = $0["quantization"] as! [String: Any]
        alternate["bits"] = 8
        $0["quantization_config"] = alternate
    }
    #expect(throws: MLXFastError.self) {
        _ = try loadQwenConfigJSON(mismatchedForms)
    }
}

@Test
func qwen35ConfigRejectsFrozenInvariantChanges() throws {
    let valid = fullQwenConfigJSON()
    let replacements: [(String, String)] = [
        (#""model_type": "qwen3_5_text""#, #""model_type": "qwen3_6_text""#),
        (#""vocab_size": 248320"#, #""vocab_size": 248321"#),
        (#""hidden_size": 5120"#, #""hidden_size": 4096"#),
        (#""intermediate_size": 17408"#, #""intermediate_size": 16384"#),
        (#""num_hidden_layers": 64"#, #""num_hidden_layers": 63"#),
        (#""num_attention_heads": 24"#, #""num_attention_heads": 32"#),
        (#""num_key_value_heads": 4"#, #""num_key_value_heads": 8"#),
        (#""linear_num_value_heads": 48"#, #""linear_num_value_heads": 32"#),
        (#""linear_num_key_heads": 16"#, #""linear_num_key_heads": 8"#),
        (#""linear_value_head_dim": 128"#, #""linear_value_head_dim": 64"#),
        (#""linear_key_head_dim": 128"#, #""linear_key_head_dim": 64"#),
        (#""linear_conv_kernel_dim": 4"#, #""linear_conv_kernel_dim": 3"#),
        (#""full_attention_interval": 4"#, #""full_attention_interval": 8"#),
        (#""hidden_act": "silu""#, #""hidden_act": "gelu""#),
        (#""attn_output_gate": true"#, #""attn_output_gate": false"#),
        (#""output_gate_type": "swish""#, #""output_gate_type": "sigmoid""#),
        (#""tie_word_embeddings": false"#, #""tie_word_embeddings": true"#),
        (#""rope_theta": 10000000"#, #""rope_theta": 1000000"#),
        (#""mrope_section": [11, 11, 10]"#, #""mrope_section": [16, 16]"#),
        (#""mode": "affine""#, #""mode": "symmetric""#),
        (#""mtp_num_hidden_layers": 1"#, #""mtp_num_hidden_layers": 0"#),
    ]

    for (expected, replacement) in replacements {
        let invalid = valid.replacingOccurrences(of: expected, with: replacement)
        #expect(invalid != valid, "missing fixture field \(expected)")
        #expect(throws: MLXFastError.self, "replacement \(replacement)") {
            _ = try loadQwenConfigJSON(invalid)
        }
    }
}

@Test
func qwen35ConfigRejectsWrongLayerPatternAndCount() throws {
    let allLinear = fullQwenConfigJSON().replacingOccurrences(
        of: #""full_attention""#,
        with: #""linear_attention""#
    )
    #expect(throws: MLXFastError.self) {
        _ = try loadQwenConfigJSON(allLinear)
    }

    let oneLayer = fullQwenConfigJSON().replacingOccurrences(
        of: qwenLayerTypesJSON(),
        with: #"["linear_attention"]"#
    )
    #expect(throws: MLXFastError.self) {
        _ = try loadQwenConfigJSON(oneLayer)
    }
}

@Test
func qwen35ConfigRejectsMismatchedPartialRotaryFields() throws {
    let valid = fullQwenConfigJSON()
    let range = try #require(
        valid.range(of: #""partial_rotary_factor": 0.25,"#)
    )
    let invalid = valid.replacingCharacters(
        in: range,
        with: #""partial_rotary_factor": 0.5,"#
    )

    #expect(throws: MLXFastError.self) {
        _ = try loadQwenConfigJSON(invalid)
    }
}

@Test
func qwen35ConfigRejectsUnsafeNumericRepresentations() throws {
    let valid = fullQwenConfigJSON()
    let replacements: [(String, String)] = [
        (#""num_key_value_heads": 4"#, #""num_key_value_heads": 1.5"#),
        (
            #""num_key_value_heads": 4"#,
            #""num_key_value_heads": 9223372036854775808"#
        ),
        (#""mrope_section": [11, 11, 10]"#, #""mrope_section": [11, 11, 10.5]"#),
        (#""attn_output_gate": true"#, #""attn_output_gate": 1"#),
    ]

    for (expected, replacement) in replacements {
        let invalid = valid.replacingOccurrences(of: expected, with: replacement)
        #expect(throws: MLXFastError.self, "replacement \(replacement)") {
            _ = try loadQwenConfigJSON(invalid)
        }
    }
}

@Test
func qwen35ConfigRejectsUnsafeLayerCountBeforeDefaultAllocation() {
    for layerCount in ["-1", "1000000000"] {
        let json = """
        {
          "num_hidden_layers": \(layerCount),
          "vocab_size": \(MLXFastConstants.vocabSize),
          "hidden_size": \(MLXFastConstants.hiddenSize),
          "intermediate_size": \(MLXFastConstants.intermediateSize),
          "num_attention_heads": \(MLXFastConstants.attentionHeads)
        }
        """
        #expect(throws: MLXFastError.self) {
            _ = try loadQwenConfigJSON(json)
        }
    }
}

@Test
func qwen35ConfigRejectsNonFiniteNumbers() throws {
    let invalid = fullQwenConfigJSON().replacingOccurrences(
        of: #""rms_norm_eps": 1e-6"#,
        with: #""rms_norm_eps": 1e400"#
    )

    #expect(throws: Error.self) {
        _ = try loadQwenConfigJSON(invalid)
    }
}

private func fullQwenConfigJSON() -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "vocab_size": 248320,
      "hidden_size": 5120,
      "intermediate_size": 17408,
      "num_hidden_layers": 64,
      "num_attention_heads": 24,
      "num_key_value_heads": 4,
      "head_dim": 256,
      "linear_num_value_heads": 48,
      "linear_num_key_heads": 16,
      "linear_value_head_dim": 128,
      "linear_key_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "full_attention_interval": 4,
      "layer_types": \(qwenLayerTypesJSON()),
      "rms_norm_eps": 1e-6,
      "hidden_act": "silu",
      "max_position_embeddings": 262144,
      "attention_bias": false,
      "attention_dropout": 0.0,
      "attn_output_gate": true,
      "output_gate_type": "swish",
      "tie_word_embeddings": false,
      "mamba_ssm_dtype": "float32",
      "dtype": "bfloat16",
      "use_cache": true,
      "partial_rotary_factor": 0.25,
      "rope_parameters": {
        "rope_theta": 10000000,
        "rope_type": "default",
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": true,
        "mrope_section": [11, 11, 10]
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false
    }
    """
}

private func qwenLayerTypesJSON() -> String {
    "[" + (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 3 ? #""full_attention""# : #""linear_attention""#
    }.joined(separator: ",") + "]"
}

private func loadQwenConfigJSON(_ json: String) throws -> Qwen35Config {
    let root = try qwenConfigTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try json.write(
        to: root.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    return try Qwen35Config.load(from: root.path)
}

private func mutateQwenConfigJSON(
    _ mutation: (inout [String: Any]) -> Void
) throws -> String {
    let data = try #require(fullQwenConfigJSON().data(using: .utf8))
    var root = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    mutation(&root)
    let output = try JSONSerialization.data(
        withJSONObject: root,
        options: [.sortedKeys, .withoutEscapingSlashes]
    )
    return try #require(String(data: output, encoding: .utf8))
}

private func qwenConfigTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
