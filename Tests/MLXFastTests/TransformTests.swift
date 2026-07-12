import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastHarness
@testable import MLXFastTransform

@Test
func transformSelectsTextTowerTensorsAndDropsVisionTensors() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)

    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )
    try "{% for message in messages %}{{ message.content }}{% endfor %}".write(
        to: reference.appendingPathComponent("chat_template.jinja"),
        atomically: true,
        encoding: .utf8
    )

    let linearAttentionName =
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
    let fullAttentionName =
        "language_model.model.layers.3.self_attn.q_proj.weight"
    let languageModelHeadName = "language_model.lm_head.weight"
    let visionName = "vision_tower.blocks.0.attn.qkv.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(
                name: linearAttentionName,
                dtype: "U8",
                shape: [4],
                data: Data([1, 2, 3, 4])
            ),
            TensorFixture(
                name: fullAttentionName,
                dtype: "U8",
                shape: [3],
                data: Data([5, 6, 7])
            ),
            TensorFixture(
                name: languageModelHeadName,
                dtype: "U8",
                shape: [2],
                data: Data([8, 9])
            ),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try """
    {
      "metadata": {"total_size": 12},
      "weight_map": {
        "\(linearAttentionName)": "\(shardName)",
        "\(fullAttentionName)": "\(shardName)",
        "\(languageModelHeadName)": "\(shardName)",
        "\(visionName)": "\(shardName)"
      }
    }
    """.write(
        to: reference.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 3)
    #expect(report.denseShardCount == 1)
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("config.json").path))
    try validateRuntimeWorkerPinnedConfigurationData(
        Data(contentsOf: output.appendingPathComponent("config.json"))
    )
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("tokenizer.json").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("chat_template.jinja").path))
    #expect(!FileManager.default.fileExists(atPath: output.appendingPathComponent("experts").path))

    let outputShard = output.appendingPathComponent(shardName)
    let outputHeader = try Safetensors.readHeader(outputShard)
    #expect(
        Set(outputHeader.tensors.keys)
            == Set([linearAttentionName, fullAttentionName, languageModelHeadName])
    )
    #expect(
        try tensorBytes(
            outputShard,
            header: outputHeader,
            name: linearAttentionName
        ) == Data([1, 2, 3, 4])
    )
    #expect(
        try tensorBytes(
            outputShard,
            header: outputHeader,
            name: fullAttentionName
        ) == Data([5, 6, 7])
    )
    #expect(
        try tensorBytes(
            outputShard,
            header: outputHeader,
            name: languageModelHeadName
        ) == Data([8, 9])
    )

    let strippedIndexData = try Data(
        contentsOf: output.appendingPathComponent("model.safetensors.index.json")
    )
    let strippedIndex = try JSONSerialization.jsonObject(with: strippedIndexData) as? [String: Any]
    let weightMap = try #require(strippedIndex?["weight_map"] as? [String: String])
    #expect(
        weightMap == [
            linearAttentionName: shardName,
            fullAttentionName: shardName,
            languageModelHeadName: shardName,
        ]
    )
    let metadata = try #require(strippedIndex?["metadata"] as? [String: Any])
    #expect(metadata["total_size"] as? Int == 9)
}

@Test
func transformWritesFlattenedRuntimeConfigFromTextConfig() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.embed_tokens.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [textName: shardName]
    )

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    #expect(config?["model_type"] as? String == "qwen3_5_text")
    #expect(config?["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(config?["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(config?["text_config"] == nil)
    let quantization = try #require(config?["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
    #expect(try Qwen35Config.load(from: output.path).layerTypes == Qwen35Config.expectedLayerTypes)
}

@Test
func transformRejectsConflictingQuantizationSchemas() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")
    var config = try #require(
        JSONSerialization.jsonObject(
            with: Data(referenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    config["quantization_config"] = [
        "group_size": 128,
        "bits": 4,
        "mode": "affine",
    ]
    try JSONSerialization.data(
        withJSONObject: config,
        options: [.sortedKeys]
    ).write(to: configPath)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.makeRuntimeConfigData(
            sourceConfigPath: configPath
        )
    }

    config["quantization_config"] = config["quantization"]
    try JSONSerialization.data(
        withJSONObject: config,
        options: [.sortedKeys]
    ).write(to: configPath)
    let runtime = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigPath: configPath
    )
    let object = try #require(
        JSONSerialization.jsonObject(with: runtime) as? [String: Any]
    )
    #expect(object["quantization"] != nil)
    #expect(object["quantization_config"] == nil)
}

@Test
func transformRuntimeConfigCaptureDoesNotRereadChangedSource() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")
    try referenceConfigJSON().write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    let captured = try SwiftTransform.makeRuntimeConfigData(sourceConfigPath: configPath)
    try #"{"text_config":{"num_hidden_layers":1}}"#.write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    let object = try JSONSerialization.jsonObject(with: captured) as? [String: Any]
    #expect(object?["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
}

@Test
func transformRejectsSourceMetadataMutationsBeforePublishingOutput() throws {
    enum Mutation: CaseIterable {
        case config
        case index
        case tokenizer
    }

    for mutation in Mutation.allCases {
        let expectedError: String
        switch mutation {
        case .config:
            expectedError = "reference config changed while transform was running"
        case .index:
            expectedError = "checkpoint index changed while transform was running"
        case .tokenizer:
            expectedError = "reference tokenizer metadata changed while transform was running"
        }
        let fixture = try writeTransformFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.createDirectory(
            at: fixture.output,
            withIntermediateDirectories: true
        )
        let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
        try "preserve \(mutation)".write(to: sentinel, atomically: true, encoding: .utf8)

        var rejection: MLXFastError?
        do {
            _ = try SwiftTransform.run(
                TransformOptions(
                    referencePath: fixture.reference.path,
                    outputPath: fixture.output.path
                ),
                beforeSourceRevalidation: {
                    switch mutation {
                    case .config:
                        try #"{"text_config":{"num_hidden_layers":1}}"#.write(
                            to: fixture.reference.appendingPathComponent("config.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    case .index:
                        try writeCheckpointIndex(
                            fixture.reference.appendingPathComponent(
                                "model.safetensors.index.json"
                            ),
                            weightMap: [
                                "language_model.model.layers.0.linear_attn.in_proj_qkv.weight":
                                    "model-00001-of-00001.safetensors",
                            ],
                            metadata: ["generation": 2]
                        )
                    case .tokenizer:
                        try #"{"tokenizer":"changed"}"#.write(
                            to: fixture.reference.appendingPathComponent("tokenizer.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    }
                }
            )
        } catch let error as MLXFastError {
            rejection = error
        } catch {
            throw error
        }
        #expect(rejection?.description == expectedError, "mutation: \(mutation)")

        #expect(
            try String(contentsOf: sentinel, encoding: .utf8) == "preserve \(mutation)",
            "mutation: \(mutation)"
        )
        #expect(
            try transformStagingDirectories(nextTo: fixture.output).isEmpty,
            "mutation: \(mutation)"
        )
    }
}

@Test
func transformRejectsSymlinkedTokenizerMetadataAndPreservesOutput() throws {
    let fixture = try writeTransformFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let tokenizer = fixture.reference.appendingPathComponent("tokenizer.json")
    let tokenizerTarget = fixture.root.appendingPathComponent("tokenizer-target.json")
    try #"{"tokenizer":"external"}"#.write(
        to: tokenizerTarget,
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.removeItem(at: tokenizer)
    try FileManager.default.createSymbolicLink(at: tokenizer, withDestinationURL: tokenizerTarget)

    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("reference metadata is not a regular file") == true)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    #expect(try transformStagingDirectories(nextTo: fixture.output).isEmpty)
}

@Test
func transformRejectsReferenceAsOutputWithoutMutatingCheckpoint() throws {
    let fixture = try writeTransformFixture()
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.reference.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.reference.appendingPathComponent("model-00001-of-00001.safetensors").path
        )
    )
}

@Test
func transformRejectsSymlinkAliasOfReferenceAsOutput() throws {
    let fixture = try writeTransformFixture()
    let outputAlias = fixture.root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createSymbolicLink(
        at: outputAlias,
        withDestinationURL: fixture.reference
    )
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputAlias.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
}

@Test
func transformRejectsOutputThatContainsWorkingDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let workingDirectory = root.appendingPathComponent("workspace/project")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: root,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsOutputInsideReferenceDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let output = reference.appendingPathComponent("weights")
    let workingDirectory = root.appendingPathComponent("workspace")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: output,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsExistingRegularFileOutputWithoutReplacingIt() throws {
    let fixture = try writeTransformFixture()
    let outputFile = fixture.root.appendingPathComponent("existing-output")
    try "preserve me".write(to: outputFile, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputFile.path
            )
        )
    }

    #expect(try String(contentsOf: outputFile, encoding: .utf8) == "preserve me")
}

@Test
func transformAtomicallyReplacesExistingOutputAndRemovesStaleFiles() throws {
    let fixture = try writeTransformFixture()
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let staleFile = fixture.output.appendingPathComponent("stale.txt")
    try "stale".write(to: staleFile, atomically: true, encoding: .utf8)

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(!FileManager.default.fileExists(atPath: staleFile.path))
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("model.safetensors.index.json").path
        )
    )
}

@Test
func transformFailurePreservesExistingOutputAndRemovesStagingDirectory() throws {
    let fixture = try writeTransformFixture()
    try #"{"vision_config":{}}"#.write(
        to: fixture.reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "original".write(to: sentinel, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    }

    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "original")
    let siblings = try FileManager.default.contentsOfDirectory(
        at: fixture.output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    )
    #expect(
        !siblings.contains {
            $0.lastPathComponent.hasPrefix(".weights.mlxfast-transform-")
        }
    )
}

@Test
func checkpointIndexBuildRejectsDuplicateTensorNamesAcrossShards() throws {
    let root = try temporaryDirectory()
    let tensorName = "language_model.model.embed_tokens.weight"
    try writeSafetensors(
        root.appendingPathComponent("model-00001-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeSafetensors(
        root.appendingPathComponent("model-00002-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([2]))]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndex.buildFromSafetensors(in: root)
    }
}

@Test
func transformRejectsCheckpointWithoutTextTowerTensorsBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let visionName = "vision_tower.blocks.0.attn.qkv.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: visionName, dtype: "U8", shape: [2], data: Data([1, 2]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [visionName: shardName]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformVerifierAcceptsFreshSubmittedTransformOutputAndIgnoresLocalCacheMarkers() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "cache\n".write(
        to: fixture.output.appendingPathComponent(".benchmark-source.sha256"),
        atomically: true,
        encoding: .utf8
    )
    FileManager.default.createFile(
        atPath: fixture.output.appendingPathComponent(".gitkeep").path,
        contents: Data()
    )

    let report = try TransformVerifier.verify(
        TransformVerificationOptions(
            referencePath: fixture.reference.path,
            weightsPath: fixture.output.path,
            temporaryParentPath: fixture.root.path
        )
    )

    #expect(report.referencePath == fixture.reference.path)
    #expect(report.weightsPath == fixture.output.path)
    #expect(report.fileCount > 0)
    #expect(report.byteCount > 0)
    #expect(report.maxByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.sha256.count == 64)
    #expect(report.deterministic)
}

@Test
func transformVerifierRejectsOutputThatDiffersFromFreshTransformRun() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "changed".write(
        to: fixture.output.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsStaleExtraGeneratedFile() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "extra".write(
        to: fixture.output.appendingPathComponent("extra.txt"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsOutputAboveConfiguredByteLimit() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path,
                maxByteCount: 1
            )
        )
    }
}

@Test
func transformRejectsUnsupportedIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight":
                "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsUnsafeIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight":
                "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsIndexTensorMissingFromShardHeaderBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(
                name: "language_model.model.layers.3.self_attn.k_proj.weight",
                dtype: "U8",
                shape: [2],
                data: Data([1, 2])
            ),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "language_model.model.layers.3.self_attn.q_proj.weight": shardName,
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformAcceptsSparseShardLargerThanInt32() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.lm_head.weight"
    let visionName = "vision_tower.blocks.0.attn.qkv.weight"
    let shardName = "model-00001-of-00001.safetensors"
    let shard = reference.appendingPathComponent(shardName)
    try writeSafetensors(
        shard,
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [1], data: Data([8])),
        ]
    )
    try truncateFile(shard, toByteCount: Int64(Int32.max) + 1024)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private struct TransformFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
}

private func writeTransformFixture() throws -> TransformFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try referenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
    let visionName = "vision_tower.blocks.0.attn.qkv.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    return TransformFixturePaths(root: root, reference: reference, output: output)
}

private func referenceConfigJSON() -> String {
    """
    {
      "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
        "vocab_size": \(MLXFastConstants.vocabSize),
        "hidden_size": \(MLXFastConstants.hiddenSize),
        "intermediate_size": \(MLXFastConstants.intermediateSize),
        "num_attention_heads": \(MLXFastConstants.attentionHeads),
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "layer_types": \(transformQwenLayerTypesJSON()),
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 262144,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "output_gate_type": "swish",
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "initializer_range": 0.02,
        "pad_token_id": null,
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
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """
}

private func transformQwenLayerTypesJSON() -> String {
    "[" + (0..<MLXFastConstants.numHiddenLayers).map {
        $0 % 4 == 3 ? #""full_attention""# : #""linear_attention""#
    }.joined(separator: ",") + "]"
}

private func writeCheckpointIndex(
    _ path: URL,
    weightMap: [String: String],
    metadata: [String: Any]? = nil
) throws {
    var object: [String: Any] = ["weight_map": weightMap]
    object["metadata"] = metadata
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func transformStagingDirectories(nextTo output: URL) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(
        at: output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    ).filter {
        $0.lastPathComponent.hasPrefix(".\(output.lastPathComponent).mlxfast-transform-")
    }
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func truncateFile(_ path: URL, toByteCount byteCount: Int64) throws {
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func tensorBytes(_ path: URL, header: SafetensorsHeader, name: String) throws -> Data {
    let info = try #require(header.tensors[name])
    let data = try Data(contentsOf: path)
    let start = Int(header.dataBaseOffset) + info.dataStart
    return data.subdata(in: start..<(start + info.byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
