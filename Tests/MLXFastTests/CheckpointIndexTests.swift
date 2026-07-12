import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastTransform

@Test
func checkpointIndexToolsReturnsQwenShardsUniquelyAndSorted() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight":
                "model-00001-of-00003.safetensors",
            "language_model.model.layers.3.self_attn.q_proj.weight":
                "model-00001-of-00003.safetensors",
            "language_model.model.layers.17.mlp.down_proj.weight":
                "model-00002-of-00003.safetensors",
            "language_model.lm_head.weight":
                "model-00003-of-00003.safetensors",
        ]
    )

    let shards = try CheckpointIndexTools.safetensorShardNames(from: index.path)

    #expect(
        shards == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
    )
}

@Test
func checkpointIndexToolsRejectsUnsupportedShard() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "a": "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndexTools.safetensorShardNames(from: index.path)
    }
}

@Test
func checkpointIndexToolsRejectsUnsafeShardPath() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let index = root.appendingPathComponent("model.safetensors.index.json")
    try writeCheckpointIndex(
        index,
        weightMap: [
            "a": "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndexTools.safetensorShardNames(from: index.path)
    }
}

private func writeCheckpointIndex(_ path: URL, weightMap: [String: String]) throws {
    let data = try JSONSerialization.data(
        withJSONObject: ["weight_map": weightMap],
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
