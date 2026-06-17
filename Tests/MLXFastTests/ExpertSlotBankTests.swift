import Foundation
import Testing
@testable import MLXFastDeepSeek

@Test
func expertSlotBankReadsExactByteRanges() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = root.appendingPathComponent("weights/experts", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let shard = reference.appendingPathComponent("model-00001.safetensors")
    try Data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).write(to: shard)

    let firstName = "model.layers.0.ffn.switch_mlp.7.gate_proj.weight"
    let secondName = "model.layers.0.ffn.switch_mlp.8.up_proj.weight"
    try manifestJSON(
        referencePath: reference.path,
        records: [
            record(name: firstName, shard: shard.lastPathComponent, offset: 2, length: 4),
            record(name: secondName, shard: shard.lastPathComponent, offset: 6, length: 3),
        ]
    ).write(
        to: experts.appendingPathComponent("manifest.json"),
        atomically: true,
        encoding: .utf8
    )

    let bank = try ExpertSlotBank(
        manifestPath: experts.appendingPathComponent("manifest.json").path,
        capacity: 2
    )

    #expect(try bank.tensorBytes(named: firstName) == Data([2, 3, 4, 5]))
    #expect(try bank.tensorBytes(named: secondName) == Data([6, 7, 8]))
    #expect(bank.record(named: firstName)?.layerIndex == 0)
    #expect(bank.record(named: firstName)?.expertIndex == 7)
    #expect(bank.record(named: firstName)?.projection == "gate_proj")

    let tensor = try bank.materializedTensor(named: firstName)
    #expect(tensor.name == firstName)
    #expect(tensor.dtype == .u8)
    #expect(tensor.shape == [4])
    #expect(try tensor.uint8Values() == [2, 3, 4, 5])
}

@Test
func expertSlotBankEvictsLeastRecentlyUsedTensor() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = root.appendingPathComponent("weights/experts", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let shard = reference.appendingPathComponent("model-00001.safetensors")
    try Data([10, 11, 12, 13, 14, 15]).write(to: shard)

    let a = "model.layers.0.ffn.switch_mlp.1.gate_proj.weight"
    let b = "model.layers.0.ffn.switch_mlp.2.gate_proj.weight"
    let c = "model.layers.0.ffn.switch_mlp.3.gate_proj.weight"
    try manifestJSON(
        referencePath: reference.path,
        records: [
            record(name: a, shard: shard.lastPathComponent, offset: 0, length: 2),
            record(name: b, shard: shard.lastPathComponent, offset: 2, length: 2),
            record(name: c, shard: shard.lastPathComponent, offset: 4, length: 2),
        ]
    ).write(
        to: experts.appendingPathComponent("manifest.json"),
        atomically: true,
        encoding: .utf8
    )

    let bank = try ExpertSlotBank(
        manifestPath: experts.appendingPathComponent("manifest.json").path,
        capacity: 2
    )

    _ = try bank.tensorBytes(named: a)
    _ = try bank.tensorBytes(named: b)
    #expect(bank.cachedTensorNames == [a, b])

    _ = try bank.tensorBytes(named: a)
    #expect(bank.cachedTensorNames == [b, a])

    _ = try bank.tensorBytes(named: c)
    #expect(bank.cachedTensorNames == [a, c])
}

@Test
func expertSlotBankRecordsStreamingMetrics() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = root.appendingPathComponent("weights/experts", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let shard = reference.appendingPathComponent("model-00001.safetensors")
    try Data([21, 22, 23, 24]).write(to: shard)

    let tensorName = "model.layers.0.ffn.switch_mlp.1.gate_proj.weight"
    try manifestJSON(
        referencePath: reference.path,
        records: [
            record(name: tensorName, shard: shard.lastPathComponent, offset: 1, length: 2),
        ]
    ).write(
        to: experts.appendingPathComponent("manifest.json"),
        atomically: true,
        encoding: .utf8
    )

    let metrics = ExpertStreamingMetrics()
    let bank = try ExpertSlotBank(
        manifestPath: experts.appendingPathComponent("manifest.json").path,
        capacity: 1,
        metrics: metrics
    )

    #expect(try bank.tensorBytes(named: tensorName) == Data([22, 23]))
    #expect(try bank.tensorBytes(named: tensorName) == Data([22, 23]))

    let snapshot = metrics.snapshot()
    #expect(snapshot.cacheMisses == 1)
    #expect(snapshot.cacheHits == 1)
    #expect(snapshot.bytesRead == 2)
    #expect(snapshot.readNanoseconds > 0)
    #expect(snapshot.totalLookups == 2)
    #expect(snapshot.hitRate == 0.5)
}

@Test
func expertStreamingConfigParsesEnvironment() {
    let config = ExpertStreamingConfig.fromEnvironment([
        "MLXFAST_EXPERT_CACHE_EXPERTS": "7",
        "MLXFAST_EXPERT_STREAM_METRICS": "yes",
    ])

    #expect(config.mode == .directNVMe)
    #expect(config.tensorCacheCapacity == 21)
    #expect(config.recordsMetrics)
}

private func manifestJSON(referencePath: String, records: [String]) -> String {
    """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(referencePath)",
      "expert_tensors": [
        \(records.joined(separator: ",\n        "))
      ]
    }
    """
}

private func record(name: String, shard: String, offset: Int, length: Int) -> String {
    """
    {
      "name": "\(name)",
      "shard": "\(shard)",
      "dtype": "U8",
      "shape": [\(length)],
      "data_offsets": [0, \(length)],
      "byte_offset": \(offset),
      "byte_length": \(length)
    }
    """
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
