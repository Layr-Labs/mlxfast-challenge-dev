import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastDeepSeek
@testable import MLXFastTransform

@Test
func transformCopiesDenseTensorsAndWritesExpertManifest() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)

    try #"{"num_hidden_layers":43}"#.write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let denseName = "model.layers.0.self_attn.q_proj.weight"
    let expertName = "model.layers.0.ffn.switch_mlp.gate_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: denseName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: expertName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try """
    {
      "metadata": {"total_size": 7},
      "weight_map": {
        "\(denseName)": "\(shardName)",
        "\(expertName)": "\(shardName)"
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

    #expect(report.denseTensorCount == 1)
    #expect(report.expertTensorCount == 1)
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("config.json").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("tokenizer.json").path))

    let outputShard = output.appendingPathComponent(shardName)
    let outputHeader = try Safetensors.readHeader(outputShard)
    #expect(outputHeader.tensors.keys.sorted() == [denseName])
    #expect(try tensorBytes(outputShard, header: outputHeader, name: denseName) == Data([1, 2, 3, 4]))

    let strippedIndexData = try Data(
        contentsOf: output.appendingPathComponent("model.safetensors.index.json")
    )
    let strippedIndex = try JSONSerialization.jsonObject(with: strippedIndexData) as? [String: Any]
    let weightMap = try #require(strippedIndex?["weight_map"] as? [String: String])
    #expect(weightMap == [denseName: shardName])

    let manifestData = try Data(contentsOf: output.appendingPathComponent("experts/manifest.json"))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let records = try #require(manifest?["expert_tensors"] as? [[String: Any]])
    #expect(records.count == 1)
    #expect(records[0]["name"] as? String == expertName)
    #expect(records[0]["shard"] as? String == shardName)
    #expect(records[0]["byte_length"] as? Int == 3)

    let bank = try ExpertSlotBank(
        manifestPath: output.appendingPathComponent("experts/manifest.json").path,
        capacity: 1
    )
    #expect(try bank.tensorBytes(named: expertName) == Data([9, 8, 7]))
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
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
