import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastDeepSeek
@testable import MLXFastTransform

@Test
func denseTensorStoreLoadsMetadataAndReadsBytes() throws {
    let root = try temporaryDirectory()
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    let shardName = "model-00001.safetensors"
    let first = "model.embed_tokens.weight"
    let second = "model.layers.0.self_attn.q_proj.weight"
    try writeSafetensors(
        weights.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: first, dtype: "U8", shape: [2], data: Data([1, 2])),
            TensorFixture(name: second, dtype: "U8", shape: [3], data: Data([3, 4, 5])),
        ]
    )
    try """
    {
      "weight_map": {
        "\(first)": "\(shardName)",
        "\(second)": "\(shardName)"
      }
    }
    """.write(
        to: weights.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let store = try DenseTensorStore(weightsPath: weights.path)

    #expect(store.tensorNames == [first, second])
    #expect(store.record(named: second)?.dtype == "U8")
    #expect(store.record(named: second)?.shape == [3])
    #expect(try store.tensorBytes(named: first) == Data([1, 2]))
    #expect(try store.tensorBytes(named: second) == Data([3, 4, 5]))

    let tensor = try store.materializedTensor(named: second)
    #expect(tensor.name == second)
    #expect(tensor.dtype == .u8)
    #expect(tensor.shape == [3])
    #expect(try tensor.uint8Values() == [3, 4, 5])
}

@Test
func denseTensorStoreRejectsByteLengthMismatch() throws {
    let root = try temporaryDirectory()
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    let shardName = "model-00001.safetensors"
    let tensorName = "model.embed_tokens.weight"
    try writeSafetensors(
        weights.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: tensorName, dtype: "U8", shape: [2], data: Data([1])),
        ]
    )
    try """
    {
      "weight_map": {
        "\(tensorName)": "\(shardName)"
      }
    }
    """.write(
        to: weights.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let store = try DenseTensorStore(weightsPath: weights.path)
    #expect(throws: MLXFastError.self) {
        try store.validateReadableByteRanges()
    }
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

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
