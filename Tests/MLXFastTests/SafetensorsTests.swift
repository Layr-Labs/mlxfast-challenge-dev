import Foundation
import Testing
@testable import MLXFastCore

@Test
func safetensorsCopySubsetPreservesBytesForUnsortedTensorNames() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
            TensorFixture(name: "b.weight", dtype: "U8", shape: [3], data: Data([3, 4, 5])),
            TensorFixture(name: "c.weight", dtype: "U8", shape: [2], data: Data([6, 7])),
        ]
    )

    let copied = try Safetensors.copySubset(
        from: source,
        to: destination,
        tensorNames: ["c.weight", "a.weight"]
    )

    #expect(copied == 2)
    let header = try Safetensors.readHeader(destination)
    #expect(header.tensors.keys.sorted() == ["a.weight", "c.weight"])
    #expect(try tensorBytes(destination, header: header, name: "a.weight") == Data([1, 2]))
    #expect(try tensorBytes(destination, header: header, name: "c.weight") == Data([6, 7]))
}

@Test
func safetensorsCopySubsetRejectsMissingRequestedTensor() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: destination,
            tensorNames: ["missing.weight"]
        )
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
