import Foundation
import MLXFastCore

struct GeneratedCombinedGateUpReport {
    let weightMap: [String: String]
    let replacedKeys: Set<String>
    let tensorByteCount: Int

    var tensorCount: Int { weightMap.count }
    var shardCount: Int { weightMap.isEmpty ? 0 : 1 }
}

/// Rewrites the production gate/up affine tensors into one row-major physical
/// allocation per payload kind. Safetensors stores each tensor contiguously, so
/// writing gate bytes followed by up bytes gives view-only runtime row slices.
enum CombinedGateUpCoding {
    static let shardName = "mlxfast-combined-gate-up-prefill.safetensors"
    static let layerCount = 60
    static let rows = 21_504

    private struct Tensor {
        let name: String
        let sourceNames: [String]
        let dtype: TensorDType
        let shape: [Int]
        let byteCount: Int
    }

    static func writeSidecar(
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedCombinedGateUpReport {
        let first = "language_model.model.layers.0.mlp.gate_proj.weight"
        guard selectedKeys.contains(first) else {
            return GeneratedCombinedGateUpReport(weightMap: [:], replacedKeys: [], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput("checkpoint already contains reserved generated shard \(shardName)")
        }

        var tensors: [Tensor] = []
        var replaced: Set<String> = []
        for layer in 0..<layerCount {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for (suffix, dtype, sourceShape, combinedShape) in [
                ("weight", TensorDType.u32, [rows, 672], [rows * 2, 672]),
                ("scales", TensorDType.bf16, [rows, 84], [rows * 2, 84]),
                ("biases", TensorDType.bf16, [rows, 84], [rows * 2, 84]),
            ] {
                let sources = ["\(prefix).gate_proj.\(suffix)", "\(prefix).up_proj.\(suffix)"]
                for source in sources {
                    guard selectedKeys.contains(source),
                          let shard = index.weightMap[source],
                          let info = sourceHeaders[shard]?.tensors[source],
                          info.dtype == dtype.rawValue,
                          info.shape == sourceShape
                    else {
                        throw MLXFastError.invalidInput("combined gate/up source has invalid dtype or shape: \(source)")
                    }
                    replaced.insert(source)
                }
                let bytes = combinedShape.reduce(1, *) * dtype.byteWidth
                tensors.append(Tensor(
                    name: "\(prefix).gate_up_prefill.\(suffix)",
                    sourceNames: sources,
                    dtype: dtype,
                    shape: combinedShape,
                    byteCount: bytes
                ))
            }
        }
        tensors.sort { $0.name < $1.name }

        var headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-combined-gate-up-prefill-v1"]
        ]
        var cursor = 0
        for tensor in tensors {
            headerObject[tensor.name] = [
                "dtype": tensor.dtype.rawValue,
                "shape": tensor.shape,
                "data_offsets": [cursor, cursor + tensor.byteCount],
            ]
            cursor += tensor.byteCount
        }
        var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
        while !header.count.isMultiple(of: 8) { header.append(0x20) }

        let destination = destinationDirectory.appendingPathComponent(shardName)
        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)
        for tensor in tensors {
            var written = 0
            for sourceName in tensor.sourceNames {
                guard let shard = index.weightMap[sourceName],
                      let sourceHeader = sourceHeaders[shard],
                      let info = sourceHeader.tensors[sourceName]
                else { throw MLXFastError.invalidInput("missing combined source \(sourceName)") }
                let input = try FileHandle(forReadingFrom: sourceDirectory.appendingPathComponent(shard))
                defer { try? input.close() }
                try input.seek(toOffset: sourceHeader.dataBaseOffset + UInt64(info.dataStart))
                let data = input.readData(ofLength: info.byteCount)
                guard data.count == info.byteCount else {
                    throw MLXFastError.invalidInput("short read while combining \(sourceName)")
                }
                try output.write(contentsOf: data)
                written += data.count
            }
            guard written == tensor.byteCount else {
                throw MLXFastError.invalidInput("combined tensor byte count mismatch for \(tensor.name)")
            }
        }
        try output.synchronize()
        return GeneratedCombinedGateUpReport(
            weightMap: Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, shardName) }),
            replacedKeys: replaced,
            tensorByteCount: cursor
        )
    }
}
