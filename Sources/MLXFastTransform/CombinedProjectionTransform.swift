import Foundation
import MLXFastCore

// Zero-dup combined projections authored by GPT 5.6 Sol through Gaj's OpenCode Harness.

struct CombinedProjectionTransformReport {
    let weightMap: [String: String]
    let prunedKeys: Set<String>
    let tensorByteCount: Int
    let sourceTensorByteCount: Int
    let indexMetadata: [String: String]

    var tensorCount: Int { weightMap.count }
    var shardCount: Int { weightMap.isEmpty ? 0 : 1 }

    static let empty = CombinedProjectionTransformReport(
        weightMap: [:],
        prunedKeys: [],
        tensorByteCount: 0,
        sourceTensorByteCount: 0,
        indexMetadata: [:]
    )
}

enum CombinedProjectionTransform {
    static let profileMetadataKey = "mlxfast_combined_projection_profile"
    static let profile = "gemma4-31b-zero-dup-attention-v1"
    static let shardName = "mlxfast-combined-attention-v1.safetensors"
    static let physicalPrefix = "mlxfast.combined_attention.layers."

    private struct Component {
        let name: String
        let shard: String
        let info: SafetensorInfo
    }

    private struct Tensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let components: [Component]

        var byteCount: Int {
            components.reduce(0) { $0 + $1.info.byteCount }
        }
    }

    struct Group {
        let physicalStem: String
        let components: [(stem: String, rows: Int)]
    }

    struct Layout: Equatable {
        let name: String
        let dtype: String
        let shape: [Int]
        let componentNames: [String]
        let byteCount: Int
    }

    static func profileLayouts() -> [Layout] {
        expectedGroups().flatMap { group in
            ["weight", "scales", "biases"].map { suffix in
                let dtype = suffix == "weight" ? "U32" : "BF16"
                let columns = suffix == "weight" ? 672 : 84
                let itemSize = suffix == "weight" ? 4 : 2
                let rows = group.components.reduce(0) { $0 + $1.rows }
                return Layout(
                    name: "\(group.physicalStem).\(suffix)",
                    dtype: dtype,
                    shape: [rows, columns],
                    componentNames: group.components.map { "\($0.stem).\(suffix)" },
                    byteCount: rows * columns * itemSize
                )
            }
        }.sorted { $0.name < $1.name }
    }

    static func writeCombinedShard(
        index: CheckpointIndex,
        sourceFiles: [String: TransformSourceFile],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> CombinedProjectionTransformReport {
        let reservedNames = index.weightMap.keys.filter { $0.hasPrefix(physicalPrefix) }
        guard reservedNames.isEmpty else {
            throw MLXFastError.invalidInput(
                "checkpoint contains reserved combined projection tensors"
            )
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }

        let groups = expectedGroups()
        let requiredKeys = requiredSourceKeys()
        // Sparse synthetic checkpoints keep the legacy layout. The production
        // profile is all-or-nothing and is only advertised after every frozen
        // component is present and validated below.
        guard requiredKeys.isSubset(of: selectedKeys) else {
            return .empty
        }

        var tensors: [Tensor] = []
        tensors.reserveCapacity(groups.count * 3)
        for group in groups {
            for suffix in ["weight", "scales", "biases"] {
                let expectedDType = suffix == "weight" ? "U32" : "BF16"
                let expectedColumns = suffix == "weight" ? 672 : 84
                var components: [Component] = []
                components.reserveCapacity(group.components.count)
                var combinedRows = 0
                for component in group.components {
                    let name = "\(component.stem).\(suffix)"
                    guard let shard = index.weightMap[name],
                          let info = sourceFiles[shard]?.header.tensors[name]
                    else {
                        throw MLXFastError.invalidInput(
                            "combined projection source tensor is missing: \(name)"
                        )
                    }
                    guard info.dtype == expectedDType,
                          info.shape == [component.rows, expectedColumns]
                    else {
                        throw MLXFastError.invalidInput(
                            "combined projection source tensor \(name) has invalid dtype or shape"
                        )
                    }
                    let expectedBytes = try expectedTensorByteCount(
                        name: name,
                        dtype: try TensorDType.parse(info.dtype),
                        shape: info.shape
                    )
                    guard info.byteCount == expectedBytes else {
                        throw MLXFastError.invalidInput(
                            "combined projection source tensor \(name) has invalid byte length"
                        )
                    }
                    components.append(Component(name: name, shard: shard, info: info))
                    let (nextRows, overflow) = combinedRows.addingReportingOverflow(component.rows)
                    guard !overflow else {
                        throw MLXFastError.invalidInput(
                            "combined projection row count overflows for \(group.physicalStem)"
                        )
                    }
                    combinedRows = nextRows
                }
                tensors.append(Tensor(
                    name: "\(group.physicalStem).\(suffix)",
                    dtype: expectedDType,
                    shape: [combinedRows, expectedColumns],
                    components: components
                ))
            }
        }
        tensors.sort { $0.name < $1.name }

        let sourceTensorByteCount = try checkedSum(
            tensors.flatMap(\.components).map(\.info.byteCount),
            context: "combined source tensor byte count"
        )
        let tensorByteCount = try checkedSum(
            tensors.map(\.byteCount),
            context: "combined tensor byte count"
        )
        guard sourceTensorByteCount == tensorByteCount else {
            throw MLXFastError.invalidInput(
                "combined projection payload size differs from pruned source payload size"
            )
        }

        try writeShard(
            destinationDirectory.appendingPathComponent(shardName),
            tensors: tensors,
            sourceFiles: sourceFiles
        )
        return CombinedProjectionTransformReport(
            weightMap: Dictionary(
                uniqueKeysWithValues: tensors.map { ($0.name, shardName) }
            ),
            prunedKeys: requiredKeys,
            tensorByteCount: tensorByteCount,
            sourceTensorByteCount: sourceTensorByteCount,
            indexMetadata: [profileMetadataKey: profile]
        )
    }

    static func sourcePayloadKeys(selectedKeys: Set<String>) -> Set<String> {
        let requiredKeys = requiredSourceKeys()
        return requiredKeys.isSubset(of: selectedKeys) ? requiredKeys : []
    }

    static func expectedGroups() -> [Group] {
        (0..<MLXFastConstants.numHiddenLayers).map { layerIndex in
            let sourcePrefix = "language_model.model.layers.\(layerIndex)"
            let physicalPrefix = "\(physicalPrefix)\(layerIndex)"
            if layerIndex % 6 == 5 {
                return Group(
                    physicalStem: "\(physicalPrefix).self_attn.qk",
                    components: [
                        ("\(sourcePrefix).self_attn.q_proj", 16_384),
                        ("\(sourcePrefix).self_attn.k_proj", 2_048),
                    ]
                )
            }
            return Group(
                physicalStem: "\(physicalPrefix).self_attn.qkv",
                components: [
                    ("\(sourcePrefix).self_attn.q_proj", 8_192),
                    ("\(sourcePrefix).self_attn.k_proj", 4_096),
                    ("\(sourcePrefix).self_attn.v_proj", 4_096),
                ]
            )
        }
    }

    private static func requiredSourceKeys() -> Set<String> {
        Set(expectedGroups().flatMap { group in
            group.components.flatMap { component in
                ["weight", "scales", "biases"].map {
                    "\(component.stem).\($0)"
                }
            }
        })
    }

    private static func writeShard(
        _ destination: URL,
        tensors: [Tensor],
        sourceFiles: [String: TransformSourceFile]
    ) throws {
        var headerObject: [String: Any] = [
            "__metadata__": [
                "format": profile,
                "row_order": "q-k-v;q-k",
            ]
        ]
        var cursor = 0
        for tensor in tensors {
            let (end, overflow) = cursor.addingReportingOverflow(tensor.byteCount)
            guard !overflow else {
                throw MLXFastError.invalidInput("combined projection shard size overflows Int")
            }
            headerObject[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }
        var header = try JSONSerialization.data(
            withJSONObject: headerObject,
            options: [.sortedKeys]
        )
        while !header.count.isMultiple(of: 8) {
            header.append(0x20)
        }

        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer {
            try? output.close()
        }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)

        for tensor in tensors {
            for component in tensor.components {
                guard let sourceFile = sourceFiles[component.shard] else {
                    throw MLXFastError.invalidInput(
                        "missing generation snapshot descriptor for \(component.shard)"
                    )
                }
                try sourceFile.copyTensor(named: component.name, to: output)
            }
        }
        try output.synchronize()
    }

    private static func checkedSum(_ values: [Int], context: String) throws -> Int {
        var total = 0
        for value in values {
            let (next, overflow) = total.addingReportingOverflow(value)
            guard !overflow else {
                throw MLXFastError.invalidInput("\(context) overflows Int")
            }
            total = next
        }
        return total
    }
}
