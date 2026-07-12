import Foundation
import MLXFastCore

public struct DenseTensorRecord: Equatable {
    public let name: String
    public let shard: String
    public let dtype: String
    public let shape: [Int]
    public let byteOffset: Int
    public let byteLength: Int
}

public final class DenseTensorStore {
    public let weightsPath: String
    let combinedProjectionProfile: String?
    private let physicalRecordsByName: [String: DenseTensorRecord]
    private let recordsByName: [String: DenseTensorRecord]

    public init(weightsPath: String) throws {
        self.weightsPath = weightsPath
        let inventory = try DenseTensorStore.loadRecords(weightsPath: weightsPath)
        self.combinedProjectionProfile = inventory.combinedProjectionProfile
        self.physicalRecordsByName = inventory.physicalRecordsByName
        self.recordsByName = inventory.recordsByName
    }

    public var tensorNames: [String] {
        recordsByName.keys.sorted()
    }

    var shardNames: [String] {
        Set(physicalRecordsByName.values.map(\.shard)).sorted()
    }

    func tensorNames(inShard shard: String) -> Set<String> {
        Set(physicalRecordsByName.values.lazy.filter { $0.shard == shard }.map(\.name))
    }

    public func record(named name: String) -> DenseTensorRecord? {
        recordsByName[name]
    }

    public func tensorBytes(named name: String) throws -> Data {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }

        let shardURL = URL(fileURLWithPath: weightsPath).appendingPathComponent(record.shard)
        let handle = try FileHandle(forReadingFrom: shardURL)
        defer {
            try? handle.close()
        }
        guard let byteOffset = UInt64(exactly: record.byteOffset) else {
            throw MLXFastError.invalidInput("negative byte offset for dense tensor \(name)")
        }
        try handle.seek(toOffset: byteOffset)
        let data = handle.readData(ofLength: record.byteLength)
        guard data.count == record.byteLength else {
            throw MLXFastError.invalidInput(
                "short read for dense tensor \(name): \(data.count)/\(record.byteLength)"
            )
        }
        return data
    }

    public func materializedTensor(named name: String) throws -> MaterializedTensor {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("dense tensor not found: \(name)")
        }
        return try materializeTensor(
            name: record.name,
            dtype: record.dtype,
            shape: record.shape,
            bytes: tensorBytes(named: name)
        )
    }

    public func validateReadableByteRanges(fileManager: FileManager = .default) throws {
        let recordsByShard = Dictionary(grouping: recordsByName.values) { $0.shard }
        for shard in recordsByShard.keys.sorted() {
            let shardPath = URL(fileURLWithPath: weightsPath).appendingPathComponent(shard).path
            let attributes = try fileManager.attributesOfItem(atPath: shardPath)
            let byteCount = try fileSizeByteCount(from: attributes, path: shardPath)
            for record in recordsByShard[shard, default: []] {
                let dtype = try TensorDType.parse(record.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: record.name,
                    dtype: dtype,
                    shape: record.shape
                )
                guard record.byteLength == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "dense tensor \(record.name) byte length \(record.byteLength) does not match dtype \(record.dtype) and shape \(record.shape) expected \(expectedByteLength)"
                    )
                }
                let (end, overflow) = record.byteOffset.addingReportingOverflow(record.byteLength)
                guard
                    !overflow,
                    record.byteOffset >= 0,
                    record.byteLength > 0,
                    end <= byteCount
                else {
                    throw MLXFastError.invalidInput(
                        "dense tensor \(record.name) byte range \(record.byteOffset)..<\(end) exceeds shard size \(byteCount)"
                    )
                }
            }
        }
    }

    private struct LoadedIndex {
        let weightMap: [String: String]
        let combinedProjectionProfile: String?
    }

    private struct Inventory {
        let physicalRecordsByName: [String: DenseTensorRecord]
        let recordsByName: [String: DenseTensorRecord]
        let combinedProjectionProfile: String?
    }

    private static func loadRecords(weightsPath: String) throws -> Inventory {
        let weightsURL = URL(fileURLWithPath: weightsPath)
        try requireFile(
            weightsURL.appendingPathComponent("model.safetensors.index.json").path,
            description: "dense safetensors index"
        )

        let loadedIndex = try loadIndex(
            weightsURL.appendingPathComponent("model.safetensors.index.json")
        )
        let weightMap = loadedIndex.weightMap
        for shard in Set(weightMap.values).sorted() {
            try validateSafetensorsShardName(shard, context: "dense safetensors index")
        }
        let keysByShard = Dictionary(grouping: weightMap.keys) { key in
            weightMap[key] ?? ""
        }

        var records: [String: DenseTensorRecord] = [:]
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shard in keysByShard.keys.sorted() {
            let shardURL = weightsURL.appendingPathComponent(shard)
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shard] = header
            for key in keysByShard[shard, default: []] {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "tensor \(key) is listed in dense index but missing from \(shard)"
                    )
                }
                guard let baseOffset = Int(exactly: header.dataBaseOffset) else {
                    throw MLXFastError.invalidInput("safetensors header offset exceeds Int range in \(shard)")
                }
                let (byteOffset, overflow) = baseOffset.addingReportingOverflow(info.dataStart)
                guard !overflow else {
                    throw MLXFastError.invalidInput("dense tensor byte offset overflows Int for \(key)")
                }
                records[key] = DenseTensorRecord(
                    name: key,
                    shard: shard,
                    dtype: info.dtype,
                    shape: info.shape,
                    byteOffset: byteOffset,
                    byteLength: info.byteCount
                )
            }
        }

        guard !records.isEmpty else {
            throw MLXFastError.invalidInput("dense tensor store contains no safetensors tensors")
        }
        let logicalRecords = try addCombinedProjectionAliases(
            to: records,
            profile: loadedIndex.combinedProjectionProfile,
            headersByShard: headersByShard
        )
        return Inventory(
            physicalRecordsByName: records,
            recordsByName: logicalRecords,
            combinedProjectionProfile: loadedIndex.combinedProjectionProfile
        )
    }

    private static func loadIndex(_ path: URL) throws -> LoadedIndex {
        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard
            let root = object as? [String: Any],
            let weightMap = root["weight_map"] as? [String: String]
        else {
            throw MLXFastError.invalidInput("dense safetensors index missing weight_map")
        }
        let metadata = root["metadata"] as? [String: Any] ?? [:]
        let profile: String?
        if let value = metadata[Gemma4CombinedProjectionProfile.metadataKey] {
            guard let string = value as? String else {
                throw MLXFastError.invalidInput(
                    "combined projection profile must be a string"
                )
            }
            guard string == Gemma4CombinedProjectionProfile.value else {
                throw MLXFastError.invalidInput(
                    "unsupported combined projection profile: \(string)"
                )
            }
            profile = string
        } else {
            profile = nil
        }
        return LoadedIndex(weightMap: weightMap, combinedProjectionProfile: profile)
    }

    static func addCombinedProjectionAliases(
        to physicalRecords: [String: DenseTensorRecord],
        profile: String?,
        headersByShard: [String: SafetensorsHeader]
    ) throws -> [String: DenseTensorRecord] {
        let reservedNames = Set(physicalRecords.keys.filter {
            $0.hasPrefix(Gemma4CombinedProjectionProfile.physicalPrefix)
        })
        guard let profile else {
            guard reservedNames.isEmpty,
                  !headersByShard.keys.contains(Gemma4CombinedProjectionProfile.shardName)
            else {
                throw MLXFastError.invalidInput(
                    "combined projection tensors require an index profile"
                )
            }
            return physicalRecords
        }
        guard profile == Gemma4CombinedProjectionProfile.value else {
            throw MLXFastError.invalidInput(
                "unsupported combined projection profile: \(profile)"
            )
        }

        let expectedNames = Set(Gemma4CombinedProjectionProfile.groups.flatMap { group in
            ["weight", "scales", "biases"].map { "\(group.physicalStem).\($0)" }
        })
        guard reservedNames == expectedNames else {
            let missing = expectedNames.subtracting(reservedNames).sorted()
            let extra = reservedNames.subtracting(expectedNames).sorted()
            throw MLXFastError.invalidInput(
                "combined projection profile inventory mismatch; missing=\(missing) extra=\(extra)"
            )
        }
        guard let combinedHeader = headersByShard[Gemma4CombinedProjectionProfile.shardName],
              combinedHeader.metadata["format"] == Gemma4CombinedProjectionProfile.value,
              combinedHeader.metadata["row_order"] == Gemma4CombinedProjectionProfile.rowOrder
        else {
            throw MLXFastError.invalidInput(
                "combined projection shard metadata does not match its profile"
            )
        }

        var logical = physicalRecords
        for group in Gemma4CombinedProjectionProfile.groups {
            for suffix in ["weight", "scales", "biases"] {
                let parentName = "\(group.physicalStem).\(suffix)"
                guard let parent = physicalRecords[parentName],
                      parent.shard == Gemma4CombinedProjectionProfile.shardName
                else {
                    throw MLXFastError.invalidInput(
                        "combined projection parent is missing or in the wrong shard: \(parentName)"
                    )
                }
                let columns = suffix == "weight" ? 672 : 84
                let dtype = suffix == "weight" ? "U32" : "BF16"
                guard parent.dtype == dtype, parent.shape == [group.rows, columns] else {
                    throw MLXFastError.invalidInput(
                        "combined projection parent has invalid dtype or shape: \(parentName)"
                    )
                }
                let itemSize = try TensorDType.parse(dtype).byteWidth
                let (rowBytes, rowOverflow) = columns.multipliedReportingOverflow(by: itemSize)
                let (expectedParentBytes, parentBytesOverflow) = group.rows
                    .multipliedReportingOverflow(by: rowBytes)
                guard !rowOverflow, !parentBytesOverflow,
                      parent.byteLength == expectedParentBytes
                else {
                    throw MLXFastError.invalidInput(
                        "combined projection parent byte length is invalid for \(parentName)"
                    )
                }
                var startRow = 0
                for component in group.components {
                    let componentName = "\(component.diskStem).\(suffix)"
                    guard physicalRecords[componentName] == nil else {
                        throw MLXFastError.invalidInput(
                            "combined projection profile mixes parent and component tensor \(componentName)"
                        )
                    }
                    let (rowOffset, offsetOverflow) = startRow
                        .multipliedReportingOverflow(by: rowBytes)
                    let (byteOffset, byteOffsetOverflow) = parent.byteOffset
                        .addingReportingOverflow(rowOffset)
                    let (byteLength, lengthOverflow) = component.rows
                        .multipliedReportingOverflow(by: rowBytes)
                    guard !offsetOverflow, !byteOffsetOverflow, !lengthOverflow else {
                        throw MLXFastError.invalidInput(
                            "combined projection alias byte range overflows for \(componentName)"
                        )
                    }
                    logical[componentName] = DenseTensorRecord(
                        name: componentName,
                        shard: parent.shard,
                        dtype: dtype,
                        shape: [component.rows, columns],
                        byteOffset: byteOffset,
                        byteLength: byteLength
                    )
                    startRow += component.rows
                }
            }
        }
        return logical
    }
}
