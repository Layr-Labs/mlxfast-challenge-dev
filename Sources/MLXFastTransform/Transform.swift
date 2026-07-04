import Foundation
import MLXFastCore

public struct TransformOptions: Equatable {
    public let referencePath: String
    public let outputPath: String

    public init(referencePath: String, outputPath: String) {
        self.referencePath = referencePath
        self.outputPath = outputPath
    }
}

public struct TransformReport: Equatable {
    public let referencePath: String
    public let outputPath: String
    public let denseTensorCount: Int
    public let expertTensorCount: Int
    public let denseShardCount: Int
    public let manifestPath: String
}

private let decodeCodeBundleLayerRange = 2..<6
private let decodeCodeBundleShardName = "decode-code-bundles.safetensors"
private let decodeCodeBundleManifestName = "decode-code-bundles.manifest.json"

public enum SwiftTransform {
    public static func run(_ options: TransformOptions) throws -> TransformReport {
        let referenceDirectory = try findReferenceDirectory(
            URL(fileURLWithPath: options.referencePath)
        )
        let outputDirectory = URL(fileURLWithPath: options.outputPath)
        let expertsDirectory = outputDirectory.appendingPathComponent("experts", isDirectory: true)

        try requireFile(
            referenceDirectory.appendingPathComponent("config.json").path,
            description: "DeepSeek V4 Flash reference config"
        )

        let index = try loadIndex(referenceDirectory)
        try validateCheckpointIndex(index, referenceDirectory: referenceDirectory)
        let denseKeys = Set(index.weightMap.keys.filter { !isExpertKey($0) })
        let expertKeys = Set(index.weightMap.keys.filter { isExpertKey($0) })
        guard !denseKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no dense tensors")
        }
        guard !expertKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no routed expert tensors")
        }

        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: expertsDirectory,
            withIntermediateDirectories: true
        )

        let denseKeysByShard = Dictionary(grouping: denseKeys) { key in
            index.weightMap[key] ?? ""
        }

        var copiedDenseTensors = 0
        for shardName in denseKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = outputDirectory.appendingPathComponent(shardName)
            copiedDenseTensors += try Safetensors.copySubset(
                from: source,
                to: destination,
                tensorNames: denseKeysByShard[shardName, default: []].sorted()
            )
        }

        try copyTokenizerAndConfigFiles(
            from: referenceDirectory,
            to: outputDirectory
        )
        try index.writeStripped(
            to: outputDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: denseKeys
        )

        let manifestPath = expertsDirectory.appendingPathComponent("manifest.json")
        try writeExpertManifest(
            referenceDirectory: referenceDirectory,
            manifestPath: manifestPath,
            expertKeys: expertKeys,
            index: index
        )
        try writeDecodeCodeBundles(
            referenceDirectory: referenceDirectory,
            expertsDirectory: expertsDirectory,
            expertKeys: expertKeys,
            index: index
        )

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedDenseTensors,
            expertTensorCount: expertKeys.count,
            denseShardCount: denseKeysByShard.count,
            manifestPath: manifestPath.path
        )
    }

    private static func loadIndex(_ referenceDirectory: URL) throws -> CheckpointIndex {
        let indexPath = referenceDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexPath.path) {
            return try CheckpointIndex.load(from: indexPath)
        }
        return try CheckpointIndex.buildFromSafetensors(in: referenceDirectory)
    }

    private static func validateCheckpointIndex(
        _ index: CheckpointIndex,
        referenceDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            let attributes = try fileManager.attributesOfItem(atPath: shardURL.path)
            let byteCount = try fileSizeByteCount(from: attributes, path: shardURL.path)
            guard header.dataBaseOffset <= UInt64(Int.max) else {
                throw MLXFastError.invalidInput("checkpoint shard header is too large: \(shardName)")
            }
            let baseOffset = Int(header.dataBaseOffset)

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "checkpoint index lists tensor \(key) in \(shardName), but the shard header does not contain it"
                    )
                }
                let dtype = try TensorDType.parse(info.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: key,
                    dtype: dtype,
                    shape: info.shape
                )
                guard info.byteCount == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte length \(info.byteCount) does not match dtype \(info.dtype) and shape \(info.shape) expected \(expectedByteLength)"
                    )
                }
                let end = baseOffset + info.dataEnd
                guard info.dataStart >= 0, info.byteCount > 0, end <= byteCount else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte range \(info.dataStart)..<\(info.dataEnd) exceeds shard size \(byteCount)"
                    )
                }
            }
        }
    }

    private static func findReferenceDirectory(_ base: URL) throws -> URL {
        if FileManager.default.fileExists(
            atPath: base.appendingPathComponent("config.json").path
        ) {
            return base
        }

        guard let enumerator = FileManager.default.enumerator(
            at: base,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw MLXFastError.missingFile("reference path not found at \(base.path)")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "config.json" {
                return url.deletingLastPathComponent()
            }
        }

        throw MLXFastError.missingFile(
            "no config.json found under \(base.path); place the DeepSeek V4 Flash checkpoint there"
        )
    }

    static func isExpertKey(_ key: String) -> Bool {
        (key.contains(".ffn.experts.") || key.contains(".ffn.switch_mlp."))
            && !key.contains(".shared_experts.")
    }

    private static func copyTokenizerAndConfigFiles(from source: URL, to destination: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey]
        )
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let target = destination.appendingPathComponent(file.lastPathComponent)
                if FileManager.default.fileExists(atPath: target.path) {
                    try FileManager.default.removeItem(at: target)
                }
                try FileManager.default.copyItem(at: file, to: target)
            }
        }
    }

    private static func shouldCopyMetadataFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if name.hasSuffix(".safetensors") {
            return false
        }
        switch url.pathExtension {
        case "json", "model", "tiktoken", "txt":
            return true
        default:
            return name == "tokenizer" || name == "vocab"
        }
    }

    private static func writeExpertManifest(
        referenceDirectory: URL,
        manifestPath: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws {
        var records: [[String: Any]] = []
        let expertKeysByShard = Dictionary(grouping: expertKeys) { key in
            index.weightMap[key] ?? ""
        }

        for shardName in expertKeysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            for key in expertKeysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "expert tensor \(key) is listed in index but missing from \(shardName)"
                    )
                }
                records.append([
                    "name": key,
                    "shard": shardName,
                    "dtype": info.dtype,
                    "shape": info.shape,
                    "data_offsets": [info.dataStart, info.dataEnd],
                    "byte_offset": Int(header.dataBaseOffset) + info.dataStart,
                    "byte_length": info.byteCount,
                ])
            }
        }

        let object: [String: Any] = [
            "version": 1,
            "source": "safetensors",
            "reference_path": referenceDirectory.path,
            "expert_tensors": records,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: manifestPath)
    }

    private struct DecodeBundleSourceRecord {
        let name: String
        let shard: String
        let dtype: String
        let shape: [Int]
        let byteOffset: Int
        let byteLength: Int
    }

    private struct DecodeBundleProjectionPlan {
        let projection: String
        let source: DecodeBundleSourceRecord
        let sliceShape: [Int]
        let sliceByteLength: Int
        let bundleOffset: Int
    }

    private struct DecodeBundleLayerPlan {
        let layerIndex: Int
        let expertCount: Int
        let projections: [DecodeBundleProjectionPlan]
        let bundleByteLength: Int
    }

    private struct DecodeBundleTensorInfo {
        let name: String
        let dataStart: Int
        let dataEnd: Int
        let byteLength: Int
    }

    private static func writeDecodeCodeBundles(
        referenceDirectory: URL,
        expertsDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws {
        let manifestURL = expertsDirectory.appendingPathComponent(decodeCodeBundleManifestName)
        let shardURL = expertsDirectory.appendingPathComponent(decodeCodeBundleShardName)
        let sourceRecords = try loadDecodeBundleSourceRecords(
            referenceDirectory: referenceDirectory,
            expertKeys: expertKeys,
            index: index
        )
        let plans = decodeCodeBundleLayerRange.compactMap {
            makeDecodeBundleLayerPlan(layerIndex: $0, sourceRecords: sourceRecords)
        }
        guard !plans.isEmpty else {
            try? FileManager.default.removeItem(at: manifestURL)
            try? FileManager.default.removeItem(at: shardURL)
            return
        }

        let tensors = try writeDecodeBundleShard(
            plans: plans,
            referenceDirectory: referenceDirectory,
            shardURL: shardURL
        )
        try writeDecodeBundleManifest(
            plans: plans,
            tensors: tensors,
            expertsDirectory: expertsDirectory,
            manifestURL: manifestURL
        )
    }

    private static func loadDecodeBundleSourceRecords(
        referenceDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws -> [String: DecodeBundleSourceRecord] {
        var records: [String: DecodeBundleSourceRecord] = [:]
        let expertKeysByShard = Dictionary(grouping: expertKeys) { key in
            index.weightMap[key] ?? ""
        }
        for shardName in expertKeysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            for key in expertKeysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    continue
                }
                records[key] = DecodeBundleSourceRecord(
                    name: key,
                    shard: shardName,
                    dtype: info.dtype,
                    shape: info.shape,
                    byteOffset: Int(header.dataBaseOffset) + info.dataStart,
                    byteLength: info.byteCount
                )
            }
        }
        return records
    }

    private static func makeDecodeBundleLayerPlan(
        layerIndex: Int,
        sourceRecords: [String: DecodeBundleSourceRecord]
    ) -> DecodeBundleLayerPlan? {
        var projections: [DecodeBundleProjectionPlan] = []
        projections.reserveCapacity(3)
        var bundleOffset = 0
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            guard let source = findStackedDecodeCodeRecord(
                layerIndex: layerIndex,
                projection: projection,
                sourceRecords: sourceRecords
            ),
                  source.dtype == "U32",
                  source.shape.count == 3,
                  source.shape.first == MLXFastConstants.routedExperts,
                  source.byteLength % MLXFastConstants.routedExperts == 0
            else {
                return nil
            }
            let sliceByteLength = source.byteLength / MLXFastConstants.routedExperts
            projections.append(
                DecodeBundleProjectionPlan(
                    projection: projection,
                    source: source,
                    sliceShape: Array(source.shape.dropFirst()),
                    sliceByteLength: sliceByteLength,
                    bundleOffset: bundleOffset
                )
            )
            bundleOffset += sliceByteLength
        }
        return DecodeBundleLayerPlan(
            layerIndex: layerIndex,
            expertCount: MLXFastConstants.routedExperts,
            projections: projections,
            bundleByteLength: bundleOffset
        )
    }

    private static func findStackedDecodeCodeRecord(
        layerIndex: Int,
        projection: String,
        sourceRecords: [String: DecodeBundleSourceRecord]
    ) -> DecodeBundleSourceRecord? {
        let layerNeedle = ".layers.\(layerIndex)."
        let aliases: [String]
        switch projection {
        case "gate_proj":
            aliases = ["gate_proj", "w1"]
        case "down_proj":
            aliases = ["down_proj", "w2"]
        case "up_proj":
            aliases = ["up_proj", "w3"]
        default:
            aliases = [projection]
        }
        return sourceRecords.values
            .filter { record in
                guard record.name.contains(layerNeedle),
                      record.name.hasSuffix(".weight"),
                      record.shape.count == 3,
                      record.shape.first == MLXFastConstants.routedExperts
                else {
                    return false
                }
                return aliases.contains { alias in
                    record.name.contains(".ffn.switch_mlp.\(alias).")
                        || record.name.contains(".ffn.experts.\(alias).")
                }
            }
            .sorted { $0.name < $1.name }
            .first
    }

    private static func writeDecodeBundleShard(
        plans: [DecodeBundleLayerPlan],
        referenceDirectory: URL,
        shardURL: URL
    ) throws -> [DecodeBundleTensorInfo] {
        var tensors: [DecodeBundleTensorInfo] = []
        var cursor = 0
        for plan in plans {
            for expertIndex in 0..<plan.expertCount {
                let end = cursor + plan.bundleByteLength
                tensors.append(
                    DecodeBundleTensorInfo(
                        name: decodeBundleName(layerIndex: plan.layerIndex, expertIndex: expertIndex),
                        dataStart: cursor,
                        dataEnd: end,
                        byteLength: plan.bundleByteLength
                    )
                )
                cursor = end
            }
        }

        let headerData = try makeDecodeBundleSafetensorsHeader(tensors: tensors)
        if FileManager.default.fileExists(atPath: shardURL.path) {
            try FileManager.default.removeItem(at: shardURL)
        }
        try Data().write(to: shardURL, options: [])
        let output = try FileHandle(forWritingTo: shardURL)
        defer {
            try? output.close()
        }
        var headerLength = UInt64(headerData.count).littleEndian
        output.write(Data(bytes: &headerLength, count: 8))
        output.write(headerData)

        var handles: [String: FileHandle] = [:]
        defer {
            for handle in handles.values {
                try? handle.close()
            }
        }
        for plan in plans {
            for expertIndex in 0..<plan.expertCount {
                for projection in plan.projections {
                    let handle = try inputHandle(
                        shard: projection.source.shard,
                        referenceDirectory: referenceDirectory,
                        handles: &handles
                    )
                    let offset = projection.source.byteOffset + expertIndex * projection.sliceByteLength
                    try handle.seek(toOffset: UInt64(offset))
                    let data = handle.readData(ofLength: projection.sliceByteLength)
                    guard data.count == projection.sliceByteLength else {
                        throw MLXFastError.invalidInput(
                            "short read while writing decode bundle \(projection.source.name)[\(expertIndex)]"
                        )
                    }
                    output.write(data)
                }
            }
        }
        return tensors
    }

    private static func inputHandle(
        shard: String,
        referenceDirectory: URL,
        handles: inout [String: FileHandle]
    ) throws -> FileHandle {
        if let handle = handles[shard] {
            return handle
        }
        let handle = try FileHandle(forReadingFrom: referenceDirectory.appendingPathComponent(shard))
        handles[shard] = handle
        return handle
    }

    private static func makeDecodeBundleSafetensorsHeader(
        tensors: [DecodeBundleTensorInfo]
    ) throws -> Data {
        var object: [String: Any] = [
            "__metadata__": [
                "format": "mlxfast_decode_code_bundles",
            ],
        ]
        for tensor in tensors {
            object[tensor.name] = [
                "dtype": "U8",
                "shape": [tensor.byteLength],
                "data_offsets": [tensor.dataStart, tensor.dataEnd],
            ]
        }
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while data.count % 8 != 0 {
            data.append(0x20)
        }
        return data
    }

    private static func writeDecodeBundleManifest(
        plans: [DecodeBundleLayerPlan],
        tensors: [DecodeBundleTensorInfo],
        expertsDirectory: URL,
        manifestURL: URL
    ) throws {
        let headerData = try makeDecodeBundleSafetensorsHeader(tensors: tensors)
        let dataBaseOffset = 8 + headerData.count
        let expertTensorRecords: [[String: Any]] = tensors.map { tensor in
            [
                "name": tensor.name,
                "shard": decodeCodeBundleShardName,
                "dtype": "U8",
                "shape": [tensor.byteLength],
                "data_offsets": [tensor.dataStart, tensor.dataEnd],
                "byte_offset": dataBaseOffset + tensor.dataStart,
                "byte_length": tensor.byteLength,
            ]
        }
        let layerRecords: [[String: Any]] = plans.map { plan in
            [
                "layer_index": plan.layerIndex,
                "expert_count": plan.expertCount,
                "bundle_byte_length": plan.bundleByteLength,
                "records": plan.projections.map { projection in
                    [
                        "projection": projection.projection,
                        "name": projection.source.name,
                        "dtype": projection.source.dtype,
                        "slice_shape": projection.sliceShape,
                        "slice_byte_length": projection.sliceByteLength,
                        "bundle_offset": projection.bundleOffset,
                    ] as [String: Any]
                },
            ]
        }
        let object: [String: Any] = [
            "version": 1,
            "source": "safetensors",
            "reference_path": expertsDirectory.path,
            "expert_tensors": expertTensorRecords,
            "decode_code_bundles": [
                "version": 1,
                "layers": layerRecords,
            ],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: manifestURL)
    }

    private static func decodeBundleName(layerIndex: Int, expertIndex: Int) -> String {
        "mlxfast.decode_code_bundle.layer_\(layerIndex).expert_\(expertIndex)"
    }
}
