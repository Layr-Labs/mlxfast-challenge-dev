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
}
