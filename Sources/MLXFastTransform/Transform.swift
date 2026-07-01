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

        // Pack U8 expert scale tensors to 2-bit format. The E8M0 scales typically
        // have only 4 unique values, so each byte maps to a 2-bit index, giving 4x
        // compression (8.6 GiB → ~2.1 GiB). The packed data is stored alongside
        // the manifest so the runtime can load it from local disk instead of seeking
        // into the reference checkpoint shards.
        try packExpertScales(
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

    /// Pack all U8 expert scale tensors from the reference checkpoint into a
    /// compact 4-bit-per-value (nibble) format. The E8M0 scales have ~9 unique
    /// values globally, so each byte maps to a 4-bit palette index, giving 2x
    /// compression (8.6 GiB → ~4.3 GiB).
    ///
    /// Format:
    ///   - 4 bytes: magic "ES4B" (expert scales 4-bit)
    ///   - 4 bytes: little-endian uint32 tensor count
    ///   - 4 bytes: little-endian uint32 palette size (<= 16)
    ///   - palette_size bytes: the unique U8 values
    ///   - For each tensor:
    ///     - 256 bytes: null-terminated tensor name (padded)
    ///     - 4 bytes: little-endian uint32 original byte count
    ///     - 4 bytes: little-endian uint32 packed byte count
    ///     - 4 bytes: little-endian uint32 shape dimension count
    ///     - shape_dims × 4 bytes: shape values
    ///   - Packed data for all tensors concatenated (2 values per byte, nibble-packed)
    private static func packExpertScales(
        referenceDirectory: URL,
        expertsDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws {
        // Find all scale tensor keys (expert keys ending in .scales)
        let scaleKeys = expertKeys.filter { $0.hasSuffix(".scales") }.sorted()
        guard !scaleKeys.isEmpty else { return }

        // Read all scale tensor data and find the global palette
        var tensorEntries: [(name: String, data: Data, shape: [Int])] = []
        var allValues = Set<UInt8>()

        let scaleKeysByShard = Dictionary(grouping: scaleKeys) { key in
            index.weightMap[key] ?? ""
        }

        for shardName in scaleKeysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            let fileHandle = try FileHandle(forReadingFrom: shardURL)
            defer { fileHandle.closeFile() }

            for key in scaleKeysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else { continue }
                guard info.dtype == "U8" else { continue }

                let absoluteOffset = UInt64(header.dataBaseOffset) + UInt64(info.dataStart)
                fileHandle.seek(toFileOffset: absoluteOffset)
                let data = fileHandle.readData(ofLength: info.byteCount)
                guard data.count == info.byteCount else {
                    throw MLXFastError.invalidInput("failed to read scale tensor \(key) from \(shardName)")
                }

                for byte in data { allValues.insert(byte) }
                tensorEntries.append((name: key, data: data, shape: info.shape))
            }
        }

        let palette = allValues.sorted()
        guard palette.count <= 16 else {
            throw MLXFastError.invalidInput("expert scales have \(palette.count) unique values; expected <= 16 for nibble packing")
        }

        // Build reverse lookup: value -> 4-bit index
        var valueToIndex = [UInt8: UInt8]()
        for (i, v) in palette.enumerated() {
            valueToIndex[v] = UInt8(i)
        }

        // Write the packed file
        let outputPath = expertsDirectory.appendingPathComponent("packed_scales.bin")
        var output = Data()

        // Header: magic + tensor count + palette
        output.append(contentsOf: [0x45, 0x53, 0x34, 0x42]) // "ES4B"
        appendUInt32(&output, UInt32(tensorEntries.count))
        appendUInt32(&output, UInt32(palette.count))
        output.append(contentsOf: palette)

        // Per-tensor metadata
        for entry in tensorEntries {
            // Name: 256 bytes, null-padded
            var nameBytes = Array(entry.name.utf8.prefix(255))
            nameBytes.append(contentsOf: Array(repeating: UInt8(0), count: 256 - nameBytes.count))
            output.append(contentsOf: nameBytes)
            appendUInt32(&output, UInt32(entry.data.count))
            let packedCount = (entry.data.count + 1) / 2  // ceil(count / 2) for nibble packing
            appendUInt32(&output, UInt32(packedCount))
            appendUInt32(&output, UInt32(entry.shape.count))
            for dim in entry.shape {
                appendUInt32(&output, UInt32(dim))
            }
        }

        // Packed data: 2 values per byte (4 bits / nibble each)
        for entry in tensorEntries {
            let raw = entry.data
            let packedCount = (raw.count + 1) / 2
            var packed = Data(count: packedCount)
            packed.withUnsafeMutableBytes { packedBuf in
                raw.withUnsafeBytes { rawBuf in
                    let rawPtr = rawBuf.bindMemory(to: UInt8.self)
                    let packPtr = packedBuf.bindMemory(to: UInt8.self)
                    for i in 0..<raw.count {
                        let idx = valueToIndex[rawPtr[i]] ?? 0
                        let bytePos = i / 2
                        if i % 2 == 0 {
                            packPtr[bytePos] = idx          // low nibble
                        } else {
                            packPtr[bytePos] |= idx << 4    // high nibble
                        }
                    }
                }
            }
            output.append(packed)
        }

        try output.write(to: outputPath)
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var le = value.littleEndian
        data.append(Data(bytes: &le, count: 4))
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
