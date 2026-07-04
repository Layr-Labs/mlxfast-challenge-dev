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
        let fusedExpertsDirectory = outputDirectory.appendingPathComponent("fused_experts", isDirectory: true)

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
        if FileManager.default.fileExists(atPath: fusedExpertsDirectory.path) {
            try FileManager.default.removeItem(at: fusedExpertsDirectory)
        }

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
        try writeFusedGateUpExperts(
            referenceDirectory: referenceDirectory,
            fusedDirectory: fusedExpertsDirectory,
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

    private static let fusedGateUpLayerCount = 7
    private static let fusedChunkSize = 8 * 1024 * 1024

    private struct FusedSourceTensor {
        let name: String
        let shardName: String
        let shardURL: URL
        let info: SafetensorInfo
        let dataBaseOffset: UInt64
    }

    private struct FusedTensorPlan {
        let name: String
        let dtype: String
        let shape: [Int]
        let gate: FusedSourceTensor
        let up: FusedSourceTensor
        let byteCount: Int
        let gateSliceByteCount: Int
        let upSliceByteCount: Int
        let expertCount: Int
    }

    private static func writeFusedGateUpExperts(
        referenceDirectory: URL,
        fusedDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws {
        var records: [[String: Any]] = []
        var headerCache: [String: SafetensorsHeader] = [:]

        for layerIndex in 0..<fusedGateUpLayerCount {
            guard
                let gateWeight = try resolveSourceTensor(
                    layerIndex: layerIndex,
                    projection: "gate_proj",
                    suffix: "weight",
                    referenceDirectory: referenceDirectory,
                    expertKeys: expertKeys,
                    index: index,
                    headerCache: &headerCache
                ),
                let upWeight = try resolveSourceTensor(
                    layerIndex: layerIndex,
                    projection: "up_proj",
                    suffix: "weight",
                    referenceDirectory: referenceDirectory,
                    expertKeys: expertKeys,
                    index: index,
                    headerCache: &headerCache
                ),
                let gateScales = try resolveSourceTensor(
                    layerIndex: layerIndex,
                    projection: "gate_proj",
                    suffix: "scales",
                    referenceDirectory: referenceDirectory,
                    expertKeys: expertKeys,
                    index: index,
                    headerCache: &headerCache
                ),
                let upScales = try resolveSourceTensor(
                    layerIndex: layerIndex,
                    projection: "up_proj",
                    suffix: "scales",
                    referenceDirectory: referenceDirectory,
                    expertKeys: expertKeys,
                    index: index,
                    headerCache: &headerCache
                )
            else {
                continue
            }
            _ = (gateScales, upScales)

            let fusedWeightName = fusedGateUpName(from: gateWeight.name, suffix: "weight")
            let plans = [
                try fusedPlan(
                    name: fusedWeightName,
                    gate: gateWeight,
                    up: upWeight
                ),
            ]

            if let _ = try resolveSourceTensor(
                layerIndex: layerIndex,
                projection: "gate_proj",
                suffix: "biases",
                referenceDirectory: referenceDirectory,
                expertKeys: expertKeys,
                index: index,
                headerCache: &headerCache
            ) {
                throw MLXFastError.invalidInput(
                    "fused gate/up sidecar does not support quantized expert biases in layer \(layerIndex)"
                )
            }

            try FileManager.default.createDirectory(
                at: fusedDirectory,
                withIntermediateDirectories: true
            )
            let shardName = String(format: "fused-gate-up-layer-%02d.safetensors", layerIndex)
            let shardURL = fusedDirectory.appendingPathComponent(shardName)
            records += try writeFusedSafetensors(
                plans: plans,
                shardName: shardName,
                destination: shardURL
            )
        }

        guard !records.isEmpty else {
            if FileManager.default.fileExists(atPath: fusedDirectory.path) {
                try FileManager.default.removeItem(at: fusedDirectory)
            }
            return
        }

        let object: [String: Any] = [
            "version": 1,
            "source": "safetensors",
            "reference_path": fusedDirectory.path,
            "expert_tensors": records,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: fusedDirectory.appendingPathComponent("manifest.json"))
    }

    private static func resolveSourceTensor(
        layerIndex: Int,
        projection: String,
        suffix: String,
        referenceDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex,
        headerCache: inout [String: SafetensorsHeader]
    ) throws -> FusedSourceTensor? {
        let marker = ".layers.\(layerIndex)."
        let suffixText = ".\(projection).\(suffix)"
        let candidates = expertKeys
            .filter { $0.contains(marker) && $0.hasSuffix(suffixText) }
            .sorted()
        for name in candidates {
            guard let shardName = index.weightMap[name] else {
                continue
            }
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header: SafetensorsHeader
            if let cached = headerCache[shardName] {
                header = cached
            } else {
                header = try Safetensors.readHeader(shardURL)
                headerCache[shardName] = header
            }
            guard let info = header.tensors[name], info.shape.count == 3 else {
                continue
            }
            return FusedSourceTensor(
                name: name,
                shardName: shardName,
                shardURL: shardURL,
                info: info,
                dataBaseOffset: header.dataBaseOffset
            )
        }
        return nil
    }

    private static func fusedGateUpName(from gateName: String, suffix: String) -> String {
        if gateName.hasSuffix(".gate_proj.\(suffix)") {
            return String(gateName.dropLast(".gate_proj.\(suffix)".count)) + ".gate_up_proj.\(suffix)"
        }
        return gateName + ".gate_up_proj.\(suffix)"
    }

    private static func fusedPlan(
        name: String,
        gate: FusedSourceTensor,
        up: FusedSourceTensor
    ) throws -> FusedTensorPlan {
        guard gate.info.dtype == up.info.dtype else {
            throw MLXFastError.invalidInput(
                "cannot fuse \(gate.name) and \(up.name): dtype mismatch \(gate.info.dtype) vs \(up.info.dtype)"
            )
        }
        guard gate.info.shape.count == 3, up.info.shape.count == 3 else {
            throw MLXFastError.invalidInput("fused gate/up tensors must be stacked rank-3 tensors")
        }
        let gateShape = gate.info.shape
        let upShape = up.info.shape
        guard gateShape[0] == upShape[0], gateShape[2] == upShape[2] else {
            throw MLXFastError.invalidInput(
                "cannot fuse \(gate.name) shape \(gateShape) with \(up.name) shape \(upShape)"
            )
        }
        let expertCount = gateShape[0]
        guard expertCount > 0,
              gate.info.byteCount % expertCount == 0,
              up.info.byteCount % expertCount == 0
        else {
            throw MLXFastError.invalidInput("fused gate/up tensors must be evenly sliceable by expert")
        }
        let shape = [expertCount, gateShape[1] + upShape[1], gateShape[2]]
        return FusedTensorPlan(
            name: name,
            dtype: gate.info.dtype,
            shape: shape,
            gate: gate,
            up: up,
            byteCount: gate.info.byteCount + up.info.byteCount,
            gateSliceByteCount: gate.info.byteCount / expertCount,
            upSliceByteCount: up.info.byteCount / expertCount,
            expertCount: expertCount
        )
    }

    private static func writeFusedSafetensors(
        plans: [FusedTensorPlan],
        shardName: String,
        destination: URL
    ) throws -> [[String: Any]] {
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try Data().write(to: destination, options: [])

        let (headerData, offsets) = try fusedHeaderData(plans: plans)
        let dataBaseOffset = 8 + headerData.count
        let output = try FileHandle(forWritingTo: destination)
        defer {
            try? output.close()
        }

        var headerLength = UInt64(headerData.count).littleEndian
        output.write(Data(bytes: &headerLength, count: 8))
        output.write(headerData)

        var readers: [String: FileHandle] = [:]
        defer {
            for reader in readers.values {
                try? reader.close()
            }
        }
        for plan in plans {
            for expertIndex in 0..<plan.expertCount {
                try copySlice(
                    source: plan.gate,
                    expertIndex: expertIndex,
                    sliceByteCount: plan.gateSliceByteCount,
                    readers: &readers,
                    output: output
                )
                try copySlice(
                    source: plan.up,
                    expertIndex: expertIndex,
                    sliceByteCount: plan.upSliceByteCount,
                    readers: &readers,
                    output: output
                )
            }
        }

        return plans.map { plan in
            let start = offsets[plan.name, default: 0]
            return [
                "name": plan.name,
                "shard": shardName,
                "dtype": plan.dtype,
                "shape": plan.shape,
                "data_offsets": [start, start + plan.byteCount],
                "byte_offset": dataBaseOffset + start,
                "byte_length": plan.byteCount,
            ]
        }
    }

    private static func fusedHeaderData(
        plans: [FusedTensorPlan]
    ) throws -> (Data, [String: Int]) {
        var object: [String: Any] = [:]
        var offsets: [String: Int] = [:]
        var cursor = 0
        for plan in plans {
            offsets[plan.name] = cursor
            object[plan.name] = [
                "dtype": plan.dtype,
                "shape": plan.shape,
                "data_offsets": [cursor, cursor + plan.byteCount],
            ]
            cursor += plan.byteCount
        }

        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while data.count % 8 != 0 {
            data.append(0x20)
        }
        return (data, offsets)
    }

    private static func copySlice(
        source: FusedSourceTensor,
        expertIndex: Int,
        sliceByteCount: Int,
        readers: inout [String: FileHandle],
        output: FileHandle
    ) throws {
        let key = source.shardURL.path
        let reader: FileHandle
        if let existing = readers[key] {
            reader = existing
        } else {
            reader = try FileHandle(forReadingFrom: source.shardURL)
            readers[key] = reader
        }
        let offset = source.dataBaseOffset
            + UInt64(source.info.dataStart)
            + UInt64(expertIndex * sliceByteCount)
        try reader.seek(toOffset: offset)
        var remaining = sliceByteCount
        while remaining > 0 {
            let data = reader.readData(ofLength: min(fusedChunkSize, remaining))
            if data.isEmpty {
                throw MLXFastError.invalidInput("unexpected EOF while writing fused expert tensor")
            }
            output.write(data)
            remaining -= data.count
        }
    }
}
