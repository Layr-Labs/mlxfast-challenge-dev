import Darwin
import CryptoKit
import Foundation
import MLXFastCore

final class TransformSourceFile {
    private struct Identity: Equatable {
        let device: UInt64
        let inode: UInt64
        let byteCount: Int
        let modificationSeconds: Int64
        let modificationNanoseconds: Int64
        let statusChangeSeconds: Int64
        let statusChangeNanoseconds: Int64
        let linkCount: UInt64
    }

    let header: SafetensorsHeader
    private let handle: FileHandle
    private let identity: Identity
    private let capturedTensors: [String: Data]?

    init(
        path: URL,
        expectedHeader: SafetensorsHeader,
        tensorNames: [String],
        trustedSHA256: String? = nil,
        afterDescriptorOpen: (() throws -> Void)? = nil,
        afterHeaderValidation: (() throws -> Void)? = nil,
        afterContentCapture: (() throws -> Void)? = nil
    ) throws {
        let descriptor = path.path.withCString {
            Darwin.open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        do {
            let validatedIdentity = try Self.fileIdentity(handle)
            try afterDescriptorOpen?()
            try Safetensors.validateSourceIdentity(
                URL(fileURLWithPath: "/dev/fd/\(descriptor)"),
                against: expectedHeader
            )
            try afterHeaderValidation?()
            guard try Self.fileIdentity(handle) == validatedIdentity else {
                throw MLXFastError.invalidInput(
                    "generation source descriptor changed while being validated"
                )
            }

            for name in tensorNames {
                guard expectedHeader.tensors[name] != nil else {
                    throw MLXFastError.invalidInput(
                        "generation source tensor is missing: \(name)"
                    )
                }
            }
            let capturedTensors: [String: Data]?
            if let trustedSHA256 {
                capturedTensors = try Self.captureTrustedTensors(
                    handle: handle,
                    expectedHeader: expectedHeader,
                    tensorNames: tensorNames,
                    trustedSHA256: trustedSHA256,
                    fileByteCount: validatedIdentity.byteCount
                )
                try afterContentCapture?()
                guard try Self.fileIdentity(handle) == validatedIdentity else {
                    throw MLXFastError.invalidInput(
                        "generation source descriptor changed while content was captured"
                    )
                }
            } else {
                capturedTensors = nil
            }
            self.header = expectedHeader
            self.handle = handle
            self.identity = validatedIdentity
            self.capturedTensors = capturedTensors
        } catch {
            try? handle.close()
            throw error
        }
    }

    deinit {
        try? handle.close()
    }

    func readTensor(named name: String) throws -> Data {
        guard let info = header.tensors[name] else {
            throw MLXFastError.invalidInput(
                "missing generation source tensor \(name)"
            )
        }
        if let capturedTensors {
            guard let bytes = capturedTensors[name], bytes.count == info.byteCount else {
                throw MLXFastError.invalidInput(
                    "trusted generation source tensor is missing: \(name)"
                )
            }
            return bytes
        }
        try validateIdentity()
        let offset = try absoluteOffset(info, name: name)
        try handle.seek(toOffset: offset)
        let bytes = handle.readData(ofLength: info.byteCount)
        guard bytes.count == info.byteCount else {
            throw MLXFastError.invalidInput(
                "short read from generation source tensor \(name)"
            )
        }
        try validateIdentity()
        return bytes
    }

    func copyTensor(named name: String, to output: FileHandle) throws {
        guard let info = header.tensors[name] else {
            throw MLXFastError.invalidInput(
                "missing generation source tensor \(name)"
            )
        }
        if let capturedTensors {
            guard let bytes = capturedTensors[name], bytes.count == info.byteCount else {
                throw MLXFastError.invalidInput(
                    "trusted generation source tensor is missing: \(name)"
                )
            }
            try output.write(contentsOf: bytes)
            return
        }
        try validateIdentity()
        try handle.seek(toOffset: try absoluteOffset(info, name: name))
        var remaining = info.byteCount
        while remaining > 0 {
            let bytes = handle.readData(ofLength: min(8 * 1024 * 1024, remaining))
            guard !bytes.isEmpty else {
                throw MLXFastError.invalidInput(
                    "unexpected EOF in generation source tensor \(name)"
                )
            }
            try output.write(contentsOf: bytes)
            remaining -= bytes.count
        }
        try validateIdentity()
    }

    func writeSubset(to destination: URL, tensorNames: [String]) throws -> Int {
        let tensors = try tensorNames.map { name in
            guard let info = header.tensors[name] else {
                throw MLXFastError.invalidInput(
                    "tensor \(name) is missing from trusted generation source"
                )
            }
            return info
        }.sorted { $0.name < $1.name }
        guard !tensors.isEmpty else {
            return 0
        }
        let headerData = try Self.makeHeaderData(
            tensors: tensors,
            metadata: header.metadata
        )
        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(headerData.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: headerData)
        for tensor in tensors {
            try copyTensor(named: tensor.name, to: output)
        }
        try output.synchronize()
        return tensors.count
    }

    private func absoluteOffset(_ info: SafetensorInfo, name: String) throws -> UInt64 {
        let (offset, overflow) = header.dataBaseOffset.addingReportingOverflow(
            UInt64(info.dataStart)
        )
        guard !overflow else {
            throw MLXFastError.invalidInput(
                "generation source tensor offset overflows for \(name)"
            )
        }
        return offset
    }

    private func validateIdentity() throws {
        guard try Self.fileIdentity(handle) == identity else {
            throw MLXFastError.invalidInput(
                "generation source descriptor changed while being read"
            )
        }
    }

    private struct Capture {
        let name: String
        let start: UInt64
        let end: UInt64
        var bytes: Data
    }

    private static func captureTrustedTensors(
        handle: FileHandle,
        expectedHeader: SafetensorsHeader,
        tensorNames: [String],
        trustedSHA256: String,
        fileByteCount: Int
    ) throws -> [String: Data] {
        try handle.seek(toOffset: 0)
        var hasher = SHA256()
        let prefix = try readExactly(handle, count: 8)
        hasher.update(data: prefix)
        let rawHeaderLength = prefix.withUnsafeBytes { raw in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard rawHeaderLength > 0,
              rawHeaderLength <= UInt64(Safetensors.maximumHeaderByteCount),
              let headerLength = Int(exactly: rawHeaderLength),
              headerLength <= fileByteCount - 8
        else {
            throw MLXFastError.invalidInput(
                "pinned generation source has an invalid safetensors header"
            )
        }
        let headerData = try readExactly(handle, count: headerLength)
        hasher.update(data: headerData)
        let header = try parseHeader(
            headerData,
            headerLength: headerLength,
            fileByteCount: fileByteCount
        )
        guard header == expectedHeader else {
            throw MLXFastError.invalidInput(
                "pinned generation source header changed during validation"
            )
        }

        var captures: [Capture] = try tensorNames.map { name in
            guard let info = header.tensors[name] else {
                throw MLXFastError.invalidInput(
                    "trusted generation source tensor is missing: \(name)"
                )
            }
            let (start, startOverflow) = header.dataBaseOffset.addingReportingOverflow(
                UInt64(info.dataStart)
            )
            let (end, endOverflow) = header.dataBaseOffset.addingReportingOverflow(
                UInt64(info.dataEnd)
            )
            guard !startOverflow, !endOverflow, end >= start else {
                throw MLXFastError.invalidInput(
                    "trusted generation source tensor range overflows: \(name)"
                )
            }
            var bytes = Data()
            bytes.reserveCapacity(info.byteCount)
            return Capture(name: name, start: start, end: end, bytes: bytes)
        }
        captures.sort { $0.start < $1.start }

        var fileOffset = header.dataBaseOffset
        while true {
            let chunk = handle.readData(ofLength: 8 * 1024 * 1024)
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
            let chunkStart = fileOffset
            let (chunkEnd, overflow) = chunkStart.addingReportingOverflow(UInt64(chunk.count))
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "trusted generation source byte count overflows"
                )
            }
            for index in captures.indices {
                let overlapStart = max(chunkStart, captures[index].start)
                let overlapEnd = min(chunkEnd, captures[index].end)
                guard overlapStart < overlapEnd else {
                    continue
                }
                let lower = Int(overlapStart - chunkStart)
                let upper = Int(overlapEnd - chunkStart)
                captures[index].bytes.append(contentsOf: chunk[lower..<upper])
            }
            fileOffset = chunkEnd
        }

        guard fileOffset == UInt64(fileByteCount) else {
            throw MLXFastError.invalidInput(
                "pinned generation source changed length while being captured"
            )
        }
        let actualSHA256 = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard actualSHA256 == trustedSHA256.lowercased() else {
            throw MLXFastError.invalidInput(
                "generation source content does not match the pinned checkpoint"
            )
        }
        for capture in captures {
            guard let info = header.tensors[capture.name],
                  capture.bytes.count == info.byteCount
            else {
                throw MLXFastError.invalidInput(
                    "trusted generation source tensor is incomplete: \(capture.name)"
                )
            }
        }
        return Dictionary(uniqueKeysWithValues: captures.map { ($0.name, $0.bytes) })
    }

    private static func readExactly(_ handle: FileHandle, count: Int) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            let chunk = handle.readData(ofLength: count - result.count)
            guard !chunk.isEmpty else {
                throw MLXFastError.invalidInput(
                    "pinned generation source ended unexpectedly"
                )
            }
            result.append(chunk)
        }
        return result
    }

    private static func parseHeader(
        _ data: Data,
        headerLength: Int,
        fileByteCount: Int
    ) throws -> SafetensorsHeader {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any] else {
            throw MLXFastError.invalidInput(
                "pinned generation source header is not an object"
            )
        }
        let dataByteCount = fileByteCount - 8 - headerLength
        var metadata: [String: String] = [:]
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary {
            if name == "__metadata__" {
                metadata = value as? [String: String] ?? [:]
                continue
            }
            guard let tensor = value as? [String: Any],
                  let dtype = tensor["dtype"] as? String,
                  let shapeValues = tensor["shape"] as? [NSNumber],
                  let offsetValues = tensor["data_offsets"] as? [NSNumber],
                  offsetValues.count == 2
            else {
                throw MLXFastError.invalidInput(
                    "pinned generation source has invalid tensor metadata for \(name)"
                )
            }
            let shape = try shapeValues.map { value in
                guard let result = Int(value.stringValue), result >= 0 else {
                    throw MLXFastError.invalidInput(
                        "pinned generation source has invalid shape for \(name)"
                    )
                }
                return result
            }
            guard let start = Int(offsetValues[0].stringValue),
                  let end = Int(offsetValues[1].stringValue),
                  start >= 0,
                  end >= start,
                  end <= dataByteCount
            else {
                throw MLXFastError.invalidInput(
                    "pinned generation source has invalid offsets for \(name)"
                )
            }
            tensors[name] = SafetensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataStart: start,
                dataEnd: end
            )
        }
        return SafetensorsHeader(
            headerLength: headerLength,
            metadata: metadata,
            tensors: tensors
        )
    }

    private static func makeHeaderData(
        tensors: [SafetensorInfo],
        metadata: [String: String]
    ) throws -> Data {
        var object: [String: Any] = [:]
        if !metadata.isEmpty {
            object["__metadata__"] = metadata
        }
        var cursor = 0
        for tensor in tensors {
            let (end, overflow) = cursor.addingReportingOverflow(tensor.byteCount)
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "trusted subset offsets overflow for \(tensor.name)"
                )
            }
            object[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while !data.count.isMultiple(of: 8) {
            data.append(0x20)
        }
        return data
    }

    private static func fileIdentity(_ handle: FileHandle) throws -> Identity {
        var status = stat()
        guard Darwin.fstat(handle.fileDescriptor, &status) == 0,
              (status.st_mode & S_IFMT) == S_IFREG,
              status.st_size >= 0,
              let byteCount = Int(exactly: status.st_size)
        else {
            throw MLXFastError.invalidInput(
                "could not inspect generation source descriptor"
            )
        }
        return Identity(
            device: UInt64(status.st_dev),
            inode: UInt64(status.st_ino),
            byteCount: byteCount,
            modificationSeconds: Int64(status.st_mtimespec.tv_sec),
            modificationNanoseconds: Int64(status.st_mtimespec.tv_nsec),
            statusChangeSeconds: Int64(status.st_ctimespec.tv_sec),
            statusChangeNanoseconds: Int64(status.st_ctimespec.tv_nsec),
            linkCount: UInt64(status.st_nlink)
        )
    }
}

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
    public let denseShardCount: Int
    public let configPath: String
    public let indexPath: String

    public init(
        referencePath: String,
        outputPath: String,
        denseTensorCount: Int,
        denseShardCount: Int,
        configPath: String,
        indexPath: String
    ) {
        self.referencePath = referencePath
        self.outputPath = outputPath
        self.denseTensorCount = denseTensorCount
        self.denseShardCount = denseShardCount
        self.configPath = configPath
        self.indexPath = indexPath
    }
}

/// Offline transform for the Gemma 4 31B 4-bit checkpoint: selects ONLY the
/// text-tower tensors (the `language_model.` prefix in the source index),
/// drops every vision/audio/multimodal-projector tensor, and rewrites the
/// selected tensors into dense safetensors shard(s) plus a
/// `model.safetensors.index.json` and a runtime-authored `config.json`
/// (the flattened `text_config` fields the runtime needs, plus the
/// checkpoint's quantization metadata). Text/audio/vision are the only kinds
/// of tensors the reference checkpoint ships; there is no expert manifest --
/// the whole selected tree is one flat set of dense tensors, matching how a
/// single dense model (no MoE, no expert streaming) is loaded fully into RAM
/// at runtime init.
public enum SwiftTransform {
    /// Tensor name prefix that marks a checkpoint tensor as part of the text
    /// tower. Every other prefix (`vision_tower.`, `embed_vision.`,
    /// `audio_tower.`, `multi_modal_projector.`, ...) is vision/audio/
    /// multimodal-glue and is out of scope for this text-only challenge.
    static let textTowerPrefix = "language_model."

    private struct GenerationSourceSnapshot {
        let files: [String: TransformSourceFile]
    }

    private static let pinnedProductionShardSHA256: [String: String] = [
        "model-00001-of-00004.safetensors":
            "48deb52fe002cd2bbc0b49c02c152b93ad3255de5e7335faa0d2afd6f4728e7f",
        "model-00002-of-00004.safetensors":
            "125c49c624651d420a09786dffb1934d4177e422da0cb3a95a089119eb8f70c2",
        "model-00003-of-00004.safetensors":
            "bdbfe71f055e388bf04c5e8953524f948457ece60c5f9bb39025bca731bb5fc8",
        "model-00004-of-00004.safetensors":
            "0215f14cd483ee8e589e7a1a3807afa22e7e12e5d4c22d1b373928002cd511ad",
    ]

    public static func run(_ options: TransformOptions) throws -> TransformReport {
        try run(
            options,
            beforeSidecarGeneration: nil,
            beforeSourceRevalidation: nil
        )
    }

    static func run(
        _ options: TransformOptions,
        beforeSidecarGeneration: (() throws -> Void)? = nil,
        beforeSourceRevalidation: (() throws -> Void)?
    ) throws -> TransformReport {
        let referenceDirectory = canonicalURL(
            try findReferenceDirectory(URL(fileURLWithPath: options.referencePath))
        )
        let outputDirectory = canonicalURL(URL(fileURLWithPath: options.outputPath))
        try validateDistinctDirectories(
            referenceDirectory: referenceDirectory,
            outputDirectory: outputDirectory
        )
        var outputIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: outputDirectory.path,
            isDirectory: &outputIsDirectory
        ), !outputIsDirectory.boolValue {
            throw MLXFastError.invalidInput(
                "transform output exists and is not a directory: \(outputDirectory.path)"
            )
        }

        let referenceConfigPath = referenceDirectory.appendingPathComponent("config.json")
        try requireFile(
            referenceConfigPath.path,
            description: "Gemma 4 31B 4-bit reference config"
        )
        let runtimeConfigData = try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
        let metadataSnapshot = try captureMetadataFiles(from: referenceDirectory)

        let index = try loadIndex(referenceDirectory)
        let indexSnapshot = try index.canonicalData()
        let validatedHeaders = try validateCheckpointIndex(
            index,
            referenceDirectory: referenceDirectory
        )
        let textKeys = Set(index.weightMap.keys.filter(isTextTowerKey))
        guard !textKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no text-tower tensors")
        }

        var totalTensorByteCount = 0
        for key in textKeys.sorted() {
            guard let shardName = index.weightMap[key],
                  let info = validatedHeaders[shardName]?.tensors[key]
            else {
                throw MLXFastError.invalidInput(
                    "missing validated tensor metadata for \(key)"
                )
            }
            let (nextTotal, overflow) = totalTensorByteCount.addingReportingOverflow(
                info.byteCount
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "transformed tensor byte count overflows Int"
                )
            }
            totalTensorByteCount = nextTotal
        }

        let fileManager = FileManager.default
        let stagingDirectory = outputDirectory.deletingLastPathComponent().appendingPathComponent(
            ".\(outputDirectory.lastPathComponent).mlxfast-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: false)
        var installed = false
        defer {
            if !installed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        let combinedSourceKeys = CombinedProjectionTransform.sourcePayloadKeys(
            selectedKeys: textKeys
        )
        let generationSourceKeys = AffineMetadataCoding.sourcePayloadKeys(
            selectedKeys: textKeys
        ).union(combinedSourceKeys)
        let requirePinnedContent = !combinedSourceKeys.isEmpty
        let generationSnapshot = try openGenerationSources(
            sourceDirectory: referenceDirectory,
            index: index,
            sourceHeaders: validatedHeaders,
            sourceKeys: requirePinnedContent ? textKeys : generationSourceKeys,
            requirePinnedContent: requirePinnedContent
        )

        let copiedKeys = textKeys.subtracting(combinedSourceKeys)
        let copiedKeysByShard = Dictionary(grouping: copiedKeys) { key in
            index.weightMap[key] ?? ""
        }

        var copiedTensors = 0
        for shardName in copiedKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = stagingDirectory.appendingPathComponent(shardName)
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput("missing validated header for checkpoint shard \(shardName)")
            }
            let tensorNames = copiedKeysByShard[shardName, default: []].sorted()
            if requirePinnedContent {
                guard let sourceFile = generationSnapshot.files[shardName] else {
                    throw MLXFastError.invalidInput(
                        "missing pinned production source shard \(shardName)"
                    )
                }
                copiedTensors += try sourceFile.writeSubset(
                    to: destination,
                    tensorNames: tensorNames
                )
            } else {
                copiedTensors += try Safetensors.copySubset(
                    from: source,
                    to: destination,
                    tensorNames: tensorNames,
                    validatedHeader: header
                )
            }
        }

        try beforeSidecarGeneration?()
        let generatedMetadata = try AffineMetadataCoding.writeProjectionSidecar(
            index: index,
            sourceFiles: generationSnapshot.files,
            metadataHeaders: validatedHeaders,
            selectedKeys: textKeys,
            destinationDirectory: stagingDirectory
        )
        let combined = try CombinedProjectionTransform.writeCombinedShard(
            index: index,
            sourceFiles: generationSnapshot.files,
            selectedKeys: textKeys,
            destinationDirectory: stagingDirectory
        )
        guard combined.prunedKeys == combinedSourceKeys else {
            throw MLXFastError.invalidInput(
                "combined projection source inventory changed during transform"
            )
        }
        let retainedTensorByteCount = totalTensorByteCount
            .subtractingReportingOverflow(combined.sourceTensorByteCount)
        guard !retainedTensorByteCount.overflow else {
            throw MLXFastError.invalidInput("transformed tensor byte count underflows Int")
        }
        let physicalTensorByteCount = retainedTensorByteCount.partialValue
            .addingReportingOverflow(combined.tensorByteCount)
        guard !physicalTensorByteCount.overflow,
              physicalTensorByteCount.partialValue == totalTensorByteCount
        else {
            throw MLXFastError.invalidInput(
                "combined transform changed tensor payload byte accounting"
            )
        }
        let generatedWeightMap = generatedMetadata.weightMap.merging(
            combined.weightMap
        ) { key, _ in key }
        guard generatedWeightMap.count
                == generatedMetadata.weightMap.count + combined.weightMap.count
        else {
            throw MLXFastError.invalidInput(
                "combined projection tensor names collide with indexed metadata"
            )
        }
        let (outputTensorByteCount, generatedSizeOverflow) =
            physicalTensorByteCount.partialValue.addingReportingOverflow(
                generatedMetadata.tensorByteCount
            )
        guard !generatedSizeOverflow else {
            throw MLXFastError.invalidInput("transformed tensor byte count overflows Int")
        }

        try writeMetadataFiles(metadataSnapshot, to: stagingDirectory)
        try index.writeStripped(
            to: stagingDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: copiedKeys,
            totalTensorByteCount: outputTensorByteCount,
            additionalWeightMap: generatedWeightMap,
            additionalMetadata: combined.indexMetadata
        )

        try runtimeConfigData.write(
            to: stagingDirectory.appendingPathComponent("config.json")
        )
        try beforeSourceRevalidation?()
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        for shardName in validatedHeaders.keys.sorted() {
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput(
                    "missing validated header for checkpoint shard \(shardName)"
                )
            }
            try Safetensors.validateSourceIdentity(
                referenceDirectory.appendingPathComponent(shardName),
                against: header
            )
        }
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        try installTransformedDirectory(
            stagingDirectory,
            at: outputDirectory,
            fileManager: fileManager
        )
        installed = true

        let indexPath = outputDirectory.appendingPathComponent("model.safetensors.index.json")
        let configPath = outputDirectory.appendingPathComponent("config.json")

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedTensors + generatedMetadata.tensorCount
                + combined.tensorCount,
            denseShardCount: copiedKeysByShard.count + generatedMetadata.shardCount
                + combined.shardCount,
            configPath: configPath.path,
            indexPath: indexPath.path
        )
    }

    private static func openGenerationSources(
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader],
        sourceKeys: Set<String>,
        requirePinnedContent: Bool
    ) throws -> GenerationSourceSnapshot {
        let keysByShard = Dictionary(grouping: sourceKeys) { key in
            index.weightMap[key] ?? ""
        }
        var files: [String: TransformSourceFile] = [:]
        do {
            for shardName in keysByShard.keys.sorted() {
                guard !shardName.isEmpty,
                      let sourceHeader = sourceHeaders[shardName]
                else {
                    throw MLXFastError.invalidInput(
                        "missing validated generation source shard"
                    )
                }
                let tensorNames = keysByShard[shardName, default: []].sorted()
                let trustedSHA256 = pinnedProductionShardSHA256[shardName]
                if requirePinnedContent, trustedSHA256 == nil {
                    throw MLXFastError.invalidInput(
                        "production generation source shard is not pinned: \(shardName)"
                    )
                }
                files[shardName] = try TransformSourceFile(
                    path: sourceDirectory.appendingPathComponent(shardName),
                    expectedHeader: sourceHeader,
                    tensorNames: tensorNames,
                    trustedSHA256: requirePinnedContent ? trustedSHA256 : nil
                )
            }
        } catch {
            files.removeAll()
            throw error
        }
        return GenerationSourceSnapshot(files: files)
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
        referenceDirectory: URL
    ) throws -> [String: SafetensorsHeader] {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shardName] = header

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
                // readHeader validated every data range against the opened
                // target descriptor. Do not recheck via the shard pathname:
                // FileManager reports a symlink's own size, not its target's.
                guard info.dataStart >= 0, info.byteCount > 0 else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) has an empty or invalid byte range"
                    )
                }
            }
        }
        return headersByShard
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validateDistinctDirectories(
        referenceDirectory: URL,
        outputDirectory: URL,
        workingDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
    ) throws {
        guard referenceDirectory.path != outputDirectory.path else {
            throw MLXFastError.invalidInput(
                "transform reference and output directories must be different: \(referenceDirectory.path)"
            )
        }
        let outputPrefix = outputDirectory.path == "/" ? "/" : outputDirectory.path + "/"
        guard !referenceDirectory.path.hasPrefix(outputPrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the reference directory: \(outputDirectory.path)"
            )
        }
        let referencePrefix = referenceDirectory.path == "/"
            ? "/"
            : referenceDirectory.path + "/"
        guard !outputDirectory.path.hasPrefix(referencePrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot be inside the reference directory: \(outputDirectory.path)"
            )
        }

        let canonicalWorkingDirectory = canonicalURL(workingDirectory)
        guard canonicalWorkingDirectory.path != outputDirectory.path,
              !canonicalWorkingDirectory.path.hasPrefix(outputPrefix)
        else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the current working directory: \(outputDirectory.path)"
            )
        }
    }

    private static func installTransformedDirectory(
        _ stagedDirectory: URL,
        at outputDirectory: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: outputDirectory.path) {
            _ = try fileManager.replaceItemAt(outputDirectory, withItemAt: stagedDirectory)
        } else {
            try fileManager.moveItem(at: stagedDirectory, to: outputDirectory)
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
            "no config.json found under \(base.path); place the Gemma 4 31B 4-bit checkpoint there"
        )
    }

    static func isTextTowerKey(_ key: String) -> Bool {
        key.hasPrefix(textTowerPrefix)
    }

    private static func captureMetadataFiles(from source: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        var snapshot: [String: Data] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" || file.lastPathComponent == "config.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let values = try file.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
                )
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw MLXFastError.invalidInput(
                        "reference metadata is not a regular file: \(file.path)"
                    )
                }
                snapshot[file.lastPathComponent] = try Data(contentsOf: file)
            }
        }
        return snapshot
    }

    private static func writeMetadataFiles(
        _ snapshot: [String: Data],
        to destination: URL
    ) throws {
        for name in snapshot.keys.sorted() {
            try snapshot[name]?.write(to: destination.appendingPathComponent(name))
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

    /// Writes the runtime's `config.json`: the source checkpoint's
    /// `text_config` fields flattened to the top level (the exact schema
    /// `Gemma4Config.load` reads), plus the checkpoint-wide `quantization`
    /// block. The runtime controls this schema directly, so it carries only
    /// what `Gemma4Config`/`Gemma4WeightLoader` actually need -- no vision or
    /// audio config, no architecture/tokenizer metadata duplicated from
    /// `tokenizer_config.json`.
    static func makeRuntimeConfigData(sourceConfigPath: URL) throws -> Data {
        let data = try Data(contentsOf: sourceConfigPath)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json must be a JSON object")
        }
        guard let textConfig = root["text_config"] as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json is missing text_config")
        }

        var runtimeConfig = textConfig
        if let quantization = root["quantization"] {
            runtimeConfig["quantization"] = quantization
        } else if let quantizationConfig = root["quantization_config"] {
            runtimeConfig["quantization"] = quantizationConfig
        }

        return try JSONSerialization.data(
            withJSONObject: runtimeConfig,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    private static func validateConfigAndIndexSnapshot(
        referenceDirectory: URL,
        referenceConfigPath: URL,
        runtimeConfigData: Data,
        indexSnapshot: Data,
        metadataSnapshot: [String: Data]
    ) throws {
        guard try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
            == runtimeConfigData
        else {
            throw MLXFastError.invalidInput(
                "reference config changed while transform was running"
            )
        }
        guard try loadIndex(referenceDirectory).canonicalData() == indexSnapshot else {
            throw MLXFastError.invalidInput(
                "checkpoint index changed while transform was running"
            )
        }
        guard try captureMetadataFiles(from: referenceDirectory) == metadataSnapshot else {
            throw MLXFastError.invalidInput(
                "reference tokenizer metadata changed while transform was running"
            )
        }
    }
}
