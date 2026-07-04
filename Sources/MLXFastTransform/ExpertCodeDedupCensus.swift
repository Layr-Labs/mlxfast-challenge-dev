import Foundation
import MLXFastCore

struct ExpertCodeDedupCensus {
    static let defaultBlockSize = 64 * 1024

    static func run(
        referenceDirectory: URL,
        outputDirectory: URL,
        expertKeysByShard: [String: [String]],
        blockSize: Int = defaultBlockSize
    ) throws {
        var census = CensusState(referenceDirectory: referenceDirectory, blockSize: blockSize)

        for shardName in expertKeysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            let handle = try FileHandle(forReadingFrom: shardURL)
            defer { try? handle.close() }
            let shardID = census.internShard(shardName)

            for key in expertKeysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else { continue }
                guard isEligibleExpertCodeTensor(key: key, info: info) else { continue }
                let tensorID = census.internTensor(key)
                let expertCount = info.shape[0]
                let sliceByteLength = info.byteCount / expertCount
                let tensorDataOffset = header.dataBaseOffset + UInt64(info.dataStart)
                for expertIndex in 0..<expertCount {
                    let sliceOffset = tensorDataOffset + UInt64(expertIndex * sliceByteLength)
                    try census.scanSlice(
                        handle: handle,
                        shardID: shardID,
                        tensorID: tensorID,
                        expertIndex: expertIndex,
                        sliceAbsoluteOffset: sliceOffset,
                        sliceByteLength: sliceByteLength
                    )
                }
            }
        }

        let report = census.makeReport()
        let reportURL = outputDirectory
            .appendingPathComponent("experts", isDirectory: true)
            .appendingPathComponent("exact-dedup-census.json")
        let data = try JSONSerialization.data(
            withJSONObject: report.jsonObject,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: reportURL)

        let exactRatio = String(format: "%.6f", report.duplicateRatioExact)
        let realizableRatio = String(format: "%.6f", report.realizableSavedRatio)
        print(
            "exact_dedup_census eligible=\(report.eligibleBytes) duplicate=\(report.duplicateBytesExact) exact_ratio=\(exactRatio) realizable=\(report.realizableSavedBytes) realizable_ratio=\(realizableRatio) full_slice_aliases=\(report.fullSliceAliases) block_size=\(blockSize)"
        )
    }

    private static func isEligibleExpertCodeTensor(key: String, info: SafetensorInfo) -> Bool {
        guard info.dtype == "U32" else { return false }
        guard SwiftTransform.isExpertKey(key) else { return false }
        if key.contains(".scales") || key.contains(".biases") || key.hasSuffix("scales") || key.hasSuffix("biases") {
            return false
        }
        guard info.shape.count == 3, info.shape[0] > 1 else { return false }
        guard info.byteCount % info.shape[0] == 0 else { return false }
        return true
    }
}

private struct CensusReport {
    let eligibleBytes: Int64
    let duplicateBytesExact: Int64
    let duplicateRatioExact: Double
    let fullSliceAliases: Int
    let realizableSavedBytes: Int64
    let realizableSavedRatio: Double
    let jsonObject: [String: Any]
}

private struct CensusState {
    let referenceDirectory: URL
    let blockSize: Int

    var shardNames: [String] = []
    var shardIDs: [String: Int] = [:]
    var tensorNames: [String] = []
    var tensorIDs: [String: Int] = [:]

    var nextSliceSerial: Int = 0
    var eligibleBytes: Int64 = 0
    var duplicateBytesExact: Int64 = 0
    var intraSliceDuplicateBytes: Int64 = 0
    var crossSlicePartialDuplicateBytes: Int64 = 0
    var fullSliceAliases: Int = 0
    var fullSliceAliasBytes: Int64 = 0

    var canonicalBlocks: [CanonicalBlock] = []
    var firstBlockByDigest: [BlockDigestKey: Int] = [:]
    var collisionBlocksByDigest: [BlockDigestKey: [Int]] = [:]
    var duplicateMetadata: [Int: DuplicateMetadata] = [:]
    var sliceAliases: [SliceSignature: SliceLocation] = [:]
    var confirmHandles = FileHandleCache()

    mutating func internShard(_ name: String) -> Int {
        if let id = shardIDs[name] { return id }
        let id = shardNames.count
        shardNames.append(name)
        shardIDs[name] = id
        return id
    }

    mutating func internTensor(_ name: String) -> Int {
        if let id = tensorIDs[name] { return id }
        let id = tensorNames.count
        tensorNames.append(name)
        tensorIDs[name] = id
        return id
    }

    mutating func scanSlice(
        handle: FileHandle,
        shardID: Int,
        tensorID: Int,
        expertIndex: Int,
        sliceAbsoluteOffset: UInt64,
        sliceByteLength: Int
    ) throws {
        eligibleBytes += Int64(sliceByteLength)
        let sliceSerial = nextSliceSerial
        nextSliceSerial += 1
        let blockCount = (sliceByteLength + blockSize - 1) / blockSize
        var signatureBlocks: [Int] = []
        signatureBlocks.reserveCapacity(blockCount)

        try handle.seek(toOffset: sliceAbsoluteOffset)
        var remaining = sliceByteLength
        var blockOrdinal = 0
        while remaining > 0 {
            try autoreleasepool {
                let length = min(blockSize, remaining)
                let data = handle.readData(ofLength: length)
                guard data.count == length else {
                    throw MLXFastError.invalidInput(
                        "unexpected EOF while scanning exact dedup block for \(tensorNames[tensorID]) expert \(expertIndex)"
                    )
                }
                let location = BlockLocation(
                    shardID: shardID,
                    tensorID: tensorID,
                    expertIndex: expertIndex,
                    sliceAbsoluteOffset: sliceAbsoluteOffset,
                    blockOrdinal: blockOrdinal,
                    absoluteOffset: sliceAbsoluteOffset + UInt64(blockOrdinal * blockSize),
                    length: length
                )
                let canonicalID = try canonicalID(for: data, location: location, sliceSerial: sliceSerial)
                signatureBlocks.append(canonicalID)
                remaining -= length
                blockOrdinal += 1
            }
        }

        let signature = SliceSignature(blocks: signatureBlocks)
        if let previous = sliceAliases[signature] {
            fullSliceAliases += 1
            fullSliceAliasBytes += Int64(sliceByteLength)
            if let firstBlock = signatureBlocks.first {
                duplicateMetadata[firstBlock, default: DuplicateMetadata()].aliasExamples.appendIfRoom(
                    SliceLocation(shardID: shardID, tensorID: tensorID, expertIndex: expertIndex, sliceAbsoluteOffset: sliceAbsoluteOffset, byteLength: sliceByteLength),
                    limit: 3
                )
            }
            _ = previous
        } else {
            sliceAliases[signature] = SliceLocation(
                shardID: shardID,
                tensorID: tensorID,
                expertIndex: expertIndex,
                sliceAbsoluteOffset: sliceAbsoluteOffset,
                byteLength: sliceByteLength
            )
        }
    }

    private mutating func canonicalID(for data: Data, location: BlockLocation, sliceSerial: Int) throws -> Int {
        let digest = deterministicDigest(data)
        let key = BlockDigestKey(length: data.count, hash1: digest.0, hash2: digest.1)
        if let firstID = firstBlockByDigest[key] {
            var candidateIDs = [firstID]
            if let collisionIDs = collisionBlocksByDigest[key] {
                candidateIDs.append(contentsOf: collisionIDs)
            }
            for id in candidateIDs {
                let canonicalData = try canonicalBytes(for: canonicalBlocks[id])
                if canonicalData == data {
                    canonicalBlocks[id].count += 1
                    canonicalBlocks[id].duplicateBytes += Int64(data.count)
                    duplicateMetadata[id, default: DuplicateMetadata()].examples.appendIfRoom(location, limit: 4)
                    duplicateBytesExact += Int64(data.count)
                    if canonicalBlocks[id].firstSliceSerial == sliceSerial {
                        intraSliceDuplicateBytes += Int64(data.count)
                    } else {
                        crossSlicePartialDuplicateBytes += Int64(data.count)
                    }
                    return id
                }
            }
            let id = appendCanonicalBlock(key: key, location: location, sliceSerial: sliceSerial)
            collisionBlocksByDigest[key, default: []].append(id)
            return id
        }

        let id = appendCanonicalBlock(key: key, location: location, sliceSerial: sliceSerial)
        firstBlockByDigest[key] = id
        return id
    }

    private mutating func appendCanonicalBlock(key: BlockDigestKey, location: BlockLocation, sliceSerial: Int) -> Int {
        let id = canonicalBlocks.count
        canonicalBlocks.append(
            CanonicalBlock(
                key: key,
                firstLocation: location,
                firstSliceSerial: sliceSerial,
                count: 1,
                duplicateBytes: 0
            )
        )
        return id
    }

    private mutating func canonicalBytes(for block: CanonicalBlock) throws -> Data {
        let location = block.firstLocation
        let url = referenceDirectory.appendingPathComponent(shardNames[location.shardID])
        let handle = try confirmHandles.handle(for: url)
        return try readExact(handle: handle, offset: location.absoluteOffset, length: location.length)
    }

    func makeReport() -> CensusReport {
        let realizableSavedBytes = intraSliceDuplicateBytes + fullSliceAliasBytes
        let duplicateRatio = eligibleBytes > 0 ? Double(duplicateBytesExact) / Double(eligibleBytes) : 0
        let realizableRatio = eligibleBytes > 0 ? Double(realizableSavedBytes) / Double(eligibleBytes) : 0
        let top = canonicalBlocks.enumerated()
            .filter { $0.element.count > 1 }
            .sorted {
                if $0.element.duplicateBytes == $1.element.duplicateBytes { return $0.element.count > $1.element.count }
                return $0.element.duplicateBytes > $1.element.duplicateBytes
            }
            .prefix(50)
            .map { id, block -> [String: Any] in
                var object: [String: Any] = [
                    "digest": block.key.digestString,
                    "block_length": block.key.length,
                    "count": block.count,
                    "duplicate_bytes": block.duplicateBytes,
                    "total_bytes": Int64(block.count * block.key.length),
                    "first": exampleObject(block.firstLocation),
                    "examples": duplicateMetadata[id]?.examples.map { exampleObject($0) } ?? [],
                ]
                if let aliases = duplicateMetadata[id]?.aliasExamples, !aliases.isEmpty {
                    object["example_full_slice_aliases"] = aliases.map { exampleObject($0) }
                }
                return object
            }

        let json: [String: Any] = [
            "block_size": blockSize,
            "eligible_bytes": eligibleBytes,
            "duplicate_bytes_exact": duplicateBytesExact,
            "duplicate_ratio_exact": duplicateRatio,
            "intra_slice_duplicate_bytes": intraSliceDuplicateBytes,
            "cross_slice_partial_duplicate_bytes": crossSlicePartialDuplicateBytes,
            "full_slice_aliases": fullSliceAliases,
            "full_slice_alias_bytes": fullSliceAliasBytes,
            "realizable_saved_bytes": realizableSavedBytes,
            "realizable_saved_ratio": realizableRatio,
            "top_duplicated_blocks": Array(top),
            "realizable_policy": "full-slice aliases plus intra-slice duplicate blocks only; cross-slice partial matches are reported but excluded to avoid random-read-heavy layouts",
        ]

        return CensusReport(
            eligibleBytes: eligibleBytes,
            duplicateBytesExact: duplicateBytesExact,
            duplicateRatioExact: duplicateRatio,
            fullSliceAliases: fullSliceAliases,
            realizableSavedBytes: realizableSavedBytes,
            realizableSavedRatio: realizableRatio,
            jsonObject: json
        )
    }

    private func exampleObject(_ location: BlockLocation) -> [String: Any] {
        [
            "tensor": tensorNames[location.tensorID],
            "expert_index": location.expertIndex,
            "shard": shardNames[location.shardID],
            "slice_absolute_offset": Int64(location.sliceAbsoluteOffset),
            "block_ordinal": location.blockOrdinal,
            "absolute_offset": Int64(location.absoluteOffset),
            "block_length": location.length,
        ]
    }

    private func exampleObject(_ location: SliceLocation) -> [String: Any] {
        [
            "tensor": tensorNames[location.tensorID],
            "expert_index": location.expertIndex,
            "shard": shardNames[location.shardID],
            "slice_absolute_offset": Int64(location.sliceAbsoluteOffset),
            "byte_length": location.byteLength,
        ]
    }
}

private final class FileHandleCache {
    private var handles: [String: FileHandle] = [:]

    deinit {
        for handle in handles.values { try? handle.close() }
    }

    func handle(for url: URL) throws -> FileHandle {
        let path = url.path
        if let existing = handles[path] { return existing }
        let handle = try FileHandle(forReadingFrom: url)
        handles[path] = handle
        return handle
    }
}

private struct BlockDigestKey: Hashable {
    let length: Int
    let hash1: UInt64
    let hash2: UInt64

    var digestString: String {
        String(format: "%016llx%016llx", hash1, hash2)
    }
}

private struct SliceSignature: Hashable {
    let blocks: [Int]
}

private struct SliceLocation: Hashable {
    let shardID: Int
    let tensorID: Int
    let expertIndex: Int
    let sliceAbsoluteOffset: UInt64
    let byteLength: Int
}

private struct BlockLocation {
    let shardID: Int
    let tensorID: Int
    let expertIndex: Int
    let sliceAbsoluteOffset: UInt64
    let blockOrdinal: Int
    let absoluteOffset: UInt64
    let length: Int
}

private struct CanonicalBlock {
    let key: BlockDigestKey
    let firstLocation: BlockLocation
    let firstSliceSerial: Int
    var count: Int
    var duplicateBytes: Int64
}

private struct DuplicateMetadata {
    var examples: [BlockLocation] = []
    var aliasExamples: [SliceLocation] = []
}

private extension Array {
    mutating func appendIfRoom(_ element: Element, limit: Int) {
        if count < limit { append(element) }
    }
}

private func readExact(handle: FileHandle, offset: UInt64, length: Int) throws -> Data {
    try handle.seek(toOffset: offset)
    let data = handle.readData(ofLength: length)
    guard data.count == length else {
        throw MLXFastError.invalidInput("unexpected EOF while reading canonical exact-dedup block")
    }
    return data
}

private func deterministicDigest(_ data: Data) -> (UInt64, UInt64) {
    var hash1: UInt64 = 0xcbf29ce484222325
    var hash2: UInt64 = 0x84222325cbf29ce4
    let prime1: UInt64 = 0x100000001b3
    let prime2: UInt64 = 0x100000001d3

    data.withUnsafeBytes { rawBuffer in
        guard let base = rawBuffer.baseAddress else { return }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for i in 0..<rawBuffer.count {
            let byte = UInt64(bytes[i])
            hash1 ^= byte
            hash1 = hash1 &* prime1
            hash2 ^= byte &+ UInt64((i & 0xff) + 1)
            hash2 = (hash2 &* prime2).rotatedLeft(13)
        }
    }
    hash1 ^= UInt64(data.count)
    hash2 ^= UInt64(data.count) &* 0x9e3779b185ebca87
    return (hash1, hash2)
}

private extension UInt64 {
    func rotatedLeft(_ amount: UInt64) -> UInt64 {
        (self << amount) | (self >> (64 - amount))
    }
}
