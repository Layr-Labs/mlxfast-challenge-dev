import Foundation
import MLXFastCore

/// Transform-side, runtime-invisible census for exact MXFP4 code-byte group elision.
///
/// MXFP4 routed-expert code tensors store four UInt32 code words for every U8
/// scale group.  This scanner classifies those exact 16-byte / 32-nibble groups
/// and estimates a small random-access superblock representation, but does not
/// alter any emitted runtime-visible weights or manifests.
enum MXFP4GroupElisionCensus {
    static let groupsPerSuperblock = 256
    private static let bytesPerGroup = 16
    private static let codeWordsPerScaleGroup = 4
    private static let readChunkBytes = 4 * 1024 * 1024

    struct Report: Codable {
        var version: Int
        var groupsPerSuperblock: Int
        var eligibleCodeBytes: Int64
        var eligibleTensors: Int
        var eligibleGroups: Int64
        var eligibleSuperblocks: Int64
        var skippedTensors: [SkippedTensor]
        var zeroGroups: Int64
        var splatNibbleGroups: Int64
        var twoSymbolGroups: Int64
        var rawGroups: Int64
        var zeroGroupByteShare: Double
        var splatNibbleGroupByteShare: Double
        var twoSymbolGroupByteShare: Double
        var rawGroupByteShare: Double
        var compressedProfitableSuperblocks: Int64
        var totalSuperblocks: Int64
        var estimatedRawBytes: Int64
        var estimatedBytesAfterGroupCoding: Int64
        var estimatedBytesAfterGroupCodingNoIndex: Int64
        var estimatedIndexAndHeaderOverheadBytes: Int64
        var estimatedSavedBytes: Int64
        var estimatedSavedRatio: Double
        var optimisticSavedBytesNoIndex: Int64
        var optimisticSavedRatioNoIndex: Double
        var topTensorsBySavedBytes: [TensorSummary]
        var runtimeIntegrationRecommended: Bool
    }

    struct SkippedTensor: Codable {
        var name: String
        var reason: String
    }

    struct TensorSummary: Codable {
        var name: String
        var shard: String
        var shape: [Int]
        var scalesShape: [Int]
        var eligibleCodeBytes: Int64
        var eligibleGroups: Int64
        var eligibleSuperblocks: Int64
        var zeroGroups: Int64
        var splatNibbleGroups: Int64
        var twoSymbolGroups: Int64
        var rawGroups: Int64
        var compressedProfitableSuperblocks: Int64
        var estimatedRawBytes: Int64
        var estimatedBytesAfterGroupCoding: Int64
        var estimatedSavedBytes: Int64
        var estimatedSavedRatio: Double
        var optimisticSavedBytesNoIndex: Int64
        var optimisticSavedRatioNoIndex: Double
    }

    private struct MutableSummary {
        var name: String
        var shard: String
        var shape: [Int]
        var scalesShape: [Int]
        var eligibleCodeBytes: Int64 = 0
        var eligibleGroups: Int64 = 0
        var eligibleSuperblocks: Int64 = 0
        var zeroGroups: Int64 = 0
        var splatNibbleGroups: Int64 = 0
        var twoSymbolGroups: Int64 = 0
        var rawGroups: Int64 = 0
        var compressedProfitableSuperblocks: Int64 = 0
        var estimatedRawBytes: Int64 = 0
        var estimatedBytesAfterGroupCoding: Int64 = 0
        var estimatedBytesAfterGroupCodingNoIndex: Int64 = 0
        var estimatedIndexAndHeaderOverheadBytes: Int64 = 0
        var estimatedSavedBytes: Int64 = 0
        var optimisticSavedBytesNoIndex: Int64 = 0

        func frozen() -> TensorSummary {
            TensorSummary(
                name: name,
                shard: shard,
                shape: shape,
                scalesShape: scalesShape,
                eligibleCodeBytes: eligibleCodeBytes,
                eligibleGroups: eligibleGroups,
                eligibleSuperblocks: eligibleSuperblocks,
                zeroGroups: zeroGroups,
                splatNibbleGroups: splatNibbleGroups,
                twoSymbolGroups: twoSymbolGroups,
                rawGroups: rawGroups,
                compressedProfitableSuperblocks: compressedProfitableSuperblocks,
                estimatedRawBytes: estimatedRawBytes,
                estimatedBytesAfterGroupCoding: estimatedBytesAfterGroupCoding,
                estimatedSavedBytes: estimatedSavedBytes,
                estimatedSavedRatio: ratio(estimatedSavedBytes, estimatedRawBytes),
                optimisticSavedBytesNoIndex: optimisticSavedBytesNoIndex,
                optimisticSavedRatioNoIndex: ratio(optimisticSavedBytesNoIndex, estimatedRawBytes)
            )
        }
    }

    private struct SuperblockAccumulator {
        var groups: Int = 0
        var codedPayloadBytes: Int64 = 0
        var zeroGroups: Int64 = 0
        var splatGroups: Int64 = 0
        var twoSymbolGroups: Int64 = 0
        var rawGroups: Int64 = 0

        mutating func add(_ classification: GroupClassification) {
            groups += 1
            switch classification {
            case .zero:
                zeroGroups += 1
            case .splatNibble:
                splatGroups += 1
                codedPayloadBytes += 1
            case .twoSymbol:
                twoSymbolGroups += 1
                codedPayloadBytes += 6
            case .raw:
                rawGroups += 1
                codedPayloadBytes += 16
            }
        }

        mutating func reset() {
            groups = 0
            codedPayloadBytes = 0
            zeroGroups = 0
            splatGroups = 0
            twoSymbolGroups = 0
            rawGroups = 0
        }
    }

    private enum GroupClassification {
        case zero
        case splatNibble
        case twoSymbol
        case raw
    }

    static func run(
        referenceDirectory: URL,
        expertsDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws -> Report {
        var skipped: [SkippedTensor] = []
        var summaries: [TensorSummary] = []
        var eligibleTensorCount = 0
        var totals = MutableSummary(name: "__total__", shard: "", shape: [], scalesShape: [])

        let candidateKeys = expertKeys
            .filter { $0.hasSuffix(".weight") }
            .sorted()

        var headerCache: [String: SafetensorsHeader] = [:]
        func header(for shardName: String) throws -> SafetensorsHeader {
            if let cached = headerCache[shardName] {
                return cached
            }
            let loaded = try Safetensors.readHeader(referenceDirectory.appendingPathComponent(shardName))
            headerCache[shardName] = loaded
            return loaded
        }

        for key in candidateKeys {
            guard let shardName = index.weightMap[key] else {
                skipped.append(SkippedTensor(name: key, reason: "missing from checkpoint index"))
                continue
            }
            let shardHeader = try header(for: shardName)
            guard let codeInfo = shardHeader.tensors[key] else {
                skipped.append(SkippedTensor(name: key, reason: "missing from shard header \(shardName)"))
                continue
            }
            guard codeInfo.dtype == "U32" else {
                skipped.append(SkippedTensor(name: key, reason: "dtype \(codeInfo.dtype) is not U32"))
                continue
            }

            let scalesKey = String(key.dropLast(".weight".count)) + ".scales"
            guard let scalesShardName = index.weightMap[scalesKey] else {
                skipped.append(SkippedTensor(name: key, reason: "missing companion scales tensor \(scalesKey) in checkpoint index"))
                continue
            }
            guard scalesShardName == shardName else {
                skipped.append(SkippedTensor(name: key, reason: "companion scales tensor is in different shard \(scalesShardName)"))
                continue
            }
            guard let scalesInfo = shardHeader.tensors[scalesKey] else {
                skipped.append(SkippedTensor(name: key, reason: "missing companion scales tensor \(scalesKey) in shard header \(shardName)"))
                continue
            }
            guard scalesInfo.dtype == "U8" else {
                skipped.append(SkippedTensor(name: key, reason: "companion scales dtype \(scalesInfo.dtype) is not U8"))
                continue
            }

            guard codeInfo.shape.count == 3, scalesInfo.shape.count == 3 else {
                skipped.append(SkippedTensor(name: key, reason: "expected rank-3 code/scales tensors, got \(codeInfo.shape.count)/\(scalesInfo.shape.count)"))
                continue
            }
            guard codeInfo.shape[0] == scalesInfo.shape[0], codeInfo.shape[1] == scalesInfo.shape[1] else {
                skipped.append(SkippedTensor(name: key, reason: "code/scales leading dimensions do not match: \(codeInfo.shape) vs \(scalesInfo.shape)"))
                continue
            }
            let packedInput = codeInfo.shape[2]
            let scaleGroups = scalesInfo.shape[2]
            guard packedInput == scaleGroups * codeWordsPerScaleGroup else {
                skipped.append(SkippedTensor(name: key, reason: "packedInput \(packedInput) != scaleGroups \(scaleGroups) * \(codeWordsPerScaleGroup)"))
                continue
            }
            guard codeInfo.byteCount == product(codeInfo.shape) * 4 else {
                skipped.append(SkippedTensor(name: key, reason: "U32 code byte count does not match shape"))
                continue
            }
            guard scalesInfo.byteCount == product(scalesInfo.shape) else {
                skipped.append(SkippedTensor(name: key, reason: "U8 scales byte count does not match shape"))
                continue
            }

            var summary = MutableSummary(
                name: key,
                shard: shardName,
                shape: codeInfo.shape,
                scalesShape: scalesInfo.shape
            )
            try scanCodeTensor(
                referenceDirectory: referenceDirectory,
                shardName: shardName,
                header: shardHeader,
                info: codeInfo,
                summary: &summary
            )
            let frozen = summary.frozen()
            summaries.append(frozen)
            eligibleTensorCount += 1
            add(summary, to: &totals)
        }

        summaries.sort {
            if $0.estimatedSavedBytes == $1.estimatedSavedBytes {
                return $0.name < $1.name
            }
            return $0.estimatedSavedBytes > $1.estimatedSavedBytes
        }
        if summaries.count > 20 {
            summaries = Array(summaries.prefix(20))
        }

        let savedRatio = ratio(totals.estimatedSavedBytes, totals.estimatedRawBytes)
        let optimisticRatio = ratio(totals.optimisticSavedBytesNoIndex, totals.estimatedRawBytes)
        return Report(
            version: 1,
            groupsPerSuperblock: groupsPerSuperblock,
            eligibleCodeBytes: totals.eligibleCodeBytes,
            eligibleTensors: eligibleTensorCount,
            eligibleGroups: totals.eligibleGroups,
            eligibleSuperblocks: totals.eligibleSuperblocks,
            skippedTensors: skipped,
            zeroGroups: totals.zeroGroups,
            splatNibbleGroups: totals.splatNibbleGroups,
            twoSymbolGroups: totals.twoSymbolGroups,
            rawGroups: totals.rawGroups,
            zeroGroupByteShare: ratio(totals.zeroGroups, totals.eligibleGroups),
            splatNibbleGroupByteShare: ratio(totals.splatNibbleGroups, totals.eligibleGroups),
            twoSymbolGroupByteShare: ratio(totals.twoSymbolGroups, totals.eligibleGroups),
            rawGroupByteShare: ratio(totals.rawGroups, totals.eligibleGroups),
            compressedProfitableSuperblocks: totals.compressedProfitableSuperblocks,
            totalSuperblocks: totals.eligibleSuperblocks,
            estimatedRawBytes: totals.estimatedRawBytes,
            estimatedBytesAfterGroupCoding: totals.estimatedBytesAfterGroupCoding,
            estimatedBytesAfterGroupCodingNoIndex: totals.estimatedBytesAfterGroupCodingNoIndex,
            estimatedIndexAndHeaderOverheadBytes: totals.estimatedIndexAndHeaderOverheadBytes,
            estimatedSavedBytes: totals.estimatedSavedBytes,
            estimatedSavedRatio: savedRatio,
            optimisticSavedBytesNoIndex: totals.optimisticSavedBytesNoIndex,
            optimisticSavedRatioNoIndex: optimisticRatio,
            topTensorsBySavedBytes: summaries,
            runtimeIntegrationRecommended: savedRatio >= 0.04 && totals.estimatedSavedBytes >= 5_000_000_000
        )
    }

    private static func scanCodeTensor(
        referenceDirectory: URL,
        shardName: String,
        header: SafetensorsHeader,
        info: SafetensorInfo,
        summary: inout MutableSummary
    ) throws {
        let shardURL = referenceDirectory.appendingPathComponent(shardName)
        let handle = try FileHandle(forReadingFrom: shardURL)
        defer { try? handle.close() }

        let tensorOffset = header.dataBaseOffset + UInt64(info.dataStart)
        try handle.seek(toOffset: tensorOffset)

        summary.eligibleCodeBytes = Int64(info.byteCount)
        summary.eligibleGroups = Int64(info.byteCount / bytesPerGroup)

        var remaining = info.byteCount
        var superblock = SuperblockAccumulator()
        var totalGroupsScanned: Int64 = 0
        let chunkSize = max(bytesPerGroup, (readChunkBytes / bytesPerGroup) * bytesPerGroup)

        while remaining > 0 {
            try autoreleasepool {
                let bytesToRead = min(chunkSize, remaining)
                let data = handle.readData(ofLength: bytesToRead)
                guard data.count == bytesToRead else {
                    throw MLXFastError.invalidInput("unexpected EOF while scanning \(info.name)")
                }
                data.withUnsafeBytes { rawBuffer in
                    guard let base = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                        return
                    }
                    var offset = 0
                    while offset < data.count {
                        let classification = classifyGroup(base + offset)
                        superblock.add(classification)
                        totalGroupsScanned += 1
                        if superblock.groups == groupsPerSuperblock {
                            flush(&superblock, into: &summary)
                        }
                        offset += bytesPerGroup
                    }
                }
                remaining -= bytesToRead
            }
        }
        if superblock.groups > 0 {
            flush(&superblock, into: &summary)
        }
        guard totalGroupsScanned == summary.eligibleGroups else {
            throw MLXFastError.invalidInput("internal MXFP4 group count mismatch for \(info.name): scanned \(totalGroupsScanned) expected \(summary.eligibleGroups)")
        }
    }

    private static func classifyGroup(_ p: UnsafePointer<UInt8>) -> GroupClassification {
        var allZero = true
        for i in 0..<bytesPerGroup {
            if p[i] != 0 {
                allZero = false
                break
            }
        }
        if allZero {
            return .zero
        }

        let firstNibble = p[0] & 0x0f
        var splat = true
        var seenMask: UInt16 = 0
        for i in 0..<bytesPerGroup {
            let byte = p[i]
            let low = byte & 0x0f
            let high = byte >> 4
            if low != firstNibble || high != firstNibble {
                splat = false
            }
            seenMask |= UInt16(1) << UInt16(low)
            seenMask |= UInt16(1) << UInt16(high)
        }
        if splat {
            return .splatNibble
        }
        if seenMask.nonzeroBitCount == 2 {
            return .twoSymbol
        }
        return .raw
    }

    private static func flush(_ superblock: inout SuperblockAccumulator, into summary: inout MutableSummary) {
        guard superblock.groups > 0 else { return }
        let rawBytes = Int64(superblock.groups * bytesPerGroup)
        let tagBytes = Int64((2 * superblock.groups + 7) / 8)
        let codedNoOverhead = tagBytes + superblock.codedPayloadBytes
        let headerBytes: Int64 = 4
        let indexBytes: Int64 = 8
        let codedWithHeader = codedNoOverhead + headerBytes
        let codedWithIndex = codedWithHeader + indexBytes

        summary.eligibleSuperblocks += 1
        summary.zeroGroups += superblock.zeroGroups
        summary.splatNibbleGroups += superblock.splatGroups
        summary.twoSymbolGroups += superblock.twoSymbolGroups
        summary.rawGroups += superblock.rawGroups
        summary.estimatedRawBytes += rawBytes

        if codedWithIndex < rawBytes {
            summary.compressedProfitableSuperblocks += 1
            summary.estimatedBytesAfterGroupCoding += codedWithIndex
            summary.estimatedIndexAndHeaderOverheadBytes += headerBytes + indexBytes
            summary.estimatedSavedBytes += rawBytes - codedWithIndex
        } else {
            summary.estimatedBytesAfterGroupCoding += rawBytes
        }

        if codedWithHeader < rawBytes {
            summary.estimatedBytesAfterGroupCodingNoIndex += codedWithHeader
            summary.optimisticSavedBytesNoIndex += rawBytes - codedWithHeader
        } else {
            summary.estimatedBytesAfterGroupCodingNoIndex += rawBytes
        }

        superblock.reset()
    }

    private static func add(_ summary: MutableSummary, to total: inout MutableSummary) {
        total.eligibleCodeBytes += summary.eligibleCodeBytes
        total.eligibleGroups += summary.eligibleGroups
        total.eligibleSuperblocks += summary.eligibleSuperblocks
        total.zeroGroups += summary.zeroGroups
        total.splatNibbleGroups += summary.splatNibbleGroups
        total.twoSymbolGroups += summary.twoSymbolGroups
        total.rawGroups += summary.rawGroups
        total.compressedProfitableSuperblocks += summary.compressedProfitableSuperblocks
        total.estimatedRawBytes += summary.estimatedRawBytes
        total.estimatedBytesAfterGroupCoding += summary.estimatedBytesAfterGroupCoding
        total.estimatedBytesAfterGroupCodingNoIndex += summary.estimatedBytesAfterGroupCodingNoIndex
        total.estimatedIndexAndHeaderOverheadBytes += summary.estimatedIndexAndHeaderOverheadBytes
        total.estimatedSavedBytes += summary.estimatedSavedBytes
        total.optimisticSavedBytesNoIndex += summary.optimisticSavedBytesNoIndex
    }

    private static func product(_ shape: [Int]) -> Int {
        shape.reduce(1, *)
    }

    private static func ratio(_ numerator: Int64, _ denominator: Int64) -> Double {
        guard denominator != 0 else { return 0 }
        return Double(numerator) / Double(denominator)
    }
}
