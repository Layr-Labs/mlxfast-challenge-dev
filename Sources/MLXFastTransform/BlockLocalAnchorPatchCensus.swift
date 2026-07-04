import Foundation
import MLXFastCore

enum BlockLocalAnchorPatchCensus {
    private static let version = 1
    private static let mechanism = "blockLocalMultiAnchorSparsePatch"
    private static let blockSize = 64 * 1024
    private static let targetAnchorCount = 4
    private static let blockHeaderBytes: Int64 = 16
    private static let patchBytesPerChangedWord: Int64 = 6
    private static let profitabilitySlackBytes: Int64 = 256
    private static let blockIndexBytesPerBlock: Int64 = 8
    private static let minSavedRatio = 0.06
    private static let minSavedBytes: Int64 = 8_589_934_592

    struct Summary: Codable {
        struct Gate: Codable {
            let minSavedRatio: Double
            let minSavedBytes: Int64
        }

        struct TensorSummary: Codable {
            let name: String
            let shard: String
            let shape: [Int]
            let expertCount: Int
            let sliceByteLength: Int
            let anchorExpertIds: [Int]
            let rawBytes: Int64
            let anchorDictionaryBytesResident: Int64
            let rawDemandBytesNonAnchor: Int64
            let estimatedDemandBytesAfterPatchOrRaw: Int64
            let blockIndexOverheadBytes: Int64
            let estimatedDemandSavedBytesAfterDictionary: Int64
            let estimatedDemandSavedRatioAfterDictionary: Double
            let patchProfitableBlocks: Int64
            let rawFallbackBlocks: Int64
            let totalBlocks: Int64
            let meanChangedWordRatioOnProfitableBlocks: Double
        }

        let version: Int
        let mechanism: String
        let blockSize: Int
        let anchorCount: Int
        let eligibleTensors: Int
        let eligibleRawBytes: Int64
        let anchorDictionaryBytesResident: Int64
        let rawDemandBytesNonAnchor: Int64
        let estimatedDemandBytesAfterPatchOrRaw: Int64
        let blockIndexOverheadBytes: Int64
        let estimatedDemandSavedBytesAfterDictionary: Int64
        let estimatedDemandSavedRatioAfterDictionary: Double
        let patchProfitableBlocks: Int64
        let rawFallbackBlocks: Int64
        let totalBlocks: Int64
        let meanChangedWordRatioOnProfitableBlocks: Double
        let runtimeIntegrationRecommended: Bool
        let gate: Gate
        let tensors: [TensorSummary]
    }

    private struct Manifest: Decodable {
        let expertTensors: [Record]

        enum CodingKeys: String, CodingKey {
            case expertTensors = "expert_tensors"
        }
    }

    private struct Record: Decodable {
        let name: String
        let shard: String
        let dtype: String
        let shape: [Int]
        let byteOffset: Int
        let byteLength: Int

        enum CodingKeys: String, CodingKey {
            case name
            case shard
            case dtype
            case shape
            case byteOffset = "byte_offset"
            case byteLength = "byte_length"
        }
    }

    static func run(referenceDirectory: URL, manifestPath: URL, outputDirectory: URL) throws -> Summary {
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        let eligible = manifest.expertTensors.filter(isEligible(_:))

        var tensorSummaries: [Summary.TensorSummary] = []
        tensorSummaries.reserveCapacity(eligible.count)

        var eligibleRawBytes: Int64 = 0
        var anchorDictionaryBytesResident: Int64 = 0
        var rawDemandBytesNonAnchor: Int64 = 0
        var estimatedDemandBytesAfterPatchOrRaw: Int64 = 0
        var blockIndexOverheadBytes: Int64 = 0
        var patchProfitableBlocks: Int64 = 0
        var rawFallbackBlocks: Int64 = 0
        var totalBlocks: Int64 = 0
        var profitableChangedWords: Int64 = 0
        var profitableTotalWords: Int64 = 0

        for record in eligible.sorted(by: { $0.name < $1.name }) {
            let scan = try scanTensor(record, referenceDirectory: referenceDirectory)
            let tensor = scan.summary
            tensorSummaries.append(tensor)

            eligibleRawBytes += tensor.rawBytes
            anchorDictionaryBytesResident += tensor.anchorDictionaryBytesResident
            rawDemandBytesNonAnchor += tensor.rawDemandBytesNonAnchor
            estimatedDemandBytesAfterPatchOrRaw += tensor.estimatedDemandBytesAfterPatchOrRaw
            blockIndexOverheadBytes += tensor.blockIndexOverheadBytes
            patchProfitableBlocks += tensor.patchProfitableBlocks
            rawFallbackBlocks += tensor.rawFallbackBlocks
            totalBlocks += tensor.totalBlocks
            profitableChangedWords += scan.profitableChangedWords
            profitableTotalWords += scan.profitableTotalWords
        }

        let savedBytes = eligibleRawBytes
            - anchorDictionaryBytesResident
            - estimatedDemandBytesAfterPatchOrRaw
            - blockIndexOverheadBytes
        let savedRatio = eligibleRawBytes > 0 ? Double(savedBytes) / Double(eligibleRawBytes) : 0.0
        let meanRatio = profitableTotalWords > 0
            ? Double(profitableChangedWords) / Double(profitableTotalWords)
            : 0.0
        let recommended = savedRatio >= minSavedRatio && savedBytes >= minSavedBytes

        let summary = Summary(
            version: version,
            mechanism: mechanism,
            blockSize: blockSize,
            anchorCount: targetAnchorCount,
            eligibleTensors: tensorSummaries.count,
            eligibleRawBytes: eligibleRawBytes,
            anchorDictionaryBytesResident: anchorDictionaryBytesResident,
            rawDemandBytesNonAnchor: rawDemandBytesNonAnchor,
            estimatedDemandBytesAfterPatchOrRaw: estimatedDemandBytesAfterPatchOrRaw,
            blockIndexOverheadBytes: blockIndexOverheadBytes,
            estimatedDemandSavedBytesAfterDictionary: savedBytes,
            estimatedDemandSavedRatioAfterDictionary: savedRatio,
            patchProfitableBlocks: patchProfitableBlocks,
            rawFallbackBlocks: rawFallbackBlocks,
            totalBlocks: totalBlocks,
            meanChangedWordRatioOnProfitableBlocks: meanRatio,
            runtimeIntegrationRecommended: recommended,
            gate: Summary.Gate(minSavedRatio: minSavedRatio, minSavedBytes: minSavedBytes),
            tensors: tensorSummaries
        )

        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let output = try encoder.encode(summary)
        try output.write(to: outputDirectory.appendingPathComponent("block-local-anchor-patch-census.json"))
        return summary
    }

    private struct TensorScan {
        let summary: Summary.TensorSummary
        let profitableChangedWords: Int64
        let profitableTotalWords: Int64
    }

    private static func isEligible(_ record: Record) -> Bool {
        guard record.dtype == "U32" else { return false }
        guard record.shape.count == 3 else { return false }
        guard let expertCount = record.shape.first, expertCount > targetAnchorCount else { return false }
        guard record.byteLength > 0, record.byteLength % expertCount == 0 else { return false }
        return true
    }

    private static func scanTensor(
        _ record: Record,
        referenceDirectory: URL
    ) throws -> TensorScan {
        let expertCount = record.shape[0]
        let sliceByteLength = record.byteLength / expertCount
        let anchorIds = selectAnchorIds(expertCount: expertCount)
        let anchorSet = Set(anchorIds)
        let shardURL = referenceDirectory.appendingPathComponent(record.shard)

        let attributes = try FileManager.default.attributesOfItem(atPath: shardURL.path)
        let shardByteCount = try fileSizeByteCount(from: attributes, path: shardURL.path)
        guard record.byteOffset >= 0,
              record.byteLength >= 0,
              record.byteOffset + record.byteLength <= shardByteCount
        else {
            throw MLXFastError.invalidInput(
                "manifest byte range for \(record.name) exceeds shard size: \(record.shard)"
            )
        }

        let handle = try FileHandle(forReadingFrom: shardURL)
        defer { try? handle.close() }

        var anchors: [Data] = []
        anchors.reserveCapacity(anchorIds.count)
        for expertId in anchorIds {
            anchors.append(try readSlice(
                handle: handle,
                tensorByteOffset: record.byteOffset,
                sliceByteLength: sliceByteLength,
                expertId: expertId
            ))
        }

        var estimatedDemand: Int64 = 0
        var rawDemand: Int64 = 0
        var tensorPatchProfitableBlocks: Int64 = 0
        var tensorRawFallbackBlocks: Int64 = 0
        var tensorTotalBlocks: Int64 = 0
        var profitableChangedWords: Int64 = 0
        var profitableTotalWords: Int64 = 0

        for expertId in 0..<expertCount {
            if anchorSet.contains(expertId) {
                continue
            }
            try autoreleasepool {
                let candidate = try readSlice(
                    handle: handle,
                    tensorByteOffset: record.byteOffset,
                    sliceByteLength: sliceByteLength,
                    expertId: expertId
                )
                rawDemand += Int64(sliceByteLength)

                try candidate.withUnsafeBytes { candidateBytes in
                    guard candidateBytes.count == sliceByteLength else {
                        throw MLXFastError.invalidInput("short candidate slice while scanning \(record.name)")
                    }
                    for blockOffset in stride(from: 0, to: sliceByteLength, by: blockSize) {
                        let blockByteCount = min(blockSize, sliceByteLength - blockOffset)
                        tensorTotalBlocks += 1

                        guard blockByteCount % 4 == 0 else {
                            estimatedDemand += Int64(blockByteCount)
                            tensorRawFallbackBlocks += 1
                            continue
                        }

                        let wordCount = blockByteCount / 4
                        let profitableChangedLimit = maxProfitableChangedWords(rawBlockByteCount: blockByteCount)
                        var bestChanged = profitableChangedLimit + 1

                        for anchor in anchors {
                            anchor.withUnsafeBytes { anchorBytes in
                                let changed = countChangedWords(
                                    candidateBytes: candidateBytes,
                                    anchorBytes: anchorBytes,
                                    blockOffset: blockOffset,
                                    wordCount: wordCount,
                                    stopAt: bestChanged
                                )
                                if changed < bestChanged {
                                    bestChanged = changed
                                }
                            }
                        }

                        if bestChanged <= profitableChangedLimit {
                            let patchBytes = blockHeaderBytes + Int64(bestChanged) * patchBytesPerChangedWord
                            estimatedDemand += patchBytes
                            tensorPatchProfitableBlocks += 1
                            profitableChangedWords += Int64(bestChanged)
                            profitableTotalWords += Int64(wordCount)
                        } else {
                            estimatedDemand += Int64(blockByteCount)
                            tensorRawFallbackBlocks += 1
                        }
                    }
                }
            }
        }

        let rawBytes = Int64(record.byteLength)
        let anchorBytes = Int64(anchorIds.count * sliceByteLength)
        let indexBytes = tensorTotalBlocks * blockIndexBytesPerBlock
        let savedBytes = rawBytes - anchorBytes - estimatedDemand - indexBytes
        let savedRatio = rawBytes > 0 ? Double(savedBytes) / Double(rawBytes) : 0.0
        let meanChangedRatio = profitableTotalWords > 0
            ? Double(profitableChangedWords) / Double(profitableTotalWords)
            : 0.0
        let summary = Summary.TensorSummary(
            name: record.name,
            shard: record.shard,
            shape: record.shape,
            expertCount: expertCount,
            sliceByteLength: sliceByteLength,
            anchorExpertIds: anchorIds,
            rawBytes: rawBytes,
            anchorDictionaryBytesResident: anchorBytes,
            rawDemandBytesNonAnchor: rawDemand,
            estimatedDemandBytesAfterPatchOrRaw: estimatedDemand,
            blockIndexOverheadBytes: indexBytes,
            estimatedDemandSavedBytesAfterDictionary: savedBytes,
            estimatedDemandSavedRatioAfterDictionary: savedRatio,
            patchProfitableBlocks: tensorPatchProfitableBlocks,
            rawFallbackBlocks: tensorRawFallbackBlocks,
            totalBlocks: tensorTotalBlocks,
            meanChangedWordRatioOnProfitableBlocks: meanChangedRatio
        )
        return TensorScan(
            summary: summary,
            profitableChangedWords: profitableChangedWords,
            profitableTotalWords: profitableTotalWords
        )
    }

    private static func selectAnchorIds(expertCount: Int) -> [Int] {
        let preferred = [0, 64, 128, 192]
        var ids: [Int] = []
        ids.reserveCapacity(min(targetAnchorCount, expertCount))
        for expertId in preferred {
            let clamped = min(max(expertId, 0), expertCount - 1)
            if !ids.contains(clamped) {
                ids.append(clamped)
            }
        }
        var fallback = 0
        while ids.count < min(targetAnchorCount, expertCount) {
            if !ids.contains(fallback) {
                ids.append(fallback)
            }
            fallback += 1
        }
        return ids.sorted()
    }

    private static func readSlice(
        handle: FileHandle,
        tensorByteOffset: Int,
        sliceByteLength: Int,
        expertId: Int
    ) throws -> Data {
        let offset = tensorByteOffset + expertId * sliceByteLength
        try handle.seek(toOffset: UInt64(offset))
        let data = handle.readData(ofLength: sliceByteLength)
        guard data.count == sliceByteLength else {
            throw MLXFastError.invalidInput("unexpected EOF while reading expert slice")
        }
        return data
    }

    private static func maxProfitableChangedWords(rawBlockByteCount: Int) -> Int {
        let budget = Int64(rawBlockByteCount) - blockHeaderBytes - profitabilitySlackBytes
        if budget < 0 { return -1 }
        return Int(budget / patchBytesPerChangedWord)
    }

    private static func countChangedWords(
        candidateBytes: UnsafeRawBufferPointer,
        anchorBytes: UnsafeRawBufferPointer,
        blockOffset: Int,
        wordCount: Int,
        stopAt: Int
    ) -> Int {
        if stopAt < 0 { return 0 }
        var changed = 0
        var byteOffset = blockOffset
        for _ in 0..<wordCount {
            let candidateWord = candidateBytes.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)
            let anchorWord = anchorBytes.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)
            if candidateWord != anchorWord {
                changed += 1
                if changed >= stopAt {
                    return changed
                }
            }
            byteOffset += 4
        }
        return changed
    }
}
