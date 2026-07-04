import Foundation
import MLXFastCore

enum MXFP4EntropyCensus {
    private static let perSliceOverheadBytes: UInt64 = 64

    struct Summary: Codable, Equatable {
        let eligibleBytes: UInt64
        let eligibleSlices: Int
        let profitableSlices: Int
        let rawPayloadBytesIfAllEncoded: UInt64
        let estimatedPayloadBytesIfAllEncoded: UInt64
        let estimatedBytesForProfitableSlices: UInt64
        let estimatedSavedBytesProfitableOnly: UInt64
        let estimatedSavedRatioProfitableOnly: Double
        let bestSliceSavedRatio: Double
        let worstSliceSavedRatio: Double
        let symbolCounts: [UInt64]
    }

    private struct SliceEstimate {
        let rawBytes: UInt64
        let encodedBytes: UInt64
        let profitable: Bool

        var savedBytes: UInt64 {
            rawBytes > encodedBytes ? rawBytes - encodedBytes : 0
        }

        var savedRatio: Double {
            guard rawBytes > 0 else { return 0 }
            return Double(Int64(rawBytes) - Int64(encodedBytes)) / Double(rawBytes)
        }
    }

    private struct QueueItem {
        let count: UInt64
        let symbol: Int
        let symbols: [Int]
    }

    static func run(
        referenceDirectory: URL,
        outputDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws -> Summary {
        let expertKeysByShard = Dictionary(grouping: expertKeys) { key in
            index.weightMap[key] ?? ""
        }

        var eligibleBytes: UInt64 = 0
        var eligibleSlices = 0
        var profitableSlices = 0
        var rawPayloadBytesIfAllEncoded: UInt64 = 0
        var estimatedPayloadBytesIfAllEncoded: UInt64 = 0
        var estimatedBytesForProfitableSlices: UInt64 = 0
        var estimatedSavedBytesProfitableOnly: UInt64 = 0
        var rawBytesForProfitableSlices: UInt64 = 0
        var bestSliceSavedRatio = -Double.infinity
        var worstSliceSavedRatio = Double.infinity
        var globalSymbolCounts = Array(repeating: UInt64(0), count: 16)

        for shardName in expertKeysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            let handle = try FileHandle(forReadingFrom: shardURL)
            defer {
                try? handle.close()
            }

            for key in expertKeysByShard[shardName, default: []].sorted() {
                guard SwiftTransform.isExpertKey(key), key.hasSuffix(".weight") else {
                    continue
                }
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "expert tensor \(key) is listed in index but missing from \(shardName)"
                    )
                }
                guard info.dtype == "U32",
                      info.shape.count >= 3,
                      let expertCount = info.shape.first,
                      expertCount > 0,
                      info.byteCount % expertCount == 0
                else {
                    continue
                }

                let sliceByteLength = info.byteCount / expertCount
                guard sliceByteLength > 0 else {
                    continue
                }

                for expertIndex in 0..<expertCount {
                    let offset = UInt64(Int(header.dataBaseOffset) + info.dataStart + expertIndex * sliceByteLength)
                    try handle.seek(toOffset: offset)

                    var byteCounts = Array(repeating: UInt64(0), count: 256)
                    var remaining = sliceByteLength
                    while remaining > 0 {
                        try autoreleasepool {
                            let chunkLength = min(16 * 1024 * 1024, remaining)
                            let data = handle.readData(ofLength: chunkLength)
                            guard data.count == chunkLength else {
                                throw MLXFastError.invalidInput(
                                    "unexpected EOF while reading \(key)[\(expertIndex)] from \(shardName)"
                                )
                            }
                            countBytes(in: data, into: &byteCounts)
                            remaining -= data.count
                        }
                    }

                    var counts = Array(repeating: UInt64(0), count: 16)
                    for byte in 0..<256 {
                        let frequency = byteCounts[byte]
                        if frequency == 0 { continue }
                        counts[byte & 0x0f] += frequency
                        counts[byte >> 4] += frequency
                    }

                    for symbol in 0..<16 {
                        globalSymbolCounts[symbol] += counts[symbol]
                    }

                    let payloadBits = huffmanPayloadBits(for: counts)
                    let payloadBytes = (payloadBits + 7) / 8
                    let encodedBytes = payloadBytes + perSliceOverheadBytes
                    let rawBytes = UInt64(sliceByteLength)
                    let profitable = encodedBytes <= rawBytes * 97 / 100
                    let estimate = SliceEstimate(
                        rawBytes: rawBytes,
                        encodedBytes: encodedBytes,
                        profitable: profitable
                    )

                    eligibleBytes += rawBytes
                    eligibleSlices += 1
                    rawPayloadBytesIfAllEncoded += rawBytes
                    estimatedPayloadBytesIfAllEncoded += encodedBytes
                    bestSliceSavedRatio = max(bestSliceSavedRatio, estimate.savedRatio)
                    worstSliceSavedRatio = min(worstSliceSavedRatio, estimate.savedRatio)

                    if profitable {
                        profitableSlices += 1
                        rawBytesForProfitableSlices += rawBytes
                        estimatedBytesForProfitableSlices += encodedBytes
                        estimatedSavedBytesProfitableOnly += estimate.savedBytes
                    }
                }
            }
        }

        if eligibleSlices == 0 {
            bestSliceSavedRatio = 0
            worstSliceSavedRatio = 0
        }

        let estimatedSavedRatioProfitableOnly: Double
        if rawBytesForProfitableSlices > 0 {
            estimatedSavedRatioProfitableOnly = Double(estimatedSavedBytesProfitableOnly) / Double(rawBytesForProfitableSlices)
        } else {
            estimatedSavedRatioProfitableOnly = 0
        }

        let summary = Summary(
            eligibleBytes: eligibleBytes,
            eligibleSlices: eligibleSlices,
            profitableSlices: profitableSlices,
            rawPayloadBytesIfAllEncoded: rawPayloadBytesIfAllEncoded,
            estimatedPayloadBytesIfAllEncoded: estimatedPayloadBytesIfAllEncoded,
            estimatedBytesForProfitableSlices: estimatedBytesForProfitableSlices,
            estimatedSavedBytesProfitableOnly: estimatedSavedBytesProfitableOnly,
            estimatedSavedRatioProfitableOnly: estimatedSavedRatioProfitableOnly,
            bestSliceSavedRatio: bestSliceSavedRatio,
            worstSliceSavedRatio: worstSliceSavedRatio,
            symbolCounts: globalSymbolCounts
        )

        let expertsDirectory = outputDirectory.appendingPathComponent("experts", isDirectory: true)
        try FileManager.default.createDirectory(at: expertsDirectory, withIntermediateDirectories: true)
        let outputURL = expertsDirectory.appendingPathComponent("mxfp4-entropy-census.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let json = try encoder.encode(summary)
        try json.write(to: outputURL)

        print(
            String(
                format: "MXFP4 entropy census: eligible=%llu slices=%d profitable=%d saved=%llu ratio=%.6f best_slice=%.6f",
                eligibleBytes,
                eligibleSlices,
                profitableSlices,
                estimatedSavedBytesProfitableOnly,
                estimatedSavedRatioProfitableOnly,
                bestSliceSavedRatio
            )
        )

        return summary
    }

    private static func countBytes(in data: Data, into counts: inout [UInt64]) {
        data.withUnsafeBytes { rawBuffer in
            for byte in rawBuffer.bindMemory(to: UInt8.self) {
                counts[Int(byte)] += 1
            }
        }
    }

    private static func huffmanPayloadBits(for counts: [UInt64]) -> UInt64 {
        let lengths = huffmanCodeLengths(for: counts)
        var bits: UInt64 = 0
        for symbol in 0..<min(counts.count, lengths.count) {
            bits += counts[symbol] * UInt64(lengths[symbol])
        }
        return bits
    }

    private static func huffmanCodeLengths(for counts: [UInt64]) -> [Int] {
        var queue: [QueueItem] = []
        for symbol in 0..<min(counts.count, 16) where counts[symbol] > 0 {
            queue.append(QueueItem(count: counts[symbol], symbol: symbol, symbols: [symbol]))
        }

        var lengths = Array(repeating: 0, count: 16)
        if queue.isEmpty {
            return lengths
        }
        if queue.count == 1 {
            lengths[queue[0].symbol] = 1
            return lengths
        }

        while queue.count > 1 {
            queue.sort {
                if $0.count != $1.count { return $0.count < $1.count }
                return $0.symbol < $1.symbol
            }
            let first = queue.removeFirst()
            let second = queue.removeFirst()
            for symbol in first.symbols {
                lengths[symbol] += 1
            }
            for symbol in second.symbols {
                lengths[symbol] += 1
            }
            let mergedSymbols = first.symbols + second.symbols
            queue.append(
                QueueItem(
                    count: first.count + second.count,
                    symbol: min(first.symbol, second.symbol),
                    symbols: mergedSymbols
                )
            )
        }

        return lengths
    }
}
