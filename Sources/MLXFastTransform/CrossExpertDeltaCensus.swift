import Foundation
import MLXFastCore

/// Full-corpus transform-side probe for cross-expert XOR-delta coding of stacked
/// MXFP4 U32 expert code tensors. This is intentionally runtime-invisible: it
/// only writes a JSON report under weights/experts/.
enum CrossExpertDeltaCensus {
    private static let version = 1
    private static let mechanism = "cross_expert_anchor_xor_delta"
    private static let blockSize = 1 << 20
    private static let prefixLimit = 16 << 20
    private static let candidateAnchors = [0, 64, 128, 192]
    private static let metadataBytesPerBlock = 16
    private static let alignmentPenaltyPerBlock = 4096
    private static let minDemandSavedRatio = 0.06
    private static let minSavedBytes: Int64 = 8_589_934_592

    struct TensorCandidate {
        let name: String
        let shardName: String
        let layerIndex: Int
        let projection: String
        let shape: [Int]
        let byteOffset: UInt64
        let byteCount: Int
        let expertCount: Int
        let sliceLength: Int
    }

    struct TensorSummary {
        let name: String
        let layerIndex: Int
        let projection: String
        let anchorExpert: Int
        let eligibleBytes: Int64
        let rawDemandBytes: Int64
        let dictionaryBytes: Int64
        let compressedDemandBytes: Int64
        let savedBytes: Int64
        let savedRatio: Double
        let model: String
        let residualEntropyBitsPerByte: Double
        let blockCount: Int64

        var jsonObject: [String: Any] {
            [
                "name": name,
                "layerIndex": layerIndex,
                "projection": projection,
                "anchorExpert": anchorExpert,
                "eligibleBytes": eligibleBytes,
                "rawDemandBytes": rawDemandBytes,
                "dictionaryBytes": dictionaryBytes,
                "compressedDemandBytes": compressedDemandBytes,
                "savedBytes": savedBytes,
                "savedRatio": savedRatio,
                "selectedModel": model,
                "residualEntropyBitsPerByte": residualEntropyBitsPerByte,
                "compressedBlockCount": blockCount,
            ]
        }
    }

    struct ProjectionStats {
        var residualBytes: Int64 = 0
        var byteHistogram = [UInt64](repeating: 0, count: 256)
        var nibbleHistogram = [UInt64](repeating: 0, count: 16)
    }

    public static func run(
        referenceDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex,
        outputDirectory: URL
    ) throws {
        if ProcessInfo.processInfo.environment["MLXFAST_CROSS_EXPERT_DELTA_CENSUS"] == "0" {
            try writeDisabledReport(outputDirectory: outputDirectory)
            return
        }

        let outputPath = outputDirectory
            .appendingPathComponent("experts", isDirectory: true)
            .appendingPathComponent("cross-expert-delta-census.json")
        try FileManager.default.createDirectory(
            at: outputPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let tensors = try discoverEligibleTensors(
            referenceDirectory: referenceDirectory,
            expertKeys: expertKeys,
            index: index
        )

        var summaries: [TensorSummary] = []
        summaries.reserveCapacity(tensors.count)
        var globalByteHistogram = [UInt64](repeating: 0, count: 256)
        var globalNibbleHistogram = [UInt64](repeating: 0, count: 16)
        var projectionStats: [String: ProjectionStats] = [:]
        var blockBucketHistogram: [[UInt64]] = Array(
            repeating: [UInt64](repeating: 0, count: 256),
            count: 16
        )
        var modelCounts: [String: Int] = [:]

        for tensor in tensors.sorted(by: tensorSortKey) {
            let result = try processTensor(
                tensor,
                referenceDirectory: referenceDirectory,
                globalByteHistogram: &globalByteHistogram,
                globalNibbleHistogram: &globalNibbleHistogram,
                projectionStats: &projectionStats,
                blockBucketHistogram: &blockBucketHistogram
            )
            summaries.append(result)
            modelCounts[result.model, default: 0] += 1
        }

        let eligibleBytes = summaries.reduce(Int64(0)) { $0 + $1.eligibleBytes }
        let dictionaryBytes = summaries.reduce(Int64(0)) { $0 + $1.dictionaryBytes }
        let rawDemandBytes = summaries.reduce(Int64(0)) { $0 + $1.rawDemandBytes }
        let compressedDemandBytes = summaries.reduce(Int64(0)) { $0 + $1.compressedDemandBytes }
        let savedBytes = rawDemandBytes - compressedDemandBytes
        let demandSavedRatio = rawDemandBytes > 0 ? Double(savedBytes) / Double(rawDemandBytes) : 0
        let onDiskBytes = dictionaryBytes + compressedDemandBytes
        let onDiskSavedRatio = eligibleBytes > 0 ? Double(eligibleBytes - onDiskBytes) / Double(eligibleBytes) : 0
        let recommended = demandSavedRatio >= minDemandSavedRatio && savedBytes >= minSavedBytes

        let topPositive = Array(summaries.sorted { $0.savedBytes > $1.savedBytes }.prefix(12))
        let topNegative = Array(summaries.sorted { $0.savedBytes < $1.savedBytes }.prefix(12))

        let projectionJSON = projectionStats.keys.sorted().reduce(into: [String: Any]()) { out, key in
            guard let stats = projectionStats[key] else { return }
            out[key] = [
                "residualBytes": stats.residualBytes,
                "byteEntropyBitsPerByte": entropyBitsPerSymbol(stats.byteHistogram),
                "nibbleEntropyBitsPerNibble": entropyBitsPerSymbol(stats.nibbleHistogram),
            ]
        }

        let bucketJSON = blockBucketHistogram.enumerated().map { bucket, hist in
            [
                "bucket": bucket,
                "residualBytes": Int64(hist.reduce(UInt64(0), +)),
                "byteEntropyBitsPerByte": entropyBitsPerSymbol(hist),
            ] as [String: Any]
        }

        let object: [String: Any] = [
            "version": version,
            "mechanism": mechanism,
            "eligible_bytes": eligibleBytes,
            "eligible_tensors": summaries.count,
            "dictionary_bytes_resident": dictionaryBytes,
            "estimated_demand_bytes_raw": rawDemandBytes,
            "estimated_demand_bytes_compressed": compressedDemandBytes,
            "estimated_demand_saved_bytes": savedBytes,
            "estimated_demand_saved_ratio": demandSavedRatio,
            "estimated_on_disk_saved_ratio_including_dictionary": onDiskSavedRatio,
            "selected_models": [
                "counts": modelCounts,
                "global_byte_entropy_bits_per_byte": entropyBitsPerSymbol(globalByteHistogram),
                "global_nibble_entropy_bits_per_nibble": entropyBitsPerSymbol(globalNibbleHistogram),
                "per_projection": projectionJSON,
                "per_block_position_bucket": bucketJSON,
            ],
            "runtimeIntegrationRecommended": recommended,
            "threshold": [
                "minDemandSavedRatio": minDemandSavedRatio,
                "minSavedBytes": minSavedBytes,
            ],
            "topPositiveTensors": topPositive.map { $0.jsonObject },
            "topNegativeTensors": topNegative.map { $0.jsonObject },
        ]

        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: outputPath)
    }

    private static func discoverEligibleTensors(
        referenceDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex
    ) throws -> [TensorCandidate] {
        let keysByShard = Dictionary(grouping: expertKeys) { key in
            index.weightMap[key] ?? ""
        }
        var tensors: [TensorCandidate] = []
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            for key in keysByShard[shardName, default: []].sorted() {
                guard !key.contains(".scales") else { continue }
                guard let info = header.tensors[key] else { continue }
                guard info.dtype == "U32", info.shape.count >= 2, info.shape.first == 256 else { continue }
                guard info.byteCount % info.shape[0] == 0 else { continue }
                guard let layerIndex = parseLayerIndex(key), let projection = parseProjection(key) else { continue }
                tensors.append(TensorCandidate(
                    name: key,
                    shardName: shardName,
                    layerIndex: layerIndex,
                    projection: projection,
                    shape: info.shape,
                    byteOffset: header.dataBaseOffset + UInt64(info.dataStart),
                    byteCount: info.byteCount,
                    expertCount: info.shape[0],
                    sliceLength: info.byteCount / info.shape[0]
                ))
            }
        }
        return tensors
    }

    private static func processTensor(
        _ tensor: TensorCandidate,
        referenceDirectory: URL,
        globalByteHistogram: inout [UInt64],
        globalNibbleHistogram: inout [UInt64],
        projectionStats: inout [String: ProjectionStats],
        blockBucketHistogram: inout [[UInt64]]
    ) throws -> TensorSummary {
        let shardURL = referenceDirectory.appendingPathComponent(tensor.shardName)
        let handle = try FileHandle(forReadingFrom: shardURL)
        defer { try? handle.close() }

        let candidates = candidateAnchors.filter { $0 < tensor.expertCount }
        guard !candidates.isEmpty else {
            throw MLXFastError.invalidInput("no anchor candidates for tensor \(tensor.name)")
        }
        let anchor = try chooseAnchor(
            tensor: tensor,
            candidates: candidates,
            handle: handle
        )

        var byteHistogram = [UInt64](repeating: 0, count: 256)
        var nibbleHistogram = [UInt64](repeating: 0, count: 16)
        var blockCount: Int64 = 0
        let blocksPerSlice = (tensor.sliceLength + blockSize - 1) / blockSize

        for blockIndex in 0..<blocksPerSlice {
            try autoreleasepool {
                let offsetInSlice = blockIndex * blockSize
                let count = min(blockSize, tensor.sliceLength - offsetInSlice)
                let anchorData = try readBlock(
                    handle: handle,
                    offset: tensor.byteOffset + UInt64(anchor * tensor.sliceLength + offsetInSlice),
                    count: count
                )
                try anchorData.withUnsafeBytes { anchorRaw in
                    guard let anchorPtr = anchorRaw.bindMemory(to: UInt8.self).baseAddress else { return }
                    for expert in 0..<tensor.expertCount where expert != anchor {
                        try autoreleasepool {
                            let currentData = try readBlock(
                                handle: handle,
                                offset: tensor.byteOffset + UInt64(expert * tensor.sliceLength + offsetInSlice),
                                count: count
                            )
                            blockCount += 1
                            let bucket = blockIndex % blockBucketHistogram.count
                            currentData.withUnsafeBytes { currentRaw in
                                guard let currentPtr = currentRaw.bindMemory(to: UInt8.self).baseAddress else { return }
                                for i in 0..<count {
                                    let value = Int(currentPtr[i] ^ anchorPtr[i])
                                    byteHistogram[value] += 1
                                    nibbleHistogram[value & 0x0f] += 1
                                    nibbleHistogram[value >> 4] += 1
                                    blockBucketHistogram[bucket][value] += 1
                                }
                            }
                        }
                    }
                }
            }
        }

        for i in 0..<256 { globalByteHistogram[i] += byteHistogram[i] }
        for i in 0..<16 { globalNibbleHistogram[i] += nibbleHistogram[i] }
        var pstats = projectionStats[tensor.projection] ?? ProjectionStats()
        for i in 0..<256 { pstats.byteHistogram[i] += byteHistogram[i] }
        for i in 0..<16 { pstats.nibbleHistogram[i] += nibbleHistogram[i] }
        let residualBytes = Int64(byteHistogram.reduce(UInt64(0), +))
        pstats.residualBytes += residualBytes
        projectionStats[tensor.projection] = pstats

        let modelEstimate = estimateCompressedBytes(
            byteHistogram: byteHistogram,
            nibbleHistogram: nibbleHistogram,
            blockCount: blockCount
        )
        let rawDemandBytes = Int64(tensor.byteCount - tensor.sliceLength)
        let savedBytes = rawDemandBytes - modelEstimate.bytes
        let savedRatio = rawDemandBytes > 0 ? Double(savedBytes) / Double(rawDemandBytes) : 0
        let entropy = entropyBitsPerSymbol(byteHistogram)
        return TensorSummary(
            name: tensor.name,
            layerIndex: tensor.layerIndex,
            projection: tensor.projection,
            anchorExpert: anchor,
            eligibleBytes: Int64(tensor.byteCount),
            rawDemandBytes: rawDemandBytes,
            dictionaryBytes: Int64(tensor.sliceLength),
            compressedDemandBytes: modelEstimate.bytes,
            savedBytes: savedBytes,
            savedRatio: savedRatio,
            model: modelEstimate.model,
            residualEntropyBitsPerByte: entropy,
            blockCount: blockCount
        )
    }

    private static func chooseAnchor(
        tensor: TensorCandidate,
        candidates: [Int],
        handle: FileHandle
    ) throws -> Int {
        let prefixBytes = min(prefixLimit, tensor.sliceLength)
        let prefixBlocks = (prefixBytes + blockSize - 1) / blockSize
        var histograms = Array(repeating: [UInt64](repeating: 0, count: 256), count: candidates.count)

        for blockIndex in 0..<prefixBlocks {
            try autoreleasepool {
                let offsetInSlice = blockIndex * blockSize
                let count = min(blockSize, prefixBytes - offsetInSlice)
                var anchorBlocks: [Data] = []
                anchorBlocks.reserveCapacity(candidates.count)
                for candidate in candidates {
                    anchorBlocks.append(try readBlock(
                        handle: handle,
                        offset: tensor.byteOffset + UInt64(candidate * tensor.sliceLength + offsetInSlice),
                        count: count
                    ))
                }

                for expert in 0..<tensor.expertCount {
                    try autoreleasepool {
                        let currentData = try readBlock(
                            handle: handle,
                            offset: tensor.byteOffset + UInt64(expert * tensor.sliceLength + offsetInSlice),
                            count: count
                        )
                        currentData.withUnsafeBytes { currentRaw in
                            guard let currentPtr = currentRaw.bindMemory(to: UInt8.self).baseAddress else { return }
                            for cidx in 0..<candidates.count where candidates[cidx] != expert {
                                anchorBlocks[cidx].withUnsafeBytes { anchorRaw in
                                    guard let anchorPtr = anchorRaw.bindMemory(to: UInt8.self).baseAddress else { return }
                                    for i in 0..<count {
                                        histograms[cidx][Int(currentPtr[i] ^ anchorPtr[i])] += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        var bestIndex = 0
        var bestBits = Double.greatestFiniteMagnitude
        for i in 0..<candidates.count {
            let bits = entropyTotalBits(histograms[i])
            if bits < bestBits {
                bestBits = bits
                bestIndex = i
            }
        }
        return candidates[bestIndex]
    }

    private static func estimateCompressedBytes(
        byteHistogram: [UInt64],
        nibbleHistogram: [UInt64],
        blockCount: Int64
    ) -> (model: String, bytes: Int64) {
        let residualBytes = Int64(byteHistogram.reduce(UInt64(0), +))
        let randomAccessOverhead = blockCount * Int64(metadataBytesPerBlock + alignmentPenaltyPerBlock)
        let byteHuffman = ceilDiv(Int64(ceil(staticCodeBits(byteHistogram))), 8) + 512 + randomAccessOverhead
        let nibbleHuffman = ceilDiv(Int64(ceil(staticCodeBits(nibbleHistogram))), 8) + 64 + randomAccessOverhead
        let ransIdeal = ceilDiv(Int64(ceil(entropyTotalBits(byteHistogram) * 1.015)), 8) + 1024 + randomAccessOverhead
        let raw = residualBytes + randomAccessOverhead
        let options = [
            ("raw_residual_bytes", raw),
            ("canonical_static_huffman_bytes", byteHuffman),
            ("canonical_static_huffman_nibbles", nibbleHuffman),
            ("rans_ideal_plus_1p5pct", ransIdeal),
        ]
        return options.min { $0.1 < $1.1 } ?? ("raw_residual_bytes", raw)
    }

    private static func staticCodeBits(_ histogram: [UInt64]) -> Double {
        let total = histogram.reduce(UInt64(0), +)
        guard total > 0 else { return 0 }
        let totalDouble = Double(total)
        var bits = 0.0
        let nonZero = histogram.filter { $0 > 0 }.count
        if nonZero <= 1 {
            return totalDouble
        }
        for count in histogram where count > 0 {
            let p = Double(count) / totalDouble
            bits += Double(count) * ceil(-log2(p))
        }
        return bits
    }

    private static func entropyTotalBits(_ histogram: [UInt64]) -> Double {
        let total = histogram.reduce(UInt64(0), +)
        guard total > 0 else { return 0 }
        let totalDouble = Double(total)
        var bits = 0.0
        for count in histogram where count > 0 {
            let p = Double(count) / totalDouble
            bits += Double(count) * -log2(p)
        }
        return bits
    }

    private static func entropyBitsPerSymbol(_ histogram: [UInt64]) -> Double {
        let total = histogram.reduce(UInt64(0), +)
        guard total > 0 else { return 0 }
        return entropyTotalBits(histogram) / Double(total)
    }

    private static func readBlock(handle: FileHandle, offset: UInt64, count: Int) throws -> Data {
        try handle.seek(toOffset: offset)
        let data = handle.readData(ofLength: count)
        guard data.count == count else {
            throw MLXFastError.invalidInput("unexpected EOF during cross-expert delta census")
        }
        return data
    }

    private static func ceilDiv(_ value: Int64, _ divisor: Int64) -> Int64 {
        guard value > 0 else { return 0 }
        return (value + divisor - 1) / divisor
    }

    private static func parseLayerIndex(_ name: String) -> Int? {
        parseInt(name, after: ".layers.")
    }

    private static func parseProjection(_ name: String) -> String? {
        for candidate in ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"] {
            if name.contains(".\(candidate).") {
                switch candidate {
                case "w1": return "gate_proj"
                case "w2": return "down_proj"
                case "w3": return "up_proj"
                default: return candidate
                }
            }
        }
        return nil
    }

    private static func parseInt(_ name: String, after marker: String) -> Int? {
        guard let range = name.range(of: marker) else { return nil }
        let suffix = name[range.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return Int(digits)
    }

    private static func tensorSortKey(_ lhs: TensorCandidate, _ rhs: TensorCandidate) -> Bool {
        if lhs.layerIndex != rhs.layerIndex { return lhs.layerIndex < rhs.layerIndex }
        if lhs.projection != rhs.projection { return lhs.projection < rhs.projection }
        return lhs.name < rhs.name
    }

    private static func writeDisabledReport(outputDirectory: URL) throws {
        let outputPath = outputDirectory
            .appendingPathComponent("experts", isDirectory: true)
            .appendingPathComponent("cross-expert-delta-census.json")
        try FileManager.default.createDirectory(
            at: outputPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let object: [String: Any] = [
            "version": version,
            "mechanism": mechanism,
            "disabled": true,
            "eligible_bytes": 0,
            "eligible_tensors": 0,
            "dictionary_bytes_resident": 0,
            "estimated_demand_bytes_raw": 0,
            "estimated_demand_bytes_compressed": 0,
            "estimated_demand_saved_bytes": 0,
            "estimated_demand_saved_ratio": 0.0,
            "estimated_on_disk_saved_ratio_including_dictionary": 0.0,
            "selected_models": [:],
            "runtimeIntegrationRecommended": false,
            "threshold": [
                "minDemandSavedRatio": minDemandSavedRatio,
                "minSavedBytes": minSavedBytes,
            ],
            "topPositiveTensors": [],
            "topNegativeTensors": [],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: outputPath)
    }
}
