import Foundation
import MLXFastCore

struct MXFP4PredictiveEntropyCensus {
    private static let blockSizeBytes = 1 * 1024 * 1024
    private static let chunkSizeBytes = 8 * 1024 * 1024
    private static let defaultScanBudgetBytes: Int64 = 16 * 1024 * 1024
    private static let projectionCount = 3
    private static let laneCount = 8

    struct Report: Codable {
        struct Family: Codable {
            let rawU32Bytes: Int64
            let idealArithmeticBytes: Int64
            let finiteStateBytes: Int64
            let codebookBytes: Int64
            let randomAccessOverheadBytes: Int64
            let estimatedCompressedBytes: Int64
            let estimatedSavedBytes: Int64
            let estimatedSavedRatio: Double
        }

        struct Summary: Codable {
            let key: String
            let rawU32Bytes: Int64
            let estimatedSavedBytes: Int64
            let estimatedSavedRatio: Double
        }

        struct SliceSummary: Codable {
            let tensor: String
            let slice: Int
            let layer: Int?
            let projection: String
            let rawU32Bytes: Int64
            let blockCount: Int64
            let estimatedSavedBytes: Int64
            let estimatedSavedRatio: Double
        }

        let version: Int
        let rawU32Bytes: Int64
        let eligibleTensorCount: Int
        let eligibleSliceCount: Int
        let blockSizeBytes: Int
        let blockCount: Int64
        let families: [String: Family]
        let selectedFamily: String
        let perProjection: [Summary]
        let perLayer: [Summary]
        let runtimeIntegrationRecommended: Bool
        let topSlices: [SliceSummary]
        let worstSlices: [SliceSummary]
    }

    private struct SliceMetrics {
        let tensor: String
        let slice: Int
        let layer: Int?
        let projection: Int
        let rawBytes: Int64
        let blockCount: Int64
        let staticIdealBytes: Int64
        let markovIdealBytes: Int64
        let splitIdealBytes: Int64
    }

    private final class Histogram {
        var staticSymbol = Array(repeating: Int64(0), count: projectionCount * laneCount * 16)
        var markov = Array(repeating: Int64(0), count: projectionCount * laneCount * 17 * 16)
        var splitMag = Array(repeating: Int64(0), count: projectionCount * laneCount * 9 * 8)
        var splitSignResidual = Array(repeating: Int64(0), count: projectionCount * laneCount * 9 * 2)

        func record(projection: Int, lane: Int, symbol: Int, previousSymbol: Int, previousMag: Int, previousSign: Int) {
            let mag = symbol & 0x7
            let sign = symbol >> 3
            staticSymbol[((projection * laneCount + lane) * 16) + symbol] += 1

            let previousSymbolContext = previousSymbol < 0 ? 16 : previousSymbol
            markov[(((projection * laneCount + lane) * 17 + previousSymbolContext) * 16) + symbol] += 1

            let previousMagContext = previousMag < 0 ? 8 : previousMag
            splitMag[(((projection * laneCount + lane) * 9 + previousMagContext) * 8) + mag] += 1

            // The residual is cold-started as the sign itself, but still conditioned by current magnitude.
            let residual = previousSign < 0 ? sign : (sign ^ previousSign)
            splitSignResidual[(((projection * laneCount + lane) * 9 + mag) * 2) + residual] += 1
        }
    }

    static func run(
        referenceDirectory: URL,
        expertKeys: Set<String>,
        index: CheckpointIndex,
        outputDirectory: URL
    ) throws -> Report {
        let expertsDirectory = outputDirectory.appendingPathComponent("experts", isDirectory: true)
        try FileManager.default.createDirectory(at: expertsDirectory, withIntermediateDirectories: true)

        let global = Histogram()
        var sliceMetrics: [SliceMetrics] = []
        var rawBytes: Int64 = 0
        var eligibleTensorCount = 0
        var eligibleSliceCount = 0
        var totalBlockCount: Int64 = 0
        let scanBudgetBytes = configuredScanBudgetBytes()
        var scannedBytes: Int64 = 0

        let keysByShard = Dictionary(grouping: expertKeys.filter { SwiftTransform.isExpertKey($0) }) { key in
            index.weightMap[key] ?? ""
        }

        for shardName in keysByShard.keys.sorted() {
            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            let header = try Safetensors.readHeader(shardURL)
            guard header.dataBaseOffset <= UInt64(Int.max) else {
                throw MLXFastError.invalidInput("safetensors data base offset is too large: \(shardName)")
            }
            let baseOffset = Int64(header.dataBaseOffset)
            let handle = try FileHandle(forReadingFrom: shardURL)
            defer { try? handle.close() }

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput("expert tensor \(key) is listed in index but missing from \(shardName)")
                }
                guard isEligibleTensor(key: key, info: info) else {
                    continue
                }
                guard info.byteCount % 256 == 0 else {
                    throw MLXFastError.invalidInput("eligible U32 tensor \(key) byte count is not divisible by 256")
                }
                guard info.byteCount % 4 == 0, info.dataStart % 4 == 0, info.dataEnd % 4 == 0 else {
                    throw MLXFastError.invalidInput("eligible U32 tensor \(key) has unaligned byte range")
                }

                eligibleTensorCount += 1
                let projection = parseProjection(key)
                let layer = parseLayer(key)
                let sliceBytes = info.byteCount / 256
                let tensorDataOffset = baseOffset + Int64(info.dataStart)

                for slice in 0..<256 {
                    if scannedBytes + Int64(sliceBytes) > scanBudgetBytes {
                        break
                    }
                    let offset = tensorDataOffset + Int64(slice * sliceBytes)
                    let local = Histogram()
                    let blocks = try scanSlice(
                        handle: handle,
                        absoluteOffset: offset,
                        byteCount: sliceBytes,
                        projection: projection,
                        global: global,
                        local: local
                    )
                    let sliceRawBytes = Int64(sliceBytes)
                    rawBytes += sliceRawBytes
                    scannedBytes += sliceRawBytes
                    eligibleSliceCount += 1
                    totalBlockCount += blocks

                    sliceMetrics.append(SliceMetrics(
                        tensor: key,
                        slice: slice,
                        layer: layer,
                        projection: projection,
                        rawBytes: sliceRawBytes,
                        blockCount: blocks,
                        staticIdealBytes: entropyBytes(local.staticSymbol, alphabetSize: 16),
                        markovIdealBytes: entropyBytes(local.markov, alphabetSize: 16),
                        splitIdealBytes: entropyBytes(local.splitMag, alphabetSize: 8) + entropyBytes(local.splitSignResidual, alphabetSize: 2)
                    ))
                }
            }
        }

        let randomAccessOverhead = Int64(totalBlockCount * 16 + Int64(eligibleSliceCount) * 64)
        let staticFamily = makeFamily(
            rawBytes: rawBytes,
            idealBytes: entropyBytes(global.staticSymbol, alphabetSize: 16),
            contextCount: projectionCount * laneCount,
            alphabetSize: 16,
            randomAccessOverhead: randomAccessOverhead
        )
        let markovFamily = makeFamily(
            rawBytes: rawBytes,
            idealBytes: entropyBytes(global.markov, alphabetSize: 16),
            contextCount: projectionCount * laneCount * 17,
            alphabetSize: 16,
            randomAccessOverhead: randomAccessOverhead
        )
        let splitFamily = makeFamily(
            rawBytes: rawBytes,
            idealBytes: entropyBytes(global.splitMag, alphabetSize: 8) + entropyBytes(global.splitSignResidual, alphabetSize: 2),
            contextCount: projectionCount * laneCount * 9 + projectionCount * laneCount * 9,
            alphabetSize: 8, // conservative-ish combined proxy; codebook is floored at 16 KiB below.
            randomAccessOverhead: randomAccessOverhead
        )

        let markovImprovement = markovFamily.estimatedSavedRatio - splitFamily.estimatedSavedRatio
        let selectedName = markovImprovement >= 0.0075 ? "markov_projection_lane_prevSymbol" : "split_projection_lane_prevMag"

        let selectedSlices = sliceMetrics.map { metric in
            makeSliceSummary(metric, selectedFamily: selectedName)
        }
        let topSlices = selectedSlices.sorted {
            if $0.estimatedSavedRatio == $1.estimatedSavedRatio { return $0.estimatedSavedBytes > $1.estimatedSavedBytes }
            return $0.estimatedSavedRatio > $1.estimatedSavedRatio
        }.prefix(20).map { $0 }
        let worstSlices = selectedSlices.sorted {
            if $0.estimatedSavedRatio == $1.estimatedSavedRatio { return $0.estimatedSavedBytes < $1.estimatedSavedBytes }
            return $0.estimatedSavedRatio < $1.estimatedSavedRatio
        }.prefix(20).map { $0 }

        let perProjection = summarize(selectedSlices) { $0.projection }
        let perLayer = summarize(selectedSlices) { summary in
            if let layer = summary.layer { return String(layer) }
            return "unknown"
        }

        let selectedFamily = selectedName == "markov_projection_lane_prevSymbol" ? markovFamily : splitFamily
        let report = Report(
            version: 1,
            rawU32Bytes: rawBytes,
            eligibleTensorCount: eligibleTensorCount,
            eligibleSliceCount: eligibleSliceCount,
            blockSizeBytes: blockSizeBytes,
            blockCount: totalBlockCount,
            families: [
                "static_projection_lane_symbol": staticFamily,
                "markov_projection_lane_prevSymbol": markovFamily,
                "split_projection_lane_prevMag": splitFamily,
            ],
            selectedFamily: selectedName,
            perProjection: perProjection,
            perLayer: perLayer,
            runtimeIntegrationRecommended: selectedFamily.estimatedSavedRatio >= 0.04 && selectedFamily.estimatedSavedBytes >= 5_000_000_000,
            topSlices: Array(topSlices),
            worstSlices: Array(worstSlices)
        )

        if scannedBytes >= scanBudgetBytes {
            print("MXFP4 predictive entropy census: stopped at scan budget \(scanBudgetBytes) bytes")
        }
        let outputURL = expertsDirectory.appendingPathComponent("mxfp4-predictive-entropy-census.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(to: outputURL)
        print("MXFP4 predictive entropy census: selected \(selectedName), saved \(selectedFamily.estimatedSavedBytes) bytes (\(selectedFamily.estimatedSavedRatio))")
        return report
    }

    private static func configuredScanBudgetBytes() -> Int64 {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_MXFP4_PREDICTIVE_CENSUS_BYTES"] else {
            return defaultScanBudgetBytes
        }
        guard let value = Int64(raw), value > 0 else {
            return defaultScanBudgetBytes
        }
        return value
    }

    private static func isEligibleTensor(key: String, info: SafetensorInfo) -> Bool {
        guard SwiftTransform.isExpertKey(key) else { return false }
        guard info.dtype == TensorDType.u32.rawValue else { return false }
        guard info.shape.count >= 3, info.shape.first == 256 else { return false }
        return true
    }

    private static func scanSlice(
        handle: FileHandle,
        absoluteOffset: Int64,
        byteCount: Int,
        projection: Int,
        global: Histogram,
        local: Histogram
    ) throws -> Int64 {
        var processed = 0
        var blocks: Int64 = 0
        while processed < byteCount {
            let blockBytes = min(blockSizeBytes, byteCount - processed)
            var remainingInBlock = blockBytes
            var blockCursor = 0
            var prevSymbolByLane = Array(repeating: -1, count: laneCount)
            var prevMagByLane = Array(repeating: -1, count: laneCount)
            var prevSignByLane = Array(repeating: -1, count: laneCount)
            blocks += 1

            try handle.seek(toOffset: UInt64(absoluteOffset + Int64(processed)))
            while remainingInBlock > 0 {
                let data = handle.readData(ofLength: min(chunkSizeBytes, remainingInBlock))
                if data.isEmpty {
                    throw MLXFastError.invalidInput("unexpected EOF while scanning MXFP4 entropy census")
                }
                data.withUnsafeBytes { rawBuffer in
                    guard let base = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                    for index in 0..<data.count {
                        let byte = Int(base[index])
                        let byteOffsetInSlice = processed + blockCursor + index
                        let laneBase = (byteOffsetInSlice & 3) * 2
                        recordSymbol(byte & 0x0f, lane: laneBase, projection: projection, global: global, local: local, previousSymbols: &prevSymbolByLane, previousMags: &prevMagByLane, previousSigns: &prevSignByLane)
                        recordSymbol(byte >> 4, lane: laneBase + 1, projection: projection, global: global, local: local, previousSymbols: &prevSymbolByLane, previousMags: &prevMagByLane, previousSigns: &prevSignByLane)
                    }
                }
                remainingInBlock -= data.count
                blockCursor += data.count
            }
            processed += blockBytes
        }
        return blocks
    }

    private static func recordSymbol(
        _ symbol: Int,
        lane: Int,
        projection: Int,
        global: Histogram,
        local: Histogram,
        previousSymbols: inout [Int],
        previousMags: inout [Int],
        previousSigns: inout [Int]
    ) {
        let previousSymbol = previousSymbols[lane]
        let previousMag = previousMags[lane]
        let previousSign = previousSigns[lane]
        global.record(projection: projection, lane: lane, symbol: symbol, previousSymbol: previousSymbol, previousMag: previousMag, previousSign: previousSign)
        local.record(projection: projection, lane: lane, symbol: symbol, previousSymbol: previousSymbol, previousMag: previousMag, previousSign: previousSign)
        previousSymbols[lane] = symbol
        previousMags[lane] = symbol & 0x7
        previousSigns[lane] = symbol >> 3
    }

    private static func makeFamily(rawBytes: Int64, idealBytes: Int64, contextCount: Int, alphabetSize: Int, randomAccessOverhead: Int64) -> Report.Family {
        let codebookBytes = max(Int64(16 * 1024), Int64(contextCount * alphabetSize * 2))
        let finiteStateBytes = idealBytes + Int64(ceil(Double(rawBytes) * 0.0075)) + codebookBytes
        let estimatedCompressedBytes = finiteStateBytes + randomAccessOverhead
        let saved = rawBytes - estimatedCompressedBytes
        let ratio = rawBytes > 0 ? Double(saved) / Double(rawBytes) : 0
        return Report.Family(
            rawU32Bytes: rawBytes,
            idealArithmeticBytes: idealBytes,
            finiteStateBytes: finiteStateBytes,
            codebookBytes: codebookBytes,
            randomAccessOverheadBytes: randomAccessOverhead,
            estimatedCompressedBytes: estimatedCompressedBytes,
            estimatedSavedBytes: saved,
            estimatedSavedRatio: ratio
        )
    }

    private static func makeSliceSummary(_ metric: SliceMetrics, selectedFamily: String) -> Report.SliceSummary {
        let ideal: Int64
        let contextCount: Int
        let alphabetSize: Int
        if selectedFamily == "markov_projection_lane_prevSymbol" {
            ideal = metric.markovIdealBytes
            contextCount = laneCount * 17
            alphabetSize = 16
        } else if selectedFamily == "static_projection_lane_symbol" {
            ideal = metric.staticIdealBytes
            contextCount = laneCount
            alphabetSize = 16
        } else {
            ideal = metric.splitIdealBytes
            contextCount = laneCount * 9 + laneCount * 9
            alphabetSize = 8
        }
        let overhead = metric.blockCount * 16 + 64
        let codebook = max(Int64(16 * 1024), Int64(contextCount * alphabetSize * 2))
        let finite = ideal + Int64(ceil(Double(metric.rawBytes) * 0.0075)) + codebook
        let compressed = finite + overhead
        let saved = metric.rawBytes - compressed
        let ratio = metric.rawBytes > 0 ? Double(saved) / Double(metric.rawBytes) : 0
        return Report.SliceSummary(
            tensor: metric.tensor,
            slice: metric.slice,
            layer: metric.layer,
            projection: projectionName(metric.projection),
            rawU32Bytes: metric.rawBytes,
            blockCount: metric.blockCount,
            estimatedSavedBytes: saved,
            estimatedSavedRatio: ratio
        )
    }

    private static func summarize(_ slices: [Report.SliceSummary], key: (Report.SliceSummary) -> String) -> [Report.Summary] {
        var raw: [String: Int64] = [:]
        var saved: [String: Int64] = [:]
        for slice in slices {
            let k = key(slice)
            raw[k, default: 0] += slice.rawU32Bytes
            saved[k, default: 0] += slice.estimatedSavedBytes
        }
        return raw.keys.sorted { lhs, rhs in
            if let li = Int(lhs), let ri = Int(rhs) { return li < ri }
            return lhs < rhs
        }.map { k in
            let r = raw[k, default: 0]
            let s = saved[k, default: 0]
            return Report.Summary(key: k, rawU32Bytes: r, estimatedSavedBytes: s, estimatedSavedRatio: r > 0 ? Double(s) / Double(r) : 0)
        }
    }

    private static func entropyBytes(_ counts: [Int64], alphabetSize: Int) -> Int64 {
        var bits = 0.0
        var start = 0
        while start < counts.count {
            let end = start + alphabetSize
            let total = counts[start..<end].reduce(Int64(0), +)
            if total > 0 {
                let totalDouble = Double(total)
                for index in start..<end {
                    let count = counts[index]
                    if count > 0 {
                        bits += Double(count) * (log2(totalDouble) - log2(Double(count)))
                    }
                }
            }
            start = end
        }
        return Int64(ceil(bits / 8.0))
    }

    private static func parseProjection(_ key: String) -> Int {
        if key.contains("gate_proj") || key.contains("w1") { return 0 }
        if key.contains("up_proj") || key.contains("w3") { return 1 }
        if key.contains("down_proj") || key.contains("w2") { return 2 }
        return 0
    }

    private static func projectionName(_ projection: Int) -> String {
        switch projection {
        case 0: return "gate"
        case 1: return "up"
        case 2: return "down"
        default: return "unknown"
        }
    }

    private static func parseLayer(_ key: String) -> Int? {
        guard let range = key.range(of: ".layers.") else { return nil }
        var index = range.upperBound
        var digits = ""
        while index < key.endIndex {
            let character = key[index]
            guard character >= "0" && character <= "9" else { break }
            digits.append(character)
            index = key.index(after: index)
        }
        return Int(digits)
    }
}
