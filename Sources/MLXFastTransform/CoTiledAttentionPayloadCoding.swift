import Foundation
import MLXFastCore

/// Authors decode-only attention payloads whose bytes are ordered exactly as
/// the fused sliding-QKV and full-QK Metal kernels consume them. Each payload
/// retains the original quantized weights for the prefill path; it is an
/// additional, input-independent decode layout only.
///
/// Metadata is packed directly from the canonical U16 `.metadata_indices`
/// tensors produced by `AffineMetadataCoding`. There is deliberately no
/// standalone fixed12/fixed13 sidecar: the compact representation exists only
/// inside this co-tiled payload and is exhaustively checked against U16 again
/// by the runtime before any MLX arrays are materialized.
enum CoTiledAttentionPayloadCoding {
    static let shardName = "mlxfast-attention-cotiled-payload.safetensors"
    static let slidingSuffix = ".self_attn.qkv_cotiled_payload"
    static let fullSuffix = ".self_attn.qk_cotiled_payload"

    static let groupsPerRow = 84
    static let weightWordsPerRow = 672
    static let weightBytesPerRow = 2_688
    static let blockBytes = 128
    static let pairCount = 10
    static let rowsPerSlot = 4

    struct ProjectionSource {
        let weight: Data
        let indices: Data
        let rows: Int
        let indexBits: Int
        let lutCount: Int
        let slots: Int
    }

    static func indexBits(lutCount: Int) -> Int? {
        if (1...4_096).contains(lutCount) { return 12 }
        if (4_097...8_192).contains(lutCount) { return 13 }
        return nil
    }

    static func wordsPerThreadgroup(slotIndexBits: [Int]) -> Int? {
        guard !slotIndexBits.isEmpty,
              slotIndexBits.allSatisfy({ $0 == 12 || $0 == 13 })
        else {
            return nil
        }
        let slots = slotIndexBits.count
        let pairMetadataBytes = slotIndexBits.reduce(0) { $0 + rowsPerSlot * $1 }
        return pairCount * (256 * slots + pairMetadataBytes / 4) + 136 * slots
    }

    static func slotIndexBits(projections: [ProjectionSource]) -> [Int] {
        projections.flatMap {
            [Int](repeating: $0.indexBits, count: $0.slots)
        }
    }

    static func threadgroupCount(projections: [ProjectionSource]) -> Int? {
        var result: Int?
        for projection in projections {
            guard projection.slots > 0,
                  projection.rows > 0,
                  projection.rows.isMultiple(of: rowsPerSlot * projection.slots)
            else {
                return nil
            }
            let count = projection.rows / (rowsPerSlot * projection.slots)
            if let result, result != count { return nil }
            result = count
        }
        return result
    }

    /// Builds one payload. Pair metadata uses the fixed12/fixed13 byte planes:
    /// four even low bytes, four even-high/odd-low nibbles, four odd high
    /// bytes, and (for 13-bit indexes) one top-bit byte. The tail stores the
    /// four remaining U16 indexes lane-major without padding.
    static func makePayload(projections: [ProjectionSource]) -> Data? {
        guard !projections.isEmpty,
              let threadgroups = threadgroupCount(projections: projections),
              let wordsPerThreadgroup = wordsPerThreadgroup(
                  slotIndexBits: slotIndexBits(projections: projections)
              )
        else {
            return nil
        }
        for projection in projections {
            let (weightBytes, weightOverflow) = projection.rows
                .multipliedReportingOverflow(by: weightBytesPerRow)
            let (indexElements, indexOverflow) = projection.rows
                .multipliedReportingOverflow(by: groupsPerRow)
            let (indexBytes, indexByteOverflow) = indexElements
                .multipliedReportingOverflow(by: 2)
            guard !weightOverflow, !indexOverflow, !indexByteOverflow,
                  projection.weight.count == weightBytes,
                  projection.indices.count == indexBytes,
                  indexBits(lutCount: projection.lutCount) == projection.indexBits
            else {
                return nil
            }
        }

        var slotProjection: [Int] = []
        var slotWithinProjection: [Int] = []
        for (projectionIndex, projection) in projections.enumerated() {
            for slot in 0..<projection.slots {
                slotProjection.append(projectionIndex)
                slotWithinProjection.append(slot)
            }
        }
        let slots = slotProjection.count
        var metadataOffsets: [Int] = []
        var metadataCursor = 0
        for slot in 0..<slots {
            metadataOffsets.append(metadataCursor)
            metadataCursor += rowsPerSlot
                * projections[slotProjection[slot]].indexBits
        }
        let pairTileBytes = 1_024 * slots + metadataCursor
        let tailTileBytes = 544 * slots
        let threadgroupBytes = pairCount * pairTileBytes + tailTileBytes
        guard threadgroupBytes == wordsPerThreadgroup * 4 else { return nil }
        let (payloadBytes, overflow) = threadgroups
            .multipliedReportingOverflow(by: threadgroupBytes)
        guard !overflow else { return nil }

        var payload = [UInt8](repeating: 0, count: payloadBytes)
        var indexesValid = true
        payload.withUnsafeMutableBytes { output in
            guard let outputBase = output.baseAddress else {
                indexesValid = false
                return
            }
            withProjectionBytes(projections) { weightPointers, indexPointers in
                @inline(__always)
                func readIndex(
                    projectionIndex: Int,
                    row: Int,
                    group: Int
                ) -> UInt16 {
                    let offset = 2 * (row * groupsPerRow + group)
                    let bytes = indexPointers[projectionIndex]
                    return UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
                }

                for threadgroup in 0..<threadgroups {
                    let tileBase = threadgroup * threadgroupBytes
                    for pair in 0..<pairCount {
                        let pairBase = tileBase + pair * pairTileBytes
                        let oddBase = pairBase + 512 * slots
                        let metadataBase = pairBase + 1_024 * slots
                        for slot in 0..<slots {
                            let projectionIndex = slotProjection[slot]
                            let projection = projections[projectionIndex]
                            guard let weightBase = weightPointers[projectionIndex].baseAddress
                            else {
                                indexesValid = false
                                continue
                            }
                            let rowBase = threadgroup
                                * rowsPerSlot * projection.slots
                                + rowsPerSlot * slotWithinProjection[slot]
                            for row in 0..<rowsPerSlot {
                                let sourceRow = rowBase + row
                                let weightRowBase = sourceRow * weightBytesPerRow
                                (outputBase + pairBase + 512 * slot + 128 * row)
                                    .copyMemory(
                                        from: weightBase + weightRowBase
                                            + (2 * pair) * blockBytes,
                                        byteCount: blockBytes
                                    )
                                (outputBase + oddBase + 512 * slot + 128 * row)
                                    .copyMemory(
                                        from: weightBase + weightRowBase
                                            + (2 * pair + 1) * blockBytes,
                                        byteCount: blockBytes
                                    )

                                let packed = (outputBase + metadataBase
                                    + metadataOffsets[slot]
                                    + row * projection.indexBits)
                                    .assumingMemoryBound(to: UInt8.self)
                                var topBits: UInt8 = 0
                                for laneGroup in 0..<4 {
                                    let even = readIndex(
                                        projectionIndex: projectionIndex,
                                        row: sourceRow,
                                        group: 8 * pair + laneGroup
                                    )
                                    let odd = readIndex(
                                        projectionIndex: projectionIndex,
                                        row: sourceRow,
                                        group: 8 * pair + 4 + laneGroup
                                    )
                                    guard Int(even) < projection.lutCount,
                                          Int(odd) < projection.lutCount
                                    else {
                                        indexesValid = false
                                        continue
                                    }
                                    packed[laneGroup] = UInt8(truncatingIfNeeded: even)
                                    packed[4 + laneGroup] = UInt8(
                                        truncatingIfNeeded: (even >> 8)
                                            | ((odd & 0x000f) << 4)
                                    )
                                    packed[8 + laneGroup] = UInt8(
                                        truncatingIfNeeded: odd >> 4
                                    )
                                    if projection.indexBits == 13 {
                                        topBits |= UInt8((even >> 12) & 1) << laneGroup
                                        topBits |= UInt8((odd >> 12) & 1)
                                            << (4 + laneGroup)
                                    }
                                }
                                if projection.indexBits == 13 {
                                    packed[12] = topBits
                                }
                            }
                        }
                    }

                    let tailBase = tileBase + pairCount * pairTileBytes
                    let tailMetadataBase = tailBase + 512 * slots
                    for slot in 0..<slots {
                        let projectionIndex = slotProjection[slot]
                        let projection = projections[projectionIndex]
                        guard let weightBase = weightPointers[projectionIndex].baseAddress
                        else {
                            indexesValid = false
                            continue
                        }
                        let rowBase = threadgroup
                            * rowsPerSlot * projection.slots
                            + rowsPerSlot * slotWithinProjection[slot]
                        for row in 0..<rowsPerSlot {
                            let sourceRow = rowBase + row
                            (outputBase + tailBase + 512 * slot + 128 * row)
                                .copyMemory(
                                    from: weightBase + sourceRow * weightBytesPerRow
                                        + 2 * pairCount * blockBytes,
                                    byteCount: blockBytes
                                )
                            let packed = (outputBase + tailMetadataBase
                                + 32 * slot + 8 * row)
                                .assumingMemoryBound(to: UInt8.self)
                            for laneGroup in 0..<4 {
                                let value = readIndex(
                                    projectionIndex: projectionIndex,
                                    row: sourceRow,
                                    group: 80 + laneGroup
                                )
                                guard Int(value) < projection.lutCount else {
                                    indexesValid = false
                                    continue
                                }
                                packed[2 * laneGroup] = UInt8(truncatingIfNeeded: value)
                                packed[2 * laneGroup + 1] = UInt8(
                                    truncatingIfNeeded: value >> 8
                                )
                            }
                        }
                    }
                }
            }
        }
        return indexesValid ? Data(payload) : nil
    }

    private static func withProjectionBytes(
        _ projections: [ProjectionSource],
        _ body: ([UnsafeRawBufferPointer], [UnsafeRawBufferPointer]) -> Void
    ) {
        func recurse(
            _ index: Int,
            _ weights: [UnsafeRawBufferPointer],
            _ indices: [UnsafeRawBufferPointer]
        ) {
            guard index < projections.count else {
                body(weights, indices)
                return
            }
            projections[index].weight.withUnsafeBytes { weightBytes in
                projections[index].indices.withUnsafeBytes { indexBytes in
                    recurse(index + 1, weights + [weightBytes], indices + [indexBytes])
                }
            }
        }
        recurse(0, [], [])
    }

    private struct EligibleLayer {
        let layerStem: String
        let tensorName: String
        let stems: [String]
        let indexNames: [String]
        let indexBits: [Int]
        let lutCounts: [Int]
        let slots: [Int]
        let rows: [Int]
        let threadgroups: Int
        let wordsPerThreadgroup: Int
    }

    /// Shape-driven and fail-open: non-production/synthetic layers that do
    /// not match either fused decode geometry simply receive no payload.
    static func writeSidecar(
        stagingDirectory: URL,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedAffineMetadataReport {
        let metadataURL = stagingDirectory.appendingPathComponent(
            AffineMetadataCoding.shardName
        )
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        let metadataHeader = try Safetensors.readHeader(metadataURL)
        let layerPrefix = "language_model.model.layers."
        let qWeightSuffix = ".self_attn.q_proj.weight"
        let layerStems = Set(selectedKeys.compactMap { key -> String? in
            guard key.hasPrefix(layerPrefix), key.hasSuffix(qWeightSuffix) else {
                return nil
            }
            return String(key.dropLast(qWeightSuffix.count))
        })
        guard !layerStems.isEmpty else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }

        var eligible: [EligibleLayer] = []
        for layerStem in layerStems.sorted() {
            if let layer = try eligibleLayer(
                layerStem: layerStem,
                index: index,
                stagedHeaders: stagedHeaders,
                selectedKeys: selectedKeys,
                metadataHeader: metadataHeader
            ) {
                eligible.append(layer)
            }
        }
        guard !eligible.isEmpty else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }
        eligible.sort { $0.tensorName < $1.tensorName }

        var headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-attention-cotiled-u16-v2"]
        ]
        var cursor = 0
        for layer in eligible {
            let (tensorBytes, tensorOverflow) = layer.threadgroups
                .multipliedReportingOverflow(by: layer.wordsPerThreadgroup * 4)
            let (end, endOverflow) = cursor.addingReportingOverflow(tensorBytes)
            guard !tensorOverflow, !endOverflow else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload size overflows Int for \(layer.layerStem)"
                )
            }
            headerObject[layer.tensorName] = [
                "dtype": TensorDType.u32.rawValue,
                "shape": [layer.threadgroups, layer.wordsPerThreadgroup],
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }
        var header = try JSONSerialization.data(
            withJSONObject: headerObject,
            options: [.sortedKeys]
        )
        while !header.count.isMultiple(of: 8) { header.append(0x20) }

        let destination = destinationDirectory.appendingPathComponent(shardName)
        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)

        var weightMap: [String: String] = [:]
        var totalBytes = 0
        for layer in eligible {
            var projections: [ProjectionSource] = []
            for position in layer.stems.indices {
                projections.append(
                    ProjectionSource(
                        weight: try stagedTensorBytes(
                            named: "\(layer.stems[position]).weight",
                            stagingDirectory: stagingDirectory,
                            index: index,
                            stagedHeaders: stagedHeaders
                        ),
                        indices: try sidecarTensorBytes(
                            named: layer.indexNames[position],
                            sidecarURL: metadataURL,
                            header: metadataHeader
                        ),
                        rows: layer.rows[position],
                        indexBits: layer.indexBits[position],
                        lutCount: layer.lutCounts[position],
                        slots: layer.slots[position]
                    )
                )
            }
            guard let payload = makePayload(projections: projections),
                  payload.count == layer.threadgroups * layer.wordsPerThreadgroup * 4
            else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload construction failed for \(layer.layerStem)"
                )
            }
            try output.write(contentsOf: payload)
            weightMap[layer.tensorName] = shardName
            totalBytes += payload.count
        }
        try output.synchronize()
        return GeneratedAffineMetadataReport(
            weightMap: weightMap,
            tensorByteCount: totalBytes
        )
    }

    private static func eligibleLayer(
        layerStem: String,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        metadataHeader: SafetensorsHeader
    ) throws -> EligibleLayer? {
        func weightInfo(_ stem: String) -> SafetensorInfo? {
            let name = "\(stem).weight"
            guard selectedKeys.contains(name),
                  let shardName = index.weightMap[name]
            else {
                return nil
            }
            return stagedHeaders[shardName]?.tensors[name]
        }
        func indexRecord(
            _ stem: String,
            rows: Int
        ) -> (name: String, bits: Int, lutCount: Int)? {
            let indexName = "\(stem).metadata_indices"
            let lutName = "\(stem).metadata_lut"
            guard let indexInfo = metadataHeader.tensors[indexName],
                  indexInfo.dtype == TensorDType.u16.rawValue,
                  indexInfo.shape == [rows, groupsPerRow],
                  let lutInfo = metadataHeader.tensors[lutName],
                  lutInfo.dtype == TensorDType.u32.rawValue,
                  lutInfo.shape.count == 1,
                  let lutCount = lutInfo.shape.first,
                  let bits = indexBits(lutCount: lutCount)
            else {
                return nil
            }
            return (indexName, bits, lutCount)
        }
        func productionRows(_ info: SafetensorInfo) -> Int? {
            guard info.dtype == TensorDType.u32.rawValue,
                  info.shape.count == 2,
                  info.shape[1] == weightWordsPerRow,
                  info.shape[0] > 0
            else {
                return nil
            }
            return info.shape[0]
        }

        let qStem = "\(layerStem).self_attn.q_proj"
        let kStem = "\(layerStem).self_attn.k_proj"
        let vStem = "\(layerStem).self_attn.v_proj"
        guard let qInfo = weightInfo(qStem), let kInfo = weightInfo(kStem),
              let qRows = productionRows(qInfo), let kRows = productionRows(kInfo)
        else {
            return nil
        }
        let vInfo = weightInfo(vStem)
        let stems: [String]
        let slots: [Int]
        let rows: [Int]
        let tensorName: String
        if let vInfo {
            guard let vRows = productionRows(vInfo),
                  qRows == 2 * kRows, vRows == kRows,
                  qRows.isMultiple(of: 8)
            else {
                return nil
            }
            stems = [qStem, kStem, vStem]
            slots = [2, 1, 1]
            rows = [qRows, kRows, vRows]
            tensorName = "\(layerStem)\(slidingSuffix)"
        } else {
            guard qRows == 8 * kRows, kRows.isMultiple(of: 4) else { return nil }
            stems = [qStem, kStem]
            slots = [8, 1]
            rows = [qRows, kRows]
            tensorName = "\(layerStem)\(fullSuffix)"
        }

        var indexNames: [String] = []
        var widths: [Int] = []
        var lutCounts: [Int] = []
        for position in stems.indices {
            guard let record = indexRecord(stems[position], rows: rows[position]) else {
                return nil
            }
            indexNames.append(record.name)
            widths.append(record.bits)
            lutCounts.append(record.lutCount)
        }
        let slotWidths = stems.indices.flatMap {
            [Int](repeating: widths[$0], count: slots[$0])
        }
        guard let words = wordsPerThreadgroup(slotIndexBits: slotWidths) else {
            return nil
        }
        return EligibleLayer(
            layerStem: layerStem,
            tensorName: tensorName,
            stems: stems,
            indexNames: indexNames,
            indexBits: widths,
            lutCounts: lutCounts,
            slots: slots,
            rows: rows,
            threadgroups: kRows / rowsPerSlot,
            wordsPerThreadgroup: words
        )
    }

    private static func stagedTensorBytes(
        named name: String,
        stagingDirectory: URL,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader]
    ) throws -> Data {
        guard let shardName = index.weightMap[name],
              let header = stagedHeaders[shardName],
              let info = header.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing staged tensor metadata for \(name)")
        }
        return try readTensorBytes(
            from: stagingDirectory.appendingPathComponent(shardName),
            dataBaseOffset: header.dataBaseOffset,
            info: info,
            name: name
        )
    }

    private static func sidecarTensorBytes(
        named name: String,
        sidecarURL: URL,
        header: SafetensorsHeader
    ) throws -> Data {
        guard let info = header.tensors[name] else {
            throw MLXFastError.invalidInput("missing generated metadata tensor \(name)")
        }
        return try readTensorBytes(
            from: sidecarURL,
            dataBaseOffset: header.dataBaseOffset,
            info: info,
            name: name
        )
    }

    private static func readTensorBytes(
        from url: URL,
        dataBaseOffset: UInt64,
        info: SafetensorInfo,
        name: String
    ) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (offset, overflow) = dataBaseOffset.addingReportingOverflow(
            UInt64(info.dataStart)
        )
        guard !overflow else {
            throw MLXFastError.invalidInput("tensor byte offset overflows for \(name)")
        }
        try handle.seek(toOffset: offset)
        let bytes = handle.readData(ofLength: info.byteCount)
        guard bytes.count == info.byteCount else {
            throw MLXFastError.invalidInput(
                "short read while authoring co-tiled payload for \(name)"
            )
        }
        return bytes
    }
}
