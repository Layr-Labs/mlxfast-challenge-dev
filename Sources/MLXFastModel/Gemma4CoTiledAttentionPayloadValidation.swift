import Foundation
import MLXFastCore

/// CPU-visible layout shared with the transform. A payload consists of ten
/// `[even weights | odd weights | packed metadata]` pair tiles followed by a
/// `[tail weights | U16 tail metadata]` tile, all threadgroup-major.
enum Gemma4CoTiledAttentionPayloadLayout {
    static let slidingSuffix = ".self_attn.qkv_cotiled_payload"
    static let fullSuffix = ".self_attn.qk_cotiled_payload"

    static let groupsPerRow = 84
    static let weightBytesPerRow = 2_688
    static let blockBytes = 128
    static let pairCount = 10
    static let rowsPerSlot = 4
    static let slidingSlots = [2, 1, 1]
    static let fullSlots = [8, 1]

    static func indexBits(lutCount: Int) -> Int? {
        if (1...4_096).contains(lutCount) { return 12 }
        if (4_097...8_192).contains(lutCount) { return 13 }
        return nil
    }

    static func maximumLUTCount(indexBits: Int) -> Int? {
        guard indexBits == 12 || indexBits == 13 else { return nil }
        return 1 << indexBits
    }

    static func slidingSlotIndexBits(qBits: Int, kBits: Int, vBits: Int) -> [Int] {
        [qBits, qBits, kBits, vBits]
    }

    static func fullSlotIndexBits(qBits: Int, kBits: Int) -> [Int] {
        [Int](repeating: qBits, count: 8) + [kBits]
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
}

struct Gemma4CoTiledPayloadProjectionSource {
    let weight: Data
    let indices: Data
    let rows: Int
    let indexBits: Int
    let lutCount: Int
    let slots: Int
    let stem: String
}

enum Gemma4CoTiledAttentionPayloadKind: Sendable, Equatable {
    case slidingQKV
    case fullQK
}

/// Raw-byte proof descriptor. Partitioning binds this to the MLXArray loaded
/// from the same tensor name and refuses any unvalidated or missing payload.
struct ValidatedGemma4CoTiledAttentionPayload: Sendable {
    let kind: Gemma4CoTiledAttentionPayloadKind
    let threadgroups: Int
    let wordsPerThreadgroup: Int
    let qBits: Int
    let kBits: Int
    let vBits: Int?
}

/// Exhaustively proves one payload against the canonical U32 weights and U16
/// indices. Every source index is checked against its LUT and bit width, every
/// weight run is compared byte-for-byte, and every compact metadata byte is
/// reconstructed independently from U16. This runs before MLX allocation.
func validateGemma4CoTiledAttentionPayloadBytes(
    payload: Data,
    projections: [Gemma4CoTiledPayloadProjectionSource],
    name: String
) throws {
    typealias Layout = Gemma4CoTiledAttentionPayloadLayout
    guard !projections.isEmpty else {
        throw MLXFastError.invalidInput(
            "co-tiled attention payload has no projections for \(name)"
        )
    }

    var threadgroups: Int?
    var slotProjection: [Int] = []
    var slotWithinProjection: [Int] = []
    for (projectionIndex, projection) in projections.enumerated() {
        guard projection.slots > 0,
              projection.rows > 0,
              projection.rows.isMultiple(of: Layout.rowsPerSlot * projection.slots),
              Layout.indexBits(lutCount: projection.lutCount) == projection.indexBits
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload has invalid projection geometry for "
                    + "\(projection.stem) in \(name)"
            )
        }
        let (weightBytes, weightOverflow) = projection.rows
            .multipliedReportingOverflow(by: Layout.weightBytesPerRow)
        let (indexElements, indexOverflow) = projection.rows
            .multipliedReportingOverflow(by: Layout.groupsPerRow)
        let (indexBytes, indexByteOverflow) = indexElements
            .multipliedReportingOverflow(by: 2)
        guard !weightOverflow, !indexOverflow, !indexByteOverflow,
              projection.weight.count == weightBytes,
              projection.indices.count == indexBytes
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention source byte lengths are invalid for "
                    + "\(projection.stem) in \(name)"
            )
        }
        let count = projection.rows / (Layout.rowsPerSlot * projection.slots)
        if let threadgroups, threadgroups != count {
            throw MLXFastError.invalidInput(
                "co-tiled attention projections disagree on threadgroup count for \(name)"
            )
        }
        threadgroups = count
        for slot in 0..<projection.slots {
            slotProjection.append(projectionIndex)
            slotWithinProjection.append(slot)
        }
    }
    guard let threadgroups else {
        throw MLXFastError.invalidInput("co-tiled attention payload is empty for \(name)")
    }

    let slots = slotProjection.count
    var metadataOffsets: [Int] = []
    var metadataCursor = 0
    for slot in 0..<slots {
        metadataOffsets.append(metadataCursor)
        metadataCursor += Layout.rowsPerSlot
            * projections[slotProjection[slot]].indexBits
    }
    let pairTileBytes = 1_024 * slots + metadataCursor
    let tailTileBytes = 544 * slots
    let threadgroupBytes = Layout.pairCount * pairTileBytes + tailTileBytes
    let (expectedBytes, overflow) = threadgroups
        .multipliedReportingOverflow(by: threadgroupBytes)
    guard !overflow, payload.count == expectedBytes else {
        throw MLXFastError.invalidInput(
            "co-tiled attention payload byte length is invalid for \(name)"
        )
    }

    func mismatch(
        _ projection: Gemma4CoTiledPayloadProjectionSource,
        threadgroup: Int,
        pair: String,
        row: Int
    ) -> MLXFastError {
        .invalidInput(
            "co-tiled attention payload does not reconstruct \(projection.stem) "
                + "at threadgroup \(threadgroup), \(pair), row \(row) for \(name)"
        )
    }

    try payload.withUnsafeBytes { payloadBytes in
        try withGemma4CoTiledProjectionBytes(projections) {
            weightBuffers,
            indexBuffers in
            guard let payloadBase = payloadBytes.baseAddress else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload has no bytes for \(name)"
                )
            }

            @inline(__always)
            func readIndex(_ projectionIndex: Int, row: Int, group: Int) -> UInt16 {
                let offset = 2 * (row * Layout.groupsPerRow + group)
                let bytes = indexBuffers[projectionIndex]
                return UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
            }

            // Validate all 84 U16 indexes, including values represented again
            // in the compact stream below. This catches unused high bits and
            // proves every LUT read is in bounds.
            for projectionIndex in projections.indices {
                let projection = projections[projectionIndex]
                for row in 0..<projection.rows {
                    for group in 0..<Layout.groupsPerRow {
                        let value = Int(readIndex(projectionIndex, row: row, group: group))
                        guard value < projection.lutCount,
                              value < (1 << projection.indexBits)
                        else {
                            throw MLXFastError.invalidInput(
                                "co-tiled attention U16 index exceeds LUT/format for "
                                    + "\(projection.stem), row \(row), group \(group)"
                            )
                        }
                    }
                }
            }

            for threadgroup in 0..<threadgroups {
                let tileBase = threadgroup * threadgroupBytes
                for slot in 0..<slots {
                    let projectionIndex = slotProjection[slot]
                    let projection = projections[projectionIndex]
                    guard let weightBase = weightBuffers[projectionIndex].baseAddress else {
                        throw MLXFastError.invalidInput(
                            "co-tiled attention source has no bytes for \(projection.stem)"
                        )
                    }
                    let rowBase = threadgroup
                        * Layout.rowsPerSlot * projection.slots
                        + Layout.rowsPerSlot * slotWithinProjection[slot]
                    for row in 0..<Layout.rowsPerSlot {
                        let sourceRow = rowBase + row
                        let weightRowBase = sourceRow * Layout.weightBytesPerRow
                        for pair in 0..<Layout.pairCount {
                            let pairBase = tileBase + pair * pairTileBytes
                            guard memcmp(
                                payloadBase + pairBase + 512 * slot + 128 * row,
                                weightBase + weightRowBase
                                    + (2 * pair) * Layout.blockBytes,
                                Layout.blockBytes
                            ) == 0,
                            memcmp(
                                payloadBase + pairBase + 512 * slots
                                    + 512 * slot + 128 * row,
                                weightBase + weightRowBase
                                    + (2 * pair + 1) * Layout.blockBytes,
                                Layout.blockBytes
                            ) == 0
                            else {
                                throw mismatch(
                                    projection,
                                    threadgroup: threadgroup,
                                    pair: "pair \(pair) weights",
                                    row: row
                                )
                            }

                            let packed = (payloadBase + pairBase + 1_024 * slots
                                + metadataOffsets[slot]
                                + row * projection.indexBits)
                                .assumingMemoryBound(to: UInt8.self)
                            var topBits: UInt8 = 0
                            for laneGroup in 0..<4 {
                                let even = readIndex(
                                    projectionIndex,
                                    row: sourceRow,
                                    group: 8 * pair + laneGroup
                                )
                                let odd = readIndex(
                                    projectionIndex,
                                    row: sourceRow,
                                    group: 8 * pair + 4 + laneGroup
                                )
                                let expectedLow = UInt8(truncatingIfNeeded: even)
                                let expectedMiddle = UInt8(
                                    truncatingIfNeeded: (even >> 8)
                                        | ((odd & 0x000f) << 4)
                                )
                                let expectedHigh = UInt8(truncatingIfNeeded: odd >> 4)
                                guard packed[laneGroup] == expectedLow,
                                      packed[4 + laneGroup] == expectedMiddle,
                                      packed[8 + laneGroup] == expectedHigh
                                else {
                                    throw mismatch(
                                        projection,
                                        threadgroup: threadgroup,
                                        pair: "pair \(pair) metadata",
                                        row: row
                                    )
                                }
                                if projection.indexBits == 13 {
                                    topBits |= UInt8((even >> 12) & 1) << laneGroup
                                    topBits |= UInt8((odd >> 12) & 1)
                                        << (4 + laneGroup)
                                }
                            }
                            if projection.indexBits == 13, packed[12] != topBits {
                                throw mismatch(
                                    projection,
                                    threadgroup: threadgroup,
                                    pair: "pair \(pair) top bits",
                                    row: row
                                )
                            }
                        }

                        let tailBase = tileBase + Layout.pairCount * pairTileBytes
                        guard memcmp(
                            payloadBase + tailBase + 512 * slot + 128 * row,
                            weightBase + weightRowBase
                                + 2 * Layout.pairCount * Layout.blockBytes,
                            Layout.blockBytes
                        ) == 0 else {
                            throw mismatch(
                                projection,
                                threadgroup: threadgroup,
                                pair: "tail weights",
                                row: row
                            )
                        }
                        let packed = (payloadBase + tailBase + 512 * slots
                            + 32 * slot + 8 * row)
                            .assumingMemoryBound(to: UInt8.self)
                        for laneGroup in 0..<4 {
                            let value = readIndex(
                                projectionIndex,
                                row: sourceRow,
                                group: 80 + laneGroup
                            )
                            guard packed[2 * laneGroup]
                                    == UInt8(truncatingIfNeeded: value),
                                  packed[2 * laneGroup + 1]
                                    == UInt8(truncatingIfNeeded: value >> 8)
                            else {
                                throw mismatch(
                                    projection,
                                    threadgroup: threadgroup,
                                    pair: "tail metadata",
                                    row: row
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

private func withGemma4CoTiledProjectionBytes(
    _ projections: [Gemma4CoTiledPayloadProjectionSource],
    _ body: ([UnsafeRawBufferPointer], [UnsafeRawBufferPointer]) throws -> Void
) throws {
    func recurse(
        _ index: Int,
        _ weights: [UnsafeRawBufferPointer],
        _ indices: [UnsafeRawBufferPointer]
    ) throws {
        guard index < projections.count else {
            try body(weights, indices)
            return
        }
        try projections[index].weight.withUnsafeBytes { weightBytes in
            try projections[index].indices.withUnsafeBytes { indexBytes in
                try recurse(
                    index + 1,
                    weights + [weightBytes],
                    indices + [indexBytes]
                )
            }
        }
    }
    try recurse(0, [], [])
}

/// Validates every payload in a checkpoint. Absence is supported for older
/// transformed trees and returns an empty dictionary; presence is fail-closed.
func validateGemma4CoTiledAttentionPayloadBytes(
    denseStore: DenseTensorStore
) throws -> [String: ValidatedGemma4CoTiledAttentionPayload] {
    typealias Layout = Gemma4CoTiledAttentionPayloadLayout
    let runtimePrefix = "language_model."
    var validated: [String: ValidatedGemma4CoTiledAttentionPayload] = [:]

    for name in denseStore.tensorNames.sorted() {
        let kind: Gemma4CoTiledAttentionPayloadKind
        let suffix: String
        if name.hasSuffix(Layout.slidingSuffix) {
            kind = .slidingQKV
            suffix = Layout.slidingSuffix
        } else if name.hasSuffix(Layout.fullSuffix) {
            kind = .fullQK
            suffix = Layout.fullSuffix
        } else {
            continue
        }
        let layerStem = String(name.dropLast(suffix.count))
        let stems: [String]
        let slots: [Int]
        switch kind {
        case .slidingQKV:
            stems = [
                "\(layerStem).self_attn.q_proj",
                "\(layerStem).self_attn.k_proj",
                "\(layerStem).self_attn.v_proj",
            ]
            slots = Layout.slidingSlots
        case .fullQK:
            stems = [
                "\(layerStem).self_attn.q_proj",
                "\(layerStem).self_attn.k_proj",
            ]
            slots = Layout.fullSlots
        }

        guard let payloadRecord = denseStore.record(named: name),
              payloadRecord.dtype == TensorDType.u32.rawValue,
              payloadRecord.shape.count == 2,
              payloadRecord.shape[0] > 0,
              payloadRecord.shape[1] > 0
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload has invalid dtype or shape for \(name)"
            )
        }
        let threadgroups = payloadRecord.shape[0]
        var projections: [Gemma4CoTiledPayloadProjectionSource] = []
        var widths: [Int] = []
        for position in stems.indices {
            let stem = stems[position]
            let expectedRows = Layout.rowsPerSlot * slots[position] * threadgroups
            let indexName = "\(stem).metadata_indices"
            let lutName = "\(stem).metadata_lut"
            guard let weightRecord = denseStore.record(named: "\(stem).weight"),
                  weightRecord.dtype == TensorDType.u32.rawValue,
                  weightRecord.shape == [expectedRows, 672],
                  let indexRecord = denseStore.record(named: indexName),
                  indexRecord.dtype == TensorDType.u16.rawValue,
                  indexRecord.shape == [expectedRows, Layout.groupsPerRow],
                  let lutRecord = denseStore.record(named: lutName),
                  lutRecord.dtype == TensorDType.u32.rawValue,
                  lutRecord.shape.count == 1,
                  let lutCount = lutRecord.shape.first,
                  let bits = Layout.indexBits(lutCount: lutCount)
            else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload is missing U16 companions for \(stem)"
                )
            }
            projections.append(
                Gemma4CoTiledPayloadProjectionSource(
                    weight: try denseStore.tensorBytes(named: "\(stem).weight"),
                    indices: try denseStore.tensorBytes(named: indexName),
                    rows: expectedRows,
                    indexBits: bits,
                    lutCount: lutCount,
                    slots: slots[position],
                    stem: stem
                )
            )
            widths.append(bits)
        }

        let slotWidths: [Int]
        switch kind {
        case .slidingQKV:
            slotWidths = Layout.slidingSlotIndexBits(
                qBits: widths[0], kBits: widths[1], vBits: widths[2]
            )
        case .fullQK:
            slotWidths = Layout.fullSlotIndexBits(
                qBits: widths[0], kBits: widths[1]
            )
        }
        guard let words = Layout.wordsPerThreadgroup(slotIndexBits: slotWidths),
              payloadRecord.shape == [threadgroups, words]
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload shape does not match U16 formats for \(name)"
            )
        }
        try validateGemma4CoTiledAttentionPayloadBytes(
            payload: denseStore.tensorBytes(named: name),
            projections: projections,
            name: name
        )

        let runtimeName = name.hasPrefix(runtimePrefix)
            ? String(name.dropFirst(runtimePrefix.count))
            : name
        guard validated.updateValue(
            ValidatedGemma4CoTiledAttentionPayload(
                kind: kind,
                threadgroups: threadgroups,
                wordsPerThreadgroup: words,
                qBits: widths[0],
                kBits: widths[1],
                vBits: kind == .slidingQKV ? widths[2] : nil
            ),
            forKey: runtimeName
        ) == nil else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload names collide after runtime rename: \(runtimeName)"
            )
        }
    }
    return validated
}
