import Darwin
import Dispatch
import Foundation
import MLX
import MLXFastCore
import MLXNN

// Runtime-derived packed13 tied-head implementation authored by GPT 5.6 Sol
// through Gaj's OpenCode Harness. The QMV topology is preserved from the
// public packed13 implementation by saucegodbased.
private let gemma4RuntimeTiedVocabularyHeadPacked13QMV = MLXFast.metalKernel(
    name: "gemma4_runtime_tied_vocabulary_head_packed13_qmv_262144x5376_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 262144;
        constexpr int kGroupsPerRow = 84;
        constexpr int kPackedWordsPerRow = 35;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDGroupsPerThreadgroup = 4;

        const int output_row =
            threadgroup_position_in_grid.y
                * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
            + simdgroup_index_in_threadgroup * kRowsPerSIMD;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uint* row_packed_indices =
            packed_indices + output_row * kPackedWordsPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum =
                gemma4_runtime_tied_head_packed13_load_values(input, values);
            const uint metadata_column =
                block * 4 + thread_index_in_simdgroup / 8;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index =
                    gemma4_runtime_tied_head_extract_packed13(
                        row_packed_indices + row * kPackedWordsPerRow,
                        metadata_column);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_runtime_tied_head_packed13_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_runtime_tied_head_pair_scale(pair),
                    gemma4_runtime_tied_head_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline ushort gemma4_runtime_tied_head_extract_packed13(
            const device uint* words,
            uint column
        ) {
            const uint bit_offset = column * 13;
            const uint word_index = bit_offset >> 5;
            const uint shift = bit_offset & 31;
            uint value = words[word_index] >> shift;
            if (shift > 19) {
                value |= words[word_index + 1] << (32 - shift);
            }
            return static_cast<ushort>(value & 0x1fff);
        }

        inline float gemma4_runtime_tied_head_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_runtime_tied_head_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_runtime_tied_head_packed13_load_values(
            const device bfloat* input,
            thread float* values
        ) {
            float sum = 0;
            for (int index = 0; index < 8; index += 4) {
                sum += input[index] + input[index + 1]
                    + input[index + 2] + input[index + 3];
                values[index] = input[index];
                values[index + 1] = input[index + 1] / 16.0f;
                values[index + 2] = input[index + 2] / 256.0f;
                values[index + 3] = input[index + 3] / 4096.0f;
            }
            return sum;
        }

        inline float gemma4_runtime_tied_head_packed13_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
                accumulator +=
                    (values[4 * index] * (packed[index] & 0x000f)
                    + values[4 * index + 1] * (packed[index] & 0x00f0)
                    + values[4 * index + 2] * (packed[index] & 0x0f00)
                    + values[4 * index + 3] * (packed[index] & 0xf000));
            }
            return scale * accumulator + input_sum * bias;
        }
        """,
    ensureRowContiguous: true
)

private let gemma4TiedHeadRows = 262_144
private let gemma4TiedHeadGroupsPerRow = 84
private let gemma4TiedHeadPackedBits = 13
private let gemma4TiedHeadPackedWordsPerRow = 35
private let gemma4TiedHeadFrozenLUTCount = 6_224

private struct Gemma4RuntimePacked13Build {
    let packedIndices: MLXArray
    let lut: MLXArray
    let wallMilliseconds: Double
    let transientPayloadBytes: Int
    let measuredPeakResidentBytes: UInt64?
}

private func gemma4RuntimeTiedHeadError(_ message: String) -> MLXFastError {
    .invalidInput("runtime tied-head packed13: \(message)")
}

private func gemma4CheckedProduct(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else {
        throw gemma4RuntimeTiedHeadError("integer overflow computing \(label)")
    }
    return value
}

private func gemma4CheckedSum(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else {
        throw gemma4RuntimeTiedHeadError("integer overflow computing \(label)")
    }
    return value
}

private func gemma4Packed13WordsPerRow(groupsPerRow: Int) throws -> Int {
    guard groupsPerRow > 0 else {
        throw gemma4RuntimeTiedHeadError("groups per row must be positive")
    }
    let rowBits = try gemma4CheckedProduct(
        groupsPerRow,
        gemma4TiedHeadPackedBits,
        label: "packed row bits"
    )
    let roundedBits = try gemma4CheckedSum(rowBits, 31, label: "rounded packed row bits")
    return roundedBits / 32
}

func gemma4Pack13BitIndices(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int,
    lutCount: Int
) throws -> [UInt32] {
    guard rows > 0, (1...8_192).contains(lutCount) else {
        throw gemma4RuntimeTiedHeadError("invalid rows or LUT length")
    }
    let indexCount = try gemma4CheckedProduct(rows, groupsPerRow, label: "index count")
    guard indices.count == indexCount else {
        throw gemma4RuntimeTiedHeadError(
            "U16 index count \(indices.count) does not match \(indexCount)"
        )
    }
    let wordsPerRow = try gemma4Packed13WordsPerRow(groupsPerRow: groupsPerRow)
    let wordCount = try gemma4CheckedProduct(rows, wordsPerRow, label: "packed word count")
    var words = Array(repeating: UInt32(0), count: wordCount)

    for row in 0..<rows {
        let sourceBase = try gemma4CheckedProduct(row, groupsPerRow, label: "source row base")
        let wordBase = try gemma4CheckedProduct(row, wordsPerRow, label: "word row base")
        let wordEnd = try gemma4CheckedSum(wordBase, wordsPerRow, label: "word row end")
        guard wordEnd <= words.count else {
            throw gemma4RuntimeTiedHeadError("packed row exceeds payload")
        }
        for column in 0..<groupsPerRow {
            let sourceOffset = try gemma4CheckedSum(
                sourceBase,
                column,
                label: "source offset"
            )
            let index = Int(indices[sourceOffset])
            guard index < lutCount, index < (1 << gemma4TiedHeadPackedBits) else {
                throw gemma4RuntimeTiedHeadError(
                    "metadata index \(index) exceeds LUT bounds at row \(row), column \(column)"
                )
            }
            let bitOffset = try gemma4CheckedProduct(
                column,
                gemma4TiedHeadPackedBits,
                label: "column bit offset"
            )
            let localWord = bitOffset >> 5
            let shift = bitOffset & 31
            let firstWord = try gemma4CheckedSum(wordBase, localWord, label: "first word offset")
            guard firstWord < wordEnd else {
                throw gemma4RuntimeTiedHeadError("packed index starts outside its row")
            }
            words[firstWord] |= UInt32(index) << UInt32(shift)
            if shift > 19 {
                let secondWord = try gemma4CheckedSum(
                    firstWord,
                    1,
                    label: "second word offset"
                )
                guard secondWord < wordEnd else {
                    throw gemma4RuntimeTiedHeadError("packed index crosses a row boundary")
                }
                words[secondWord] |= UInt32(index) >> UInt32(32 - shift)
            }
        }
    }
    return words
}

private func gemma4Packed13Index(
    words: [UInt32],
    row: Int,
    column: Int,
    groupsPerRow: Int,
    wordsPerRow: Int
) throws -> UInt16 {
    guard row >= 0, column >= 0, column < groupsPerRow else {
        throw gemma4RuntimeTiedHeadError("invalid packed13 reconstruction coordinate")
    }
    let bitOffset = try gemma4CheckedProduct(
        column,
        gemma4TiedHeadPackedBits,
        label: "reconstruction bit offset"
    )
    let wordBase = try gemma4CheckedProduct(row, wordsPerRow, label: "reconstruction row base")
    let wordEnd = try gemma4CheckedSum(wordBase, wordsPerRow, label: "reconstruction row end")
    let localWord = bitOffset >> 5
    let shift = bitOffset & 31
    let firstWord = try gemma4CheckedSum(wordBase, localWord, label: "reconstruction word")
    guard firstWord < wordEnd, firstWord < words.count else {
        throw gemma4RuntimeTiedHeadError("packed13 reconstruction exceeds payload")
    }
    var value = words[firstWord] >> UInt32(shift)
    if shift > 19 {
        let secondWord = try gemma4CheckedSum(firstWord, 1, label: "reconstruction spill word")
        guard secondWord < wordEnd, secondWord < words.count else {
            throw gemma4RuntimeTiedHeadError("packed13 reconstruction crosses a row boundary")
        }
        value |= words[secondWord] << UInt32(32 - shift)
    }
    return UInt16(value & 0x1fff)
}

private func gemma4ValidatePacked13Indices(
    words: [UInt32],
    expected: [UInt16],
    rows: Int,
    groupsPerRow: Int,
    lutCount: Int
) throws {
    let wordsPerRow = try gemma4Packed13WordsPerRow(groupsPerRow: groupsPerRow)
    let expectedWordCount = try gemma4CheckedProduct(
        rows,
        wordsPerRow,
        label: "validation word count"
    )
    let expectedIndexCount = try gemma4CheckedProduct(
        rows,
        groupsPerRow,
        label: "validation index count"
    )
    guard words.count == expectedWordCount, expected.count == expectedIndexCount else {
        throw gemma4RuntimeTiedHeadError("packed13 validation payload has invalid dimensions")
    }
    for row in 0..<rows {
        let rowBase = try gemma4CheckedProduct(row, groupsPerRow, label: "validation row base")
        for column in 0..<groupsPerRow {
            let offset = try gemma4CheckedSum(rowBase, column, label: "validation offset")
            let reconstructed = try gemma4Packed13Index(
                words: words,
                row: row,
                column: column,
                groupsPerRow: groupsPerRow,
                wordsPerRow: wordsPerRow
            )
            guard Int(reconstructed) < lutCount, reconstructed == expected[offset] else {
                throw gemma4RuntimeTiedHeadError(
                    "packed13 reconstruction mismatch at row \(row), column \(column)"
                )
            }
        }
    }
}

private func gemma4ValidatePacked13CodingBoundaries() throws {
    let rows = gemma4TiedHeadGroupsPerRow
    let groups = gemma4TiedHeadGroupsPerRow
    var indices = Array(repeating: UInt16(0), count: rows * groups)
    for row in 0..<rows {
        for column in 0..<groups {
            let offset = row * groups + column
            indices[offset] = UInt16((row * 131 + column * 67) & 0x1fff)
        }
        indices[row * groups + row] = 8_191
    }
    let packed = try gemma4Pack13BitIndices(
        indices,
        rows: rows,
        groupsPerRow: groups,
        lutCount: 8_192
    )
    try gemma4ValidatePacked13Indices(
        words: packed,
        expected: indices,
        rows: rows,
        groupsPerRow: groups,
        lutCount: 8_192
    )
}

private func gemma4ResidentBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<integer_t>.stride
    )
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                rebound,
                &count
            )
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.resident_size) : nil
}

private func gemma4RuntimePacked13Metadata(
    scales: MLXArray,
    biases: MLXArray
) throws -> Gemma4RuntimePacked13Build {
    guard scales.dtype == .bfloat16,
          scales.shape == [gemma4TiedHeadRows, gemma4TiedHeadGroupsPerRow],
          biases.dtype == .bfloat16,
          biases.shape == scales.shape
    else {
        throw gemma4RuntimeTiedHeadError(
            "expected BF16 scales and biases [262144,84]"
        )
    }

    let start = DispatchTime.now().uptimeNanoseconds
    var measuredPeakResidentBytes = gemma4ResidentBytes()
    try gemma4ValidatePacked13CodingBoundaries()

    let scaleBits = scales.view(dtype: .uint16).asArray(UInt16.self)
    let biasBits = biases.view(dtype: .uint16).asArray(UInt16.self)
    let expectedCount = try gemma4CheckedProduct(
        gemma4TiedHeadRows,
        gemma4TiedHeadGroupsPerRow,
        label: "production metadata count"
    )
    guard scaleBits.count == expectedCount, biasBits.count == expectedCount else {
        throw gemma4RuntimeTiedHeadError("materialized affine metadata has invalid size")
    }

    var pairToIndex: [UInt32: UInt16] = [:]
    pairToIndex.reserveCapacity(8_192)
    var lut: [UInt32] = []
    lut.reserveCapacity(8_192)
    let wordsPerRow = try gemma4Packed13WordsPerRow(
        groupsPerRow: gemma4TiedHeadGroupsPerRow
    )
    guard wordsPerRow == gemma4TiedHeadPackedWordsPerRow else {
        throw gemma4RuntimeTiedHeadError("production row does not pack to 35 words")
    }
    let packedWordCount = try gemma4CheckedProduct(
        gemma4TiedHeadRows,
        wordsPerRow,
        label: "production packed word count"
    )
    var packed = Array(repeating: UInt32(0), count: packedWordCount)

    // Exact production geometry and total counts are checked above. Keeping the
    // row-local loop linear avoids tens of millions of throwing helper calls
    // during the worker's pre-handshake initialization window.
    for row in 0..<gemma4TiedHeadRows {
        let sourceBase = row * gemma4TiedHeadGroupsPerRow
        let wordBase = row * wordsPerRow
        for column in 0..<gemma4TiedHeadGroupsPerRow {
            let offset = sourceBase + column
            let pair = UInt32(scaleBits[offset]) | (UInt32(biasBits[offset]) << 16)
            let index: UInt16
            if let existingIndex = pairToIndex[pair] {
                index = existingIndex
            } else {
                guard lut.count < 8_192 else {
                    throw gemma4RuntimeTiedHeadError(
                        "affine metadata LUT exceeds 13-bit capacity"
                    )
                }
                index = UInt16(lut.count)
                pairToIndex[pair] = index
                lut.append(pair)
            }

            let bitOffset = column * gemma4TiedHeadPackedBits
            let word = wordBase + (bitOffset >> 5)
            let shift = bitOffset & 31
            packed[word] |= UInt32(index) << UInt32(shift)
            if shift > 19 {
                packed[word + 1] |= UInt32(index) >> UInt32(32 - shift)
            }
        }
    }
    guard (1...8_192).contains(lut.count) else {
        throw gemma4RuntimeTiedHeadError("affine metadata LUT length is outside 1...8192")
    }
    guard lut.count == gemma4TiedHeadFrozenLUTCount else {
        throw gemma4RuntimeTiedHeadError(
            "frozen production LUT has \(lut.count) entries, expected 6224"
        )
    }

    for row in 0..<gemma4TiedHeadRows {
        let sourceBase = row * gemma4TiedHeadGroupsPerRow
        let wordBase = row * wordsPerRow
        for column in 0..<gemma4TiedHeadGroupsPerRow {
            let bitOffset = column * gemma4TiedHeadPackedBits
            let word = wordBase + (bitOffset >> 5)
            let shift = bitOffset & 31
            var index = packed[word] >> UInt32(shift)
            if shift > 19 {
                index |= packed[word + 1] << UInt32(32 - shift)
            }
            index &= 0x1fff
            guard index < lut.count else {
                throw gemma4RuntimeTiedHeadError(
                    "packed index exceeds LUT at row \(row), column \(column)"
                )
            }
            let offset = sourceBase + column
            let pair = lut[Int(index)]
            guard UInt16(truncatingIfNeeded: pair) == scaleBits[offset],
                  UInt16(truncatingIfNeeded: pair >> 16) == biasBits[offset]
            else {
                throw gemma4RuntimeTiedHeadError(
                    "LUT reconstruction differs from affine BF16 bits at "
                        + "row \(row), column \(column)"
                )
            }
        }
    }

    let packedIndices = MLXArray(
        packed,
        [gemma4TiedHeadRows, gemma4TiedHeadPackedWordsPerRow]
    )
    let lutArray = MLXArray(lut)
    guard packedIndices.dtype == .uint32,
          packedIndices.shape == [gemma4TiedHeadRows, gemma4TiedHeadPackedWordsPerRow],
          lutArray.dtype == .uint32,
          lutArray.ndim == 1,
          lutArray.size == gemma4TiedHeadFrozenLUTCount
    else {
        throw gemma4RuntimeTiedHeadError("materialized packed metadata has invalid dtype or shape")
    }
    eval(packedIndices, lutArray)
    if let resident = gemma4ResidentBytes() {
        measuredPeakResidentBytes = max(measuredPeakResidentBytes ?? 0, resident)
    }

    let temporaryUInt16Count = try gemma4CheckedSum(
        scaleBits.count,
        biasBits.count,
        label: "temporary source bit copies"
    )
    let temporaryUInt16Bytes = try gemma4CheckedProduct(
        temporaryUInt16Count,
        MemoryLayout<UInt16>.stride,
        label: "temporary U16 bytes"
    )
    let temporaryUInt32Count = try gemma4CheckedSum(
        lut.count,
        packed.count,
        label: "temporary U32 payload"
    )
    let temporaryUInt32Bytes = try gemma4CheckedProduct(
        temporaryUInt32Count,
        MemoryLayout<UInt32>.stride,
        label: "temporary U32 bytes"
    )
    let transientPayloadBytes = try gemma4CheckedSum(
        temporaryUInt16Bytes,
        temporaryUInt32Bytes,
        label: "transient payload bytes"
    )
    let elapsed = DispatchTime.now().uptimeNanoseconds - start
    return Gemma4RuntimePacked13Build(
        packedIndices: packedIndices,
        lut: lutArray,
        wallMilliseconds: Double(elapsed) / 1_000_000,
        transientPayloadBytes: transientPayloadBytes,
        measuredPeakResidentBytes: measuredPeakResidentBytes
    )
}

func gemma4RuntimeEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) throws -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    switch raw.lowercased() {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        throw MLXFastError.invalidInput("\(name) must be 0 or 1")
    }
}

func isGemma4ProductionTiedVocabularyEmbedding(_ embedding: Embedding) -> Bool {
    guard let embedding = embedding as? QuantizedEmbedding else { return false }
    return embedding.mode == .affine
        && embedding.groupSize == 64
        && embedding.bits == 4
        && embedding.weight.dtype == .uint32
        && embedding.weight.shape == [gemma4TiedHeadRows, 672]
}

struct Gemma4RuntimeTiedVocabularyHead: @unchecked Sendable {
    let weight: MLXArray
    let packedIndices: MLXArray
    let lut: MLXArray
    let initializationMilliseconds: Double
    let transientPayloadBytes: Int
    let measuredResidentDeltaBytes: UInt64?

    init(_ embedding: Embedding) throws {
        guard let embedding = embedding as? QuantizedEmbedding,
              embedding.mode == .affine,
              embedding.groupSize == 64,
              embedding.bits == 4,
              embedding.weight.dtype == .uint32,
              embedding.weight.shape == [gemma4TiedHeadRows, 672],
              let biases = embedding.biases
        else {
            throw gemma4RuntimeTiedHeadError(
                "expected affine 4-bit QuantizedEmbedding weight U32 [262144,672]"
            )
        }
        let residentBefore = gemma4ResidentBytes()
        let metadata = try gemma4RuntimePacked13Metadata(
            scales: embedding.scales,
            biases: biases
        )
        self.weight = embedding.weight
        self.packedIndices = metadata.packedIndices
        self.lut = metadata.lut
        self.initializationMilliseconds = metadata.wallMilliseconds
        self.transientPayloadBytes = metadata.transientPayloadBytes
        if let before = residentBefore, let peak = metadata.measuredPeakResidentBytes {
            self.measuredResidentDeltaBytes = peak > before ? peak - before : 0
        } else {
            self.measuredResidentDeltaBytes = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        return gemma4RuntimeTiedVocabularyHeadPacked13QMV(
            [weight, packedIndices, lut, input],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    func verifyRawBF16(_ candidate: MLXArray, stock: MLXArray) {
        precondition(candidate.dtype == .bfloat16)
        precondition(stock.dtype == .bfloat16)
        precondition(candidate.shape == stock.shape)
        let candidateBits = candidate.view(dtype: .uint16)
        let stockBits = stock.view(dtype: .uint16)
        let matches = arrayEqual(candidateBits, stockBits)
        eval(matches)
        guard matches.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let stockValues = stockBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, stockValues).enumerated().first {
                $0.element.0 != $0.element.1
            }
            preconditionFailure(
                "runtime tied-head packed13 raw BF16 mismatch vs embedTokens.asLinear "
                    + "at index \(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), stock=\(mismatch?.element.1 ?? 0)"
            )
        }
    }
}

func reportGemma4RuntimeTiedHeadInitialization(
    _ head: Gemma4RuntimeTiedVocabularyHead
) {
    let payloadMiB = Double(head.transientPayloadBytes) / Double(1 << 20)
    let residentMiB = head.measuredResidentDeltaBytes.map {
        String(format: "%.3f", Double($0) / Double(1 << 20))
    } ?? "unavailable"
    let message = String(
        format: "mlxfast: runtime-packed13 init_ms=%.3f transient_payload_mib=%.3f ",
        head.initializationMilliseconds,
        payloadMiB
    ) + "measured_peak_rss_delta_mib=\(residentMiB) lut=\(head.lut.size)\n"
    FileHandle.standardError.write(Data(message.utf8))
}

func benchmarkGemma4RuntimeTiedHead(
    head: Gemma4RuntimeTiedVocabularyHead,
    embedding: Embedding
) throws {
    let repetitions = 24
    let hiddenValues: [Float] = (0..<5_376).map { index in
        let integerValue = (index * 73 + 19) % 257 - 128
        return Float(integerValue) / Float(64)
    }
    let hidden = MLXArray(hiddenValues, [1, 1, 5_376]).asType(.bfloat16)
    eval(hidden)
    let candidate = head(hidden)
    let stock = embedding.asLinear(hidden)
    head.verifyRawBF16(candidate, stock: stock)

    func runStock() {
        eval(embedding.asLinear(hidden))
    }
    func runPacked13() {
        eval(head(hidden))
    }
    for _ in 0..<8 {
        runStock()
        runPacked13()
    }

    func measure(_ body: () -> Void) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<repetitions {
            body()
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        return Double(elapsed) / 1_000_000 / Double(repetitions)
    }

    let stockFirst = measure(runStock)
    let packedFirst = measure(runPacked13)
    let packedSecond = measure(runPacked13)
    let stockSecond = measure(runStock)
    let stockMean = (stockFirst + stockSecond) / 2
    let packedMean = (packedFirst + packedSecond) / 2
    let ratio = stockMean / packedMean
    let saving = stockMean - packedMean
    let message = String(
        format: "mlxfast: runtime-packed13 ABBA reps=%d stock_ms=[%.6f,%.6f] "
            + "packed_ms=[%.6f,%.6f] mean_ms=[%.6f,%.6f] ratio=%.6f saving_ms=%.6f\n",
        repetitions,
        stockFirst,
        stockSecond,
        packedFirst,
        packedSecond,
        stockMean,
        packedMean,
        ratio,
        saving
    )
    FileHandle.standardError.write(Data(message.utf8))
    guard ratio > 1.01 else {
        throw gemma4RuntimeTiedHeadError("isolated ABBA ratio \(ratio) did not clear 1.01")
    }
    guard saving > 0.05 else {
        throw gemma4RuntimeTiedHeadError(
            "isolated ABBA saving \(saving) ms/token did not clear 0.05"
        )
    }
}
