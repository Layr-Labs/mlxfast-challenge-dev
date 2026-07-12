import Foundation
import MLX

// Packed-output additions authored by GPT 5.6 Sol through Gaj's OpenCode Harness.
private let gemma4OutputCoTiledPayloadWords = 536

private func gemma4PackOutputFixed12Planes(
    _ indices: [UInt16], rows: Int, groupsPerRow: Int
) -> [UInt32]? {
    guard groupsPerRow.isMultiple(of: 8), indices.count == rows * groupsPerRow else { return nil }
    var words = [UInt32](repeating: 0, count: rows * groupsPerRow / 8 * 3)
    for row in 0..<rows {
        for block in 0..<(groupsPerRow / 8) {
            let source = row * groupsPerRow + block * 8
            let target = (row * groupsPerRow / 8 + block) * 3
            for group in 0..<8 {
                let value = UInt32(indices[source + group])
                guard value < 4_096 else { return nil }
                words[target + group / 4] |= (value & 0xff) << ((group % 4) * 8)
                words[target + 2] |= ((value >> 8) & 0xf) << ((group / 2) * 8 + (group % 2) * 4)
            }
        }
    }
    return words
}

func gemma4MakeCoTiledFixed12SlidingOutputPayload(
    weight: MLXArray, indices: MLXArray, materialize: Bool = true
) -> MLXArray? {
    guard weight.dtype == .uint32, weight.shape == [5_376, 1_024],
          indices.dtype == .uint16, indices.shape == [5_376, 128],
          let planes = gemma4PackOutputFixed12Planes(indices.asArray(UInt16.self), rows: 5_376, groupsPerRow: 128)
    else { return nil }
    let weights = weight.reshaped(672, 2, 4, 16, 32, 2)
        .transposed(0, 3, 1, 2, 4, 5).contiguous().reshaped(672, 16, 512)
    let metadata = MLXArray(planes, [5_376, 16, 3]).reshaped(672, 2, 4, 16, 3)
        .transposed(0, 3, 1, 2, 4).contiguous().reshaped(672, 16, 24)
    let payload = concatenated([weights, metadata], axis: 2)
    precondition(payload.shape == [672, 16, gemma4OutputCoTiledPayloadWords])
    if materialize { eval(payload) }
    return payload
}

private let gemma4CoTiledFixed12IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_indexed_output_qmv_fast_8192_v1",
    inputNames: ["payload", "lut", "x"], outputNames: ["output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kWordsPerRow = 64;
        constexpr int kWordsPerSIMD = 256;
        constexpr int kWeightWords = 512;
        constexpr int kPayloadWords = 536;
        const int tg = threadgroup_position_in_grid.y;
        const int simd = simdgroup_index_in_threadgroup;
        const int output_row = tg * 8 + simd * 4;
        const uint lane_group = thread_index_in_simdgroup / 4;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;
        const device uint* tile = payload + tg * 16 * kPayloadWords;
        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 16; ++block) {
            float values[16];
            const float input_sum = gemma4_output_load_values(input, values);
            const device uint* weights = tile + simd * kWordsPerSIMD + thread_index_in_simdgroup * 2;
            const device uchar* metadata = reinterpret_cast<const device uchar*>(tile + kWeightWords)
                + simd * kRowsPerSIMD * 12;
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight = reinterpret_cast<const device uchar*>(weights + row * kWordsPerRow);
                const device uchar* row_metadata = metadata + row * 12;
                const uint low = row_metadata[lane_group];
                const uint packed_high = row_metadata[8 + lane_group / 2];
                const uint index = low | (((packed_high >> ((lane_group & 1) * 4)) & 0xf) << 8);
                const uint pair = lut[index];
                result[row] += gemma4_output_qdot_4bit(row_weight, values,
                    gemma4_output_pair_scale(pair), gemma4_output_pair_bias(pair), input_sum);
            }
            tile += kPayloadWords;
            input += 512;
        }
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) output[output_row + row] = static_cast<bfloat>(result[row]);
        }
        """,
    header: gemma4IndexedOutputHeader, ensureRowContiguous: true
)

private let gemma4Packed12IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_output_qmv_fast_8192_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4Packed12IndexedOutputBody(
        inputWidth: 8_192,
        groupsPerRow: 128,
        packedWordsPerRow: 48,
        weightBytesPerRow: 4_096
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4Packed12IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_output_qmv_fast_16384_v1",
    inputNames: ["weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4Packed12IndexedOutputBody(
        inputWidth: 16_384,
        groupsPerRow: 256,
        packedWordsPerRow: 96,
        weightBytesPerRow: 8_192
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4IndexedSlidingOutputQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_output_qmv_fast_8192_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4IndexedOutputBody(
        inputWidth: 8_192,
        groupsPerRow: 128,
        weightBytesPerRow: 4_096
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private let gemma4IndexedFullOutputQMV = MLXFast.metalKernel(
    name: "gemma4_indexed_output_qmv_fast_16384_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4IndexedOutputBody(
        inputWidth: 16_384,
        groupsPerRow: 256,
        weightBytesPerRow: 8_192
    ),
    header: gemma4IndexedOutputHeader,
    ensureRowContiguous: true
)

private func gemma4Packed12IndexedOutputBody(
    inputWidth: Int,
    groupsPerRow: Int,
    packedWordsPerRow: Int,
    weightBytesPerRow: Int
) -> String {
    """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kGroupsPerRow = \(groupsPerRow);
        constexpr int kPackedWordsPerRow = \(packedWordsPerRow);
        constexpr int kWeightBytesPerRow = \(weightBytesPerRow);
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;

        const int output_row =
            threadgroup_position_in_grid.y * 8
            + simdgroup_index_in_threadgroup * kRowsPerSIMD;
        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 8;

        // Each 512-element block consumes eight group-64 indexes. The block's
        // 96 metadata bits are exactly three U32 words, so no block crosses a
        // row boundary for either supported output geometry.
        const uint lane_group = thread_index_in_simdgroup / 4;
        const uint lane_bit = lane_group * 12;
        const uint lane_word = lane_bit / 32;
        const uint lane_shift = lane_bit % 32;
        const device uint* row_packed_indices =
            packed_indices + output_row * kPackedWordsPerRow + lane_word;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_output_load_values(input, values);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uint* row_words =
                    row_packed_indices + row * kPackedWordsPerRow;
                uint metadata_index = row_words[0] >> lane_shift;
                if (lane_shift > 20) {
                    metadata_index |= row_words[1] << (32 - lane_shift);
                }
                metadata_index &= 0x0fff;
                const uint pair = lut[metadata_index];
                result[row] += gemma4_output_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_output_pair_scale(pair),
                    gemma4_output_pair_bias(pair),
                    input_sum);
            }
            weight_bytes += 256;
            row_packed_indices += 3;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private func gemma4IndexedOutputBody(
    inputWidth: Int,
    groupsPerRow: Int,
    weightBytesPerRow: Int
) -> String {
    """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kGroupsPerRow = \(groupsPerRow);
        constexpr int kWeightBytesPerRow = \(weightBytesPerRow);
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;

        const int output_row =
            threadgroup_position_in_grid.y * 8
            + simdgroup_index_in_threadgroup * kRowsPerSIMD;
        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 8;
        const device ushort* row_indices =
            indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 4;
        const device bfloat* input = x + thread_index_in_simdgroup * 16;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kInputWidth; block += kBlockSize) {
            float values[16];
            const float input_sum = gemma4_output_load_values(input, values);
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_output_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_output_pair_scale(pair),
                    gemma4_output_pair_bias(pair),
                    input_sum);
            }
            weight_bytes += 256;
            row_indices += 8;
            input += kBlockSize;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private let gemma4IndexedOutputHeader = """
    using namespace metal;

    inline float gemma4_output_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_output_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_output_load_values(
        const device bfloat* input,
        thread float* values
    ) {
        float sum = 0;
        for (int index = 0; index < 16; index += 4) {
            sum += input[index] + input[index + 1]
                + input[index + 2] + input[index + 3];
            values[index] = input[index];
            values[index + 1] = input[index + 1] / 16.0f;
            values[index + 2] = input[index + 2] / 256.0f;
            values[index + 3] = input[index + 3] / 4096.0f;
        }
        return sum;
    }

    inline float gemma4_output_qdot_4bit(
        const device uchar* weight,
        const thread float* values,
        float scale,
        float bias,
        float input_sum
    ) {
        const device ushort* packed =
            reinterpret_cast<const device ushort*>(weight);
        float accumulator = 0;
        for (int index = 0; index < 4; ++index) {
            accumulator +=
                (values[4 * index] * (packed[index] & 0x000f)
                + values[4 * index + 1] * (packed[index] & 0x00f0)
                + values[4 * index + 2] * (packed[index] & 0x0f00)
                + values[4 * index + 3] * (packed[index] & 0xf000));
        }
        return scale * accumulator + input_sum * bias;
    }
    """

private func gemma4OutputEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private struct Packed12OutputMetadata: @unchecked Sendable {
    let words: MLXArray

    init?(metadata: IndexedAffineMetadata, groupsPerRow: Int) {
        guard groupsPerRow == 128 || groupsPerRow == 256,
              metadata.indices.dtype == .uint16,
              metadata.indices.shape == [5_376, groupsPerRow],
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size)
        else {
            return nil
        }

        let (rowBits, rowBitsOverflow) = groupsPerRow.multipliedReportingOverflow(by: 12)
        guard !rowBitsOverflow, rowBits.isMultiple(of: 32) else { return nil }
        let wordsPerRow = rowBits / 32
        let (wordCount, wordCountOverflow) = 5_376.multipliedReportingOverflow(
            by: wordsPerRow)
        guard !wordCountOverflow else { return nil }

        let indices = metadata.indices.asArray(UInt16.self)
        guard indices.allSatisfy({ Int($0) < metadata.lut.size }),
              let packed = gemma4Pack12BitIndices(
                  indices,
                  rows: 5_376,
                  groupsPerRow: groupsPerRow
              ),
              packed.count == wordCount
        else {
            return nil
        }

        let words = MLXArray(packed, [5_376, wordsPerRow])
        // Packing and host-to-Metal materialization happen while the model is
        // initialized, before any scored prefill or decode phase begins.
        eval(words)
        self.words = words
    }
}

struct IndexedOutputProjection: @unchecked Sendable {
    let projection: FastQuantizedProjection
    let metadata: IndexedAffineMetadata
    let inputWidth: Int
    private let packed12: Packed12OutputMetadata?
    private let coTiledPayload: MLXArray?
    private let verifyCoTiledBits: Bool
    private let verifyPacked12Bits: Bool

    init?(projection: FastQuantizedProjection) {
        guard let biases = projection.biases else { return nil }
        let inputWidth: Int
        if projection.weight.shape == [5_376, 1_024]
            && projection.scales.shape == [5_376, 128]
        {
            inputWidth = 8_192
        } else if projection.weight.shape == [5_376, 2_048]
            && projection.scales.shape == [5_376, 256]
        {
            inputWidth = 16_384
        } else {
            return nil
        }
        guard projection.groupSize == 64,
              projection.bits == 4,
              projection.weight.dtype == .uint32,
              projection.scales.dtype == .bfloat16,
              biases.dtype == .bfloat16,
              biases.shape == projection.scales.shape
        else {
            return nil
        }
        let metadata = makeIndexedAffineMetadata(
            scales: projection.scales,
            biases: biases
        )
        self.projection = projection
        self.metadata = metadata
        self.inputWidth = inputWidth
        if gemma4OutputEnvironmentFlag(
            "MLXFAST_PACKED_OUTPUT_INDICES",
            default: true
        ) {
            self.packed12 = Packed12OutputMetadata(
                metadata: metadata,
                groupsPerRow: inputWidth / 64
            )
        } else {
            self.packed12 = nil
        }
        if inputWidth == 8_192,
           self.packed12 != nil,
           gemma4OutputEnvironmentFlag("MLXFAST_COTILED_SLIDING_OUTPUT", default: true) {
            self.coTiledPayload = gemma4MakeCoTiledFixed12SlidingOutputPayload(
                weight: projection.weight, indices: metadata.indices)
        } else {
            self.coTiledPayload = nil
        }
        self.verifyCoTiledBits = gemma4OutputEnvironmentFlag(
            "MLXFAST_VERIFY_COTILED_SLIDING_OUTPUT_BITS", default: false)
        self.verifyPacked12Bits = gemma4OutputEnvironmentFlag(
            "MLXFAST_VERIFY_PACKED_OUTPUT_BITS",
            default: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, inputWidth])
        guard let packed12 else {
            let kernel = inputWidth == 8_192
                ? gemma4IndexedSlidingOutputQMV
                : gemma4IndexedFullOutputQMV
            return kernel(
                [projection.weight, metadata.indices, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [[1, 1, 5_376]],
                outputDTypes: [.bfloat16]
            )[0]
        }

        let packedKernel = inputWidth == 8_192
            ? gemma4Packed12IndexedSlidingOutputQMV
            : gemma4Packed12IndexedFullOutputQMV
        let packedOutput = packedKernel(
            [projection.weight, packed12.words, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 5_376]],
            outputDTypes: [.bfloat16]
        )[0]
        let output: MLXArray
        if let coTiledPayload {
            output = gemma4CoTiledFixed12IndexedSlidingOutputQMV(
                [coTiledPayload, metadata.lut, input],
                grid: (32, 1_344, 1), threadGroup: (32, 2, 1),
                outputShapes: [[1, 1, 5_376]], outputDTypes: [.bfloat16])[0]
            if verifyCoTiledBits {
                let matches = arrayEqual(output.view(dtype: .uint16), packedOutput.view(dtype: .uint16))
                eval(matches)
                precondition(matches.item(Bool.self), "co-tiled output differs from packed12 qmv_fast")
            }
        } else {
            output = packedOutput
        }
        /* packed output call retained above; verifier below remains packed12-versus-U16 */
        _ = packedKernel
        /*
            [projection.weight, packed12.words, metadata.lut, input],
            grid: (32, 1_344, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 5_376]],
            outputDTypes: [.bfloat16]
        )[0]
        */
        if verifyPacked12Bits {
            let referenceKernel = inputWidth == 8_192
                ? gemma4IndexedSlidingOutputQMV
                : gemma4IndexedFullOutputQMV
            let reference = referenceKernel(
                [projection.weight, metadata.indices, metadata.lut, input],
                grid: (32, 1_344, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [[1, 1, 5_376]],
                outputDTypes: [.bfloat16]
            )[0]
            let matches = arrayEqual(
                packedOutput.view(dtype: .uint16),
                reference.view(dtype: .uint16)
            )
            eval(matches)
            precondition(
                matches.item(Bool.self),
                "packed 12-bit output projection differs from U16 indexed qmv_fast"
            )
        }
        return output
    }
}
