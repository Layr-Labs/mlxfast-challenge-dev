import Foundation
import MLX
import MLXFastCore
import MLXNN

private let gemma4TiedHeadCoTiledRowsPerThreadgroup = 16
private let gemma4TiedHeadCoTiledGroupsPerBlock = 4
private let gemma4TiedHeadCoTiledBlocks = 21
private let gemma4TiedHeadCoTiledWeightWordsPerTile = 512
private let gemma4TiedHeadCoTiledMetadataWordsPerTile = 32
private let gemma4TiedHeadCoTiledPayloadWordsPerTile = 544
private let gemma4TiedHeadCoTiledPayloadByteCount = 748_683_264

private func gemma4TiedHeadEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4TiedHeadCoTiledU16Enabled = gemma4TiedHeadEnvironmentFlag(
    "DARKBLOOM_TIED_HEAD_COTILED_U16",
    default: true
)

private let gemma4VerifyTiedHeadCoTiledU16Bits = gemma4TiedHeadEnvironmentFlag(
    "DARKBLOOM_VERIFY_TIED_HEAD_COTILED_U16_BITS",
    default: false
)

/// Losslessly expand independently row-packed 13-bit LUT indexes to U16.
/// Returning nil is fail-closed for malformed shapes, arithmetic overflow, or
/// an index outside the supplied LUT.
func gemma4DecodeTiedHeadPacked13Indices(
    _ packed: [UInt32],
    rows: Int,
    groupsPerRow: Int,
    lutCount: Int
) -> [UInt16]? {
    guard rows > 0,
          groupsPerRow > 0,
          (1...8_192).contains(lutCount)
    else {
        return nil
    }
    let (rowBits, rowBitsOverflow) = groupsPerRow.multipliedReportingOverflow(
        by: 13)
    guard !rowBitsOverflow else { return nil }
    let (paddedRowBits, paddedRowBitsOverflow) = rowBits.addingReportingOverflow(
        31)
    guard !paddedRowBitsOverflow else { return nil }
    let wordsPerRow = paddedRowBits / 32
    let (wordCount, wordCountOverflow) = rows.multipliedReportingOverflow(
        by: wordsPerRow)
    let (indexCount, indexCountOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !wordCountOverflow,
          !indexCountOverflow,
          packed.count == wordCount
    else {
        return nil
    }

    var indices = [UInt16](repeating: 0, count: indexCount)
    for row in 0..<rows {
        let packedRow = row * wordsPerRow
        let indexRow = row * groupsPerRow
        for group in 0..<groupsPerRow {
            let bitOffset = group * 13
            let word = bitOffset / 32
            let shift = bitOffset % 32
            var index = packed[packedRow + word] >> UInt32(shift)
            if shift > 19 {
                index |= packed[packedRow + word + 1]
                    << UInt32(32 - shift)
            }
            index &= 0x1fff
            guard index < UInt32(lutCount) else { return nil }
            indices[indexRow + group] = UInt16(index)
        }
    }
    return indices
}

/// Reorder direct U16 LUT indexes into the exact threadgroup/K-block ownership
/// of the co-tiled kernel. Two adjacent U16 indexes are stored per U32 word.
func gemma4PackTiedHeadCoTiledU16Metadata(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt32]? {
    guard rows > 0,
          rows.isMultiple(of: gemma4TiedHeadCoTiledRowsPerThreadgroup),
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: gemma4TiedHeadCoTiledGroupsPerBlock)
    else {
        return nil
    }
    let (indexCount, indexCountOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !indexCountOverflow, indices.count == indexCount else { return nil }

    let threadgroups = rows / gemma4TiedHeadCoTiledRowsPerThreadgroup
    let blocks = groupsPerRow / gemma4TiedHeadCoTiledGroupsPerBlock
    let (tileCount, tileCountOverflow) = threadgroups.multipliedReportingOverflow(
        by: blocks)
    let (wordCount, wordCountOverflow) = tileCount.multipliedReportingOverflow(
        by: gemma4TiedHeadCoTiledMetadataWordsPerTile)
    guard !tileCountOverflow, !wordCountOverflow else { return nil }

    var words = [UInt32](repeating: 0, count: wordCount)
    for threadgroup in 0..<threadgroups {
        for block in 0..<blocks {
            let outputTile = (threadgroup * blocks + block)
                * gemma4TiedHeadCoTiledMetadataWordsPerTile
            for localRow in 0..<gemma4TiedHeadCoTiledRowsPerThreadgroup {
                let inputRow = (threadgroup
                    * gemma4TiedHeadCoTiledRowsPerThreadgroup + localRow)
                    * groupsPerRow
                let inputBlock = inputRow
                    + block * gemma4TiedHeadCoTiledGroupsPerBlock
                let outputRow = outputTile + localRow * 2
                for pair in 0..<2 {
                    let low = UInt32(indices[inputBlock + pair * 2])
                    let high = UInt32(indices[inputBlock + pair * 2 + 1])
                    words[outputRow + pair] = low | (high << 16)
                }
            }
        }
    }
    return words
}

private func gemma4MakeTiedHeadCoTiledU16Payload(
    weight: MLXArray,
    metadata: Gemma4TiedHeadPacked13Metadata,
    materialize: Bool = true
) -> MLXArray? {
    guard weight.dtype == .uint32,
          weight.shape == [262_144, 672],
          metadata.packedIndices.dtype == .uint32,
          metadata.packedIndices.shape == [262_144, 35],
          metadata.lut.dtype == .uint32,
          metadata.lut.ndim == 1,
          (1...8_192).contains(metadata.lut.size),
          let indices = gemma4DecodeTiedHeadPacked13Indices(
              metadata.packedIndices.asArray(UInt32.self),
              rows: 262_144,
              groupsPerRow: 84,
              lutCount: metadata.lut.size
          ),
          let metadataWords = gemma4PackTiedHeadCoTiledU16Metadata(
              indices,
              rows: 262_144,
              groupsPerRow: 84
          )
    else {
        return nil
    }

    // A tile owns 16 output rows and one 256-wide K block. Its payload is
    // [16 rows * 32 weight U32, 16 rows * 4 direct U16 indexes].
    let weightPayload = weight
        .reshaped(16_384, 16, gemma4TiedHeadCoTiledBlocks, 32)
        .transposed(0, 2, 1, 3)
        .contiguous()
        .reshaped(
            16_384,
            gemma4TiedHeadCoTiledBlocks,
            gemma4TiedHeadCoTiledWeightWordsPerTile
        )
    let metadataPayload = MLXArray(
        metadataWords,
        [
            16_384,
            gemma4TiedHeadCoTiledBlocks,
            gemma4TiedHeadCoTiledMetadataWordsPerTile,
        ]
    )
    let payload = concatenated([weightPayload, metadataPayload], axis: 2)
    guard payload.dtype == .uint32,
          payload.shape == [
              16_384,
              gemma4TiedHeadCoTiledBlocks,
              gemma4TiedHeadCoTiledPayloadWordsPerTile,
          ],
          payload.size * MemoryLayout<UInt32>.size
              == gemma4TiedHeadCoTiledPayloadByteCount
    else {
        return nil
    }
    if materialize {
        // This input-independent sidecar is built during untimed model init.
        eval(payload)
    }
    return payload
}

private let gemma4TiedVocabularyHeadQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_qmv_262144x5376_v1",
    inputNames: ["weight", "scales", "biases", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 262144;
        constexpr int kGroupsPerRow = 84;
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
        const device bfloat* row_scales =
            scales + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* row_biases =
            biases + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum =
                gemma4_tied_head_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                result[row] += gemma4_tied_head_qdot_4bit(
                    row_weight,
                    values,
                    static_cast<float>(row_scales[row * kGroupsPerRow]),
                    static_cast<float>(row_biases[row * kGroupsPerRow]),
                    input_sum);
            }

            weight_bytes += 128;
            row_scales += 4;
            row_biases += 4;
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

        inline float gemma4_tied_head_load_values(
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

        inline float gemma4_tied_head_qdot_4bit(
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

private let gemma4TiedVocabularyHeadPacked13SoftcapQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_packed13_softcap_qmv_262144x5376_v2",
    inputNames: ["weight", "packed_indices", "lut", "x", "cap"],
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
                gemma4_tied_head_packed13_load_values(input, values);
            const uint metadata_column =
                block * 4 + thread_index_in_simdgroup / 8;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index =
                    gemma4_tied_head_extract_packed13(
                        row_packed_indices + row * kPackedWordsPerRow,
                        metadata_column);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_tied_head_packed13_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_tied_head_pair_scale(pair),
                    gemma4_tied_head_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                // Match the stock graph's materialized BF16 projection before
                // applying its Float32 softcap suffix in the same command.
                const bfloat projected = static_cast<bfloat>(result[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
        }
        """,
    header: """
        using namespace metal;

        inline ushort gemma4_tied_head_extract_packed13(
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

        inline float gemma4_tied_head_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_tied_head_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_tied_head_packed13_load_values(
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

        inline float gemma4_tied_head_packed13_qdot_4bit(
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

private let gemma4TiedVocabularyHeadCoTiledU16SoftcapQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_cotiled_u16_softcap_262144x5376_v1",
    inputNames: ["cotiled_payload", "lut", "x", "cap"],
    outputNames: ["output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDGroupsPerThreadgroup = 4;
        constexpr int kBlocks = 21;
        constexpr int kWeightWordsPerRow = 32;
        constexpr int kWeightWordsPerSIMD =
            kRowsPerSIMD * kWeightWordsPerRow;
        constexpr int kWeightWordsPerTile = 512;
        constexpr int kMetadataGroupsPerRow = 4;
        constexpr int kMetadataWordsPerTile = 32;
        constexpr int kPayloadWords =
            kWeightWordsPerTile + kMetadataWordsPerTile;
        constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

        const int threadgroup_row = threadgroup_position_in_grid.y;
        const int simd_group = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_row * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
            + simd_group * kRowsPerSIMD;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const device uint* tile_words =
            cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < kBlocks; ++block) {
            float values[8];
            const float input_sum =
                gemma4_tied_head_cotiled_u16_load_values(input, values);
            const device uint* weight_words = tile_words
                + simd_group * kWeightWordsPerSIMD
                + thread_index_in_simdgroup;
            const device ushort* metadata =
                reinterpret_cast<const device ushort*>(
                    tile_words + kWeightWordsPerTile)
                + simd_group * kRowsPerSIMD * kMetadataGroupsPerRow;
            const uint metadata_column = thread_index_in_simdgroup / 8;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        weight_words + row * kWeightWordsPerRow);
                const ushort metadata_index =
                    metadata[row * kMetadataGroupsPerRow + metadata_column];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_tied_head_cotiled_u16_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_tied_head_cotiled_u16_pair_scale(pair),
                    gemma4_tied_head_cotiled_u16_pair_bias(pair),
                    input_sum);
            }

            tile_words += kPayloadWords;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                // Preserve the stock fused kernel's BF16 projection boundary
                // and precise Float32 softcap suffix exactly.
                const bfloat projected = static_cast<bfloat>(result[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_tied_head_cotiled_u16_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_tied_head_cotiled_u16_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_tied_head_cotiled_u16_load_values(
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

        inline float gemma4_tied_head_cotiled_u16_qdot_4bit(
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

private let gemma4TiedHeadMaximumPacked13LUTCount = 8_192

struct Gemma4TiedHeadPacked13Metadata: @unchecked Sendable {
    let packedIndices: MLXArray
    let lut: MLXArray
}

func validateGemma4TiedHeadPacked13MetadataLayout(
    _ metadata: Gemma4TiedHeadPacked13Metadata,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray
) throws {
    guard weight.dtype == .uint32,
          weight.shape == [262_144, 672],
          scales.dtype == .bfloat16,
          scales.shape == [262_144, 84],
          biases.dtype == .bfloat16,
          biases.shape == scales.shape,
          metadata.packedIndices.dtype == .uint32,
          metadata.packedIndices.shape == [262_144, 35],
          metadata.lut.dtype == .uint32,
          metadata.lut.ndim == 1,
          (1...gemma4TiedHeadMaximumPacked13LUTCount).contains(
              metadata.lut.size
          )
    else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata has invalid dtype or shape"
        )
    }
}

func supportsGemma4TiedVocabularyHead(
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?,
    groupSize: Int,
    bits: Int
) -> Bool {
    guard let biases else { return false }
    return groupSize == 64
        && bits == 4
        && weight.dtype == .uint32
        && weight.shape == [262_144, 672]
        && scales.dtype == .bfloat16
        && scales.shape == [262_144, 84]
        && biases.dtype == .bfloat16
        && biases.shape == [262_144, 84]
}

func isGemma4ProductionTiedVocabularyHead(_ embedding: Embedding) -> Bool {
    guard let embedding = embedding as? QuantizedEmbedding else { return false }
    return embedding.mode == .affine
        && embedding.groupSize == 64
        && embedding.bits == 4
        && embedding.weight.dtype == .uint32
        && embedding.weight.shape == [262_144, 672]
}

struct Gemma4TiedVocabularyHead: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let packed13Metadata: Gemma4TiedHeadPacked13Metadata?
    private let coTiledU16Payload: MLXArray?

    init?(
        _ embedding: Embedding,
        packed13Metadata: Gemma4TiedHeadPacked13Metadata?
    ) {
        guard let embedding = embedding as? QuantizedEmbedding,
              embedding.mode == .affine,
              let biases = embedding.biases,
              supportsGemma4TiedVocabularyHead(
                weight: embedding.weight,
                scales: embedding.scales,
                biases: biases,
                groupSize: embedding.groupSize,
                bits: embedding.bits
              )
        else { return nil }
        self.weight = embedding.weight
        self.scales = embedding.scales
        self.biases = biases
        self.packed13Metadata = packed13Metadata
        if gemma4TiedHeadCoTiledU16Enabled
            || gemma4VerifyTiedHeadCoTiledU16Bits
        {
            guard let packed13Metadata else {
                preconditionFailure(
                    "requested tied-head co-tiled U16 path requires "
                        + "validated packed13 metadata"
                )
            }
            guard let coTiledU16Payload =
                gemma4MakeTiedHeadCoTiledU16Payload(
                    weight: embedding.weight,
                    metadata: packed13Metadata
                )
            else {
                preconditionFailure(
                    "requested tied-head co-tiled U16 payload could not be "
                        + "derived losslessly from packed13 metadata"
                )
            }
            self.coTiledU16Payload = coTiledU16Payload
        } else {
            self.coTiledU16Payload = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        return gemma4TiedVocabularyHeadQMV(
            [weight, scales, biases, input],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    func packed13Softcapped(_ input: MLXArray, cap: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        precondition(cap.dtype == .float32)
        precondition(cap.size == 1)
        guard let packed13Metadata else {
            preconditionFailure("tied vocabulary packed13 metadata was not prepared")
        }

        var verifiedStock: MLXArray?
        if let coTiledU16Payload {
            let candidate = gemma4TiedVocabularyHeadCoTiledU16SoftcapQMV(
                [coTiledU16Payload, packed13Metadata.lut, input, cap],
                grid: (32, 65_536, 1),
                threadGroup: (32, 4, 1),
                outputShapes: [[1, 1, 262_144]],
                outputDTypes: [.float32]
            )[0]
            if gemma4VerifyTiedHeadCoTiledU16Bits {
                let stock = packed13SoftcappedStock(
                    input,
                    cap: cap,
                    metadata: packed13Metadata
                )
                verifyRawFloat32(
                    candidate,
                    stock: stock,
                    candidateName: "co-tiled direct-U16 fused-softcap",
                    stockName: "packed13 fused-softcap"
                )
                verifiedStock = stock
            }
            if gemma4TiedHeadCoTiledU16Enabled {
                return candidate
            }
        }

        return verifiedStock ?? packed13SoftcappedStock(
            input,
            cap: cap,
            metadata: packed13Metadata
        )
    }

    private func packed13SoftcappedStock(
        _ input: MLXArray,
        cap: MLXArray,
        metadata: Gemma4TiedHeadPacked13Metadata
    ) -> MLXArray {
        gemma4TiedVocabularyHeadPacked13SoftcapQMV(
            [weight, metadata.packedIndices, metadata.lut, input, cap],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.float32]
        )[0]
    }

    func verifyRawFloat32(
        _ candidate: MLXArray,
        stock: MLXArray,
        candidateName: String,
        stockName: String
    ) {
        precondition(candidate.dtype == .float32)
        precondition(stock.dtype == .float32)
        precondition(candidate.shape == stock.shape)
        let candidateBits = candidate.view(dtype: .uint32)
        let stockBits = stock.view(dtype: .uint32)
        let equal = arrayEqual(candidateBits, stockBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt32.self)
            let stockValues = stockBits.asArray(UInt32.self)
            let mismatch = zip(candidateValues, stockValues).enumerated().first {
                $0.element.0 != $0.element.1
            }
            preconditionFailure(
                "tied vocabulary head raw Float32 mismatch \(candidateName) vs "
                    + "\(stockName) at index "
                    + "\(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), stock="
                    + "\(mismatch?.element.1 ?? 0)"
            )
        }
    }

    func verifyRawBF16(
        _ candidate: MLXArray,
        stock: MLXArray,
        candidateName: String,
        stockName: String
    ) {
        precondition(candidate.dtype == .bfloat16)
        precondition(stock.dtype == .bfloat16)
        precondition(candidate.shape == stock.shape)
        let candidateBits = candidate.view(dtype: .uint16)
        let stockBits = stock.view(dtype: .uint16)
        let equal = arrayEqual(candidateBits, stockBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let stockValues = stockBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, stockValues).enumerated().first {
                $0.element.0 != $0.element.1
            }
            preconditionFailure(
                "tied vocabulary head raw BF16 mismatch \(candidateName) vs "
                    + "\(stockName) at index "
                    + "\(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), stock="
                    + "\(mismatch?.element.1 ?? 0)"
            )
        }
    }
}
