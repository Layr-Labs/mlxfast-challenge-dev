import Foundation
import MLX
import MLXFastCore
import MLXNN

private let gemma4TiedHeadRows = 262_144
private let gemma4TiedHeadGroupsPerRow = 84
private let gemma4TiedHeadPackedWordsPerRow = 35
private let gemma4TiedHeadRowsPerThreadgroup = 16
private let gemma4TiedHeadGroupsPerBlock = 4
private let gemma4TiedHeadBlocks = 21
private let gemma4TiedHeadWeightWordsPerTile = 512
private let gemma4TiedHeadFixed13MetadataBytesPerTile = 104
private let gemma4TiedHeadFixed13MetadataWordsPerTile = 26
private let gemma4TiedHeadFixed13PayloadWordsPerTile = 538
private let gemma4TiedHeadFixed13PayloadByteCount = 740_425_728

private func gemma4TiedHeadEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4TiedHeadCoTiledFixed13Enabled =
    gemma4TiedHeadEnvironmentFlag(
        "DARKBLOOM_TIED_HEAD_COTILED_FIXED13",
        default: true
    )

private let gemma4VerifyTiedHeadCoTiledFixed13Bits =
    gemma4TiedHeadEnvironmentFlag(
        "DARKBLOOM_VERIFY_TIED_HEAD_COTILED_FIXED13_BITS",
        default: false
    )

/// Repack independently row-padded packed13 indexes directly into the exact
/// `(threadgroup, K-block, adjacent-row-pair)` ownership of the co-tiled
/// kernel. Each pair contributes eight indexes in the promoted output
/// fixed13 format: eight low bytes, four middle-nibble bytes, and one byte of
/// top bits. Thus a 16-row tile occupies exactly 104 bytes (26 U32) without a
/// U16 expansion or per-row padding.
func gemma4PackTiedHeadCoTileFixed13Metadata(
    _ packed: [UInt32],
    rows: Int,
    groupsPerRow: Int,
    lutCount: Int
) -> [UInt32]? {
    guard rows > 0,
          rows.isMultiple(of: gemma4TiedHeadRowsPerThreadgroup),
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: gemma4TiedHeadGroupsPerBlock),
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
    let packedWordsPerRow = paddedRowBits / 32
    let (packedWordCount, packedWordCountOverflow) =
        rows.multipliedReportingOverflow(by: packedWordsPerRow)
    guard !packedWordCountOverflow, packed.count == packedWordCount else {
        return nil
    }

    let threadgroups = rows / gemma4TiedHeadRowsPerThreadgroup
    let blocks = groupsPerRow / gemma4TiedHeadGroupsPerBlock
    let (tileCount, tileCountOverflow) = threadgroups.multipliedReportingOverflow(
        by: blocks)
    let (wordCount, wordCountOverflow) = tileCount.multipliedReportingOverflow(
        by: gemma4TiedHeadFixed13MetadataWordsPerTile)
    let (_, byteCountOverflow) = wordCount.multipliedReportingOverflow(
        by: MemoryLayout<UInt32>.size)
    guard !tileCountOverflow, !wordCountOverflow, !byteCountOverflow else {
        return nil
    }

    var words = [UInt32](repeating: 0, count: wordCount)
    for threadgroup in 0..<threadgroups {
        for block in 0..<blocks {
            let tile = threadgroup * blocks + block
            let tileByteBase = tile
                * gemma4TiedHeadFixed13MetadataBytesPerTile
            for localRow in 0..<gemma4TiedHeadRowsPerThreadgroup {
                let sourceRow = threadgroup
                    * gemma4TiedHeadRowsPerThreadgroup + localRow
                let sourceWordBase = sourceRow * packedWordsPerRow
                let pairByteBase = tileByteBase + (localRow / 2) * 13
                for metadataColumn in 0..<gemma4TiedHeadGroupsPerBlock {
                    let sourceColumn = block
                        * gemma4TiedHeadGroupsPerBlock + metadataColumn
                    let bitOffset = sourceColumn * 13
                    let sourceWord = bitOffset / 32
                    let sourceShift = bitOffset % 32
                    var value = packed[sourceWordBase + sourceWord]
                        >> UInt32(sourceShift)
                    if sourceShift > 19 {
                        value |= packed[sourceWordBase + sourceWord + 1]
                            << UInt32(32 - sourceShift)
                    }
                    value &= 0x1fff
                    guard value < UInt32(lutCount) else { return nil }

                    let pairIndex = (localRow % 2)
                        * gemma4TiedHeadGroupsPerBlock + metadataColumn
                    let lowByte = pairByteBase + pairIndex
                    words[lowByte / 4] |= (value & 0xff)
                        << UInt32((lowByte % 4) * 8)

                    let middleByte = pairByteBase + 8 + pairIndex / 2
                    let middleShift = (middleByte % 4) * 8
                        + (pairIndex % 2) * 4
                    words[middleByte / 4] |= ((value >> 8) & 0x0f)
                        << UInt32(middleShift)

                    let topByte = pairByteBase + 12
                    let topShift = (topByte % 4) * 8 + pairIndex
                    words[topByte / 4] |= ((value >> 12) & 1)
                        << UInt32(topShift)
                }
            }
        }
    }
    return words
}

private func gemma4MakeTiedHeadCoTiledFixed13Payload(
    weight: MLXArray,
    metadata: Gemma4TiedHeadPacked13Metadata,
    materialize: Bool = true
) -> MLXArray? {
    guard weight.dtype == .uint32,
          weight.shape == [gemma4TiedHeadRows, 672],
          metadata.packedIndices.dtype == .uint32,
          metadata.packedIndices.shape == [
              gemma4TiedHeadRows,
              gemma4TiedHeadPackedWordsPerRow,
          ],
          metadata.lut.dtype == .uint32,
          metadata.lut.ndim == 1,
          (1...8_192).contains(metadata.lut.size),
          let fixed13Metadata = gemma4PackTiedHeadCoTileFixed13Metadata(
              metadata.packedIndices.asArray(UInt32.self),
              rows: gemma4TiedHeadRows,
              groupsPerRow: gemma4TiedHeadGroupsPerRow,
              lutCount: metadata.lut.size
          )
    else {
        return nil
    }

    // row = 16 * threadgroup + 4 * SIMD-group + row-within-SIMD
    // word = 32 * K-block + lane
    let threadgroups = gemma4TiedHeadRows
        / gemma4TiedHeadRowsPerThreadgroup
    let weightPayload = weight
        .reshaped(
            threadgroups,
            gemma4TiedHeadRowsPerThreadgroup,
            gemma4TiedHeadBlocks,
            32
        )
        .transposed(0, 2, 1, 3)
        .contiguous()
        .reshaped(
            threadgroups,
            gemma4TiedHeadBlocks,
            gemma4TiedHeadWeightWordsPerTile
        )
    let metadataPayload = MLXArray(
        fixed13Metadata,
        [
            threadgroups,
            gemma4TiedHeadBlocks,
            gemma4TiedHeadFixed13MetadataWordsPerTile,
        ]
    )
    let payload = concatenated([weightPayload, metadataPayload], axis: 2)
    guard payload.dtype == .uint32,
          payload.shape == [
              threadgroups,
              gemma4TiedHeadBlocks,
              gemma4TiedHeadFixed13PayloadWordsPerTile,
          ],
          payload.size * MemoryLayout<UInt32>.size
              == gemma4TiedHeadFixed13PayloadByteCount
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

private let gemma4TiedVocabularyHeadCoTiledFixed13SoftcapQMV =
    MLXFast.metalKernel(
        name: "gemma4_tied_vocabulary_head_cotiled_fixed13_softcap_262144x5376_v1",
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
            constexpr int kMetadataBytesPerPair = 13;
            constexpr int kMetadataBytesPerSIMD = 26;
            constexpr int kMetadataWordsPerTile = 26;
            constexpr int kPayloadWords =
                kWeightWordsPerTile + kMetadataWordsPerTile;
            constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

            const int threadgroup_row = threadgroup_position_in_grid.y;
            const int simd_group = simdgroup_index_in_threadgroup;
            const int output_row =
                threadgroup_row * kRowsPerSIMD * kSIMDGroupsPerThreadgroup
                + simd_group * kRowsPerSIMD;
            const uint metadata_column = thread_index_in_simdgroup / 8;
            const device bfloat* input =
                x + thread_index_in_simdgroup * 8;
            const device uint* tile_words =
                cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

            float result[kRowsPerSIMD] = {0};
            for (int block = 0; block < kBlocks; ++block) {
                float values[8];
                const float input_sum =
                    gemma4_tied_head_cotiled_fixed13_load_values(
                        input, values);
                const device uint* weight_words = tile_words
                    + simd_group * kWeightWordsPerSIMD
                    + thread_index_in_simdgroup;
                const device uchar* metadata_bytes =
                    reinterpret_cast<const device uchar*>(
                        tile_words + kWeightWordsPerTile)
                    + simd_group * kMetadataBytesPerSIMD;

                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    const device uchar* row_weight =
                        reinterpret_cast<const device uchar*>(
                            weight_words + row * kWeightWordsPerRow);
                    const device uchar* pair_metadata = metadata_bytes
                        + (row / 2) * kMetadataBytesPerPair;
                    const uint pair_index =
                        (row & 1) * 4 + metadata_column;
                    const uint low = pair_metadata[pair_index];
                    const uint packed_middle =
                        pair_metadata[8 + pair_index / 2];
                    const uint middle =
                        (packed_middle >> ((pair_index & 1) * 4)) & 0x0f;
                    const uint top =
                        (pair_metadata[12] >> pair_index) & 1;
                    const uint metadata_index =
                        low | (middle << 8) | (top << 12);
                    const uint pair = lut[metadata_index];
                    result[row] +=
                        gemma4_tied_head_cotiled_fixed13_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_tied_head_cotiled_fixed13_pair_scale(pair),
                            gemma4_tied_head_cotiled_fixed13_pair_bias(pair),
                            input_sum);
                }

                tile_words += kPayloadWords;
                input += 256;
            }

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                result[row] = simd_sum(result[row]);
                if (thread_index_in_simdgroup == 0) {
                    // Preserve the promoted packed13 kernel's materialized
                    // BF16 projection boundary and precise Float32 suffix.
                    const bfloat projected =
                        static_cast<bfloat>(result[row]);
                    const float quotient =
                        static_cast<float>(projected) / cap;
                    const float softened =
                        metal::precise::tanh(quotient);
                    output[output_row + row] = softened * cap;
                }
            }
            """,
        header: """
            using namespace metal;

            inline float gemma4_tied_head_cotiled_fixed13_pair_scale(
                uint pair
            ) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_tied_head_cotiled_fixed13_pair_bias(
                uint pair
            ) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_tied_head_cotiled_fixed13_load_values(
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

            inline float gemma4_tied_head_cotiled_fixed13_qdot_4bit(
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
                        + values[4 * index + 1] *
                            (packed[index] & 0x00f0)
                        + values[4 * index + 2] *
                            (packed[index] & 0x0f00)
                        + values[4 * index + 3] *
                            (packed[index] & 0xf000));
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
    private let coTiledFixed13Payload: MLXArray?

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
        if gemma4TiedHeadCoTiledFixed13Enabled
            || gemma4VerifyTiedHeadCoTiledFixed13Bits
        {
            guard let packed13Metadata else {
                preconditionFailure(
                    "requested tied-head co-tiled fixed13 path requires "
                        + "validated packed13 metadata"
                )
            }
            guard let coTiledFixed13Payload =
                gemma4MakeTiedHeadCoTiledFixed13Payload(
                    weight: embedding.weight,
                    metadata: packed13Metadata
                )
            else {
                preconditionFailure(
                    "requested tied-head co-tiled fixed13 payload could not "
                        + "be derived losslessly from packed13 metadata"
                )
            }
            self.coTiledFixed13Payload = coTiledFixed13Payload
        } else {
            self.coTiledFixed13Payload = nil
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
        if let coTiledFixed13Payload {
            let candidate =
                gemma4TiedVocabularyHeadCoTiledFixed13SoftcapQMV(
                    [
                        coTiledFixed13Payload,
                        packed13Metadata.lut,
                        input,
                        cap,
                    ],
                    grid: (32, 65_536, 1),
                    threadGroup: (32, 4, 1),
                    outputShapes: [[1, 1, 262_144]],
                    outputDTypes: [.float32]
                )[0]
            if gemma4VerifyTiedHeadCoTiledFixed13Bits {
                let stock = packed13SoftcappedStock(
                    input,
                    cap: cap,
                    metadata: packed13Metadata
                )
                verifyRawFloat32(
                    candidate,
                    stock: stock,
                    candidateName: "co-tiled fixed13 fused-softcap",
                    stockName: "packed13 fused-softcap"
                )
                verifiedStock = stock
            }
            if gemma4TiedHeadCoTiledFixed13Enabled {
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
