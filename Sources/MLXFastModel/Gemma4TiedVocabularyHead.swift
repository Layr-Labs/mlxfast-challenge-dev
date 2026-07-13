import Foundation
import MLX
import MLXFastCore
import MLXNN

let gemma4TiedHeadRowsPerThreadgroup = 16
let gemma4TiedHeadBlocks = 21
let gemma4TiedHeadWeightWordsPerBlock = 512
let gemma4TiedHeadMetadataWordsPerBlock = 26
let gemma4TiedHeadPayloadWordsPerBlock = 538

func gemma4TiedHeadCoTileWeightSourceWordIndex(
    threadgroup: Int, block: Int, localRow: Int, word: Int
) -> Int? {
    guard (0..<16_384).contains(threadgroup), (0..<21).contains(block),
          (0..<16).contains(localRow), (0..<32).contains(word) else { return nil }
    return (threadgroup * 16 + localRow) * 672 + block * 32 + word
}

func gemma4PackTiedHeadCoTileMetadata(_ indices: [UInt16]) -> [UInt32]? {
    guard indices.count == 262_144 * 84,
          indices.allSatisfy({ $0 < 8_192 }) else { return nil }
    var result = [UInt32](repeating: 0, count: 16_384 * 21 * 26)
    for tg in 0..<16_384 {
        for block in 0..<21 {
            let destination = (tg * 21 + block) * 26
            for row in 0..<16 {
                let source = (tg * 16 + row) * 84 + block * 4
                for group in 0..<4 {
                    let field = row * 4 + group
                    let bit = field * 13
                    let word = bit >> 5
                    let shift = bit & 31
                    let value = UInt32(indices[source + group])
                    result[destination + word] |= value << UInt32(shift)
                    if shift > 19 {
                        result[destination + word + 1] |= value >> UInt32(32 - shift)
                    }
                }
            }
        }
    }
    return result
}

func gemma4UnpackTiedHeadCoTileMetadata(
    _ words: [UInt32], threadgroup: Int, block: Int,
    localRow: Int, group: Int
) -> UInt16? {
    guard words.count == 16_384 * 21 * 26,
          (0..<16_384).contains(threadgroup), (0..<21).contains(block),
          (0..<16).contains(localRow), (0..<4).contains(group) else { return nil }
    let bit = (localRow * 4 + group) * 13
    let base = (threadgroup * 21 + block) * 26
    let index = bit >> 5
    let shift = bit & 31
    var value = words[base + index] >> UInt32(shift)
    if shift > 19 { value |= words[base + index + 1] << UInt32(32 - shift) }
    return UInt16(value & 0x1fff)
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

private let gemma4TiedVocabularyHeadCoTiledPacked13SoftcapQMV = MLXFast.metalKernel(
    name: "gemma4_tied_vocabulary_head_cotiled_packed13_softcap_qmv_v1",
    inputNames: ["payload", "lut", "x", "cap"],
    outputNames: ["output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDGroupsPerThreadgroup = 4;
        constexpr int kTileWords = 538;
        const int tg = threadgroup_position_in_grid.y;
        const int output_row = tg * 16 + simdgroup_index_in_threadgroup * 4;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            const device uint* tile = payload + (tg * 21 + block) * kTileWords;
            const device uint* metadata = tile + 512;
            float values[8];
            const float input_sum = gemma4_tied_head_cotiled_load_values(input, values);
            const uint column = thread_index_in_simdgroup / 8;
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const int local_row = simdgroup_index_in_threadgroup * 4 + row;
                const device uchar* row_weight = reinterpret_cast<const device uchar*>(
                    tile + local_row * 32) + thread_index_in_simdgroup * 4;
                const ushort metadata_index = gemma4_tied_head_cotiled_extract13(
                    metadata, local_row * 4 + column);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_tied_head_cotiled_qdot(
                    row_weight, values,
                    gemma4_tied_head_cotiled_scale(pair),
                    gemma4_tied_head_cotiled_bias(pair), input_sum);
            }
            input += 256;
        }
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                const bfloat projected = static_cast<bfloat>(result[row]);
                const float quotient = static_cast<float>(projected) / cap;
                const float softened = metal::precise::tanh(quotient);
                output[output_row + row] = softened * cap;
            }
        }
        """,
    header: """
        using namespace metal;
        inline ushort gemma4_tied_head_cotiled_extract13(const device uint* words, uint column) {
            const uint bit = column * 13, wi = bit >> 5, shift = bit & 31;
            uint value = words[wi] >> shift;
            if (shift > 19) value |= words[wi + 1] << (32 - shift);
            return static_cast<ushort>(value & 0x1fff);
        }
        inline float gemma4_tied_head_cotiled_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }
        inline float gemma4_tied_head_cotiled_bias(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }
        inline float gemma4_tied_head_cotiled_load_values(const device bfloat* input, thread float* values) {
            float sum = 0;
            for (int index = 0; index < 8; index += 4) {
                sum += input[index] + input[index + 1] + input[index + 2] + input[index + 3];
                values[index] = input[index]; values[index + 1] = input[index + 1] / 16.0f;
                values[index + 2] = input[index + 2] / 256.0f; values[index + 3] = input[index + 3] / 4096.0f;
            }
            return sum;
        }
        inline float gemma4_tied_head_cotiled_qdot(const device uchar* weight, const thread float* values,
                                                    float scale, float bias, float input_sum) {
            const device ushort* packed = reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int index = 0; index < 2; ++index) {
                accumulator += (values[4 * index] * (packed[index] & 0x000f)
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

private func gemma4MakeCoTiledTiedHeadPayload(
    weight: MLXArray, metadata: Gemma4TiedHeadPacked13Metadata
) -> MLXArray? {
    let packed = metadata.packedIndices.asArray(UInt32.self)
    var indices = [UInt16](repeating: 0, count: 262_144 * 84)
    for row in 0..<262_144 {
        for column in 0..<84 {
            let bit = column * 13, word = bit >> 5, shift = bit & 31
            var value = packed[row * 35 + word] >> UInt32(shift)
            if shift > 19 { value |= packed[row * 35 + word + 1] << UInt32(32 - shift) }
            indices[row * 84 + column] = UInt16(value & 0x1fff)
        }
    }
    guard let metadataWords = gemma4PackTiedHeadCoTileMetadata(indices) else { return nil }
    let weights = weight.reshaped(16_384, 16, 21, 32)
        .transposed(0, 2, 1, 3).contiguous()
        .reshaped(16_384, 21, 512)
    // Each tile contains 16 rows × 32 U32 weight words = 512 words.
    let metadataArray = MLXArray(metadataWords, [16_384, 21, 26])
    let payload = concatenated([weights, metadataArray], axis: 2)
    precondition(payload.dtype == .uint32 && payload.shape == [16_384, 21, 538])
    eval(payload)
    return payload
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
    let coTiledPayload: MLXArray?
    let useCoTiled: Bool
    let verifyCoTiledBits: Bool

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
        let environment = ProcessInfo.processInfo.environment
        self.useCoTiled = environment["DARKBLOOM_TIED_HEAD_COTILED"] != "0"
        self.verifyCoTiledBits = environment["DARKBLOOM_VERIFY_TIED_HEAD_COTILED_BITS"] == "1"
        self.coTiledPayload = packed13Metadata.flatMap {
            gemma4MakeCoTiledTiedHeadPayload(weight: embedding.weight, metadata: $0)
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
        let current = {
            gemma4TiedVocabularyHeadPacked13SoftcapQMV(
                [weight, packed13Metadata.packedIndices, packed13Metadata.lut, input, cap],
                grid: (32, 65_536, 1), threadGroup: (32, 4, 1),
                outputShapes: [[1, 1, 262_144]], outputDTypes: [.float32]
            )[0]
        }
        guard useCoTiled, let coTiledPayload else { return current() }
        let candidate = gemma4TiedVocabularyHeadCoTiledPacked13SoftcapQMV(
            [coTiledPayload, packed13Metadata.lut, input, cap],
            grid: (32, 65_536, 1), threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]], outputDTypes: [.float32]
        )[0]
        if verifyCoTiledBits {
            verifyRawFloat32(candidate, stock: current(), candidateName: "co-tiled packed13 fused-softcap",
                             stockName: "packed13 fused-softcap")
            return current()
        }
        return candidate
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
