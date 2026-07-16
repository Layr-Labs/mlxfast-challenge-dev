import Foundation
import MLX
import MLXFastCore
import MLXNN

private func gemma4TiedHeadEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4TiedHeadSoftcapLUTEnabled = gemma4TiedHeadEnvironmentFlag(
    "DARKBLOOM_TIED_HEAD_SOFTCAP_LUT",
    default: true
)

private let gemma4VerifyTiedHeadSoftcapLUTBits =
    gemma4TiedHeadEnvironmentFlag(
        "DARKBLOOM_VERIFY_TIED_HEAD_SOFTCAP_LUT_BITS",
        default: false
    )

/// Verification is a stronger selection than an explicit performance
/// rollback: asking to verify this route must actually dispatch it.
private let gemma4UseTiedHeadSoftcapLUT =
    gemma4TiedHeadSoftcapLUTEnabled
        || gemma4VerifyTiedHeadSoftcapLUTBits

func gemma4TiedHeadSoftcapLUTVerificationRequested() -> Bool {
    gemma4VerifyTiedHeadSoftcapLUTBits
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

/// Materialize the complete BF16-domain softcap once. The scored tied-head
/// projection has an explicit BF16 boundary, so its Float32 suffix can observe
/// only these 65,536 raw inputs. Keeping the formula in Metal uses the exact
/// same precise tanh implementation as the production suffix.
private let gemma4TiedHeadSoftcapLUTBuilder = MLXFast.metalKernel(
    name: "gemma4_tied_head_softcap_lut_65536_v1",
    inputNames: ["cap"],
    outputNames: ["output"],
    source: """
        const uint bits = thread_position_in_grid.x;
        const bfloat projected =
            as_type<bfloat>(static_cast<ushort>(bits));
        const float quotient = static_cast<float>(projected) / cap;
        const float softened = metal::precise::tanh(quotient);
        output[bits] = softened * cap;
        """,
    header: """
        using namespace metal;
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

/// Exact packed13 QMV with the same BF16 projection boundary as the production
/// kernel. The only changed suffix is a raw-BF16 lookup into the table prepared
/// by `gemma4TiedHeadSoftcapLUTBuilder`.
private let gemma4TiedVocabularyHeadPacked13SoftcapLUTQMV =
    MLXFast.metalKernel(
        name: "gemma4_tied_vocabulary_head_packed13_softcap_lut_qmv_"
            + "262144x5376_v1",
        inputNames: ["weight", "packed_indices", "lut", "x", "softcap_lut"],
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
                    const bfloat projected = static_cast<bfloat>(result[row]);
                    const ushort projected_bits = as_type<ushort>(projected);
                    output[output_row + row] = softcap_lut[projected_bits];
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
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair)));
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

private let gemma4TiedHeadMaximumPacked13LUTCount = 8_192

private func makeGemma4TiedHeadSoftcapLUT(
    cap: MLXArray,
    verifyRawBits: Bool
) -> MLXArray {
    precondition(cap.dtype == .float32 && cap.ndim == 0 && cap.size == 1)
    let table = gemma4TiedHeadSoftcapLUTBuilder(
        [cap],
        grid: (65_536, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[65_536]],
        outputDTypes: [.float32]
    )[0]

    if verifyRawBits {
        // Normal construction deliberately remains lazy so the existing
        // constructor warmup materializes builder and consumer together after
        // FastEngine finishes all host-side preparation. Verification is the
        // only mode that synchronizes the table here.
        eval(table)
        let bitPatterns = MLXArray((0..<65_536).map { UInt16($0) })
        let projected = bitPatterns.view(dtype: .bfloat16)
        let reference = tanh(projected.asType(.float32) / cap) * cap
        let tableBits = table.view(dtype: .uint32)
        let referenceBits = reference.view(dtype: .uint32)
        let equal = arrayEqual(tableBits, referenceBits)
        eval(equal)
        guard equal.item(Bool.self) else {
            let candidateValues = tableBits.asArray(UInt32.self)
            let referenceValues = referenceBits.asArray(UInt32.self)
            let mismatch = zip(candidateValues, referenceValues)
                .enumerated().first { $0.element.0 != $0.element.1 }
            preconditionFailure(
                "tied-head softcap LUT raw Float32 mismatch at BF16 bits "
                    + "\(mismatch?.offset ?? -1): candidate="
                    + "\(mismatch?.element.0 ?? 0), reference="
                    + "\(mismatch?.element.1 ?? 0)"
            )
        }
    }

    return table
}

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
    /// The single model-configured scalar used by every tied-head suffix.
    /// Owning it here prevents a precomputed table from being paired with a
    /// different call-site cap.
    let softcap: MLXArray
    let softcapLUT: MLXArray?

    init?(
        _ embedding: Embedding,
        packed13Metadata: Gemma4TiedHeadPacked13Metadata?,
        softcap: Float
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
        precondition(softcap.isFinite && softcap > 0)
        let softcapArray = MLXArray(softcap)
        self.softcap = softcapArray
        self.softcapLUT = gemma4UseTiedHeadSoftcapLUT
            ? makeGemma4TiedHeadSoftcapLUT(
                cap: softcapArray,
                verifyRawBits: gemma4VerifyTiedHeadSoftcapLUTBits
            )
            : nil
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

    func packed13Softcapped(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        guard let packed13Metadata else {
            preconditionFailure("tied vocabulary packed13 metadata was not prepared")
        }
        if gemma4UseTiedHeadSoftcapLUT {
            guard let softcapLUT else {
                preconditionFailure("tied vocabulary softcap LUT was not prepared")
            }
            precondition(softcapLUT.dtype == .float32)
            precondition(softcapLUT.shape == [65_536])
            return gemma4TiedVocabularyHeadPacked13SoftcapLUTQMV(
                [
                    weight,
                    packed13Metadata.packedIndices,
                    packed13Metadata.lut,
                    input,
                    softcapLUT,
                ],
                grid: (32, 65_536, 1),
                threadGroup: (32, 4, 1),
                outputShapes: [[1, 1, 262_144]],
                outputDTypes: [.float32]
            )[0]
        }
        return gemma4TiedVocabularyHeadPacked13SoftcapQMV(
            [
                weight,
                packed13Metadata.packedIndices,
                packed13Metadata.lut,
                input,
                softcap,
            ],
            grid: (32, 65_536, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, 262_144]],
            outputDTypes: [.float32]
        )[0]
    }

    var supportsExactTwoVectorPacked13: Bool {
        packed13Metadata != nil
    }

    func exactTwoVectorPacked13Softcapped(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [2, 5_376])
        guard let packed13Metadata else {
            preconditionFailure("exact two-vector tied-head metadata is unavailable")
        }
        return gemma4ExactTwoVectorTiedHead(
            weight: weight,
            packedIndices: packed13Metadata.packedIndices,
            lut: packed13Metadata.lut,
            input: input,
            cap: softcap
        )
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
