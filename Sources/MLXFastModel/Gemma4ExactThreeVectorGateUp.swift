import MLX

/// Three independent token rows sharing one traversal of the production
/// fixed12 gate/up payload. Every token keeps its own Float accumulators and
/// crosses the same BF16 projection boundary before the exact GELU/product
/// epilogue used by the promoted one- and two-vector kernels.
private let gemma4ExactThreeVectorGateUpSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kVectorCount = 3;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;
    constexpr int kWordsPerProjectionBlock =
        kRowsPerSIMD * kSIMDSize;
    constexpr int kWeightWordsPerPair =
        4 * kWordsPerProjectionBlock;
    constexpr int kMetadataBytesPerProjectionPair = 48;
    constexpr int kWordsPerPair = 536;
    constexpr int kPairCount = 10;
    constexpr int kTailWeightWords =
        2 * kWordsPerProjectionBlock;
    constexpr int kTailMetadataBytesPerProjection = 32;
    constexpr int kWordsPerTail = 272;
    constexpr int kWordsPerThreadgroup =
        kPairCount * kWordsPerPair + kWordsPerTail;

    const bool is_up = simdgroup_index_in_threadgroup == 1;
    const int threadgroup_row = threadgroup_position_in_grid.y;
    const int output_row = threadgroup_row * kRowsPerSIMD;
    const uint lane = thread_index_in_simdgroup;
    const uint lane_group = lane >> 3;
    const device uint* tile_words =
        cotiled_payload + threadgroup_row * kWordsPerThreadgroup;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device uint* lut = is_up ? up_lut : gate_lut;

    float results[kVectorCount][kRowsPerSIMD] = {{0}};
    for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
        uint odd_pairs[kRowsPerSIMD];
        const device uint* even_weight_words = tile_words
            + (is_up ? kWordsPerProjectionBlock : 0)
            + lane;
        const device uint* odd_weight_words = tile_words
            + 2 * kWordsPerProjectionBlock
            + (is_up ? kWordsPerProjectionBlock : 0)
            + lane;
        const device uchar* metadata_bytes =
            reinterpret_cast<const device uchar*>(
                tile_words + kWeightWordsPerPair)
            + (is_up ? kMetadataBytesPerProjectionPair : 0);

        {
            float values[kVectorCount][8];
            float input_sums[kVectorCount];
            input_sums[0] = gemma4_exact_three_vector_gate_up_load_values(
                input0, values[0]);
            input_sums[1] = gemma4_exact_three_vector_gate_up_load_values(
                input1, values[1]);
            input_sums[2] = gemma4_exact_three_vector_gate_up_load_values(
                input2, values[2]);

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata_bytes + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                const device uint* row_weight_words =
                    even_weight_words + row * kSIMDSize;
                const uint packed_word = row_weight_words[0];
                #pragma clang loop unroll(full)
                for (int vector = 0; vector < kVectorCount; ++vector) {
                    gemma4_exact_three_vector_gate_up_qdot_4bit(
                        packed_word,
                        values[vector],
                        gemma4_exact_three_vector_gate_up_pair_scale(even_pair),
                        gemma4_exact_three_vector_gate_up_pair_bias(even_pair),
                        input_sums[vector],
                        results[vector][row]);
                }
            }
        }

        input0 += 256;
        input1 += 256;
        input2 += 256;
        {
            float values[kVectorCount][8];
            float input_sums[kVectorCount];
            input_sums[0] = gemma4_exact_three_vector_gate_up_load_values(
                input0, values[0]);
            input_sums[1] = gemma4_exact_three_vector_gate_up_load_values(
                input1, values[1]);
            input_sums[2] = gemma4_exact_three_vector_gate_up_load_values(
                input2, values[2]);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_pairs[row];
                const device uint* row_weight_words =
                    odd_weight_words + row * kSIMDSize;
                const uint packed_word = row_weight_words[0];
                #pragma clang loop unroll(full)
                for (int vector = 0; vector < kVectorCount; ++vector) {
                    gemma4_exact_three_vector_gate_up_qdot_4bit(
                        packed_word,
                        values[vector],
                        gemma4_exact_three_vector_gate_up_pair_scale(odd_pair),
                        gemma4_exact_three_vector_gate_up_pair_bias(odd_pair),
                        input_sums[vector],
                        results[vector][row]);
                }
            }
        }

        input0 += 256;
        input1 += 256;
        input2 += 256;
        tile_words += kWordsPerPair;
    }

    float tail_values[kVectorCount][8];
    float tail_input_sums[kVectorCount];
    tail_input_sums[0] = gemma4_exact_three_vector_gate_up_load_values(
        input0, tail_values[0]);
    tail_input_sums[1] = gemma4_exact_three_vector_gate_up_load_values(
        input1, tail_values[1]);
    tail_input_sums[2] = gemma4_exact_three_vector_gate_up_load_values(
        input2, tail_values[2]);
    const device uint* tail_weight_words = tile_words
        + (is_up ? kWordsPerProjectionBlock : 0)
        + lane;
    const device uchar* tail_metadata =
        reinterpret_cast<const device uchar*>(
            tile_words + kTailWeightWords)
        + (is_up ? kTailMetadataBytesPerProjection : 0);
    const uint tail_lane_offset = lane_group << 1;
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        const device uchar* row_metadata = tail_metadata + row * 8;
        const uint low = row_metadata[tail_lane_offset];
        const uint high = row_metadata[tail_lane_offset + 1];
        const uint metadata_index = low | (high << 8);
        const uint pair = lut[metadata_index];
        const device uint* row_weight_words =
            tail_weight_words + row * kSIMDSize;
        const uint packed_word = row_weight_words[0];
        #pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectorCount; ++vector) {
            gemma4_exact_three_vector_gate_up_qdot_4bit(
                packed_word,
                tail_values[vector],
                gemma4_exact_three_vector_gate_up_pair_scale(pair),
                gemma4_exact_three_vector_gate_up_pair_bias(pair),
                tail_input_sums[vector],
                results[vector][row]);
        }
    }

    threadgroup bfloat projections[2][kVectorCount][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for (int vector = 0; vector < kVectorCount; ++vector) {
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            results[vector][row] = simd_sum(results[vector][row]);
            if (thread_index_in_simdgroup == 0) {
                projections[is_up ? 1 : 0][vector][row] =
                    static_cast<bfloat>(results[vector][row]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        #pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectorCount; ++vector) {
            const bfloat gate = projections[0][vector][row];
            const bfloat up = projections[1][vector][row];
            const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
            const bfloat cubic1 = cubic0 * gate;
            const bfloat cubic2 = cubic1 * gate;
            const bfloat inner0 = gate + cubic2;
            const bfloat inner1 =
                static_cast<bfloat>(0.7978845834732056f) * inner0;
            const bfloat tanh_value =
                static_cast<bfloat>(metal::precise::tanh(inner1));
            const bfloat shifted = static_cast<bfloat>(1.0f) + tanh_value;
            const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
            const bfloat gelu = scaled * shifted;
            activated[vector * kOutputRows + output_row + row] = gelu * up;
        }
    }
    """

private let gemma4ExactThreeVectorGateUpHeader = """
    using namespace metal;

    inline float gemma4_exact_three_vector_gate_up_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_three_vector_gate_up_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_three_vector_gate_up_load_values(
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

    inline void gemma4_exact_three_vector_gate_up_qdot_4bit(
        uint packed_word,
        const thread float* values,
        float scale,
        float bias,
        float input_sum,
        thread float& result
    ) {
        float accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values[4 * index] * (packed & 0x000f)
                + values[4 * index + 1] * (packed & 0x00f0)
                + values[4 * index + 2] * (packed & 0x0f00)
                + values[4 * index + 3] * (packed & 0xf000));
        }
        result += scale * accumulator + input_sum * bias;
    }
    """

private let gemma4ExactThreeVectorGateUpKernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed12_gate_up_activation_qmv_5376_v1",
    inputNames: ["cotiled_payload", "gate_lut", "up_lut", "x"],
    outputNames: ["activated"],
    source: gemma4ExactThreeVectorGateUpSource,
    header: gemma4ExactThreeVectorGateUpHeader,
    ensureRowContiguous: true
)

func gemma4ExactThreeVectorGateUpActivated(
    payload: MLXArray,
    gateLUT: MLXArray,
    upLUT: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [5_376, 5_632])
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1)
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
    return gemma4ExactThreeVectorGateUpKernel(
        [payload, gateLUT, upLUT, input],
        grid: (32, 10_752, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[3, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}
