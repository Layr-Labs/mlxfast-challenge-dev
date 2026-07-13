import MLX

/// Exact three-row output/down projection research kernels. Each 512-wide
/// block loads its four owned rows of packed weights and affine metadata once,
/// then applies those values to three independent activation rows. The Float
/// accumulation order within every output element is identical to the
/// one-/two-vector kernels, including the final BF16 conversion boundary.
private func gemma4ExactThreeVectorCoTiledOutputDownSource(
    inputWidth: Int,
    indexBits: Int
) -> String {
    precondition(inputWidth == 8_192 || inputWidth == 16_384 || inputWidth == 21_504)
    precondition(indexBits == 12 || indexBits == 13)
    precondition(inputWidth != 21_504 || indexBits == 12)
    let metadataExtraction = indexBits == 12
        ? """
                const uint low = row_metadata[lane_group];
                const uint packed_high =
                    row_metadata[8 + lane_group / 2];
                const uint high =
                    (packed_high >> ((lane_group & 1) * 4)) & 0x0f;
                const uint metadata_index = low | (high << 8);
            """
        : """
                const uint low = row_metadata[lane_group];
                const uint packed_middle =
                    row_metadata[8 + lane_group / 2];
                const uint middle =
                    (packed_middle >> ((lane_group & 1) * 4)) & 0x0f;
                const uint top = (row_metadata[12] >> lane_group) & 1;
                const uint metadata_index =
                    low | (middle << 8) | (top << 12);
            """
    return """
        constexpr int kInputWidth = \(inputWidth);
        constexpr int kOutputRows = 5376;
        constexpr int kVectorCount = 3;
        constexpr int kRowsPerSIMD = 4;
        constexpr int kBlockSize = 512;
        constexpr int kBlocks = \(inputWidth / 512);
        constexpr int kWordsPerLane = 2;
        constexpr int kWordsPerRow = 32 * kWordsPerLane;
        constexpr int kWordsPerSIMD = kRowsPerSIMD * kWordsPerRow;
        constexpr int kWeightWordsPerTile = 2 * kWordsPerSIMD;
        constexpr int kMetadataBytesPerRow = \(indexBits);
        constexpr int kMetadataWordsPerTile = \(2 * indexBits);
        constexpr int kPayloadWords =
            kWeightWordsPerTile + kMetadataWordsPerTile;
        constexpr int kWordsPerThreadgroup = kBlocks * kPayloadWords;

        const int threadgroup_row = threadgroup_position_in_grid.y;
        const int simd_group = simdgroup_index_in_threadgroup;
        const int output_row =
            threadgroup_row * 8 + simd_group * kRowsPerSIMD;
        const uint lane_group = thread_index_in_simdgroup / 4;
        const device bfloat* input0 =
            x + thread_index_in_simdgroup * 16;
        const device bfloat* input1 =
            x + kInputWidth + thread_index_in_simdgroup * 16;
        const device bfloat* input2 =
            x + 2 * kInputWidth + thread_index_in_simdgroup * 16;
        const device uint* tile_words =
            cotiled_payload + threadgroup_row * kWordsPerThreadgroup;

        float results[kVectorCount][kRowsPerSIMD] = {{0}};
        for (int block = 0; block < kBlocks; ++block) {
            uint packed_words0[kRowsPerSIMD];
            uint packed_words1[kRowsPerSIMD];
            float scales[kRowsPerSIMD];
            float biases[kRowsPerSIMD];
            const device uint* weight_words = tile_words
                + simd_group * kWordsPerSIMD
                + thread_index_in_simdgroup * kWordsPerLane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(
                    tile_words + kWeightWordsPerTile)
                + simd_group * kRowsPerSIMD * kMetadataBytesPerRow;

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uint* row_weight_words =
                    weight_words + row * kWordsPerRow;
                const device uchar* row_metadata =
                    metadata_bytes + row * kMetadataBytesPerRow;
        \(metadataExtraction)
                const uint pair = lut[metadata_index];
                packed_words0[row] = row_weight_words[0];
                packed_words1[row] = row_weight_words[1];
                scales[row] =
                    gemma4_exact_three_vector_output_down_pair_scale(pair);
                biases[row] =
                    gemma4_exact_three_vector_output_down_pair_bias(pair);
            }

            {
                float values[16];
                const float input_sum =
                    gemma4_exact_three_vector_output_down_load_values(
                        input0, values);
                #pragma clang loop unroll(full)
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    gemma4_exact_three_vector_output_down_qdot_4bit(
                        packed_words0[row],
                        packed_words1[row],
                        values,
                        scales[row],
                        biases[row],
                        input_sum,
                        results[0][row]);
                }
            }
            {
                float values[16];
                const float input_sum =
                    gemma4_exact_three_vector_output_down_load_values(
                        input1, values);
                #pragma clang loop unroll(full)
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    gemma4_exact_three_vector_output_down_qdot_4bit(
                        packed_words0[row],
                        packed_words1[row],
                        values,
                        scales[row],
                        biases[row],
                        input_sum,
                        results[1][row]);
                }
            }
            {
                float values[16];
                const float input_sum =
                    gemma4_exact_three_vector_output_down_load_values(
                        input2, values);
                #pragma clang loop unroll(full)
                for (int row = 0; row < kRowsPerSIMD; ++row) {
                    gemma4_exact_three_vector_output_down_qdot_4bit(
                        packed_words0[row],
                        packed_words1[row],
                        values,
                        scales[row],
                        biases[row],
                        input_sum,
                        results[2][row]);
                }
            }

            tile_words += kPayloadWords;
            input0 += kBlockSize;
            input1 += kBlockSize;
            input2 += kBlockSize;
        }

        #pragma clang loop unroll(full)
        for (int vector = 0; vector < kVectorCount; ++vector) {
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                results[vector][row] = simd_sum(results[vector][row]);
                if (thread_index_in_simdgroup == 0) {
                    output[vector * kOutputRows + output_row + row] =
                        static_cast<bfloat>(results[vector][row]);
                }
            }
        }
        """
}

private let gemma4ExactThreeVectorOutputDownHeader = """
    using namespace metal;

    inline float gemma4_exact_three_vector_output_down_pair_scale(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_exact_three_vector_output_down_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_exact_three_vector_output_down_load_values(
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

    inline void gemma4_exact_three_vector_output_down_qdot_4bit(
        uint packed_word0,
        uint packed_word1,
        const thread float* values,
        float scale,
        float bias,
        float input_sum,
        thread float& result
    ) {
        const ushort packed0 = static_cast<ushort>(packed_word0);
        const ushort packed1 = static_cast<ushort>(packed_word0 >> 16);
        const ushort packed2 = static_cast<ushort>(packed_word1);
        const ushort packed3 = static_cast<ushort>(packed_word1 >> 16);
        float accumulator = 0;
        accumulator +=
            (values[0] * (packed0 & 0x000f)
            + values[1] * (packed0 & 0x00f0)
            + values[2] * (packed0 & 0x0f00)
            + values[3] * (packed0 & 0xf000));
        accumulator +=
            (values[4] * (packed1 & 0x000f)
            + values[5] * (packed1 & 0x00f0)
            + values[6] * (packed1 & 0x0f00)
            + values[7] * (packed1 & 0xf000));
        accumulator +=
            (values[8] * (packed2 & 0x000f)
            + values[9] * (packed2 & 0x00f0)
            + values[10] * (packed2 & 0x0f00)
            + values[11] * (packed2 & 0xf000));
        accumulator +=
            (values[12] * (packed3 & 0x000f)
            + values[13] * (packed3 & 0x00f0)
            + values[14] * (packed3 & 0x0f00)
            + values[15] * (packed3 & 0xf000));
        result += scale * accumulator + input_sum * bias;
    }
    """

private let gemma4ExactThreeVectorSlidingFixed12Output = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed12_indexed_output_qmv_8192_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorCoTiledOutputDownSource(
        inputWidth: 8_192,
        indexBits: 12
    ),
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

private let gemma4ExactThreeVectorSlidingFixed13Output = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed13_indexed_output_qmv_8192_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorCoTiledOutputDownSource(
        inputWidth: 8_192,
        indexBits: 13
    ),
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

private let gemma4ExactThreeVectorFullFixed12Output = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed12_indexed_output_qmv_16384_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorCoTiledOutputDownSource(
        inputWidth: 16_384,
        indexBits: 12
    ),
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

private let gemma4ExactThreeVectorFullFixed13Output = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed13_indexed_output_qmv_16384_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorCoTiledOutputDownSource(
        inputWidth: 16_384,
        indexBits: 13
    ),
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

func gemma4ExactThreeVectorIndexedOutput(
    payload: MLXArray,
    lut: MLXArray,
    input: MLXArray,
    inputWidth: Int,
    indexBits: Int
) -> MLXArray {
    let blockCount = inputWidth / 512
    precondition(payload.dtype == .uint32)
    precondition(payload.shape == [672, blockCount, 512 + 2 * indexBits])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [3, inputWidth])
    let kernel: MLXFast.MLXFastKernel
    switch (inputWidth, indexBits) {
    case (8_192, 12): kernel = gemma4ExactThreeVectorSlidingFixed12Output
    case (8_192, 13): kernel = gemma4ExactThreeVectorSlidingFixed13Output
    case (16_384, 12): kernel = gemma4ExactThreeVectorFullFixed12Output
    case (16_384, 13): kernel = gemma4ExactThreeVectorFullFixed13Output
    default: preconditionFailure("unsupported exact three-vector output layout")
    }
    return kernel(
        [payload, lut, input],
        grid: (32, 1_344, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[3, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let gemma4ExactThreeVectorDownKernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed12_indexed_down_qmv_21504_v1",
    inputNames: ["cotiled_payload", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorCoTiledOutputDownSource(
        inputWidth: 21_504,
        indexBits: 12
    ),
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

func gemma4ExactThreeVectorIndexedDown(
    payload: MLXArray,
    lut: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [672, 42, 536])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition(input.dtype == .bfloat16 && input.shape == [3, 21_504])
    return gemma4ExactThreeVectorDownKernel(
        [payload, lut, input],
        grid: (32, 1_344, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[3, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}

private let gemma4ExactThreeVectorU16DownSource = """
    constexpr int kInputWidth = 21504;
    constexpr int kOutputRows = 5376;
    constexpr int kVectorCount = 3;
    constexpr int kGroupsPerRow = 336;
    constexpr int kWeightBytesPerRow = 10752;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kBlockSize = 512;
    constexpr int kBlocks = 42;

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
    const device bfloat* input0 =
        x + thread_index_in_simdgroup * 16;
    const device bfloat* input1 =
        x + kInputWidth + thread_index_in_simdgroup * 16;
    const device bfloat* input2 =
        x + 2 * kInputWidth + thread_index_in_simdgroup * 16;

    float results[kVectorCount][kRowsPerSIMD] = {{0}};
    for (int block = 0; block < kBlocks; ++block) {
        uint packed_words0[kRowsPerSIMD];
        uint packed_words1[kRowsPerSIMD];
        float scales[kRowsPerSIMD];
        float biases[kRowsPerSIMD];
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uint* row_weight_words =
                reinterpret_cast<const device uint*>(
                    weight_bytes + row * kWeightBytesPerRow);
            const ushort metadata_index =
                row_indices[row * kGroupsPerRow];
            const uint pair = lut[metadata_index];
            packed_words0[row] = row_weight_words[0];
            packed_words1[row] = row_weight_words[1];
            scales[row] =
                gemma4_exact_three_vector_output_down_pair_scale(pair);
            biases[row] =
                gemma4_exact_three_vector_output_down_pair_bias(pair);
        }

        {
            float values[16];
            const float input_sum =
                gemma4_exact_three_vector_output_down_load_values(
                    input0, values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                gemma4_exact_three_vector_output_down_qdot_4bit(
                    packed_words0[row], packed_words1[row], values,
                    scales[row], biases[row], input_sum, results[0][row]);
            }
        }
        {
            float values[16];
            const float input_sum =
                gemma4_exact_three_vector_output_down_load_values(
                    input1, values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                gemma4_exact_three_vector_output_down_qdot_4bit(
                    packed_words0[row], packed_words1[row], values,
                    scales[row], biases[row], input_sum, results[1][row]);
            }
        }
        {
            float values[16];
            const float input_sum =
                gemma4_exact_three_vector_output_down_load_values(
                    input2, values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                gemma4_exact_three_vector_output_down_qdot_4bit(
                    packed_words0[row], packed_words1[row], values,
                    scales[row], biases[row], input_sum, results[2][row]);
            }
        }

        weight_bytes += 256;
        row_indices += 8;
        input0 += kBlockSize;
        input1 += kBlockSize;
        input2 += kBlockSize;
    }

    #pragma clang loop unroll(full)
    for (int vector = 0; vector < kVectorCount; ++vector) {
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            results[vector][row] = simd_sum(results[vector][row]);
            if (thread_index_in_simdgroup == 0) {
                output[vector * kOutputRows + output_row + row] =
                    static_cast<bfloat>(results[vector][row]);
            }
        }
    }
    """

private let gemma4ExactThreeVectorU16DownKernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_u16_indexed_down_qmv_21504_v1",
    inputNames: ["weight", "indices", "lut", "x"],
    outputNames: ["output"],
    source: gemma4ExactThreeVectorU16DownSource,
    header: gemma4ExactThreeVectorOutputDownHeader,
    ensureRowContiguous: true
)

func gemma4ExactThreeVectorU16IndexedDown(
    weight: MLXArray,
    indices: MLXArray,
    lut: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(weight.dtype == .uint32 && weight.shape == [5_376, 2_688])
    precondition(indices.dtype == .uint16 && indices.shape == [5_376, 336])
    precondition(lut.dtype == .uint32 && lut.ndim == 1)
    precondition((4_097...65_536).contains(lut.size))
    precondition(input.dtype == .bfloat16 && input.shape == [3, 21_504])
    return gemma4ExactThreeVectorU16DownKernel(
        [weight, indices, lut, input],
        grid: (32, 1_344, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[3, 5_376]],
        outputDTypes: [.bfloat16]
    )[0]
}
