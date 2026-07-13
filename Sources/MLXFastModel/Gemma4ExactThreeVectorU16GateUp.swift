import MLX

// Research-only exact-three path for the direct UInt16 indexed representation.
// Keeping it separate from the production exact-two dispatch makes the kernel
// straightforward to differential-test and benchmark before integration.
private let gemma4ExactThreeVectorU16GateUpSource = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kGroupsPerRow = 84;
    constexpr int kWeightBytesPerRow = 2688;
    constexpr int kRowsPerSIMD = 4;

    const bool is_up = simdgroup_index_in_threadgroup == 1;
    const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
    const uint lane = thread_index_in_simdgroup;
    const device uint* weight = is_up ? up_weight : gate_weight;
    const device ushort* indices = is_up ? up_indices : gate_indices;

    const device uchar* weight_bytes =
        reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow
        + lane * 4;
    const device ushort* row_indices =
        indices + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;

    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        float values0[8];
        float values1[8];
        float values2[8];
        const float input_sum0 =
            gemma4_exact_three_vector_gate_up_load_values(input0, values0);
        const float input_sum1 =
            gemma4_exact_three_vector_gate_up_load_values(input1, values1);
        const float input_sum2 =
            gemma4_exact_three_vector_gate_up_load_values(input2, values2);

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const ushort metadata_index =
                row_indices[row * kGroupsPerRow];
            const uint pair = is_up
                ? up_lut[metadata_index]
                : gate_lut[metadata_index];
            const uint packed_word =
                *reinterpret_cast<const device uint*>(row_weight);
            gemma4_exact_three_vector_gate_up_qdot_4bit(
                packed_word,
                values0,
                values1,
                values2,
                gemma4_exact_three_vector_gate_up_pair_scale(pair),
                gemma4_exact_three_vector_gate_up_pair_bias(pair),
                input_sum0,
                input_sum1,
                input_sum2,
                result0[row],
                result1[row],
                result2[row]);
        }

        weight_bytes += 128;
        row_indices += 4;
        input0 += 256;
        input1 += 256;
        input2 += 256;
    }

    threadgroup bfloat projections[2][3][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][0][row] =
                static_cast<bfloat>(result0[row]);
        }
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result1[row] = simd_sum(result1[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][1][row] =
                static_cast<bfloat>(result1[row]);
        }
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result2[row] = simd_sum(result2[row]);
        if (thread_index_in_simdgroup == 0) {
            projections[is_up ? 1 : 0][2][row] =
                static_cast<bfloat>(result2[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0
        && thread_index_in_simdgroup < kRowsPerSIMD
    ) {
        const int row = thread_index_in_simdgroup;
        {
            const bfloat gate = projections[0][0][row];
            const bfloat up = projections[1][0][row];
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
            activated[output_row + row] = gelu * up;
        }
        {
            const bfloat gate = projections[0][1][row];
            const bfloat up = projections[1][1][row];
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
            activated[kOutputRows + output_row + row] = gelu * up;
        }
        {
            const bfloat gate = projections[0][2][row];
            const bfloat up = projections[1][2][row];
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
            activated[2 * kOutputRows + output_row + row] = gelu * up;
        }
    }
    """

private let gemma4ExactThreeVectorU16GateUpHeader = """
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
        const thread float* values0,
        const thread float* values1,
        const thread float* values2,
        float scale,
        float bias,
        float input_sum0,
        float input_sum1,
        float input_sum2,
        thread float& result0,
        thread float& result1,
        thread float& result2
    ) {
        float accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values0[4 * index] * (packed & 0x000f)
                + values0[4 * index + 1] * (packed & 0x00f0)
                + values0[4 * index + 2] * (packed & 0x0f00)
                + values0[4 * index + 3] * (packed & 0xf000));
        }
        result0 += scale * accumulator + input_sum0 * bias;

        accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values1[4 * index] * (packed & 0x000f)
                + values1[4 * index + 1] * (packed & 0x00f0)
                + values1[4 * index + 2] * (packed & 0x0f00)
                + values1[4 * index + 3] * (packed & 0xf000));
        }
        result1 += scale * accumulator + input_sum1 * bias;

        accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(
                packed_word >> (index * 16));
            accumulator +=
                (values2[4 * index] * (packed & 0x000f)
                + values2[4 * index + 1] * (packed & 0x00f0)
                + values2[4 * index + 2] * (packed & 0x0f00)
                + values2[4 * index + 3] * (packed & 0xf000));
        }
        result2 += scale * accumulator + input_sum2 * bias;
    }
    """

private let gemma4ExactThreeVectorU16GateUpKernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_u16_indexed_gate_up_activation_qmv_5376_v1",
    inputNames: [
        "gate_weight", "gate_indices", "gate_lut",
        "up_weight", "up_indices", "up_lut", "x",
    ],
    outputNames: ["activated"],
    source: gemma4ExactThreeVectorU16GateUpSource,
    header: gemma4ExactThreeVectorU16GateUpHeader,
    ensureRowContiguous: true
)

func gemma4ExactThreeVectorU16GateUpActivated(
    gateWeight: MLXArray,
    gateIndices: MLXArray,
    gateLUT: MLXArray,
    upWeight: MLXArray,
    upIndices: MLXArray,
    upLUT: MLXArray,
    input: MLXArray
) -> MLXArray {
    precondition(gateWeight.dtype == .uint32)
    precondition(gateWeight.shape == [21_504, 672])
    precondition(upWeight.dtype == .uint32 && upWeight.shape == gateWeight.shape)
    precondition(gateIndices.dtype == .uint16)
    precondition(gateIndices.shape == [21_504, 84])
    precondition(upIndices.dtype == .uint16 && upIndices.shape == gateIndices.shape)
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1)
    precondition((1...65_536).contains(gateLUT.size))
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1)
    precondition((1...65_536).contains(upLUT.size))
    precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
    return gemma4ExactThreeVectorU16GateUpKernel(
        [
            gateWeight, gateIndices, gateLUT,
            upWeight, upIndices, upLUT, input,
        ],
        grid: (32, 10_752, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[3, 21_504]],
        outputDTypes: [.bfloat16]
    )[0]
}

extension FusedGateUpProjection {
    var supportsExactThreeVectorU16: Bool {
        guard metadataMode == .indexed,
              let indexedGate,
              let indexedUp
        else {
            return false
        }
        return gate.weight.dtype == .uint32
            && gate.weight.shape == [21_504, 672]
            && up.weight.dtype == .uint32
            && up.weight.shape == gate.weight.shape
            && indexedGate.indices.dtype == .uint16
            && indexedGate.indices.shape == [21_504, 84]
            && indexedUp.indices.dtype == .uint16
            && indexedUp.indices.shape == indexedGate.indices.shape
            && indexedGate.lut.dtype == .uint32
            && indexedGate.lut.ndim == 1
            && (1...65_536).contains(indexedGate.lut.size)
            && indexedUp.lut.dtype == .uint32
            && indexedUp.lut.ndim == 1
            && (1...65_536).contains(indexedUp.lut.size)
    }

    func exactThreeVectorU16Activated(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
        guard supportsExactThreeVectorU16,
              let indexedGate,
              let indexedUp
        else {
            preconditionFailure(
                "exact-three UInt16 gate/up payload is unavailable"
            )
        }
        return gemma4ExactThreeVectorU16GateUpActivated(
            gateWeight: gate.weight,
            gateIndices: indexedGate.indices,
            gateLUT: indexedGate.lut,
            upWeight: up.weight,
            upIndices: indexedUp.indices,
            upLUT: indexedUp.lut,
            input: input
        )
    }
}
