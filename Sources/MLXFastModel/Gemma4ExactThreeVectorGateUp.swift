import MLX

// Qualification-only three-row counterpart of the promoted exact two-row
// primitive. It is intentionally not reachable from model execution.
private let gemma4ExactThreeVectorHeader = """
    using namespace metal;
    inline float gemma4_exact_three_vector_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }
    inline float gemma4_exact_three_vector_pair_bias(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }
    inline float gemma4_exact_three_vector_load_values(
        const device bfloat* input, thread float* values
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
    inline void gemma4_exact_three_vector_qdot_4bit(
        uint packed_word, const thread float* values, float scale, float bias,
        float input_sum, thread float& result
    ) {
        float accumulator = 0;
        for (int index = 0; index < 2; ++index) {
            const ushort packed = static_cast<ushort>(packed_word >> (index * 16));
            accumulator +=
                (values[4 * index] * (packed & 0x000f)
                + values[4 * index + 1] * (packed & 0x00f0)
                + values[4 * index + 2] * (packed & 0x0f00)
                + values[4 * index + 3] * (packed & 0xf000));
        }
        result += scale * accumulator + input_sum * bias;
    }
"""

private let gemma4ExactThreeVectorU16Source = """
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
    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight)
        + output_row * kWeightBytesPerRow + lane * 4;
    const device ushort* row_indices = indices + output_row * kGroupsPerRow + lane / 8;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    float result0[kRowsPerSIMD] = {0};
    float result1[kRowsPerSIMD] = {0};
    float result2[kRowsPerSIMD] = {0};
    for (int block = 0; block < 21; ++block) {
        float values0[8], values1[8], values2[8];
        const float input_sum0 = gemma4_exact_three_vector_load_values(input0, values0);
        const float input_sum1 = gemma4_exact_three_vector_load_values(input1, values1);
        const float input_sum2 = gemma4_exact_three_vector_load_values(input2, values2);
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight = weight_bytes + row * kWeightBytesPerRow;
            const ushort metadata_index = row_indices[row * kGroupsPerRow];
            const uint pair = is_up ? up_lut[metadata_index] : gate_lut[metadata_index];
            const uint packed_word = *reinterpret_cast<const device uint*>(row_weight);
            const float scale = gemma4_exact_three_vector_pair_scale(pair);
            const float bias = gemma4_exact_three_vector_pair_bias(pair);
            gemma4_exact_three_vector_qdot_4bit(packed_word, values0, scale, bias, input_sum0, result0[row]);
            gemma4_exact_three_vector_qdot_4bit(packed_word, values1, scale, bias, input_sum1, result1[row]);
            gemma4_exact_three_vector_qdot_4bit(packed_word, values2, scale, bias, input_sum2, result2[row]);
        }
        weight_bytes += 128; row_indices += 4;
        input0 += 256; input1 += 256; input2 += 256;
    }
    threadgroup bfloat projections[2][3][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result0[row] = simd_sum(result0[row]);
        if (lane == 0) projections[is_up ? 1 : 0][0][row] = static_cast<bfloat>(result0[row]);
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result1[row] = simd_sum(result1[row]);
        if (lane == 0) projections[is_up ? 1 : 0][1][row] = static_cast<bfloat>(result1[row]);
    }
    #pragma clang loop unroll(full)
    for (int row = 0; row < kRowsPerSIMD; ++row) {
        result2[row] = simd_sum(result2[row]);
        if (lane == 0) projections[is_up ? 1 : 0][2][row] = static_cast<bfloat>(result2[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0 && lane < kRowsPerSIMD) {
        const int row = lane;
        #pragma clang loop unroll(full)
        for (int vector = 0; vector < 3; ++vector) {
            const bfloat gate = projections[0][vector][row];
            const bfloat up = projections[1][vector][row];
            const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
            const bfloat cubic1 = cubic0 * gate;
            const bfloat cubic2 = cubic1 * gate;
            const bfloat inner0 = gate + cubic2;
            const bfloat inner1 = static_cast<bfloat>(0.7978845834732056f) * inner0;
            const bfloat tanh_value = static_cast<bfloat>(metal::precise::tanh(inner1));
            const bfloat shifted = static_cast<bfloat>(1.0f) + tanh_value;
            const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
            const bfloat gelu = scaled * shifted;
            activated[vector * kOutputRows + output_row + row] = gelu * up;
        }
    }
"""

private let gemma4ExactThreeVectorFixed12Source = """
    constexpr int kInputWidth = 5376;
    constexpr int kOutputRows = 21504;
    constexpr int kRowsPerSIMD = 4;
    constexpr int kSIMDSize = 32;
    constexpr int kWordsPerProjectionBlock = kRowsPerSIMD * kSIMDSize;
    constexpr int kWeightWordsPerPair = 4 * kWordsPerProjectionBlock;
    constexpr int kMetadataBytesPerProjectionPair = 48;
    constexpr int kWordsPerPair = 536;
    constexpr int kPairCount = 10;
    constexpr int kTailWeightWords = 2 * kWordsPerProjectionBlock;
    constexpr int kTailMetadataBytesPerProjection = 32;
    constexpr int kWordsPerThreadgroup = kPairCount * kWordsPerPair + 272;
    const bool is_up = simdgroup_index_in_threadgroup == 1;
    const int threadgroup_row = threadgroup_position_in_grid.y;
    const int output_row = threadgroup_row * kRowsPerSIMD;
    const uint lane = thread_index_in_simdgroup;
    const uint lane_group = lane >> 3;
    const device uint* tile_words = cotiled_payload + threadgroup_row * kWordsPerThreadgroup;
    const device bfloat* input0 = x + lane * 8;
    const device bfloat* input1 = x + kInputWidth + lane * 8;
    const device bfloat* input2 = x + 2 * kInputWidth + lane * 8;
    const device uint* lut = is_up ? up_lut : gate_lut;
    float result0[kRowsPerSIMD] = {0}, result1[kRowsPerSIMD] = {0}, result2[kRowsPerSIMD] = {0};
    for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
        uint odd_pairs[kRowsPerSIMD];
        const device uint* even_words = tile_words + (is_up ? kWordsPerProjectionBlock : 0) + lane;
        const device uint* odd_words = tile_words + 2 * kWordsPerProjectionBlock + (is_up ? kWordsPerProjectionBlock : 0) + lane;
        const device uchar* metadata = reinterpret_cast<const device uchar*>(tile_words + kWeightWordsPerPair)
            + (is_up ? kMetadataBytesPerProjectionPair : 0);
        float v0[8], v1[8], v2[8];
        float s0 = gemma4_exact_three_vector_load_values(input0, v0);
        float s1 = gemma4_exact_three_vector_load_values(input1, v1);
        float s2 = gemma4_exact_three_vector_load_values(input2, v2);
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* m = metadata + row * 12;
            uint middle = m[4 + lane_group];
            uint even_index = m[lane_group] | ((middle & 15) << 8);
            uint odd_index = (middle >> 4) | (m[8 + lane_group] << 4);
            uint pair = lut[even_index]; odd_pairs[row] = lut[odd_index];
            uint word = even_words[row * kSIMDSize];
            float scale = gemma4_exact_three_vector_pair_scale(pair), bias = gemma4_exact_three_vector_pair_bias(pair);
            gemma4_exact_three_vector_qdot_4bit(word,v0,scale,bias,s0,result0[row]);
            gemma4_exact_three_vector_qdot_4bit(word,v1,scale,bias,s1,result1[row]);
            gemma4_exact_three_vector_qdot_4bit(word,v2,scale,bias,s2,result2[row]);
        }
        input0 += 256; input1 += 256; input2 += 256;
        float o0[8], o1[8], o2[8];
        s0 = gemma4_exact_three_vector_load_values(input0,o0);
        s1 = gemma4_exact_three_vector_load_values(input1,o1);
        s2 = gemma4_exact_three_vector_load_values(input2,o2);
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            uint pair=odd_pairs[row], word=odd_words[row*kSIMDSize];
            float scale=gemma4_exact_three_vector_pair_scale(pair), bias=gemma4_exact_three_vector_pair_bias(pair);
            gemma4_exact_three_vector_qdot_4bit(word,o0,scale,bias,s0,result0[row]);
            gemma4_exact_three_vector_qdot_4bit(word,o1,scale,bias,s1,result1[row]);
            gemma4_exact_three_vector_qdot_4bit(word,o2,scale,bias,s2,result2[row]);
        }
        input0 += 256; input1 += 256; input2 += 256; tile_words += kWordsPerPair;
    }
    float v0[8],v1[8],v2[8];
    float s0=gemma4_exact_three_vector_load_values(input0,v0), s1=gemma4_exact_three_vector_load_values(input1,v1), s2=gemma4_exact_three_vector_load_values(input2,v2);
    const device uint* words=tile_words+(is_up?kWordsPerProjectionBlock:0)+lane;
    const device uchar* metadata=reinterpret_cast<const device uchar*>(tile_words+kTailWeightWords)+(is_up?kTailMetadataBytesPerProjection:0);
    #pragma clang loop unroll(full)
    for(int row=0;row<kRowsPerSIMD;++row){const device uchar*m=metadata+row*8;uint off=lane_group<<1;uint pair=lut[m[off]|(m[off+1]<<8)],word=words[row*kSIMDSize];float scale=gemma4_exact_three_vector_pair_scale(pair),bias=gemma4_exact_three_vector_pair_bias(pair);gemma4_exact_three_vector_qdot_4bit(word,v0,scale,bias,s0,result0[row]);gemma4_exact_three_vector_qdot_4bit(word,v1,scale,bias,s1,result1[row]);gemma4_exact_three_vector_qdot_4bit(word,v2,scale,bias,s2,result2[row]);}
    threadgroup bfloat projections[2][3][kRowsPerSIMD];
    #pragma clang loop unroll(full)
    for(int row=0;row<kRowsPerSIMD;++row){result0[row]=simd_sum(result0[row]);result1[row]=simd_sum(result1[row]);result2[row]=simd_sum(result2[row]);if(lane==0){projections[is_up?1:0][0][row]=static_cast<bfloat>(result0[row]);projections[is_up?1:0][1][row]=static_cast<bfloat>(result1[row]);projections[is_up?1:0][2][row]=static_cast<bfloat>(result2[row]);}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simdgroup_index_in_threadgroup==0&&lane<kRowsPerSIMD){int row=lane;
    #pragma clang loop unroll(full)
    for(int vector=0;vector<3;++vector){const bfloat gate=projections[0][vector][row];const bfloat up=projections[1][vector][row];const bfloat cubic0=static_cast<bfloat>(0.044715f)*gate;const bfloat cubic1=cubic0*gate;const bfloat cubic2=cubic1*gate;const bfloat inner0=gate+cubic2;const bfloat inner1=static_cast<bfloat>(0.7978845834732056f)*inner0;const bfloat tanh_value=static_cast<bfloat>(metal::precise::tanh(inner1));const bfloat shifted=static_cast<bfloat>(1.0f)+tanh_value;const bfloat scaled=static_cast<bfloat>(0.5f)*gate;const bfloat gelu=scaled*shifted;activated[vector*kOutputRows+output_row+row]=gelu*up;}}
"""

private let gemma4ExactThreeVectorKernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_cotiled_fixed12_gate_up_activation_qmv_5376_v1",
    inputNames: ["cotiled_payload", "gate_lut", "up_lut", "x"], outputNames: ["activated"],
    source: gemma4ExactThreeVectorFixed12Source, header: gemma4ExactThreeVectorHeader, ensureRowContiguous: true)
private let gemma4ExactThreeVectorU16Kernel = MLXFast.metalKernel(
    name: "gemma4_exact_three_vector_u16_indexed_gate_up_activation_qmv_5376_v1",
    inputNames: ["gate_weight","gate_indices","gate_lut","up_weight","up_indices","up_lut","x"], outputNames: ["activated"],
    source: gemma4ExactThreeVectorU16Source, header: gemma4ExactThreeVectorHeader, ensureRowContiguous: true)

func gemma4ExactThreeVectorGateUpActivated(payload: MLXArray, gateLUT: MLXArray, upLUT: MLXArray, input: MLXArray) -> MLXArray {
    precondition(payload.dtype == .uint32 && payload.shape == [5_376, 5_632])
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1 && (1...4_096).contains(gateLUT.size))
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1 && (1...4_096).contains(upLUT.size))
    precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
    return gemma4ExactThreeVectorKernel([payload,gateLUT,upLUT,input], grid:(32,10_752,1), threadGroup:(32,2,1), outputShapes:[[3,21_504]], outputDTypes:[.bfloat16])[0]
}

func gemma4ExactThreeVectorU16GateUpActivated(gateWeight: MLXArray, gateIndices: MLXArray, gateLUT: MLXArray, upWeight: MLXArray, upIndices: MLXArray, upLUT: MLXArray, input: MLXArray) -> MLXArray {
    precondition(gateWeight.dtype == .uint32 && gateWeight.shape == [21_504,672])
    precondition(upWeight.dtype == .uint32 && upWeight.shape == gateWeight.shape)
    precondition(gateIndices.dtype == .uint16 && gateIndices.shape == [21_504,84])
    precondition(upIndices.dtype == .uint16 && upIndices.shape == gateIndices.shape)
    precondition(gateLUT.dtype == .uint32 && gateLUT.ndim == 1 && (1...65_536).contains(gateLUT.size))
    precondition(upLUT.dtype == .uint32 && upLUT.ndim == 1 && (1...65_536).contains(upLUT.size))
    precondition(input.dtype == .bfloat16 && input.shape == [3,5_376])
    return gemma4ExactThreeVectorU16Kernel([gateWeight,gateIndices,gateLUT,upWeight,upIndices,upLUT,input], grid:(32,10_752,1), threadGroup:(32,2,1), outputShapes:[[3,21_504]], outputDTypes:[.bfloat16])[0]
}
