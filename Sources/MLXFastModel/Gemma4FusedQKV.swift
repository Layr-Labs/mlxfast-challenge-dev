import MLX

private let gemma4IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_indexed_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut",
        "v_weight", "v_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q
            ? q_weight
            : (is_k ? k_weight : v_weight);
        const device ushort* indices = is_q
            ? q_indices
            : (is_k ? k_indices : v_indices);
        const device uint* lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q
            ? q_output
            : (is_k ? k_output : v_output);

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device ushort* row_indices =
            indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_qkv_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_qkv_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_qkv_pair_scale(pair),
                    gemma4_qkv_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            row_indices += 4;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_qkv_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_qkv_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_qkv_load_values(
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

        inline float gemma4_qkv_qdot_4bit(
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

private func supportsSlidingProjection(
    _ projection: FastQuantizedProjection,
    metadata: IndexedAffineMetadata,
    outputWidth: Int
) -> Bool {
    guard let biases = projection.biases else { return false }
    let metadataShape = [outputWidth, 84]
    return projection.groupSize == 64
        && projection.bits == 4
        && projection.weight.dtype == .uint32
        && projection.weight.shape == [outputWidth, 672]
        && projection.scales.dtype == .bfloat16
        && projection.scales.shape == metadataShape
        && biases.dtype == .bfloat16
        && biases.shape == metadataShape
        && metadata.indices.dtype == .uint16
        && metadata.indices.shape == metadataShape
        && metadata.lut.dtype == .uint32
        && metadata.lut.ndim == 1
        && (1...65_536).contains(metadata.lut.size)
}

func supportsGemma4FusedSlidingQKV(
    q: FastQuantizedProjection,
    k: FastQuantizedProjection,
    v: FastQuantizedProjection,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata,
    vMetadata: IndexedAffineMetadata
) -> Bool {
    supportsSlidingProjection(q, metadata: qMetadata, outputWidth: 8_192)
        && supportsSlidingProjection(k, metadata: kMetadata, outputWidth: 4_096)
        && supportsSlidingProjection(v, metadata: vMetadata, outputWidth: 4_096)
}

struct FusedSlidingQKVProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let v: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata
    let vMetadata: IndexedAffineMetadata

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        v: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata,
        vMetadata: IndexedAffineMetadata
    ) {
        precondition(supportsGemma4FusedSlidingQKV(
            q: q,
            k: k,
            v: v,
            qMetadata: qMetadata,
            kMetadata: kMetadata,
            vMetadata: vMetadata
        ))
        self.q = q
        self.k = k
        self.v = v
        self.qMetadata = qMetadata
        self.kMetadata = kMetadata
        self.vMetadata = vMetadata
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputs = gemma4IndexedSlidingQKV(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut,
                v.weight, vMetadata.indices, vMetadata.lut, input,
            ],
            grid: (32, 4_096, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [
                [1, 1, 8_192],
                [1, 1, 4_096],
                [1, 1, 4_096],
            ],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1], outputs[2])
    }
}

private let gemma4IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_indexed_full_qk_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: """
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q ? q_weight : k_weight;
        const device ushort* indices = is_q ? q_indices : k_indices;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device ushort* row_indices =
            indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_full_qk_load_values(input, values);

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index = row_indices[row * kGroupsPerRow];
                const uint pair = lut[metadata_index];
                result[row] += gemma4_full_qk_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_full_qk_pair_scale(pair),
                    gemma4_full_qk_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            row_indices += 4;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_full_qk_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_full_qk_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_full_qk_load_values(
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

        inline float gemma4_full_qk_qdot_4bit(
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

/// Full-attention decode projection plus attention preparation, cooperatively
/// mapped as one threadgroup per logical Q/K head. Each of 16 SIMD groups
/// computes four sequential eight-row tiles so the complete D=512 projection
/// row is available in threadgroup memory before the established RMS and
/// proportional-RoPE preparation topology runs.
private let gemma4IndexedFullQKAttentionPreparation = MLXFast.metalKernel(
    name: "gemma4_indexed_full_qk_attention_prep_head_coop_512_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut", "x",
        "q_norm_weight", "k_norm_weight", "position",
        "rope_cosines", "rope_sines",
    ],
    outputNames: ["queries", "combined_kv"],
    source: """
        constexpr uint kGroupsPerRow = 84;
        constexpr uint kWeightBytesPerRow = 2688;
        constexpr uint kRowsPerSIMD = 8;
        constexpr uint kRowsPerTile = 128;
        constexpr uint kTiles = 4;
        constexpr uint kHeadDim = 512;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 4;
        constexpr uint kKVSlabElements = kKVHeads * kHeadDim;
        constexpr uint kReads = 4;
        constexpr uint kPreparationThreads = kHeadDim / kReads;
        constexpr uint kSIMDSize = 32;

        const uint combined_head = threadgroup_position_in_grid.y;
        const bool is_q = combined_head < kQHeads;
        const uint projection_head = is_q
            ? combined_head
            : combined_head - kQHeads;
        const uint head_output_row = projection_head * kHeadDim;

        const device uint* projection_weight = is_q ? q_weight : k_weight;
        const device ushort* projection_indices = is_q
            ? q_indices
            : k_indices;
        const device uint* projection_lut = is_q ? q_lut : k_lut;

        threadgroup bfloat raw[kHeadDim];
        for (uint tile = 0; tile < kTiles; ++tile) {
            const uint tile_row = tile * kRowsPerTile
                + simdgroup_index_in_threadgroup * kRowsPerSIMD;
            const uint output_row = head_output_row + tile_row;
            const device uchar* weight_bytes =
                reinterpret_cast<const device uchar*>(projection_weight)
                + output_row * kWeightBytesPerRow
                + thread_index_in_simdgroup * 4;
            const device ushort* row_indices =
                projection_indices + output_row * kGroupsPerRow
                + thread_index_in_simdgroup / 8;
            const device bfloat* input =
                x + thread_index_in_simdgroup * 8;

            float result[kRowsPerSIMD] = {0};
            for (uint block = 0; block < 21; ++block) {
                float values[8];
                const float input_sum =
                    gemma4_full_qk_prep_load_values(input, values);

                for (uint row = 0; row < kRowsPerSIMD; ++row) {
                    const device uchar* row_weight =
                        weight_bytes + row * kWeightBytesPerRow;
                    const ushort metadata_index =
                        row_indices[row * kGroupsPerRow];
                    const uint pair = projection_lut[metadata_index];
                    result[row] += gemma4_full_qk_prep_qdot_4bit(
                        row_weight,
                        values,
                        gemma4_full_qk_prep_pair_scale(pair),
                        gemma4_full_qk_prep_pair_bias(pair),
                        input_sum);
                }

                weight_bytes += 128;
                row_indices += 4;
                input += 256;
            }

            for (uint row = 0; row < kRowsPerSIMD; ++row) {
                result[row] = simd_sum(result[row]);
                if (thread_index_in_simdgroup == 0) {
                    raw[tile_row + row] =
                        static_cast<bfloat>(result[row]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Match the current 128-thread D=512 fused-RMS topology exactly: four
        // BF16 inputs per active thread, four SIMD partials, a zero-filled
        // 32-lane secondary reduction, and precise rsqrt. The remaining SIMD
        // groups explicitly contribute zero while still crossing all barriers.
        const uint preparation_thread =
            thread_position_in_threadgroup.y * kSIMDSize
            + thread_position_in_threadgroup.x;
        float accumulator = 0;
        if (preparation_thread < kPreparationThreads) {
            const threadgroup bfloat* rms_input =
                raw + preparation_thread * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const float value = rms_input[index];
                accumulator += value * value;
            }
        }
        accumulator = simd_sum(accumulator);

        threadgroup float inverse_mean[1];
        threadgroup float local_sums[kSIMDSize];
        if (simdgroup_index_in_threadgroup == 0) {
            local_sums[thread_index_in_simdgroup] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index_in_simdgroup == 0) {
            local_sums[simdgroup_index_in_threadgroup] = accumulator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup_index_in_threadgroup == 0) {
            accumulator = simd_sum(local_sums[thread_index_in_simdgroup]);
            if (thread_index_in_simdgroup == 0) {
                inverse_mean[0] = metal::precise::rsqrt(
                    accumulator / kHeadDim + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const device bfloat* norm_weight = is_q
            ? q_norm_weight
            : k_norm_weight;
        device bfloat* output = is_q
            ? queries + projection_head * kHeadDim
            : combined_kv + projection_head * kHeadDim;
        if (preparation_thread < kPreparationThreads) {
            const device bfloat* row_weight =
                norm_weight + preparation_thread * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = preparation_thread * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    raw[dimension] * inverse_mean[0]);
                const bfloat weighted = row_weight[index] * normalized;
                raw[dimension] = weighted;
                if (!is_q) {
                    combined_kv[
                        kKVSlabElements
                        + projection_head * kHeadDim
                        + dimension
                    ] = static_cast<bfloat>(1.0f) * normalized;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (preparation_thread < kPreparationThreads) {
            constexpr uint kPairs = kHeadDim / 2;
            for (uint pair = preparation_thread;
                 pair < kPairs;
                 pair += kPreparationThreads) {
                const uint rope_index =
                    static_cast<uint>(position) * kPairs + pair;
                const float cosine = rope_cosines[rope_index];
                const float sine = rope_sines[rope_index];
                const float left = static_cast<float>(raw[pair]);
                const float right = static_cast<float>(raw[pair + kPairs]);
                output[pair] = static_cast<bfloat>(
                    left * cosine - right * sine);
                output[pair + kPairs] = static_cast<bfloat>(
                    left * sine + right * cosine);
            }
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_full_qk_prep_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_full_qk_prep_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_full_qk_prep_load_values(
            const device bfloat* input,
            thread float* values
        ) {
            float sum = 0;
            for (uint index = 0; index < 8; index += 4) {
                sum += input[index] + input[index + 1]
                    + input[index + 2] + input[index + 3];
                values[index] = input[index];
                values[index + 1] = input[index + 1] / 16.0f;
                values[index + 2] = input[index + 2] / 256.0f;
                values[index + 3] = input[index + 3] / 4096.0f;
            }
            return sum;
        }

        inline float gemma4_full_qk_prep_qdot_4bit(
            const device uchar* weight,
            const thread float* values,
            float scale,
            float bias,
            float input_sum
        ) {
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (uint index = 0; index < 2; ++index) {
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

func supportsGemma4FusedFullQK(
    q: FastQuantizedProjection,
    k: FastQuantizedProjection,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata
) -> Bool {
    supportsSlidingProjection(q, metadata: qMetadata, outputWidth: 16_384)
        && supportsSlidingProjection(k, metadata: kMetadata, outputWidth: 2_048)
}

struct FusedFullQKProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata
    ) {
        precondition(supportsGemma4FusedFullQK(
            q: q,
            k: k,
            qMetadata: qMetadata,
            kMetadata: kMetadata
        ))
        self.q = q
        self.k = k
        self.qMetadata = qMetadata
        self.kMetadata = kMetadata
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputs = gemma4IndexedFullQK(
            [
                q.weight, qMetadata.indices, qMetadata.lut,
                k.weight, kMetadata.indices, kMetadata.lut, input,
            ],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[1, 1, 16_384], [1, 1, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}

/// Opt-in single-dispatch full-attention decode projection and preparation.
/// The stock projection and preparation remain attached so bit verification
/// exercises the exact rollback graph rather than a second reference kernel.
struct FusedFullQKAttentionPreparation: @unchecked Sendable {
    let projection: FusedFullQKProjection
    let preparation: FusedAttentionRMSPreparation
    let verifyBits: Bool

    init?(
        projection: FusedFullQKProjection?,
        preparation: FusedAttentionRMSPreparation?,
        verifyBits: Bool
    ) {
        guard let projection,
              let preparation,
              !preparation.isSliding,
              preparation.headDim == 512,
              preparation.kvHeads == 4
        else { return nil }
        self.projection = projection
        self.preparation = preparation
        self.verifyBits = verifyBits
    }

    func supports(offset: Int) -> Bool {
        preparation.supports(offset: offset)
    }

    func callAsFunction(
        _ input: MLXArray,
        offset: Int
    ) -> (queries: MLXArray, combinedKV: MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        precondition(supports(offset: offset))

        let outputs = gemma4IndexedFullQKAttentionPreparation(
            [
                projection.q.weight,
                projection.qMetadata.indices,
                projection.qMetadata.lut,
                projection.k.weight,
                projection.kMetadata.indices,
                projection.kMetadata.lut,
                input,
                preparation.qNormWeight,
                preparation.kNormWeight,
                preparation.positionViews[offset],
                preparation.ropeCosines,
                preparation.ropeSines,
            ],
            grid: (32, 36 * 16, 1),
            threadGroup: (32, 16, 1),
            outputShapes: [
                [1, 32, 1, 512],
                [2, 1, 4, 1, 512],
            ],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        let candidate = (queries: outputs[0], combinedKV: outputs[1])

        if verifyBits {
            let raw = projection(input)
            let reference = preparation.callCombined(
                rawQueries: raw.0,
                rawKeys: raw.1,
                rawValues: nil,
                offset: offset
            )
            let queriesMatch = arrayEqual(
                candidate.queries.view(dtype: .uint16),
                reference.queries.view(dtype: .uint16)
            )
            let combinedKVMatch = arrayEqual(
                candidate.combinedKV.view(dtype: .uint16),
                reference.combinedKV.view(dtype: .uint16)
            )
            eval(queriesMatch, combinedKVMatch)
            precondition(
                queriesMatch.item(Bool.self)
                    && combinedKVMatch.item(Bool.self),
                "head-cooperative full QK preparation differs in raw BF16"
            )
        }

        return candidate
    }
}
