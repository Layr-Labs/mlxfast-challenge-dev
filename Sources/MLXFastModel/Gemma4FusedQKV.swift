import Foundation
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

/// Sliding decode projection plus attention preparation, cooperatively mapped
/// as one threadgroup per logical Q/K/V head. All 32 SIMD groups first retain
/// the established indexed QMV accumulation order for eight rows apiece. The
/// first two SIMD groups then retain the established D=256 RMS reduction and
/// split-half RoPE order over the BF16-staged projection row.
private let gemma4IndexedSlidingQKVAttentionPreparation = MLXFast.metalKernel(
    name: "gemma4_indexed_sliding_qkv_attention_prep_head_coop_256_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut",
        "v_weight", "v_indices", "v_lut", "x",
        "q_norm_weight", "k_norm_weight", "position",
        "rope_cosines", "rope_sines",
    ],
    outputNames: ["queries", "combined_kv"],
    source: """
        constexpr uint kGroupsPerRow = 84;
        constexpr uint kWeightBytesPerRow = 2688;
        constexpr uint kRowsPerSIMD = 8;
        constexpr uint kHeadDim = 256;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 16;
        constexpr uint kKVSlabElements = kKVHeads * kHeadDim;
        constexpr uint kReads = 4;
        constexpr uint kSIMDSize = 32;

        const uint combined_head = threadgroup_position_in_grid.y;
        const bool is_q = combined_head < kQHeads;
        const bool is_k = !is_q && combined_head < kQHeads + kKVHeads;
        const uint projection_head = is_q
            ? combined_head
            : (is_k ? combined_head - kQHeads
                    : combined_head - kQHeads - kKVHeads);
        const uint head_output_row = projection_head * kHeadDim;
        const uint output_row = head_output_row
            + simdgroup_index_in_threadgroup * kRowsPerSIMD;

        const device uint* projection_weight = is_q
            ? q_weight
            : (is_k ? k_weight : v_weight);
        const device ushort* projection_indices = is_q
            ? q_indices
            : (is_k ? k_indices : v_indices);
        const device uint* projection_lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(projection_weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device ushort* row_indices =
            projection_indices + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (uint block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum =
                gemma4_sliding_qkv_prep_load_values(input, values);

            for (uint row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const ushort metadata_index =
                    row_indices[row * kGroupsPerRow];
                const uint pair = projection_lut[metadata_index];
                result[row] += gemma4_sliding_qkv_prep_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_sliding_qkv_prep_pair_scale(pair),
                    gemma4_sliding_qkv_prep_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            row_indices += 4;
            input += 256;
        }

        threadgroup bfloat raw[kHeadDim];
        for (uint row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                raw[simdgroup_index_in_threadgroup * kRowsPerSIMD + row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Match the current 64-thread D=256 fused-RMS topology exactly: four
        // BF16 inputs per thread, two SIMD partials, a zero-filled 32-lane
        // secondary reduction, and precise rsqrt.
        const uint preparation_thread =
            thread_position_in_threadgroup.y * kSIMDSize
            + thread_position_in_threadgroup.x;
        float accumulator = 0;
        if (preparation_thread < kHeadDim / kReads) {
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

        const bool has_weight = is_q || is_k;
        const device bfloat* norm_weight = is_q
            ? q_norm_weight
            : k_norm_weight;
        device bfloat* output = is_q
            ? queries + projection_head * kHeadDim
            : combined_kv
                + (is_k ? 0 : kKVSlabElements)
                + projection_head * kHeadDim;
        if (preparation_thread < kHeadDim / kReads) {
            const device bfloat* row_weight =
                norm_weight + preparation_thread * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = preparation_thread * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    raw[dimension] * inverse_mean[0]);
                const bfloat weighted = has_weight
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
                raw[dimension] = weighted;
                if (!has_weight) {
                    output[dimension] = weighted;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has_weight && preparation_thread < kHeadDim / kReads) {
            constexpr uint kPairs = kHeadDim / 2;
            constexpr uint kPreparationThreads = kHeadDim / kReads;
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

        inline float gemma4_sliding_qkv_prep_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_sliding_qkv_prep_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_sliding_qkv_prep_load_values(
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

        inline float gemma4_sliding_qkv_prep_qdot_4bit(
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

/// Conservative head-owned sliding fusion. Sixteen SIMDgroups retain the
/// stock four-row accumulator shape and cooperatively cover one 256-wide head
/// in four sequential 64-row tiles before the exact D=256 preparation phase.
private let gemma4IndexedSlidingQKVAttentionPreparationTiled512 = MLXFast.metalKernel(
    name: "gemma4_indexed_sliding_qkv_attention_prep_tiled512_256_v1",
    inputNames: [
        "q_weight", "q_indices", "q_lut",
        "k_weight", "k_indices", "k_lut",
        "v_weight", "v_indices", "v_lut", "x",
        "q_norm_weight", "k_norm_weight", "position",
        "rope_cosines", "rope_sines",
    ],
    outputNames: ["queries", "combined_kv"],
    source: """
        constexpr uint kGroupsPerRow = 84;
        constexpr uint kWeightBytesPerRow = 2688;
        constexpr uint kHeadDim = 256;
        constexpr uint kRowsPerSIMD = 4;
        constexpr uint kSIMDGroups = 16;
        constexpr uint kRowsPerTile = kSIMDGroups * kRowsPerSIMD;
        constexpr uint kTiles = kHeadDim / kRowsPerTile;
        constexpr uint kQHeads = 32;
        constexpr uint kKVHeads = 16;
        constexpr uint kKVSlabElements = kKVHeads * kHeadDim;
        constexpr uint kReads = 4;
        constexpr uint kSIMDSize = 32;

        const uint combined_head = threadgroup_position_in_grid.y;
        const bool is_q = combined_head < kQHeads;
        const bool is_k = !is_q && combined_head < kQHeads + kKVHeads;
        const uint projection_head = is_q
            ? combined_head
            : (is_k ? combined_head - kQHeads
                    : combined_head - kQHeads - kKVHeads);
        const uint head_output_row = projection_head * kHeadDim;

        const device uint* projection_weight = is_q
            ? q_weight
            : (is_k ? k_weight : v_weight);
        const device ushort* projection_indices = is_q
            ? q_indices
            : (is_k ? k_indices : v_indices);
        const device uint* projection_lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);

        threadgroup bfloat raw[kHeadDim];
        for (uint tile = 0; tile < kTiles; ++tile) {
            const uint tile_row = tile * kRowsPerTile;
            const uint output_row = head_output_row + tile_row
                + simdgroup_index_in_threadgroup * kRowsPerSIMD;
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
                    gemma4_sliding_qkv_prep_load_values(input, values);

                for (uint row = 0; row < kRowsPerSIMD; ++row) {
                    const device uchar* row_weight =
                        weight_bytes + row * kWeightBytesPerRow;
                    const ushort metadata_index =
                        row_indices[row * kGroupsPerRow];
                    const uint pair = projection_lut[metadata_index];
                    result[row] += gemma4_sliding_qkv_prep_qdot_4bit(
                        row_weight,
                        values,
                        gemma4_sliding_qkv_prep_pair_scale(pair),
                        gemma4_sliding_qkv_prep_pair_bias(pair),
                        input_sum);
                }

                weight_bytes += 128;
                row_indices += 4;
                input += 256;
            }

            for (uint row = 0; row < kRowsPerSIMD; ++row) {
                result[row] = simd_sum(result[row]);
                if (thread_index_in_simdgroup == 0) {
                    raw[
                        tile_row
                        + simdgroup_index_in_threadgroup * kRowsPerSIMD
                        + row
                    ] = static_cast<bfloat>(result[row]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Match the current 64-thread D=256 fused-RMS topology exactly: four
        // BF16 inputs per thread, two SIMD partials, a zero-filled 32-lane
        // secondary reduction, and precise rsqrt.
        const uint preparation_thread =
            thread_position_in_threadgroup.y * kSIMDSize
            + thread_position_in_threadgroup.x;
        float accumulator = 0;
        if (preparation_thread < kHeadDim / kReads) {
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

        const bool has_weight = is_q || is_k;
        const device bfloat* norm_weight = is_q
            ? q_norm_weight
            : k_norm_weight;
        device bfloat* output = is_q
            ? queries + projection_head * kHeadDim
            : combined_kv
                + (is_k ? 0 : kKVSlabElements)
                + projection_head * kHeadDim;
        if (preparation_thread < kHeadDim / kReads) {
            const device bfloat* row_weight =
                norm_weight + preparation_thread * kReads;
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = preparation_thread * kReads + index;
                const bfloat normalized = static_cast<bfloat>(
                    raw[dimension] * inverse_mean[0]);
                const bfloat weighted = has_weight
                    ? row_weight[index] * normalized
                    : static_cast<bfloat>(1.0f) * normalized;
                raw[dimension] = weighted;
                if (!has_weight) {
                    output[dimension] = weighted;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (has_weight && preparation_thread < kHeadDim / kReads) {
            constexpr uint kPairs = kHeadDim / 2;
            constexpr uint kPreparationThreads = kHeadDim / kReads;
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

        inline float gemma4_sliding_qkv_prep_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float gemma4_sliding_qkv_prep_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float gemma4_sliding_qkv_prep_load_values(
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

        inline float gemma4_sliding_qkv_prep_qdot_4bit(
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

/// Opt-in single-dispatch sliding decode projection and preparation. Keeping
/// the stock projection and preparation objects here makes the raw-BF16
/// verifier exercise the exact rollback path rather than a second reference
/// implementation.
enum FusedSlidingQKVAttentionPreparationTopology: String, Sendable {
    case tiled512
    case head1024
}

struct FusedSlidingQKVAttentionPreparation: @unchecked Sendable {
    let projection: FusedSlidingQKVProjection
    let preparation: FusedAttentionRMSPreparation
    let useCandidate: Bool
    let verifyBits: Bool
    let topology: FusedSlidingQKVAttentionPreparationTopology

    init?(
        projection: FusedSlidingQKVProjection?,
        preparation: FusedAttentionRMSPreparation?,
        useCandidate: Bool,
        verifyBits: Bool,
        topology: FusedSlidingQKVAttentionPreparationTopology = .tiled512
    ) {
        guard useCandidate || verifyBits,
              let projection,
              let preparation,
              preparation.isSliding,
              preparation.headDim == 256,
              preparation.kvHeads == 16
        else { return nil }
        self.projection = projection
        self.preparation = preparation
        self.useCandidate = useCandidate
        self.verifyBits = verifyBits
        self.topology = topology
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

        let kernel: MLXFast.MLXFastKernel
        let grid: (Int, Int, Int)
        let threadGroup: (Int, Int, Int)
        switch topology {
        case .tiled512:
            kernel = gemma4IndexedSlidingQKVAttentionPreparationTiled512
            grid = (32, 64 * 16, 1)
            threadGroup = (32, 16, 1)
        case .head1024:
            kernel = gemma4IndexedSlidingQKVAttentionPreparation
            grid = (32, 64 * 32, 1)
            threadGroup = (32, 32, 1)
        }

        let outputs = kernel(
            [
                projection.q.weight,
                projection.qMetadata.indices,
                projection.qMetadata.lut,
                projection.k.weight,
                projection.kMetadata.indices,
                projection.kMetadata.lut,
                projection.v.weight,
                projection.vMetadata.indices,
                projection.vMetadata.lut,
                input,
                preparation.qNormWeight,
                preparation.kNormWeight,
                preparation.positionViews[offset],
                preparation.ropeCosines,
                preparation.ropeSines,
            ],
            grid: grid,
            threadGroup: threadGroup,
            outputShapes: [
                [1, 32, 1, 256],
                [2, 1, 16, 1, 256],
            ],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        let candidate = (queries: outputs[0], combinedKV: outputs[1])

        if verifyBits {
            let raw = projection(input)
            let reference = preparation.callCombined(
                rawQueries: raw.0,
                rawKeys: raw.1,
                rawValues: raw.2,
                offset: offset
            )
            verifyRawBF16(
                candidate.queries,
                reference.queries,
                output: "queries"
            )
            verifyRawBF16(
                candidate.combinedKV,
                reference.combinedKV,
                output: "combined_kv"
            )
            FileHandle.standardError.write(Data(
                (
                    "verify_fused_sliding_qkv_rms_bits arrays=2 values="
                        + "\(candidate.queries.size + candidate.combinedKV.size) "
                        + "offset=\(offset) topology=\(topology.rawValue)\n"
                ).utf8
            ))
            return useCandidate ? candidate : reference
        }

        precondition(useCandidate)
        return candidate
    }

    private func verifyRawBF16(
        _ candidate: MLXArray,
        _ reference: MLXArray,
        output: String
    ) {
        precondition(candidate.dtype == .bfloat16)
        precondition(reference.dtype == .bfloat16)
        precondition(candidate.shape == reference.shape)

        let candidateBits = candidate.view(dtype: .uint16)
        let referenceBits = reference.view(dtype: .uint16)
        let matches = arrayEqual(candidateBits, referenceBits)
        eval(matches)
        guard matches.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let referenceValues = referenceBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, referenceValues)
                .enumerated()
                .first { $0.element.0 != $0.element.1 }
            preconditionFailure(
                "head-cooperative sliding QKV preparation raw BF16 mismatch "
                    + "output=\(output) index=\(mismatch?.offset ?? -1) "
                    + "candidate=\(mismatch?.element.0 ?? 0) "
                    + "reference=\(mismatch?.element.1 ?? 0)"
            )
        }
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
