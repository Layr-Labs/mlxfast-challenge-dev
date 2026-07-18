import Foundation
import MLX

/// Number of consecutive four-row gate/up output tiles consumed by one
/// serial-decode threadgroup. Factor one preserves the qualified sidecar
/// consumer; factors two and four stage its 6,048 FP32 elements once in
/// threadgroup memory and reuse them across the strip.
func gemma4DecodeOutputStripFactor() -> Int {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_DECODE_OUTPUT_STRIP"
    ], let factor = Int(raw), [1, 2, 4].contains(factor) else {
        return 2
    }
    return factor
}

private func makeGemma4CoTiledFixed12SidecarStripQMV(
    factor: Int
) -> MLXFast.MLXFastKernel {
    precondition([2, 4].contains(factor))
    return MLXFast.metalKernel(
        name: "gemma4_cotiled_fixed12_sidecar_strip\(factor)_qmv_5376_t121_v1",
        inputNames: [
            "cotiled_payload", "gate_lut", "up_lut", "activation_sidecar",
        ],
        outputNames: ["activated"],
        source: """
            constexpr int kStrip = \(factor);
            constexpr int kInputWidth = 5376;
            constexpr int kInputGroups = kInputWidth / 8;
            constexpr int kSidecarElements = kInputWidth + kInputGroups;
            constexpr int kRowsPerTile = 4;
            constexpr int kSIMDSize = 32;
            constexpr int kWordsPerTile = 5628;

            threadgroup uint staged_sidecar[kSidecarElements];
            threadgroup bfloat projections[2 * kStrip * kRowsPerTile];

            const uint lane = thread_index_in_simdgroup;
            const uint simdgroup = simdgroup_index_in_threadgroup;
            const uint local_thread = simdgroup * kSIMDSize + lane;
            for (uint index = local_thread;
                 index < kSidecarElements;
                 index += 2 * kSIMDSize
            ) {
                staged_sidecar[index] = as_type<uint>(
                    activation_sidecar[index]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool is_up = simdgroup == 1;
            const uint lane_group = lane >> 3;
            const device uint* lut = is_up ? up_lut : gate_lut;
            const int first_tile =
                int(threadgroup_position_in_grid.y) * kStrip;

            #pragma clang loop unroll(disable)
            for (int strip_tile = 0; strip_tile < kStrip; ++strip_tile) {
                const int tile = first_tile + strip_tile;
                const device uint* tile_words =
                    cotiled_payload + tile * kWordsPerTile;
                float result[kRowsPerTile] = {0};
                gemma4_strip_accumulate_tile(
                    tile_words,
                    lut,
                    staged_sidecar,
                    is_up,
                    lane,
                    lane_group,
                    result);

                #pragma clang loop unroll(full)
                for (int row = 0; row < kRowsPerTile; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (lane == 0) {
                        const int projection_index = is_up ? 1 : 0;
                        const int index =
                            (projection_index * kStrip + strip_tile)
                            * kRowsPerTile + row;
                        projections[index] =
                            static_cast<bfloat>(result[row]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int epilogue_rows = kStrip * kRowsPerTile;
            if (simdgroup == 0 && int(lane) < epilogue_rows) {
                const int strip_tile = int(lane) / kRowsPerTile;
                const int row = int(lane) % kRowsPerTile;
                const int projection_offset = kStrip * kRowsPerTile;
                const bfloat gate =
                    projections[strip_tile * kRowsPerTile + row];
                const bfloat up = projections[
                    projection_offset + strip_tile * kRowsPerTile + row];
                const int output_row =
                    (first_tile + strip_tile) * kRowsPerTile + row;
                activated[output_row] = gemma4_strip_activated_exact(gate, up);
            }
            """,
        header: """
            using namespace metal;

            constant int kStripRowsPerTile = 4;
            constant int kStripSIMDSize = 32;
            constant int kStripWordsPerProjectionBlock =
                kStripRowsPerTile * kStripSIMDSize;
            constant int kStripWeightWordsPerPair =
                4 * kStripWordsPerProjectionBlock;
            constant int kStripMetadataBytesPerProjectionPair = 48;
            constant int kStripWordsPerPair = 536;
            constant int kStripPairCount = 10;
            constant int kStripTailWeightWords =
                2 * kStripWordsPerProjectionBlock;
            constant int kStripTailMetadataBytesPerProjection = 24;

            inline float gemma4_strip_pair_scale(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_strip_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline void gemma4_strip_load_values(
                const threadgroup uint* input,
                thread float* values
            ) {
                #pragma clang loop unroll(full)
                for (int index = 0; index < 8; ++index) {
                    values[index] = as_type<float>(input[index]);
                }
            }

            inline float gemma4_strip_qdot_4bit(
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

            inline bfloat gemma4_strip_activated_exact(
                bfloat gate,
                bfloat up
            ) {
                const bfloat cubic0 = static_cast<bfloat>(0.044715f) * gate;
                const bfloat cubic1 = cubic0 * gate;
                const bfloat cubic2 = cubic1 * gate;
                const bfloat inner0 = gate + cubic2;
                const bfloat inner1 =
                    static_cast<bfloat>(0.7978845834732056f) * inner0;
                const bfloat tanh_value =
                    static_cast<bfloat>(metal::precise::tanh(inner1));
                const bfloat shifted =
                    static_cast<bfloat>(1.0f) + tanh_value;
                const bfloat scaled = static_cast<bfloat>(0.5f) * gate;
                const bfloat gelu = scaled * shifted;
                return gelu * up;
            }

            inline void gemma4_strip_accumulate_tile(
                const device uint* tile_base,
                const device uint* lut,
                const threadgroup uint* sidecar,
                bool is_up,
                uint lane,
                uint lane_group,
                thread float* result
            ) {
                const device uint* tile_words = tile_base;
                const threadgroup uint* input_values = sidecar + lane * 8;
                const threadgroup uint* input_sums = sidecar + 5376 + lane;

                for (int block_pair = 0;
                     block_pair < kStripPairCount;
                     ++block_pair
                ) {
                    float even_values[8];
                    gemma4_strip_load_values(input_values, even_values);
                    const float even_input_sum =
                        as_type<float>(input_sums[0]);
                    uint odd_pairs[kStripRowsPerTile];

                    const device uint* even_weight_words = tile_words
                        + (is_up ? kStripWordsPerProjectionBlock : 0)
                        + lane;
                    const device uint* odd_weight_words = tile_words
                        + 2 * kStripWordsPerProjectionBlock
                        + (is_up ? kStripWordsPerProjectionBlock : 0)
                        + lane;
                    const device uchar* metadata_bytes =
                        reinterpret_cast<const device uchar*>(
                            tile_words + kStripWeightWordsPerPair)
                        + (is_up
                            ? kStripMetadataBytesPerProjectionPair
                            : 0);

                    #pragma clang loop unroll(full)
                    for (int row = 0; row < kStripRowsPerTile; ++row) {
                        const device uchar* row_metadata =
                            metadata_bytes + row * 12;
                        const uint even_low = row_metadata[lane_group];
                        const uint middle = row_metadata[4 + lane_group];
                        const uint odd_high = row_metadata[8 + lane_group];
                        const uint even_index =
                            even_low | ((middle & 0x0f) << 8);
                        const uint odd_index =
                            (middle >> 4) | (odd_high << 4);
                        const uint even_pair = lut[even_index];
                        odd_pairs[row] = lut[odd_index];
                        const device uchar* row_weight =
                            reinterpret_cast<const device uchar*>(
                                even_weight_words
                                + row * kStripSIMDSize);
                        result[row] += gemma4_strip_qdot_4bit(
                            row_weight,
                            even_values,
                            gemma4_strip_pair_scale(even_pair),
                            gemma4_strip_pair_bias(even_pair),
                            even_input_sum);
                    }

                    input_values += 256;
                    input_sums += 32;
                    float odd_values[8];
                    gemma4_strip_load_values(input_values, odd_values);
                    const float odd_input_sum =
                        as_type<float>(input_sums[0]);
                    #pragma clang loop unroll(full)
                    for (int row = 0; row < kStripRowsPerTile; ++row) {
                        const uint odd_pair = odd_pairs[row];
                        const device uchar* row_weight =
                            reinterpret_cast<const device uchar*>(
                                odd_weight_words
                                + row * kStripSIMDSize);
                        result[row] += gemma4_strip_qdot_4bit(
                            row_weight,
                            odd_values,
                            gemma4_strip_pair_scale(odd_pair),
                            gemma4_strip_pair_bias(odd_pair),
                            odd_input_sum);
                    }

                    input_values += 256;
                    input_sums += 32;
                    tile_words += kStripWordsPerPair;
                }

                float tail_values[8];
                gemma4_strip_load_values(input_values, tail_values);
                const float tail_input_sum = as_type<float>(input_sums[0]);
                const device uint* tail_weight_words = tile_words
                    + (is_up ? kStripWordsPerProjectionBlock : 0)
                    + lane;
                const device uchar* tail_metadata =
                    reinterpret_cast<const device uchar*>(
                        tile_words + kStripTailWeightWords)
                    + (is_up ? kStripTailMetadataBytesPerProjection : 0);

                #pragma clang loop unroll(full)
                for (int row = 0; row < kStripRowsPerTile; ++row) {
                    const device uchar* row_metadata =
                        tail_metadata + row * 6;
                    const uint pair_base = (lane_group >> 1) * 3;
                    const uint low =
                        row_metadata[pair_base + (lane_group & 1)];
                    const uint middle = row_metadata[pair_base + 2];
                    const uint metadata_index = (lane_group & 1) == 0
                        ? low | ((middle & 0x0f) << 8)
                        : low | ((middle >> 4) << 8);
                    const uint pair = lut[metadata_index];
                    const device uchar* row_weight =
                        reinterpret_cast<const device uchar*>(
                            tail_weight_words + row * kStripSIMDSize);
                    result[row] += gemma4_strip_qdot_4bit(
                        row_weight,
                        tail_values,
                        gemma4_strip_pair_scale(pair),
                        gemma4_strip_pair_bias(pair),
                        tail_input_sum);
                }
            }
            """,
        ensureRowContiguous: true
    )
}

let gemma4CoTiledFixed12SidecarStrip2QMV =
    makeGemma4CoTiledFixed12SidecarStripQMV(factor: 2)

let gemma4CoTiledFixed12SidecarStrip4QMV =
    makeGemma4CoTiledFixed12SidecarStripQMV(factor: 4)
