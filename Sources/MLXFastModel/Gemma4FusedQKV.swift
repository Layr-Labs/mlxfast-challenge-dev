import Foundation
import MLX

private func gemma4CoTiledEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4CoTiledSlidingQKVEnabled = gemma4CoTiledEnvironmentFlag(
    "DARKBLOOM_QKV_COTILED",
    default: true
)
private let gemma4VerifyCoTiledSlidingQKVBits = gemma4CoTiledEnvironmentFlag(
    "DARKBLOOM_VERIFY_QKV_COTILED_BITS",
    default: false
)
private let gemma4CoTiledFullQKEnabled = gemma4CoTiledEnvironmentFlag(
    "DARKBLOOM_FULL_QK_COTILED",
    default: true
)
private let gemma4VerifyCoTiledFullQKBits = gemma4CoTiledEnvironmentFlag(
    "DARKBLOOM_VERIFY_FULL_QK_COTILED_BITS",
    default: false
)

private let gemma4FusedSlidingQKVRouteEnabledForCoTileRetention =
    gemma4CoTiledEnvironmentFlag("MLXFAST_FUSED_QKV", default: true)
private let gemma4FusedFullQKRouteEnabledForCoTileRetention =
    gemma4CoTiledEnvironmentFlag("MLXFAST_FUSED_FULL_QK", default: true)

/// Resolved lazily after `Gemma4StartupMemoryPolicy.apply()`. Its existing
/// low-memory defaults turn both fused attention routes off, so the loader can
/// omit the 2.6 GiB alternate shard before creating any MLX allocations. An
/// explicit fused-route plus feature/verifier override retains and validates
/// the shard for focused testing on a memory-constrained host.
let gemma4RetainCoTiledAttentionPayloads =
    (gemma4FusedSlidingQKVRouteEnabledForCoTileRetention
        && (gemma4CoTiledSlidingQKVEnabled
            || gemma4VerifyCoTiledSlidingQKVBits))
    || (gemma4FusedFullQKRouteEnabledForCoTileRetention
        && (gemma4CoTiledFullQKEnabled
            || gemma4VerifyCoTiledFullQKBits))

struct Gemma4CoTiledAttentionPayload: @unchecked Sendable {
    let kind: Gemma4CoTiledAttentionPayloadKind
    let words: MLXArray
    let qBits: Int
    let kBits: Int
    let vBits: Int?
}

private struct Gemma4CoTiledSlidingQKVFormats: Hashable {
    let qBits: Int
    let kBits: Int
    let vBits: Int
}

private struct Gemma4CoTiledFullQKFormats: Hashable {
    let qBits: Int
    let kBits: Int
}

func supportsGemma4CoTiledSlidingQKVPayload(
    _ payload: Gemma4CoTiledAttentionPayload,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata,
    vMetadata: IndexedAffineMetadata
) -> Bool {
    guard payload.kind == .slidingQKV,
          let vBits = payload.vBits,
          let qMaximum = Gemma4CoTiledAttentionPayloadLayout.maximumLUTCount(
              indexBits: payload.qBits
          ),
          let kMaximum = Gemma4CoTiledAttentionPayloadLayout.maximumLUTCount(
              indexBits: payload.kBits
          ),
          let vMaximum = Gemma4CoTiledAttentionPayloadLayout.maximumLUTCount(
              indexBits: vBits
          ),
          let words = Gemma4CoTiledAttentionPayloadLayout.wordsPerThreadgroup(
              slotIndexBits: Gemma4CoTiledAttentionPayloadLayout.slidingSlotIndexBits(
                  qBits: payload.qBits,
                  kBits: payload.kBits,
                  vBits: vBits
              )
          )
    else {
        return false
    }
    return payload.words.dtype == .uint32
        && payload.words.shape == [1_024, words]
        && (1...qMaximum).contains(qMetadata.lut.size)
        && (1...kMaximum).contains(kMetadata.lut.size)
        && (1...vMaximum).contains(vMetadata.lut.size)
}

func supportsGemma4CoTiledFullQKPayload(
    _ payload: Gemma4CoTiledAttentionPayload,
    qMetadata: IndexedAffineMetadata,
    kMetadata: IndexedAffineMetadata
) -> Bool {
    guard payload.kind == .fullQK,
          payload.vBits == nil,
          let qMaximum = Gemma4CoTiledAttentionPayloadLayout.maximumLUTCount(
              indexBits: payload.qBits
          ),
          let kMaximum = Gemma4CoTiledAttentionPayloadLayout.maximumLUTCount(
              indexBits: payload.kBits
          ),
          let words = Gemma4CoTiledAttentionPayloadLayout.wordsPerThreadgroup(
              slotIndexBits: Gemma4CoTiledAttentionPayloadLayout.fullSlotIndexBits(
                  qBits: payload.qBits,
                  kBits: payload.kBits
              )
          )
    else {
        return false
    }
    return payload.words.dtype == .uint32
        && payload.words.shape == [512, words]
        && (1...qMaximum).contains(qMetadata.lut.size)
        && (1...kMaximum).contains(kMetadata.lut.size)
}

private func gemma4VerifyCoTiledRawBF16(
    _ candidate: MLXArray,
    reference: MLXArray,
    label: String
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
            "\(label) raw BF16 mismatch vs U16 indexed qmv at index "
                + "\(mismatch?.offset ?? -1): candidate="
                + "\(mismatch?.element.0 ?? 0), reference="
                + "\(mismatch?.element.1 ?? 0)"
        )
    }
}

private func gemma4CoTiledQKVHelperHeader(functionPrefix: String) -> String {
    """
    using namespace metal;

    inline float \(functionPrefix)_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float \(functionPrefix)_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float \(functionPrefix)_load_values(
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

    inline float \(functionPrefix)_qdot_4bit(
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
    """
}

/// The accumulation order exactly matches the U16 kernels: blocks 0...20,
/// each row in the same simdgroup, then the same `simd_sum` and BF16 cast.
private func gemma4CoTiledQKVDecodeLoops(functionPrefix: String) -> String {
    """
        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < kPairCount; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                \(functionPrefix)_load_values(input, even_values);
            uint odd_luts[kRowsPerSIMD];

            const device uint* even_weight_words = tile_words
                + projection * kSlotWeightWords + lane;
            const device uint* odd_weight_words = tile_words
                + kBlockWeightWords + projection * kSlotWeightWords + lane;
            const device uchar* metadata_bytes =
                reinterpret_cast<const device uchar*>(tile_words + kPairWeightWords)
                + slot_meta_offset;

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_tile = metadata_bytes + row * pair_stride;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                uint even_index = even_low | ((middle & 0x0f) << 8);
                uint odd_index = (middle >> 4) | (odd_high << 4);
                if (has_top_bits) {
                    const uint top = row_tile[12];
                    even_index |= ((top >> lane_group) & 1) << 12;
                    odd_index |= ((top >> (4 + lane_group)) & 1) << 12;
                }
                const uint even_pair = lut[even_index];
                odd_luts[row] = lut[odd_index];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        even_weight_words + row * kSIMDSize);
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    even_values,
                    \(functionPrefix)_pair_scale(even_pair),
                    \(functionPrefix)_pair_bias(even_pair),
                    even_input_sum);
            }

            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                \(functionPrefix)_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_luts[row];
                const device uchar* row_weight =
                    reinterpret_cast<const device uchar*>(
                        odd_weight_words + row * kSIMDSize);
                result[row] += \(functionPrefix)_qdot_4bit(
                    row_weight,
                    odd_values,
                    \(functionPrefix)_pair_scale(odd_pair),
                    \(functionPrefix)_pair_bias(odd_pair),
                    odd_input_sum);
            }

            input += 256;
            tile_words += kWordsPerPair;
        }

        float tail_values[8];
        const float tail_input_sum =
            \(functionPrefix)_load_values(input, tail_values);
        const device uint* tail_weight_words = tile_words
            + projection * kSlotWeightWords + lane;
        const device uchar* tail_metadata =
            reinterpret_cast<const device uchar*>(tile_words + kBlockWeightWords)
            + projection * 32;
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_tail = tail_metadata + row * 8;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            const device uchar* row_weight =
                reinterpret_cast<const device uchar*>(
                    tail_weight_words + row * kSIMDSize);
            result[row] += \(functionPrefix)_qdot_4bit(
                row_weight,
                tail_values,
                \(functionPrefix)_pair_scale(pair),
                \(functionPrefix)_pair_bias(pair),
                tail_input_sum);
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private func gemma4CoTiledSlidingQKVBody(
    qBits: Int,
    kBits: Int,
    vBits: Int
) -> String {
    """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDSize = 32;
        constexpr int kSlotWeightWords = kRowsPerSIMD * kSIMDSize;
        constexpr int kSlotsPerThreadgroup = 4;
        constexpr int kBlockWeightWords = kSlotsPerThreadgroup * kSlotWeightWords;
        constexpr int kPairWeightWords = 2 * kBlockWeightWords;
        constexpr int kPairCount = 10;
        constexpr int kQPairStride = \(qBits);
        constexpr int kKPairStride = \(kBits);
        constexpr int kVPairStride = \(vBits);
        constexpr bool kQHasTopBits = \(qBits == 13 ? "true" : "false");
        constexpr bool kKHasTopBits = \(kBits == 13 ? "true" : "false");
        constexpr bool kVHasTopBits = \(vBits == 13 ? "true" : "false");
        constexpr int kPairMetadataBytes =
            8 * kQPairStride + 4 * kKPairStride + 4 * kVPairStride;
        constexpr int kWordsPerPair = kPairWeightWords + kPairMetadataBytes / 4;
        constexpr int kWordsPerTail =
            kBlockWeightWords + 8 * kSlotsPerThreadgroup;
        constexpr int kWordsPerThreadgroup =
            kPairCount * kWordsPerPair + kWordsPerTail;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 8 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* lut = is_q ? q_lut : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q ? q_output : (is_k ? k_output : v_output);
        const int pair_stride = is_q
            ? kQPairStride : (is_k ? kKPairStride : kVPairStride);
        const bool has_top_bits = is_q
            ? kQHasTopBits : (is_k ? kKHasTopBits : kVHasTopBits);
        const int slot_meta_offset = is_q
            ? projection * 4 * kQPairStride
            : (is_k ? 8 * kQPairStride
                : 8 * kQPairStride + 4 * kKPairStride);
        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device uint* tile_words = cotiled_payload
            + threadgroup_position_in_grid.y * kWordsPerThreadgroup;
        const device bfloat* input = x + lane * 8;

        \(gemma4CoTiledQKVDecodeLoops(functionPrefix: "gemma4_cotiled_qkv"))
        """
}

private func gemma4CoTiledFullQKBody(qBits: Int, kBits: Int) -> String {
    """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kSIMDSize = 32;
        constexpr int kSlotWeightWords = kRowsPerSIMD * kSIMDSize;
        constexpr int kSlotsPerThreadgroup = 9;
        constexpr int kBlockWeightWords = kSlotsPerThreadgroup * kSlotWeightWords;
        constexpr int kPairWeightWords = 2 * kBlockWeightWords;
        constexpr int kPairCount = 10;
        constexpr int kQPairStride = \(qBits);
        constexpr int kKPairStride = \(kBits);
        constexpr bool kQHasTopBits = \(qBits == 13 ? "true" : "false");
        constexpr bool kKHasTopBits = \(kBits == 13 ? "true" : "false");
        constexpr int kPairMetadataBytes =
            32 * kQPairStride + 4 * kKPairStride;
        constexpr int kWordsPerPair = kPairWeightWords + kPairMetadataBytes / 4;
        constexpr int kWordsPerTail =
            kBlockWeightWords + 8 * kSlotsPerThreadgroup;
        constexpr int kWordsPerThreadgroup =
            kPairCount * kWordsPerPair + kWordsPerTail;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;
        const int pair_stride = is_q ? kQPairStride : kKPairStride;
        const bool has_top_bits = is_q ? kQHasTopBits : kKHasTopBits;
        const int slot_meta_offset = is_q
            ? projection * 4 * kQPairStride : 32 * kQPairStride;
        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device uint* tile_words = cotiled_payload
            + threadgroup_position_in_grid.y * kWordsPerThreadgroup;
        const device bfloat* input = x + lane * 8;

        \(gemma4CoTiledQKVDecodeLoops(functionPrefix: "gemma4_cotiled_full_qk"))
        """
}

private let gemma4CoTiledIndexedSlidingQKVKernels:
    [Gemma4CoTiledSlidingQKVFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4CoTiledSlidingQKVFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                for vBits in [12, 13] {
                    let formats = Gemma4CoTiledSlidingQKVFormats(
                        qBits: qBits, kBits: kBits, vBits: vBits
                    )
                    kernels[formats] = MLXFast.metalKernel(
                        name: "gemma4_cotiled_indexed_sliding_qkv_qmv_5376"
                            + "_q\(qBits)_k\(kBits)_v\(vBits)_v2",
                        inputNames: [
                            "cotiled_payload", "q_lut", "k_lut", "v_lut", "x",
                        ],
                        outputNames: ["q_output", "k_output", "v_output"],
                        source: gemma4CoTiledSlidingQKVBody(
                            qBits: qBits, kBits: kBits, vBits: vBits
                        ),
                        header: gemma4CoTiledQKVHelperHeader(
                            functionPrefix: "gemma4_cotiled_qkv"
                        ),
                        ensureRowContiguous: true
                    )
                }
            }
        }
        return kernels
    }()

private let gemma4CoTiledIndexedFullQKKernels:
    [Gemma4CoTiledFullQKFormats: MLXFast.MLXFastKernel] = {
        var kernels = [Gemma4CoTiledFullQKFormats: MLXFast.MLXFastKernel]()
        for qBits in [12, 13] {
            for kBits in [12, 13] {
                let formats = Gemma4CoTiledFullQKFormats(qBits: qBits, kBits: kBits)
                kernels[formats] = MLXFast.metalKernel(
                    name: "gemma4_cotiled_indexed_full_qk_qmv_5376"
                        + "_q\(qBits)_k\(kBits)_v2",
                    inputNames: ["cotiled_payload", "q_lut", "k_lut", "x"],
                    outputNames: ["q_output", "k_output"],
                    source: gemma4CoTiledFullQKBody(qBits: qBits, kBits: kBits),
                    header: gemma4CoTiledQKVHelperHeader(
                        functionPrefix: "gemma4_cotiled_full_qk"
                    ),
                    ensureRowContiguous: true
                )
            }
        }
        return kernels
    }()

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
    private let coTiled: CoTiledPayload?
    private let useCoTiled: Bool
    private let verifyCoTiledBits: Bool

    private struct CoTiledPayload {
        let words: MLXArray
        let formats: Gemma4CoTiledSlidingQKVFormats
    }

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        v: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata,
        vMetadata: IndexedAffineMetadata,
        coTiledPayload: Gemma4CoTiledAttentionPayload? = nil
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
        self.useCoTiled = gemma4CoTiledSlidingQKVEnabled
        self.verifyCoTiledBits = gemma4VerifyCoTiledSlidingQKVBits
        if gemma4CoTiledSlidingQKVEnabled || gemma4VerifyCoTiledSlidingQKVBits,
           let coTiledPayload,
           let vBits = coTiledPayload.vBits,
           supportsGemma4CoTiledSlidingQKVPayload(
               coTiledPayload,
               qMetadata: qMetadata,
               kMetadata: kMetadata,
               vMetadata: vMetadata
           )
        {
            self.coTiled = CoTiledPayload(
                words: coTiledPayload.words,
                formats: Gemma4CoTiledSlidingQKVFormats(
                    qBits: coTiledPayload.qBits,
                    kBits: coTiledPayload.kBits,
                    vBits: vBits
                )
            )
        } else {
            self.coTiled = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        guard let coTiled else {
            return u16Outputs(input)
        }
        let candidate = coTiledOutputs(input, coTiled: coTiled)
        if verifyCoTiledBits {
            let reference = u16Outputs(input)
            gemma4VerifyCoTiledRawBF16(
                candidate.0,
                reference: reference.0,
                label: "co-tiled sliding qkv q_proj"
            )
            gemma4VerifyCoTiledRawBF16(
                candidate.1,
                reference: reference.1,
                label: "co-tiled sliding qkv k_proj"
            )
            gemma4VerifyCoTiledRawBF16(
                candidate.2,
                reference: reference.2,
                label: "co-tiled sliding qkv v_proj"
            )
            if !useCoTiled { return reference }
        }
        return candidate
    }

    private func u16Outputs(
        _ input: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
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

    private func coTiledOutputs(
        _ input: MLXArray,
        coTiled: CoTiledPayload
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let kernel = gemma4CoTiledIndexedSlidingQKVKernels[coTiled.formats]
        else {
            preconditionFailure("missing co-tiled sliding QKV kernel variant")
        }
        let outputs = kernel(
            [
                coTiled.words,
                qMetadata.lut, kMetadata.lut, vMetadata.lut, input,
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
    private let coTiled: CoTiledPayload?
    private let useCoTiled: Bool
    private let verifyCoTiledBits: Bool

    private struct CoTiledPayload {
        let words: MLXArray
        let formats: Gemma4CoTiledFullQKFormats
    }

    init(
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        qMetadata: IndexedAffineMetadata,
        kMetadata: IndexedAffineMetadata,
        coTiledPayload: Gemma4CoTiledAttentionPayload? = nil
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
        self.useCoTiled = gemma4CoTiledFullQKEnabled
        self.verifyCoTiledBits = gemma4VerifyCoTiledFullQKBits
        if gemma4CoTiledFullQKEnabled || gemma4VerifyCoTiledFullQKBits,
           let coTiledPayload,
           supportsGemma4CoTiledFullQKPayload(
               coTiledPayload,
               qMetadata: qMetadata,
               kMetadata: kMetadata
           )
        {
            self.coTiled = CoTiledPayload(
                words: coTiledPayload.words,
                formats: Gemma4CoTiledFullQKFormats(
                    qBits: coTiledPayload.qBits,
                    kBits: coTiledPayload.kBits
                )
            )
        } else {
            self.coTiled = nil
        }
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        guard let coTiled else { return u16Outputs(input) }
        let candidate = coTiledOutputs(input, coTiled: coTiled)
        if verifyCoTiledBits {
            let reference = u16Outputs(input)
            gemma4VerifyCoTiledRawBF16(
                candidate.0,
                reference: reference.0,
                label: "co-tiled full qk q_proj"
            )
            gemma4VerifyCoTiledRawBF16(
                candidate.1,
                reference: reference.1,
                label: "co-tiled full qk k_proj"
            )
            if !useCoTiled { return reference }
        }
        return candidate
    }

    private func u16Outputs(_ input: MLXArray) -> (MLXArray, MLXArray) {
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

    private func coTiledOutputs(
        _ input: MLXArray,
        coTiled: CoTiledPayload
    ) -> (MLXArray, MLXArray) {
        guard let kernel = gemma4CoTiledIndexedFullQKKernels[coTiled.formats]
        else {
            preconditionFailure("missing co-tiled full QK kernel variant")
        }
        let outputs = kernel(
            [coTiled.words, qMetadata.lut, kMetadata.lut, input],
            grid: (32, 4_608, 1),
            threadGroup: (32, 9, 1),
            outputShapes: [[1, 1, 16_384], [1, 1, 2_048]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return (outputs[0], outputs[1])
    }
}
