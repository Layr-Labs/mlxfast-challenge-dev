import Foundation
import MLX

private func gemma4PackedQKVEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private struct Gemma4PackedAffineIndices: @unchecked Sendable {
    let words: MLXArray

    init?(metadata: IndexedAffineMetadata, bits: Int) {
        guard bits == 12 || bits == 13,
              metadata.indices.dtype == .uint16,
              metadata.indices.ndim == 2,
              metadata.indices.dim(1) == 84,
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              metadata.lut.size > 0,
              metadata.lut.size <= (1 << bits)
        else {
            return nil
        }

        let rows = metadata.indices.dim(0)
        let groupsPerRow = metadata.indices.dim(1)
        let wordsPerRow = (groupsPerRow * bits + 31) / 32
        let indices = metadata.indices.asArray(UInt16.self)
        guard indices.count == rows * groupsPerRow,
              indices.allSatisfy({ Int($0) < metadata.lut.size })
        else {
            return nil
        }

        var packed = [UInt32](repeating: 0, count: rows * wordsPerRow)
        for row in 0..<rows {
            let inputBase = row * groupsPerRow
            let outputBase = row * wordsPerRow
            for group in 0..<groupsPerRow {
                let value = UInt32(indices[inputBase + group])
                let bitOffset = group * bits
                let word = bitOffset / 32
                let shift = bitOffset % 32
                packed[outputBase + word] |= value << UInt32(shift)
                if shift + bits > 32 {
                    packed[outputBase + word + 1] |= value >> UInt32(32 - shift)
                }
            }
        }

        let words = MLXArray(packed, [rows, wordsPerRow])
        // The source U16 metadata is already materialized and validated. This
        // one-time host packing and Metal upload happen during model setup,
        // before either scored phase begins.
        eval(words)
        self.words = words
    }
}

private func gemma4PackedQKVHeader(bits: Int, prefix: String) -> String {
    let mask = (1 << bits) - 1
    return """
        using namespace metal;

        inline uint \(prefix)_packed_index(
            const device uint* words,
            uint group
        ) {
            constexpr uint kBits = \(bits);
            constexpr uint kMask = \(mask);
            const uint bit_offset = group * kBits;
            const uint word = bit_offset >> 5;
            const uint shift = bit_offset & 31;
            uint value = words[word] >> shift;
            if (shift + kBits > 32) {
                value |= words[word + 1] << (32 - shift);
            }
            return value & kMask;
        }

        inline float \(prefix)_pair_scale(uint pair) {
            return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
        }

        inline float \(prefix)_pair_bias(uint pair) {
            return static_cast<float>(
                as_type<bfloat>(static_cast<ushort>(pair >> 16)));
        }

        inline float \(prefix)_load_values(
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

        inline float \(prefix)_qdot_4bit(
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

private func gemma4PackedSlidingQKVSource(bits: Int) -> String {
    let wordsPerRow = (84 * bits + 31) / 32
    return """
        constexpr int kInputWidth = 5376;
        constexpr int kGroupsPerRow = 84;
        constexpr int kPackedWordsPerRow = \(wordsPerRow);
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
        const device uint* packed_indices = is_q
            ? q_packed_indices
            : (is_k ? k_packed_indices : v_packed_indices);
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
        const device uint* row_packed_indices =
            packed_indices + output_row * kPackedWordsPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup / 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_packed_qkv_load_values(input, values);
            const uint logical_group = block * 4 + lane_group;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint metadata_index = gemma4_packed_qkv_packed_index(
                    row_packed_indices + row * kPackedWordsPerRow,
                    logical_group);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_packed_qkv_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_packed_qkv_pair_scale(pair),
                    gemma4_packed_qkv_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private func gemma4PackedFullQKSource(bits: Int) -> String {
    let wordsPerRow = (84 * bits + 31) / 32
    return """
        constexpr int kGroupsPerRow = 84;
        constexpr int kPackedWordsPerRow = \(wordsPerRow);
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q ? q_weight : k_weight;
        const device uint* packed_indices =
            is_q ? q_packed_indices : k_packed_indices;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uint* row_packed_indices =
            packed_indices + output_row * kPackedWordsPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup / 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_packed_full_qk_load_values(input, values);
            const uint logical_group = block * 4 + lane_group;

            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint metadata_index = gemma4_packed_full_qk_packed_index(
                    row_packed_indices + row * kPackedWordsPerRow,
                    logical_group);
                const uint pair = lut[metadata_index];
                result[row] += gemma4_packed_full_qk_qdot_4bit(
                    row_weight,
                    values,
                    gemma4_packed_full_qk_pair_scale(pair),
                    gemma4_packed_full_qk_pair_bias(pair),
                    input_sum);
            }

            weight_bytes += 128;
            input += 256;
        }

        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """
}

private let gemma4Packed12IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut",
        "v_weight", "v_packed_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: gemma4PackedSlidingQKVSource(bits: 12),
    header: gemma4PackedQKVHeader(bits: 12, prefix: "gemma4_packed_qkv"),
    ensureRowContiguous: true
)

private let gemma4Packed13IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_packed13_indexed_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut",
        "v_weight", "v_packed_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: gemma4PackedSlidingQKVSource(bits: 13),
    header: gemma4PackedQKVHeader(bits: 13, prefix: "gemma4_packed_qkv"),
    ensureRowContiguous: true
)

private let gemma4Packed12IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_full_qk_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: gemma4PackedFullQKSource(bits: 12),
    header: gemma4PackedQKVHeader(bits: 12, prefix: "gemma4_packed_full_qk"),
    ensureRowContiguous: true
)

private let gemma4Packed13IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_packed13_indexed_full_qk_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: gemma4PackedFullQKSource(bits: 13),
    header: gemma4PackedQKVHeader(bits: 13, prefix: "gemma4_packed_full_qk"),
    ensureRowContiguous: true
)

private struct Gemma4PackedSlidingQKVMetadata: @unchecked Sendable {
    let bits: Int
    let q: Gemma4PackedAffineIndices
    let k: Gemma4PackedAffineIndices
    let v: Gemma4PackedAffineIndices

    init?(
        q: IndexedAffineMetadata,
        k: IndexedAffineMetadata,
        v: IndexedAffineMetadata
    ) {
        let maximumLUTSize = max(q.lut.size, k.lut.size, v.lut.size)
        let bits: Int
        if maximumLUTSize <= 4_096 {
            bits = 12
        } else if maximumLUTSize <= 8_192 {
            bits = 13
        } else {
            return nil
        }
        guard let packedQ = Gemma4PackedAffineIndices(metadata: q, bits: bits),
              let packedK = Gemma4PackedAffineIndices(metadata: k, bits: bits),
              let packedV = Gemma4PackedAffineIndices(metadata: v, bits: bits)
        else {
            return nil
        }
        self.bits = bits
        self.q = packedQ
        self.k = packedK
        self.v = packedV
    }
}

private struct Gemma4PackedFullQKMetadata: @unchecked Sendable {
    let bits: Int
    let q: Gemma4PackedAffineIndices
    let k: Gemma4PackedAffineIndices

    init?(q: IndexedAffineMetadata, k: IndexedAffineMetadata) {
        let maximumLUTSize = max(q.lut.size, k.lut.size)
        let bits: Int
        if maximumLUTSize <= 4_096 {
            bits = 12
        } else if maximumLUTSize <= 8_192 {
            bits = 13
        } else {
            return nil
        }
        guard let packedQ = Gemma4PackedAffineIndices(metadata: q, bits: bits),
              let packedK = Gemma4PackedAffineIndices(metadata: k, bits: bits)
        else {
            return nil
        }
        self.bits = bits
        self.q = packedQ
        self.k = packedK
    }
}

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
    private let packedMetadata: Gemma4PackedSlidingQKVMetadata?
    private let verifyPackedBits: Bool

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
        self.packedMetadata = gemma4PackedQKVEnvironmentFlag(
            "DARKBLOOM_PACKED_QKV_INDICES",
            default: true
        ) ? Gemma4PackedSlidingQKVMetadata(
            q: qMetadata,
            k: kMetadata,
            v: vMetadata
        ) : nil
        self.verifyPackedBits = gemma4PackedQKVEnvironmentFlag(
            "DARKBLOOM_VERIFY_PACKED_QKV_BITS",
            default: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [
            [1, 1, 8_192],
            [1, 1, 4_096],
            [1, 1, 4_096],
        ]
        let outputs: [MLXArray]
        if let packedMetadata {
            let kernel = packedMetadata.bits == 12
                ? gemma4Packed12IndexedSlidingQKV
                : gemma4Packed13IndexedSlidingQKV
            outputs = kernel(
                [
                    q.weight, packedMetadata.q.words, qMetadata.lut,
                    k.weight, packedMetadata.k.words, kMetadata.lut,
                    v.weight, packedMetadata.v.words, vMetadata.lut, input,
                ],
                grid: (32, 4_096, 1),
                threadGroup: (32, 4, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            )
            if verifyPackedBits {
                let reference = gemma4IndexedSlidingQKV(
                    [
                        q.weight, qMetadata.indices, qMetadata.lut,
                        k.weight, kMetadata.indices, kMetadata.lut,
                        v.weight, vMetadata.indices, vMetadata.lut, input,
                    ],
                    grid: (32, 4_096, 1),
                    threadGroup: (32, 4, 1),
                    outputShapes: outputShapes,
                    outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
                )
                let matches = zip(outputs, reference).map {
                    arrayEqual($0.view(dtype: .uint16), $1.view(dtype: .uint16))
                }
                eval(matches)
                precondition(
                    matches.allSatisfy({ $0.item(Bool.self) }),
                    "packed QKV projection differs from U16 indexed QKV"
                )
            }
        } else {
            outputs = gemma4IndexedSlidingQKV(
                [
                    q.weight, qMetadata.indices, qMetadata.lut,
                    k.weight, kMetadata.indices, kMetadata.lut,
                    v.weight, vMetadata.indices, vMetadata.lut, input,
                ],
                grid: (32, 4_096, 1),
                threadGroup: (32, 4, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            )
        }
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
    private let packedMetadata: Gemma4PackedFullQKMetadata?
    private let verifyPackedBits: Bool

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
        self.packedMetadata = gemma4PackedQKVEnvironmentFlag(
            "DARKBLOOM_PACKED_QKV_INDICES",
            default: true
        ) ? Gemma4PackedFullQKMetadata(q: qMetadata, k: kMetadata) : nil
        self.verifyPackedBits = gemma4PackedQKVEnvironmentFlag(
            "DARKBLOOM_VERIFY_PACKED_QKV_BITS",
            default: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [[1, 1, 16_384], [1, 1, 2_048]]
        let outputs: [MLXArray]
        if let packedMetadata {
            let kernel = packedMetadata.bits == 12
                ? gemma4Packed12IndexedFullQK
                : gemma4Packed13IndexedFullQK
            outputs = kernel(
                [
                    q.weight, packedMetadata.q.words, qMetadata.lut,
                    k.weight, packedMetadata.k.words, kMetadata.lut, input,
                ],
                grid: (32, 4_608, 1),
                threadGroup: (32, 9, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
            if verifyPackedBits {
                let reference = gemma4IndexedFullQK(
                    [
                        q.weight, qMetadata.indices, qMetadata.lut,
                        k.weight, kMetadata.indices, kMetadata.lut, input,
                    ],
                    grid: (32, 4_608, 1),
                    threadGroup: (32, 9, 1),
                    outputShapes: outputShapes,
                    outputDTypes: [.bfloat16, .bfloat16]
                )
                let matches = zip(outputs, reference).map {
                    arrayEqual($0.view(dtype: .uint16), $1.view(dtype: .uint16))
                }
                eval(matches)
                precondition(
                    matches.allSatisfy({ $0.item(Bool.self) }),
                    "packed QK projection differs from U16 indexed QK"
                )
            }
        } else {
            outputs = gemma4IndexedFullQK(
                [
                    q.weight, qMetadata.indices, qMetadata.lut,
                    k.weight, kMetadata.indices, kMetadata.lut, input,
                ],
                grid: (32, 4_608, 1),
                threadGroup: (32, 9, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
        }
        return (outputs[0], outputs[1])
    }
}
