import Foundation
import MLX

private func gemma4QKVEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

/// Losslessly packs each pair of four-group QMV blocks into a fixed 96-bit
/// tile. Gemma's 84 groups become ten 12-byte tiles plus one eight-byte tail,
/// so every row is exactly 128 bytes instead of 168 U16 bytes. The layout is
/// deliberately block-shaped: decode uses fixed byte offsets and nibble
/// operations rather than division, modulo, or cross-word branches.
func gemma4Pack12BitQKVIndices(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt8]? {
    guard rows > 0,
          groupsPerRow > 0,
          groupsPerRow.isMultiple(of: 4)
    else {
        return nil
    }
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    guard !elementOverflow, indices.count == elementCount else { return nil }

    let blockCount = groupsPerRow / 4
    let blockPairCount = blockCount / 2
    let tailBlockCount = blockCount % 2
    let (pairBytes, pairOverflow) = blockPairCount.multipliedReportingOverflow(
        by: 12)
    guard !pairOverflow else { return nil }
    let (payloadBytes, payloadOverflow) = pairBytes.addingReportingOverflow(
        tailBlockCount * 8)
    guard !payloadOverflow else { return nil }
    let (roundedBytes, roundedOverflow) = payloadBytes.addingReportingOverflow(3)
    guard !roundedOverflow else { return nil }
    let bytesPerRow = (roundedBytes / 4) * 4
    let (byteCount, byteOverflow) = rows.multipliedReportingOverflow(
        by: bytesPerRow)
    guard !byteOverflow else { return nil }

    var bytes = [UInt8](repeating: 0, count: byteCount)
    for row in 0..<rows {
        let inputBase = row * groupsPerRow
        let outputBase = row * bytesPerRow
        for blockPair in 0..<blockPairCount {
            let inputPairBase = inputBase + blockPair * 8
            let outputPairBase = outputBase + blockPair * 12
            for laneGroup in 0..<4 {
                let even = indices[inputPairBase + laneGroup]
                let odd = indices[inputPairBase + 4 + laneGroup]
                guard even < 4_096, odd < 4_096 else { return nil }
                bytes[outputPairBase + laneGroup] = UInt8(truncatingIfNeeded: even)
                bytes[outputPairBase + 4 + laneGroup] =
                    UInt8((even >> 8) | ((odd & 0x000f) << 4))
                bytes[outputPairBase + 8 + laneGroup] = UInt8(odd >> 4)
            }
        }
        if tailBlockCount == 1 {
            let inputTailBase = inputBase + blockPairCount * 8
            let outputTailBase = outputBase + blockPairCount * 12
            for laneGroup in 0..<4 {
                let value = indices[inputTailBase + laneGroup]
                guard value < 4_096 else { return nil }
                let laneOffset = outputTailBase + laneGroup * 2
                bytes[laneOffset] = UInt8(truncatingIfNeeded: value)
                bytes[laneOffset + 1] = UInt8(value >> 8)
            }
        }
    }
    return bytes
}

private struct Packed12QKVMetadata: @unchecked Sendable {
    let bytes: MLXArray

    init?(metadata: IndexedAffineMetadata, rows: Int) {
        guard metadata.indices.dtype == .uint16,
              metadata.indices.shape == [rows, 84],
              metadata.lut.dtype == .uint32,
              metadata.lut.ndim == 1,
              (1...4_096).contains(metadata.lut.size),
              let packed = gemma4Pack12BitQKVIndices(
                  metadata.indices.asArray(UInt16.self),
                  rows: rows,
                  groupsPerRow: 84
              )
        else {
            return nil
        }
        let bytes = MLXArray(packed, [rows, 128])
        eval(bytes)
        self.bytes = bytes
    }
}

private let gemma4Packed12QKVHeader = """
    using namespace metal;

    inline float gemma4_packed_qkv_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_packed_qkv_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_packed_qkv_load_values(
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

    inline float gemma4_packed_qkv_qdot_4bit(
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

private let gemma4Packed12IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut",
        "v_weight", "v_packed_indices", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: """
        constexpr int kPackedBytesPerRow = 128;
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
        const device uchar* packed_indices = is_q
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
        const device uchar* row_packed_indices =
            packed_indices + output_row * kPackedBytesPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup >> 3;

        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < 10; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                gemma4_packed_qkv_load_values(input, even_values);
            uint odd_pairs[kRowsPerSIMD];

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uchar* row_tile =
                    row_packed_indices + row * kPackedBytesPerRow;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                result[row] += gemma4_packed_qkv_qdot_4bit(
                    row_weight,
                    even_values,
                    gemma4_packed_qkv_pair_scale(even_pair),
                    gemma4_packed_qkv_pair_bias(even_pair),
                    even_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                gemma4_packed_qkv_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint odd_pair = odd_pairs[row];
                result[row] += gemma4_packed_qkv_qdot_4bit(
                    row_weight,
                    odd_values,
                    gemma4_packed_qkv_pair_scale(odd_pair),
                    gemma4_packed_qkv_pair_bias(odd_pair),
                    odd_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            row_packed_indices += 12;
        }

        float tail_values[8];
        const float tail_input_sum =
            gemma4_packed_qkv_load_values(input, tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const device uchar* row_tail =
                row_packed_indices + row * kPackedBytesPerRow;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            result[row] += gemma4_packed_qkv_qdot_4bit(
                row_weight,
                tail_values,
                gemma4_packed_qkv_pair_scale(pair),
                gemma4_packed_qkv_pair_bias(pair),
                tail_input_sum);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: gemma4Packed12QKVHeader,
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

private struct Packed12SlidingQKVMetadata: @unchecked Sendable {
    let q: Packed12QKVMetadata
    let k: Packed12QKVMetadata
    let v: Packed12QKVMetadata
}

struct FusedSlidingQKVProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let v: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata
    let vMetadata: IndexedAffineMetadata
    private let packed12Metadata: Packed12SlidingQKVMetadata?
    private let usePacked12: Bool
    private let verifyPacked12Bits: Bool

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
        let usePacked12 = gemma4QKVEnvironmentFlag(
            "DARKBLOOM_PACKED_QKV_FIXED12",
            default: true
        )
        let verifyPacked12Bits = gemma4QKVEnvironmentFlag(
            "DARKBLOOM_VERIFY_PACKED_QKV_FIXED12_BITS",
            default: false
        )
        if usePacked12 || verifyPacked12Bits,
           let packedQ = Packed12QKVMetadata(metadata: qMetadata, rows: 8_192),
           let packedK = Packed12QKVMetadata(metadata: kMetadata, rows: 4_096),
           let packedV = Packed12QKVMetadata(metadata: vMetadata, rows: 4_096)
        {
            self.packed12Metadata = Packed12SlidingQKVMetadata(
                q: packedQ,
                k: packedK,
                v: packedV
            )
        } else {
            self.packed12Metadata = nil
        }
        self.usePacked12 = usePacked12
        self.verifyPacked12Bits = verifyPacked12Bits
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [
            [1, 1, 8_192],
            [1, 1, 4_096],
            [1, 1, 4_096],
        ]
        let runPacked = usePacked12 || verifyPacked12Bits
        let packedOutputs: [MLXArray]?
        if runPacked, let packed12Metadata {
            packedOutputs = gemma4Packed12IndexedSlidingQKV(
                [
                    q.weight, packed12Metadata.q.bytes, qMetadata.lut,
                    k.weight, packed12Metadata.k.bytes, kMetadata.lut,
                    v.weight, packed12Metadata.v.bytes, vMetadata.lut, input,
                ],
                grid: (32, 4_096, 1),
                threadGroup: (32, 4, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            )
        } else {
            packedOutputs = nil
        }
        let needsPromoted = !usePacked12
            || verifyPacked12Bits
            || packed12Metadata == nil
        let promotedOutputs: [MLXArray]? = needsPromoted
            ? gemma4IndexedSlidingQKV(
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
            : nil
        if verifyPacked12Bits,
           let packedOutputs,
           let promotedOutputs
        {
            for (candidate, reference) in zip(packedOutputs, promotedOutputs) {
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "fixed-tile packed sliding QKV differs from U16 kernel"
                )
            }
            return (promotedOutputs[0], promotedOutputs[1], promotedOutputs[2])
        }
        let outputs = packedOutputs ?? promotedOutputs
        guard let outputs else {
            preconditionFailure("sliding QKV kernel was not selected")
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

private let gemma4Packed12IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_packed12_indexed_full_qk_qmv_5376_v1",
    inputNames: [
        "q_weight", "q_packed_indices", "q_lut",
        "k_weight", "k_packed_indices", "k_lut", "x",
    ],
    outputNames: ["q_output", "k_output"],
    source: """
        constexpr int kPackedBytesPerRow = 128;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int output_row = is_q
            ? threadgroup_position_in_grid.y * 32 + projection * kRowsPerSIMD
            : threadgroup_position_in_grid.y * kRowsPerSIMD;

        const device uint* weight = is_q ? q_weight : k_weight;
        const device uchar* packed_indices = is_q
            ? q_packed_indices
            : k_packed_indices;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device uchar* row_packed_indices =
            packed_indices + output_row * kPackedBytesPerRow;
        const device bfloat* input = x + thread_index_in_simdgroup * 8;
        const uint lane_group = thread_index_in_simdgroup >> 3;

        float result[kRowsPerSIMD] = {0};
        for (int block_pair = 0; block_pair < 10; ++block_pair) {
            float even_values[8];
            const float even_input_sum =
                gemma4_packed_qkv_load_values(input, even_values);
            uint odd_pairs[kRowsPerSIMD];

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device uchar* row_tile =
                    row_packed_indices + row * kPackedBytesPerRow;
                const uint even_low = row_tile[lane_group];
                const uint middle = row_tile[4 + lane_group];
                const uint odd_high = row_tile[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                result[row] += gemma4_packed_qkv_qdot_4bit(
                    row_weight,
                    even_values,
                    gemma4_packed_qkv_pair_scale(even_pair),
                    gemma4_packed_qkv_pair_bias(even_pair),
                    even_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            float odd_values[8];
            const float odd_input_sum =
                gemma4_packed_qkv_load_values(input, odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const uint odd_pair = odd_pairs[row];
                result[row] += gemma4_packed_qkv_qdot_4bit(
                    row_weight,
                    odd_values,
                    gemma4_packed_qkv_pair_scale(odd_pair),
                    gemma4_packed_qkv_pair_bias(odd_pair),
                    odd_input_sum);
            }

            weight_bytes += 128;
            input += 256;
            row_packed_indices += 12;
        }

        float tail_values[8];
        const float tail_input_sum =
            gemma4_packed_qkv_load_values(input, tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_weight =
                weight_bytes + row * kWeightBytesPerRow;
            const device uchar* row_tail =
                row_packed_indices + row * kPackedBytesPerRow;
            const uint low = row_tail[tail_lane_offset];
            const uint high = row_tail[tail_lane_offset + 1];
            const uint metadata_index = low | (high << 8);
            const uint pair = lut[metadata_index];
            result[row] += gemma4_packed_qkv_qdot_4bit(
                row_weight,
                tail_values,
                gemma4_packed_qkv_pair_scale(pair),
                gemma4_packed_qkv_pair_bias(pair),
                tail_input_sum);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: gemma4Packed12QKVHeader,
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

private struct Packed12FullQKMetadata: @unchecked Sendable {
    let q: Packed12QKVMetadata
    let k: Packed12QKVMetadata
}

struct FusedFullQKProjection: @unchecked Sendable {
    let q: FastQuantizedProjection
    let k: FastQuantizedProjection
    let qMetadata: IndexedAffineMetadata
    let kMetadata: IndexedAffineMetadata
    private let packed12Metadata: Packed12FullQKMetadata?
    private let usePacked12: Bool
    private let verifyPacked12Bits: Bool

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
        let usePacked12 = gemma4QKVEnvironmentFlag(
            "DARKBLOOM_PACKED_QKV_FIXED12",
            default: true
        )
        let verifyPacked12Bits = gemma4QKVEnvironmentFlag(
            "DARKBLOOM_VERIFY_PACKED_QKV_FIXED12_BITS",
            default: false
        )
        if usePacked12 || verifyPacked12Bits,
           let packedQ = Packed12QKVMetadata(metadata: qMetadata, rows: 16_384),
           let packedK = Packed12QKVMetadata(metadata: kMetadata, rows: 2_048)
        {
            self.packed12Metadata = Packed12FullQKMetadata(
                q: packedQ,
                k: packedK
            )
        } else {
            self.packed12Metadata = nil
        }
        self.usePacked12 = usePacked12
        self.verifyPacked12Bits = verifyPacked12Bits
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [[1, 1, 16_384], [1, 1, 2_048]]
        let runPacked = usePacked12 || verifyPacked12Bits
        let packedOutputs: [MLXArray]?
        if runPacked, let packed12Metadata {
            packedOutputs = gemma4Packed12IndexedFullQK(
                [
                    q.weight, packed12Metadata.q.bytes, qMetadata.lut,
                    k.weight, packed12Metadata.k.bytes, kMetadata.lut, input,
                ],
                grid: (32, 4_608, 1),
                threadGroup: (32, 9, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
        } else {
            packedOutputs = nil
        }
        let needsPromoted = !usePacked12
            || verifyPacked12Bits
            || packed12Metadata == nil
        let promotedOutputs: [MLXArray]? = needsPromoted
            ? gemma4IndexedFullQK(
                [
                    q.weight, qMetadata.indices, qMetadata.lut,
                    k.weight, kMetadata.indices, kMetadata.lut, input,
                ],
                grid: (32, 4_608, 1),
                threadGroup: (32, 9, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
            : nil
        if verifyPacked12Bits,
           let packedOutputs,
           let promotedOutputs
        {
            for (candidate, reference) in zip(packedOutputs, promotedOutputs) {
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "fixed-tile packed full QK differs from U16 kernel"
                )
            }
            return (promotedOutputs[0], promotedOutputs[1])
        }
        let outputs = packedOutputs ?? promotedOutputs
        guard let outputs else {
            preconditionFailure("full QK kernel was not selected")
        }
        return (outputs[0], outputs[1])
    }
}
