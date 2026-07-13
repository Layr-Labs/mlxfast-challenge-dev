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

private func gemma4QKVCoTiledFixed12Enabled() -> Bool {
    gemma4QKVEnvironmentFlag(
        "DARKBLOOM_QKV_COTILED_FIXED12",
        default: true
    )
}

private func gemma4VerifyQKVCoTiledFixed12Bits() -> Bool {
    gemma4QKVEnvironmentFlag(
        "DARKBLOOM_VERIFY_QKV_COTILED_FIXED12_BITS",
        default: false
    )
}

/// Losslessly packs every pair of four-group QMV blocks into three U32 words.
/// Each word is authored in the byte order consumed by Metal: four even low
/// bytes, four shared even-high/odd-low bytes, then four odd high bytes. The
/// unpaired final block occupies two U32 words. Gemma's 84 groups therefore
/// use exactly 32 U32 per row with no variable bit-crossing in the kernel.
func gemma4Pack12BitQKVIndexWords(
    _ indices: [UInt16],
    rows: Int,
    groupsPerRow: Int
) -> [UInt32]? {
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
    let pairCount = blockCount / 2
    let hasTail = !blockCount.isMultiple(of: 2)
    let (pairWords, pairOverflow) = pairCount.multipliedReportingOverflow(by: 3)
    guard !pairOverflow else { return nil }
    let (wordsPerRow, rowOverflow) = pairWords.addingReportingOverflow(
        hasTail ? 2 : 0)
    guard !rowOverflow else { return nil }
    let (wordCount, countOverflow) = rows.multipliedReportingOverflow(
        by: wordsPerRow)
    guard !countOverflow else { return nil }

    var words = [UInt32](repeating: 0, count: wordCount)
    for row in 0..<rows {
        let inputBase = row * groupsPerRow
        let outputBase = row * wordsPerRow
        for pair in 0..<pairCount {
            let inputPair = inputBase + pair * 8
            let outputPair = outputBase + pair * 3
            for laneGroup in 0..<4 {
                let even = UInt32(indices[inputPair + laneGroup])
                let odd = UInt32(indices[inputPair + 4 + laneGroup])
                guard even < 4_096, odd < 4_096 else { return nil }
                let shift = laneGroup * 8
                words[outputPair] |= (even & 0xff) << shift
                words[outputPair + 1] |= (
                    ((even >> 8) & 0x0f) | ((odd & 0x0f) << 4)
                ) << shift
                words[outputPair + 2] |= ((odd >> 4) & 0xff) << shift
            }
        }
        if hasTail {
            let inputTail = inputBase + pairCount * 8
            let outputTail = outputBase + pairCount * 3
            for laneGroup in 0..<4 {
                let value = UInt32(indices[inputTail + laneGroup])
                guard value < 4_096 else { return nil }
                let word = outputTail + laneGroup / 2
                let shift = (laneGroup % 2) * 16
                words[word] |= value << shift
            }
        }
    }
    return words
}

/// Reorders Q/K[/V] weights and fixed12 affine indexes into one allocation in
/// the exact threadgroup/SIMD ownership used by the established decode
/// kernels. Ten paired tiles are followed by the final 256-input tail:
///
/// `[even weights, odd weights, pair metadata, alignment padding] * 10,`
/// `[tail weights, tail metadata, threadgroup alignment padding]`.
///
/// The LUTs remain separate because each projection owns a different table;
/// all bandwidth-dominant weights and per-row indexes share this one payload.
/// Pair and threadgroup starts are 128-byte aligned so each SIMD's 128-byte
/// weight row begins on a cache-line boundary.
func gemma4MakeCoTiledFixed12QKVPayload(
    _ projections: [(
        weight: MLXArray,
        indices: MLXArray,
        simdGroups: Int
    )],
    threadgroupCount: Int,
    materialize: Bool = true
) -> MLXArray? {
    guard !projections.isEmpty, threadgroupCount > 0 else { return nil }
    let rowsPerSIMD = 4
    let blocks = 21
    let wordsPerBlock = 32
    let groupsPerRow = 84
    let packedWordsPerRow = 32

    var totalSIMDGroups = 0
    var weightTiles: [MLXArray] = []
    var metadataTiles: [MLXArray] = []
    weightTiles.reserveCapacity(projections.count)
    metadataTiles.reserveCapacity(projections.count)

    for projection in projections {
        guard projection.simdGroups > 0 else { return nil }
        let (rowsPerThreadgroup, rowOverflow) = projection.simdGroups
            .multipliedReportingOverflow(by: rowsPerSIMD)
        let (rows, countOverflow) = threadgroupCount
            .multipliedReportingOverflow(by: rowsPerThreadgroup)
        guard !rowOverflow, !countOverflow,
              projection.weight.dtype == .uint32,
              projection.weight.shape == [rows, blocks * wordsPerBlock],
              projection.indices.dtype == .uint16,
              projection.indices.shape == [rows, groupsPerRow],
              let packed = gemma4Pack12BitQKVIndexWords(
                  projection.indices.asArray(UInt16.self),
                  rows: rows,
                  groupsPerRow: groupsPerRow
              )
        else {
            return nil
        }
        let (nextSIMDGroups, simdOverflow) = totalSIMDGroups
            .addingReportingOverflow(projection.simdGroups)
        guard !simdOverflow else { return nil }
        totalSIMDGroups = nextSIMDGroups
        weightTiles.append(
            projection.weight.reshaped(
                threadgroupCount,
                projection.simdGroups,
                rowsPerSIMD,
                blocks,
                wordsPerBlock
            )
        )
        metadataTiles.append(
            MLXArray(packed, [
                threadgroupCount,
                projection.simdGroups,
                rowsPerSIMD,
                packedWordsPerRow,
            ])
        )
    }

    let weights = concatenated(weightTiles, axis: 1)
    let metadata = concatenated(metadataTiles, axis: 1)

    let pairedWeights = weights[0..., 0..., 0..., 0..<20, 0...]
        .reshaped(
            threadgroupCount,
            totalSIMDGroups,
            rowsPerSIMD,
            10,
            2,
            wordsPerBlock
        )
        .transposed(0, 3, 4, 1, 2, 5)
        .reshaped(
            threadgroupCount,
            10,
            2 * totalSIMDGroups * rowsPerSIMD * wordsPerBlock
        )
    let pairedMetadata = metadata[0..., 0..., 0..., 0..<30]
        .reshaped(
            threadgroupCount,
            totalSIMDGroups,
            rowsPerSIMD,
            10,
            3
        )
        .transposed(0, 3, 1, 2, 4)
        .reshaped(
            threadgroupCount,
            10,
            totalSIMDGroups * rowsPerSIMD * 3
        )
    let tightPairedPayload = concatenated(
        [pairedWeights, pairedMetadata],
        axis: 2
    )
    let tightPairWords = totalSIMDGroups
        * (2 * rowsPerSIMD * wordsPerBlock + 12)
    let alignedPairWords = (tightPairWords + 31) & ~31
    let pairPaddingWords = alignedPairWords - tightPairWords
    let pairedPayload: MLXArray
    if pairPaddingWords > 0 {
        pairedPayload = concatenated(
            [
                tightPairedPayload,
                MLXArray.zeros(
                    [threadgroupCount, 10, pairPaddingWords],
                    dtype: .uint32
                ),
            ],
            axis: 2
        ).reshaped(threadgroupCount, 10 * alignedPairWords)
    } else {
        pairedPayload = tightPairedPayload.reshaped(
            threadgroupCount,
            10 * alignedPairWords
        )
    }

    let tailWeights = weights[0..., 0..., 0..., 20..<21, 0...]
        .reshaped(
            threadgroupCount,
            totalSIMDGroups * rowsPerSIMD * wordsPerBlock
        )
    let tailMetadata = metadata[0..., 0..., 0..., 30..<32]
        .reshaped(
            threadgroupCount,
            totalSIMDGroups * rowsPerSIMD * 2
        )
    let tightTailPayload = concatenated([tailWeights, tailMetadata], axis: 1)
    let tightTailWords = totalSIMDGroups
        * rowsPerSIMD
        * (wordsPerBlock + 2)
    let tightThreadgroupWords = 10 * alignedPairWords + tightTailWords
    let wordsPerThreadgroup = (tightThreadgroupWords + 31) & ~31
    let threadgroupPaddingWords = wordsPerThreadgroup - tightThreadgroupWords
    let tailPayload: MLXArray
    if threadgroupPaddingWords > 0 {
        tailPayload = concatenated(
            [
                tightTailPayload,
                MLXArray.zeros(
                    [threadgroupCount, threadgroupPaddingWords],
                    dtype: .uint32
                ),
            ],
            axis: 1
        )
    } else {
        tailPayload = tightTailPayload
    }
    let payload = concatenated([pairedPayload, tailPayload], axis: 1)
    precondition(payload.dtype == .uint32)
    precondition(payload.shape == [threadgroupCount, wordsPerThreadgroup])
    if materialize {
        eval(payload)
    }
    return payload
}

private struct Gemma4CoTiledFixed12QKVPayload: @unchecked Sendable {
    let words: MLXArray

    init?(
        projections: [(
            projection: FastQuantizedProjection,
            metadata: IndexedAffineMetadata,
            simdGroups: Int
        )],
        threadgroupCount: Int
    ) {
        guard projections.allSatisfy({
            $0.metadata.lut.dtype == .uint32
                && $0.metadata.lut.ndim == 1
                && (1...4_096).contains($0.metadata.lut.size)
        }), let words = gemma4MakeCoTiledFixed12QKVPayload(
            projections.map {
                (
                    weight: $0.projection.weight,
                    indices: $0.metadata.indices,
                    simdGroups: $0.simdGroups
                )
            },
            threadgroupCount: threadgroupCount
        ) else {
            return nil
        }
        self.words = words
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

private let gemma4CoTiledFixed12QKVHeader = """
    using namespace metal;

    inline float gemma4_cotiled_qkv_pair_scale(uint pair) {
        return static_cast<float>(as_type<bfloat>(static_cast<ushort>(pair)));
    }

    inline float gemma4_cotiled_qkv_pair_bias(uint pair) {
        return static_cast<float>(
            as_type<bfloat>(static_cast<ushort>(pair >> 16)));
    }

    inline float gemma4_cotiled_qkv_load_values(
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

    inline float gemma4_cotiled_qkv_qdot_4bit(
        uint packed_word,
        const thread float* values,
        float scale,
        float bias,
        float input_sum
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
        return scale * accumulator + input_sum * bias;
    }
    """

private let gemma4CoTiledFixed12IndexedSlidingQKV = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_aligned_sliding_qkv_qmv_5376_v1",
    inputNames: [
        "cotiled_payload",
        "q_lut", "k_lut", "v_lut", "x",
    ],
    outputNames: ["q_output", "k_output", "v_output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kWordsPerRowBlock = 32;
        constexpr int kSIMDGroups = 4;
        constexpr int kWeightWordsPerBlock =
            kSIMDGroups * kRowsPerSIMD * kWordsPerRowBlock;
        constexpr int kPairMetadataWords =
            kSIMDGroups * kRowsPerSIMD * 3;
        constexpr int kPairContentWords =
            2 * kWeightWordsPerBlock + kPairMetadataWords;
        constexpr int kPairPayloadWords =
            ((kPairContentWords + 31) / 32) * 32;
        constexpr int kTailMetadataWords =
            kSIMDGroups * kRowsPerSIMD * 2;
        constexpr int kTailPayloadWords =
            kWeightWordsPerBlock + kTailMetadataWords;
        constexpr int kThreadgroupContentWords =
            10 * kPairPayloadWords + kTailPayloadWords;
        constexpr int kWordsPerThreadgroup =
            ((kThreadgroupContentWords + 31) / 32) * 32;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 2;
        const bool is_k = projection == 2;
        const int threadgroup_index = threadgroup_position_in_grid.y;
        const int output_row = is_q
            ? threadgroup_index * 8 + projection * kRowsPerSIMD
            : threadgroup_index * kRowsPerSIMD;
        const device uint* lut = is_q
            ? q_lut
            : (is_k ? k_lut : v_lut);
        device bfloat* output = is_q
            ? q_output
            : (is_k ? k_output : v_output);

        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device bfloat* input = x + lane * 8;
        const device uint* threadgroup_words =
            cotiled_payload + threadgroup_index * kWordsPerThreadgroup;

        float result[kRowsPerSIMD] = {0};
        for (int pair_index = 0; pair_index < 10; ++pair_index) {
            const device uint* pair_words =
                threadgroup_words + pair_index * kPairPayloadWords;
            const device uint* even_weight = pair_words
                + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
            const device uint* odd_weight = pair_words
                + kWeightWordsPerBlock
                + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
            const device uchar* metadata =
                reinterpret_cast<const device uchar*>(
                    pair_words + 2 * kWeightWordsPerBlock)
                + projection * kRowsPerSIMD * 12;

            float even_values[8];
            const float even_input_sum = gemma4_cotiled_qkv_load_values(
                input,
                even_values);
            uint odd_pairs[kRowsPerSIMD];
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                result[row] += gemma4_cotiled_qkv_qdot_4bit(
                    even_weight[row * kWordsPerRowBlock],
                    even_values,
                    gemma4_cotiled_qkv_pair_scale(even_pair),
                    gemma4_cotiled_qkv_pair_bias(even_pair),
                    even_input_sum);
            }

            input += 256;
            float odd_values[8];
            const float odd_input_sum = gemma4_cotiled_qkv_load_values(
                input,
                odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_pairs[row];
                result[row] += gemma4_cotiled_qkv_qdot_4bit(
                    odd_weight[row * kWordsPerRowBlock],
                    odd_values,
                    gemma4_cotiled_qkv_pair_scale(odd_pair),
                    gemma4_cotiled_qkv_pair_bias(odd_pair),
                    odd_input_sum);
            }
            input += 256;
        }

        const device uint* tail_words =
            threadgroup_words + 10 * kPairPayloadWords;
        const device uint* tail_weight = tail_words
            + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
        const device uchar* tail_metadata =
            reinterpret_cast<const device uchar*>(
                tail_words + kWeightWordsPerBlock)
            + projection * kRowsPerSIMD * 8;
        float tail_values[8];
        const float tail_input_sum = gemma4_cotiled_qkv_load_values(
            input,
            tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_metadata = tail_metadata + row * 8;
            const uint metadata_index = row_metadata[tail_lane_offset]
                | (row_metadata[tail_lane_offset + 1] << 8);
            const uint pair = lut[metadata_index];
            result[row] += gemma4_cotiled_qkv_qdot_4bit(
                tail_weight[row * kWordsPerRowBlock],
                tail_values,
                gemma4_cotiled_qkv_pair_scale(pair),
                gemma4_cotiled_qkv_pair_bias(pair),
                tail_input_sum);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: gemma4CoTiledFixed12QKVHeader,
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
    private let coTiledFixed12: Gemma4CoTiledFixed12QKVPayload?
    private let useCoTiledFixed12: Bool
    private let verifyCoTiledFixed12Bits: Bool

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
        let useCoTiledFixed12 = gemma4QKVCoTiledFixed12Enabled()
        let verifyCoTiledFixed12Bits = gemma4VerifyQKVCoTiledFixed12Bits()
        self.coTiledFixed12 = useCoTiledFixed12 || verifyCoTiledFixed12Bits
            ? Gemma4CoTiledFixed12QKVPayload(
                projections: [
                    (q, qMetadata, 2),
                    (k, kMetadata, 1),
                    (v, vMetadata, 1),
                ],
                threadgroupCount: 1_024
            )
            : nil
        self.useCoTiledFixed12 = useCoTiledFixed12
        self.verifyCoTiledFixed12Bits = verifyCoTiledFixed12Bits
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [
            [1, 1, 8_192],
            [1, 1, 4_096],
            [1, 1, 4_096],
        ]
        let coTiledOutputs: [MLXArray]?
        if let coTiledFixed12 {
            coTiledOutputs = gemma4CoTiledFixed12IndexedSlidingQKV(
                [
                    coTiledFixed12.words,
                    qMetadata.lut,
                    kMetadata.lut,
                    vMetadata.lut,
                    input,
                ],
                grid: (32, 4_096, 1),
                threadGroup: (32, 4, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16]
            )
        } else {
            coTiledOutputs = nil
        }
        let needsCurrent = !useCoTiledFixed12
            || verifyCoTiledFixed12Bits
            || coTiledOutputs == nil
        let currentOutputs: [MLXArray]? = needsCurrent
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
        if verifyCoTiledFixed12Bits,
           let coTiledOutputs,
           let currentOutputs
        {
            for (candidate, reference) in zip(coTiledOutputs, currentOutputs) {
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "co-tiled fixed12 sliding QKV differs from current kernel"
                )
            }
        }
        let outputs = useCoTiledFixed12
            ? (coTiledOutputs ?? currentOutputs)
            : (currentOutputs ?? coTiledOutputs)
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

private let gemma4CoTiledFixed12IndexedFullQK = MLXFast.metalKernel(
    name: "gemma4_cotiled_fixed12_aligned_full_qk_qmv_5376_v1",
    inputNames: ["cotiled_payload", "q_lut", "k_lut", "x"],
    outputNames: ["q_output", "k_output"],
    source: """
        constexpr int kRowsPerSIMD = 4;
        constexpr int kWordsPerRowBlock = 32;
        constexpr int kSIMDGroups = 9;
        constexpr int kWeightWordsPerBlock =
            kSIMDGroups * kRowsPerSIMD * kWordsPerRowBlock;
        constexpr int kPairMetadataWords =
            kSIMDGroups * kRowsPerSIMD * 3;
        constexpr int kPairContentWords =
            2 * kWeightWordsPerBlock + kPairMetadataWords;
        constexpr int kPairPayloadWords =
            ((kPairContentWords + 31) / 32) * 32;
        constexpr int kTailMetadataWords =
            kSIMDGroups * kRowsPerSIMD * 2;
        constexpr int kTailPayloadWords =
            kWeightWordsPerBlock + kTailMetadataWords;
        constexpr int kThreadgroupContentWords =
            10 * kPairPayloadWords + kTailPayloadWords;
        constexpr int kWordsPerThreadgroup =
            ((kThreadgroupContentWords + 31) / 32) * 32;

        const int projection = simdgroup_index_in_threadgroup;
        const bool is_q = projection < 8;
        const int threadgroup_index = threadgroup_position_in_grid.y;
        const int output_row = is_q
            ? threadgroup_index * 32 + projection * kRowsPerSIMD
            : threadgroup_index * kRowsPerSIMD;
        const device uint* lut = is_q ? q_lut : k_lut;
        device bfloat* output = is_q ? q_output : k_output;

        const uint lane = thread_index_in_simdgroup;
        const uint lane_group = lane >> 3;
        const device bfloat* input = x + lane * 8;
        const device uint* threadgroup_words =
            cotiled_payload + threadgroup_index * kWordsPerThreadgroup;

        float result[kRowsPerSIMD] = {0};
        for (int pair_index = 0; pair_index < 10; ++pair_index) {
            const device uint* pair_words =
                threadgroup_words + pair_index * kPairPayloadWords;
            const device uint* even_weight = pair_words
                + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
            const device uint* odd_weight = pair_words
                + kWeightWordsPerBlock
                + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
            const device uchar* metadata =
                reinterpret_cast<const device uchar*>(
                    pair_words + 2 * kWeightWordsPerBlock)
                + projection * kRowsPerSIMD * 12;

            float even_values[8];
            const float even_input_sum = gemma4_cotiled_qkv_load_values(
                input,
                even_values);
            uint odd_pairs[kRowsPerSIMD];
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_metadata = metadata + row * 12;
                const uint even_low = row_metadata[lane_group];
                const uint middle = row_metadata[4 + lane_group];
                const uint odd_high = row_metadata[8 + lane_group];
                const uint even_index =
                    even_low | ((middle & 0x0f) << 8);
                const uint odd_index =
                    (middle >> 4) | (odd_high << 4);
                const uint even_pair = lut[even_index];
                odd_pairs[row] = lut[odd_index];
                result[row] += gemma4_cotiled_qkv_qdot_4bit(
                    even_weight[row * kWordsPerRowBlock],
                    even_values,
                    gemma4_cotiled_qkv_pair_scale(even_pair),
                    gemma4_cotiled_qkv_pair_bias(even_pair),
                    even_input_sum);
            }

            input += 256;
            float odd_values[8];
            const float odd_input_sum = gemma4_cotiled_qkv_load_values(
                input,
                odd_values);
            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const uint odd_pair = odd_pairs[row];
                result[row] += gemma4_cotiled_qkv_qdot_4bit(
                    odd_weight[row * kWordsPerRowBlock],
                    odd_values,
                    gemma4_cotiled_qkv_pair_scale(odd_pair),
                    gemma4_cotiled_qkv_pair_bias(odd_pair),
                    odd_input_sum);
            }
            input += 256;
        }

        const device uint* tail_words =
            threadgroup_words + 10 * kPairPayloadWords;
        const device uint* tail_weight = tail_words
            + projection * kRowsPerSIMD * kWordsPerRowBlock + lane;
        const device uchar* tail_metadata =
            reinterpret_cast<const device uchar*>(
                tail_words + kWeightWordsPerBlock)
            + projection * kRowsPerSIMD * 8;
        float tail_values[8];
        const float tail_input_sum = gemma4_cotiled_qkv_load_values(
            input,
            tail_values);
        const uint tail_lane_offset = lane_group << 1;
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            const device uchar* row_metadata = tail_metadata + row * 8;
            const uint metadata_index = row_metadata[tail_lane_offset]
                | (row_metadata[tail_lane_offset + 1] << 8);
            const uint pair = lut[metadata_index];
            result[row] += gemma4_cotiled_qkv_qdot_4bit(
                tail_weight[row * kWordsPerRowBlock],
                tail_values,
                gemma4_cotiled_qkv_pair_scale(pair),
                gemma4_cotiled_qkv_pair_bias(pair),
                tail_input_sum);
        }

        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) {
                output[output_row + row] = static_cast<bfloat>(result[row]);
            }
        }
        """,
    header: gemma4CoTiledFixed12QKVHeader,
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
    private let coTiledFixed12: Gemma4CoTiledFixed12QKVPayload?
    private let useCoTiledFixed12: Bool
    private let verifyCoTiledFixed12Bits: Bool

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
        let useCoTiledFixed12 = gemma4QKVCoTiledFixed12Enabled()
        let verifyCoTiledFixed12Bits = gemma4VerifyQKVCoTiledFixed12Bits()
        self.coTiledFixed12 = useCoTiledFixed12 || verifyCoTiledFixed12Bits
            ? Gemma4CoTiledFixed12QKVPayload(
                projections: [
                    (q, qMetadata, 8),
                    (k, kMetadata, 1),
                ],
                threadgroupCount: 512
            )
            : nil
        self.useCoTiledFixed12 = useCoTiledFixed12
        self.verifyCoTiledFixed12Bits = verifyCoTiledFixed12Bits
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.shape == [1, 1, 5_376])
        let outputShapes = [[1, 1, 16_384], [1, 1, 2_048]]
        let coTiledOutputs: [MLXArray]?
        if let coTiledFixed12 {
            coTiledOutputs = gemma4CoTiledFixed12IndexedFullQK(
                [
                    coTiledFixed12.words,
                    qMetadata.lut,
                    kMetadata.lut,
                    input,
                ],
                grid: (32, 4_608, 1),
                threadGroup: (32, 9, 1),
                outputShapes: outputShapes,
                outputDTypes: [.bfloat16, .bfloat16]
            )
        } else {
            coTiledOutputs = nil
        }
        let needsCurrent = !useCoTiledFixed12
            || verifyCoTiledFixed12Bits
            || coTiledOutputs == nil
        let currentOutputs: [MLXArray]? = needsCurrent
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
        if verifyCoTiledFixed12Bits,
           let coTiledOutputs,
           let currentOutputs
        {
            for (candidate, reference) in zip(coTiledOutputs, currentOutputs) {
                let matches = arrayEqual(
                    candidate.view(dtype: .uint16),
                    reference.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "co-tiled fixed12 full QK differs from current kernel"
                )
            }
        }
        let outputs = useCoTiledFixed12
            ? (coTiledOutputs ?? currentOutputs)
            : (currentOutputs ?? coTiledOutputs)
        guard let outputs else {
            preconditionFailure("full QK kernel was not selected")
        }
        return (outputs[0], outputs[1])
    }
}
