import MLX

/// Decode-only byte permutation for Gemma 4's 5,376 -> 21,504 gate projection.
/// Original projection tensors remain authoritative for prefill and rollback.
struct Gemma4KMajorGate: @unchecked Sendable {
    static let rows = 21_504
    static let rowBytes = 2_688
    static let groupsPerRow = 84
    static let rowsPerTile = 16
    static let blocks = 21

    let weightBytes: MLXArray
    let metadataIndices: MLXArray
    let lut: MLXArray

    init?(projection: FastQuantizedProjection, metadata: IndexedAffineMetadata) {
        guard supportsGemma4KMajorGate(projection: projection, metadata: metadata) else {
            return nil
        }
        let sourceWeight = projection.weight.view(dtype: .uint8).asArray(UInt8.self)
        let sourceIndices = metadata.indices.asArray(UInt16.self)
        var packedWeight = [UInt8](repeating: 0, count: sourceWeight.count)
        var packedIndices = [UInt16](repeating: 0, count: sourceIndices.count)
        for tile in 0..<(Self.rows / Self.rowsPerTile) {
            for block in 0..<Self.blocks {
                for rowInTile in 0..<Self.rowsPerTile {
                    let row = tile * Self.rowsPerTile + rowInTile
                    let weightSource = row * Self.rowBytes + block * 128
                    let weightTarget = ((tile * Self.blocks + block) * Self.rowsPerTile + rowInTile) * 128
                    packedWeight.replaceSubrange(weightTarget..<(weightTarget + 128), with: sourceWeight[weightSource..<(weightSource + 128)])
                    let indexSource = row * Self.groupsPerRow + block * 4
                    let indexTarget = ((tile * Self.blocks + block) * Self.rowsPerTile + rowInTile) * 4
                    packedIndices.replaceSubrange(indexTarget..<(indexTarget + 4), with: sourceIndices[indexSource..<(indexSource + 4)])
                }
            }
        }
        self.weightBytes = MLXArray(packedWeight, [Self.rows / Self.rowsPerTile, Self.blocks, Self.rowsPerTile, 128])
        self.metadataIndices = MLXArray(packedIndices, [Self.rows / Self.rowsPerTile, Self.blocks, Self.rowsPerTile, 4])
        self.lut = metadata.lut
        eval(weightBytes, metadataIndices)
    }

    /// Test/debug inverse; returns row-major weight bytes and metadata indices.
    func inverse() -> ([UInt8], [UInt16]) {
        let packedWeight = weightBytes.asArray(UInt8.self)
        let packedIndices = metadataIndices.asArray(UInt16.self)
        var weight = [UInt8](repeating: 0, count: packedWeight.count)
        var indices = [UInt16](repeating: 0, count: packedIndices.count)
        for tile in 0..<(Self.rows / Self.rowsPerTile) {
            for block in 0..<Self.blocks {
                for rowInTile in 0..<Self.rowsPerTile {
                    let row = tile * Self.rowsPerTile + rowInTile
                    let weightSource = ((tile * Self.blocks + block) * Self.rowsPerTile + rowInTile) * 128
                    let weightTarget = row * Self.rowBytes + block * 128
                    weight.replaceSubrange(weightTarget..<(weightTarget + 128), with: packedWeight[weightSource..<(weightSource + 128)])
                    let indexSource = ((tile * Self.blocks + block) * Self.rowsPerTile + rowInTile) * 4
                    let indexTarget = row * Self.groupsPerRow + block * 4
                    indices.replaceSubrange(indexTarget..<(indexTarget + 4), with: packedIndices[indexSource..<(indexSource + 4)])
                }
            }
        }
        return (weight, indices)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(supportsGemma4KMajorGateInput(input))
        return gemma4KMajorGateQMV(
            [weightBytes, metadataIndices, lut, input],
            grid: (32, Self.rows / Self.rowsPerTile, 1),
            threadGroup: (32, 4, 1),
            outputShapes: [[1, 1, Self.rows]], outputDTypes: [.bfloat16]
        )[0]
    }
}

func supportsGemma4KMajorGateInput(_ input: MLXArray) -> Bool {
    input.dtype == .bfloat16 && input.shape == [1, 1, 5_376]
}

func supportsGemma4KMajorGate(projection: FastQuantizedProjection, metadata: IndexedAffineMetadata) -> Bool {
    projection.bits == 4 && projection.groupSize == 64
        && projection.weight.dtype == .uint32 && projection.weight.shape == [21_504, 672]
        && projection.scales.dtype == .bfloat16 && projection.scales.shape == [21_504, 84]
        && projection.biases?.dtype == .bfloat16 && projection.biases?.shape == [21_504, 84]
        && metadata.indices.dtype == .uint16 && metadata.indices.shape == [21_504, 84]
        && metadata.lut.dtype == .uint32 && metadata.lut.ndim == 1
}

private let gemma4KMajorGateQMV = MLXFast.metalKernel(
    name: "gemma4_kmajor_gate_qmv_5376",
    inputNames: ["packed_weight", "packed_indices", "lut", "x"],
    outputNames: ["output"],
    source: """
        constexpr int kRows = 16;
        const int tile = threadgroup_position_in_grid.y;
        const int simd = simdgroup_index_in_threadgroup;
        const int lane = thread_index_in_simdgroup;
        float result[4] = {0};
        const device bfloat* input = x + lane * 8;
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_load_qmv_values(input, values);
            for (int local = 0; local < 4; ++local) {
                const int row = simd * 4 + local;
                const ulong base = ((tile * 21 + block) * kRows + row);
                const device uchar* weight = packed_weight + base * 128 + lane * 4;
                const ushort metadata_index = packed_indices[base * 4 + lane / 8];
                const uint pair = lut[metadata_index];
                result[local] += gemma4_qdot_4bit(weight, values,
                    gemma4_pair_scale(pair), gemma4_pair_bias(pair), input_sum);
            }
            input += 256;
        }
        for (int local = 0; local < 4; ++local) {
            result[local] = simd_sum(result[local]);
            if (lane == 0) output[tile * 16 + simd * 4 + local] = static_cast<bfloat>(result[local]);
        }
    """,
    header: """
        using namespace metal;
        inline float gemma4_pair_scale(uint p) { return float(as_type<bfloat>(ushort(p))); }
        inline float gemma4_pair_bias(uint p) { return float(as_type<bfloat>(ushort(p >> 16))); }
        inline float gemma4_load_qmv_values(const device bfloat* input, thread float* values) {
            float sum = 0;
            for (int i = 0; i < 8; i += 4) {
                sum += input[i] + input[i+1] + input[i+2] + input[i+3];
                values[i] = input[i]; values[i+1] = input[i+1] / 16.0f;
                values[i+2] = input[i+2] / 256.0f; values[i+3] = input[i+3] / 4096.0f;
            }
            return sum;
        }
        inline float gemma4_qdot_4bit(const device uchar* weight, const thread float* values,
            float scale, float bias, float input_sum) {
            const device ushort* packed = reinterpret_cast<const device ushort*>(weight);
            float accumulator = 0;
            for (int i = 0; i < 2; ++i)
                accumulator += values[4*i]*(packed[i]&0x000f) + values[4*i+1]*(packed[i]&0x00f0)
                    + values[4*i+2]*(packed[i]&0x0f00) + values[4*i+3]*(packed[i]&0xf000);
            return scale * accumulator + input_sum * bias;
        }
    """,
    ensureRowContiguous: true
)
