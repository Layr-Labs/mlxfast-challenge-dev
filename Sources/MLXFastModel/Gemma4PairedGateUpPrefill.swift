import Foundation
import MLX

private func gemma4PrefillEnvironmentFlag(_ name: String) -> Bool? {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return nil
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

/// Mirrors the NAX availability contract in the pinned MLX Metal backend.
/// NAX first shipped with macOS 26.2 and requires Apple GPU generation 17,
/// except phone-class (`p`) architectures, which require generation 18.
private let gemma4PairedGateUpPrefillNAXContractSatisfied: Bool = {
    guard #available(macOS 26.2, *) else {
        return false
    }
    let architecture = Array(GPU.deviceInfo().architecture)
    guard architecture.count >= 3,
          let tens = architecture[architecture.count - 3].wholeNumberValue,
          let ones = architecture[architecture.count - 2].wholeNumberValue
    else {
        return false
    }
    let generation = tens * 10 + ones
    let minimumGeneration = architecture.last == "p" ? 18 : 17
    return generation >= minimumGeneration
}()

private let gemma4PairedGateUpPrefillEnabled: Bool = {
    gemma4PrefillEnvironmentFlag("DARKBLOOM_PAIRED_GATE_UP_PREFILL")
        ?? gemma4PairedGateUpPrefillNAXContractSatisfied
}()

private let gemma4VerifyPairedGateUpPrefillBits: Bool = {
    gemma4PrefillEnvironmentFlag(
        "DARKBLOOM_VERIFY_PAIRED_GATE_UP_PREFILL_BITS"
    ) ?? false
}()

private let gemma4PairedGateUpPrefillNAXEnabled: Bool = {
    gemma4PairedGateUpPrefillNAXContractSatisfied
        && (gemma4PrefillEnvironmentFlag(
            "DARKBLOOM_PAIRED_GATE_UP_PREFILL_NAX"
        ) ?? true)
}()

/// Paired affine-4bit QMV for the short-prefill regime used by MLX for fewer
/// than six 5,376-wide rows on the ranked large Apple GPU. Gate and up share a
/// launch and only the BF16 GELU-product is written to device memory.
private let gemma4PairedGateUpPrefillQMV = MLXFast.metalKernel(
    name: "gemma4_paired_gate_up_prefill_qmv_5376_v1",
    inputNames: [
        "gate_weight", "gate_scales", "gate_biases",
        "up_weight", "up_scales", "up_biases", "x",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 21504;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kRowsPerSIMD = 4;

        const bool is_up = simdgroup_index_in_threadgroup == 1;
        const int output_row = threadgroup_position_in_grid.y * kRowsPerSIMD;
        const int sequence_row = threadgroup_position_in_grid.z;
        const device uint* weight = is_up ? up_weight : gate_weight;
        const device bfloat* scales = is_up ? up_scales : gate_scales;
        const device bfloat* biases = is_up ? up_biases : gate_biases;

        const device uchar* weight_bytes =
            reinterpret_cast<const device uchar*>(weight)
            + output_row * kWeightBytesPerRow
            + thread_index_in_simdgroup * 4;
        const device bfloat* row_scales =
            scales + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* row_biases =
            biases + output_row * kGroupsPerRow
            + thread_index_in_simdgroup / 8;
        const device bfloat* input =
            x + sequence_row * kInputWidth + thread_index_in_simdgroup * 8;

        float result[kRowsPerSIMD] = {0};
        for (int block = 0; block < 21; ++block) {
            float values[8];
            const float input_sum = gemma4_prefill_load_qmv_values(input, values);

            #pragma clang loop unroll(full)
            for (int row = 0; row < kRowsPerSIMD; ++row) {
                const device uchar* row_weight =
                    weight_bytes + row * kWeightBytesPerRow;
                const device bfloat* scale =
                    row_scales + row * kGroupsPerRow;
                const device bfloat* bias =
                    row_biases + row * kGroupsPerRow;
                result[row] += gemma4_prefill_qdot_4bit(
                    row_weight, values, scale[0], bias[0], input_sum);
            }

            weight_bytes += 128;
            row_scales += 4;
            row_biases += 4;
            input += 256;
        }

        threadgroup bfloat projections[2][kRowsPerSIMD];
        #pragma clang loop unroll(full)
        for (int row = 0; row < kRowsPerSIMD; ++row) {
            result[row] = simd_sum(result[row]);
            if (thread_index_in_simdgroup == 0) {
                projections[is_up ? 1 : 0][row] =
                    static_cast<bfloat>(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup_index_in_threadgroup == 0
            && thread_index_in_simdgroup < kRowsPerSIMD
        ) {
            const int row = thread_index_in_simdgroup;
            const bfloat gate = projections[0][row];
            const bfloat up = projections[1][row];
            activated[sequence_row * kOutputWidth + output_row + row] =
                gemma4_prefill_gelu_bf16(gate) * up;
        }
        """,
    header: """
        using namespace metal;

        inline float gemma4_prefill_load_qmv_values(
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

        inline float gemma4_prefill_qdot_4bit(
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

        inline bfloat gemma4_prefill_gelu_bf16(bfloat gate) {
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
            return scaled * shifted;
        }
        """,
    ensureRowContiguous: true
)

/// Long-prefill prototype. A 32x32 output tile loads each 32x32 activation
/// tile once, dequantizes matching gate/up tiles together, and keeps both
/// matrix accumulators in registers until the exact BF16 GELU-product store.
///
/// MLX's public CustomKernel surface does not expose its private NAX tile
/// helpers. This uses the public Metal simdgroup-matrix primitive (MPP), the
/// same accumulation machinery as MLX's non-NAX affine QMM implementation.
private let gemma4PairedGateUpPrefillMPP = MLXFast.metalKernel(
    name: "gemma4_paired_gate_up_prefill_mpp_32x32x32_v2",
    inputNames: [
        "gate_weight", "gate_scales", "gate_biases",
        "up_weight", "up_scales", "up_biases", "x", "sequence_length",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 21504;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kBM = 32;
        constexpr int kBN = 32;
        constexpr int kBK = 32;
        constexpr int kPaddedBK = 40;

        const int lane = thread_index_in_simdgroup;
        const int simd = simdgroup_index_in_threadgroup;
        const int linear_thread = simd * 32 + lane;
        const int output_tile = threadgroup_position_in_grid.y;
        const int sequence_tile = threadgroup_position_in_grid.z;
        const int output_base = output_tile * kBN;
        const int sequence_base = sequence_tile * kBM;

        threadgroup bfloat x_tile[kBM * kPaddedBK];
        threadgroup bfloat gate_tile[kBN * kPaddedBK];
        threadgroup bfloat up_tile[kBN * kPaddedBK];

        metal::simdgroup_matrix<float, 8, 8> gate_accumulators[4];
        metal::simdgroup_matrix<float, 8, 8> up_accumulators[4];
        #pragma clang loop unroll(full)
        for (int frag_index = 0; frag_index < 4; ++frag_index) {
            gate_accumulators[frag_index].thread_elements()[0] = 0.0f;
            gate_accumulators[frag_index].thread_elements()[1] = 0.0f;
            up_accumulators[frag_index].thread_elements()[0] = 0.0f;
            up_accumulators[frag_index].thread_elements()[1] = 0.0f;
        }

        const int matrix_row = (lane >> 2 & 4) + ((lane >> 1) & 3);
        const int matrix_col = ((lane >> 2) & 2) * 2 + (lane & 1) * 2;
        const int simd_row = (simd >> 1) * 16;
        const int simd_col = (simd & 1) * 16;

        for (int k_base = 0; k_base < kInputWidth; k_base += kBK) {
            // The 128-thread group cooperatively loads 1,024 activations and
            // both 1,024-element dequantized weight tiles.
            for (int index = linear_thread; index < kBM * kBK; index += 128) {
                const int row = index / kBK;
                const int col = index - row * kBK;
                const int sequence_row = sequence_base + row;
                x_tile[row * kPaddedBK + col] = sequence_row < sequence_length
                    ? x[sequence_row * kInputWidth + k_base + col]
                    : static_cast<bfloat>(0.0f);

                const int output_row = output_base + row;
                const int packed_offset =
                    output_row * kWeightBytesPerRow + (k_base + col) / 2;
                const int metadata_offset =
                    output_row * kGroupsPerRow + k_base / 64;
                const uchar gate_code =
                    reinterpret_cast<const device uchar*>(gate_weight)[packed_offset];
                const uchar up_code =
                    reinterpret_cast<const device uchar*>(up_weight)[packed_offset];
                const bfloat gate_scale = gate_scales[metadata_offset];
                const bfloat gate_bias = gate_biases[metadata_offset];
                const bfloat up_scale = up_scales[metadata_offset];
                const bfloat up_bias = up_biases[metadata_offset];
                if ((col & 1) == 0) {
                    gate_tile[row * kPaddedBK + col] =
                        gate_scale * (gate_code & 0x0f) + gate_bias;
                    up_tile[row * kPaddedBK + col] =
                        up_scale * (up_code & 0x0f) + up_bias;
                } else {
                    gate_tile[row * kPaddedBK + col] =
                        (gate_scale / static_cast<bfloat>(16.0f))
                            * (gate_code & 0xf0) + gate_bias;
                    up_tile[row * kPaddedBK + col] =
                        (up_scale / static_cast<bfloat>(16.0f))
                            * (up_code & 0xf0) + up_bias;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (int kk = 0; kk < kBK; kk += 8) {
                metal::simdgroup_matrix<float, 8, 8> activation_matrices[2];
                metal::simdgroup_matrix<float, 8, 8> gate_matrices[2];
                metal::simdgroup_matrix<float, 8, 8> up_matrices[2];

                #pragma clang loop unroll(full)
                for (int frag_index = 0; frag_index < 2; ++frag_index) {
                    const int row_offset = simd_row + frag_index * 8;
                    activation_matrices[frag_index].thread_elements()[0] =
                        static_cast<float>(
                            x_tile[(row_offset + matrix_row) * kPaddedBK
                                + kk + matrix_col]);
                    activation_matrices[frag_index].thread_elements()[1] =
                        static_cast<float>(
                            x_tile[(row_offset + matrix_row) * kPaddedBK
                                + kk + matrix_col + 1]);

                    const int col_offset = simd_col + frag_index * 8;
                    gate_matrices[frag_index].thread_elements()[0] =
                        static_cast<float>(
                            gate_tile[(col_offset + matrix_col) * kPaddedBK
                                + kk + matrix_row]);
                    gate_matrices[frag_index].thread_elements()[1] =
                        static_cast<float>(
                            gate_tile[(col_offset + matrix_col + 1) * kPaddedBK
                                + kk + matrix_row]);
                    up_matrices[frag_index].thread_elements()[0] =
                        static_cast<float>(
                            up_tile[(col_offset + matrix_col) * kPaddedBK
                                + kk + matrix_row]);
                    up_matrices[frag_index].thread_elements()[1] =
                        static_cast<float>(
                            up_tile[(col_offset + matrix_col + 1) * kPaddedBK
                                + kk + matrix_row]);
                }

                #pragma clang loop unroll(full)
                for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                    #pragma clang loop unroll(full)
                    for (int col_fragment = 0; col_fragment < 2; ++col_fragment) {
                        const int frag_index = row_fragment * 2 + col_fragment;
                        simdgroup_multiply_accumulate(
                            gate_accumulators[frag_index],
                            activation_matrices[row_fragment],
                            gate_matrices[col_fragment],
                            gate_accumulators[frag_index]);
                        simdgroup_multiply_accumulate(
                            up_accumulators[frag_index],
                            activation_matrices[row_fragment],
                            up_matrices[col_fragment],
                            up_accumulators[frag_index]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int sequence_row = sequence_base + simd_row
                + row_fragment * 8 + matrix_row;
            #pragma clang loop unroll(full)
            for (int col_fragment = 0; col_fragment < 2; ++col_fragment) {
                const int frag_index = row_fragment * 2 + col_fragment;
                #pragma clang loop unroll(full)
                for (int element = 0; element < 2; ++element) {
                    const int output_row = output_base + simd_col
                        + col_fragment * 8 + matrix_col + element;
                    if (sequence_row < sequence_length
                        && output_row < kOutputWidth
                    ) {
                        const bfloat gate = static_cast<bfloat>(
                            gate_accumulators[frag_index]
                                .thread_elements()[element]);
                        const bfloat up = static_cast<bfloat>(
                            up_accumulators[frag_index]
                                .thread_elements()[element]);
                        activated[sequence_row * kOutputWidth + output_row] =
                            gemma4_prefill_mpp_gelu_bf16(gate) * up;
                    }
                }
            }
        }
        """,
    header: """
        #include <metal_simdgroup>
        #include <metal_simdgroup_matrix>
        #include <metal_stdlib>
        using namespace metal;

        inline bfloat gemma4_prefill_mpp_gelu_bf16(bfloat gate) {
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
            return scaled * shifted;
        }
        """,
    ensureRowContiguous: true
)

/// M5-class long-prefill route using the same 16x32x16 MPP cooperative-tensor
/// operation and 64x64x64 geometry as MLX's pinned NAX affine QMM. Each
/// activation tile is loaded once while independent gate/up accumulator tiles
/// stay in registers; only the BF16 GELU-product reaches device memory.
private let gemma4PairedGateUpPrefillNAX = MLXFast.metalKernel(
    name: "gemma4_paired_gate_up_prefill_nax_64x64x64_v1",
    inputNames: [
        "gate_weight", "gate_scales", "gate_biases",
        "up_weight", "up_scales", "up_biases", "x", "sequence_length",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int kInputWidth = 5376;
        constexpr int kOutputWidth = 21504;
        constexpr int kGroupsPerRow = 84;
        constexpr int kWeightBytesPerRow = 2688;
        constexpr int kBM = 64;
        constexpr int kBN = 64;
        constexpr int kBK = 64;
        constexpr int kPaddedBK = 72;

        const int lane = thread_index_in_simdgroup;
        const int simd = simdgroup_index_in_threadgroup;
        const int linear_thread = simd * 32 + lane;
        const int output_base = threadgroup_position_in_grid.y * kBN;
        const int sequence_base = threadgroup_position_in_grid.z * kBM;
        const int simd_row = (simd >> 1) * 32;
        const int simd_col = (simd & 1) * 32;

        threadgroup bfloat x_tile[kBM * kPaddedBK];
        threadgroup bfloat gate_tile[kBN * kPaddedBK];
        threadgroup bfloat up_tile[kBN * kPaddedBK];

        metal::vec<float, 8> gate_accumulators[4];
        metal::vec<float, 8> up_accumulators[4];
        #pragma clang loop unroll(full)
        for (int frag_index = 0; frag_index < 4; ++frag_index) {
            gate_accumulators[frag_index] = metal::vec<float, 8>(0.0f);
            up_accumulators[frag_index] = metal::vec<float, 8>(0.0f);
        }

        const short qid = lane >> 2;
        const short frag_row = (qid & 4) | ((lane >> 1) & 3);
        const short frag_col = ((qid & 2) | (lane & 1)) * 4;

        for (int k_base = 0; k_base < kInputWidth; k_base += kBK) {
            for (int index = linear_thread; index < kBM * kBK; index += 128) {
                const int row = index / kBK;
                const int col = index - row * kBK;
                const int sequence_row = sequence_base + row;
                x_tile[row * kPaddedBK + col] = sequence_row < sequence_length
                    ? x[sequence_row * kInputWidth + k_base + col]
                    : static_cast<bfloat>(0.0f);
            }

            constexpr int kPackedColumns = kBK / 2;
            for (int index = linear_thread;
                 index < kBN * kPackedColumns;
                 index += 128
            ) {
                const int row = index / kPackedColumns;
                const int packed_col = index - row * kPackedColumns;
                const int output_row = output_base + row;
                const int packed_offset = output_row * kWeightBytesPerRow
                    + k_base / 2 + packed_col;
                const int metadata_offset = output_row * kGroupsPerRow
                    + k_base / 64;
                const uchar gate_code =
                    reinterpret_cast<const device uchar*>(gate_weight)[packed_offset];
                const uchar up_code =
                    reinterpret_cast<const device uchar*>(up_weight)[packed_offset];
                const bfloat gate_scale = gate_scales[metadata_offset];
                const bfloat gate_bias = gate_biases[metadata_offset];
                const bfloat up_scale = up_scales[metadata_offset];
                const bfloat up_bias = up_biases[metadata_offset];
                const int col = packed_col * 2;
                gate_tile[row * kPaddedBK + col] =
                    gate_scale * (gate_code & 0x0f) + gate_bias;
                gate_tile[row * kPaddedBK + col + 1] =
                    (gate_scale / static_cast<bfloat>(16.0f))
                        * (gate_code & 0xf0) + gate_bias;
                up_tile[row * kPaddedBK + col] =
                    up_scale * (up_code & 0x0f) + up_bias;
                up_tile[row * kPaddedBK + col + 1] =
                    (up_scale / static_cast<bfloat>(16.0f))
                        * (up_code & 0xf0) + up_bias;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (int kk = 0; kk < kBK; kk += 32) {
                metal::vec<bfloat, 8> activation_fragments[4];
                metal::vec<bfloat, 8> gate_fragments[4];
                metal::vec<bfloat, 8> up_fragments[4];

                #pragma clang loop unroll(full)
                for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                    #pragma clang loop unroll(full)
                    for (int k_fragment = 0; k_fragment < 2; ++k_fragment) {
                        const int frag_index = row_fragment * 2 + k_fragment;
                        const int row = simd_row + row_fragment * 16 + frag_row;
                        const int col = kk + k_fragment * 16 + frag_col;
                        #pragma clang loop unroll(full)
                        for (int element_row = 0; element_row < 2; ++element_row) {
                            #pragma clang loop unroll(full)
                            for (int element_col = 0; element_col < 4; ++element_col) {
                                activation_fragments[frag_index][
                                    element_row * 4 + element_col
                                ] = x_tile[
                                    (row + element_row * 8) * kPaddedBK
                                        + col + element_col
                                ];
                            }
                        }
                    }
                }

                #pragma clang loop unroll(full)
                for (int col_fragment = 0; col_fragment < 2; ++col_fragment) {
                    #pragma clang loop unroll(full)
                    for (int k_fragment = 0; k_fragment < 2; ++k_fragment) {
                        const int frag_index = col_fragment * 2 + k_fragment;
                        const int row = simd_col + col_fragment * 16 + frag_row;
                        const int col = kk + k_fragment * 16 + frag_col;
                        #pragma clang loop unroll(full)
                        for (int element_row = 0; element_row < 2; ++element_row) {
                            #pragma clang loop unroll(full)
                            for (int element_col = 0; element_col < 4; ++element_col) {
                                const int element = element_row * 4 + element_col;
                                const int offset =
                                    (row + element_row * 8) * kPaddedBK
                                        + col + element_col;
                                gate_fragments[frag_index][element] = gate_tile[offset];
                                up_fragments[frag_index][element] = up_tile[offset];
                            }
                        }
                    }
                }

                #pragma clang loop unroll(full)
                for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                    #pragma clang loop unroll(full)
                    for (int k_fragment = 0; k_fragment < 2; ++k_fragment) {
                        gemma4_prefill_nax_mma_pair(
                            gate_accumulators[row_fragment * 2],
                            gate_accumulators[row_fragment * 2 + 1],
                            activation_fragments[row_fragment * 2 + k_fragment],
                            gate_fragments[k_fragment],
                            gate_fragments[2 + k_fragment]);
                        gemma4_prefill_nax_mma_pair(
                            up_accumulators[row_fragment * 2],
                            up_accumulators[row_fragment * 2 + 1],
                            activation_fragments[row_fragment * 2 + k_fragment],
                            up_fragments[k_fragment],
                            up_fragments[2 + k_fragment]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            #pragma clang loop unroll(full)
            for (int col_fragment = 0; col_fragment < 2; ++col_fragment) {
                const int frag_index = row_fragment * 2 + col_fragment;
                #pragma clang loop unroll(full)
                for (int element_row = 0; element_row < 2; ++element_row) {
                    const int sequence_row = sequence_base + simd_row
                        + row_fragment * 16 + frag_row + element_row * 8;
                    #pragma clang loop unroll(full)
                    for (int element_col = 0; element_col < 4; ++element_col) {
                        const int output_row = output_base + simd_col
                            + col_fragment * 16 + frag_col + element_col;
                        if (sequence_row < sequence_length
                            && output_row < kOutputWidth
                        ) {
                            const int element = element_row * 4 + element_col;
                            const bfloat gate = static_cast<bfloat>(
                                gate_accumulators[frag_index][element]);
                            const bfloat up = static_cast<bfloat>(
                                up_accumulators[frag_index][element]);
                            activated[sequence_row * kOutputWidth + output_row] =
                                gemma4_prefill_nax_gelu_bf16(gate) * up;
                        }
                    }
                }
            }
        }
        """,
    header: """
        #include <metal_simdgroup>
        #include <metal_stdlib>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        inline void gemma4_prefill_nax_mma_pair(
            thread metal::vec<float, 8>& output0,
            thread metal::vec<float, 8>& output1,
            const thread metal::vec<bfloat, 8>& lhs,
            const thread metal::vec<bfloat, 8>& rhs0,
            const thread metal::vec<bfloat, 8>& rhs1
        ) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16,
                32,
                16,
                false,
                true,
                true,
                mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            auto a = op.template get_left_input_cooperative_tensor<
                bfloat, bfloat, float>();
            auto b = op.template get_right_input_cooperative_tensor<
                bfloat, bfloat, float>();
            auto c = op.template get_destination_cooperative_tensor<
                decltype(a), decltype(b), float>();
            #pragma clang loop unroll(full)
            for (int index = 0; index < 8; ++index) {
                a[index] = lhs[index];
                b[index] = rhs0[index];
                b[8 + index] = rhs1[index];
                c[index] = output0[index];
                c[8 + index] = output1[index];
            }
            op.run(a, b, c);
            #pragma clang loop unroll(full)
            for (int index = 0; index < 8; ++index) {
                output0[index] = c[index];
                output1[index] = c[8 + index];
            }
        }

        inline bfloat gemma4_prefill_nax_gelu_bf16(bfloat gate) {
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
            return scaled * shifted;
        }
        """,
    ensureRowContiguous: true
)

struct PairedGateUpPrefillProjection: @unchecked Sendable {
    let gate: FastQuantizedProjection
    let up: FastQuantizedProjection
    let verifyBits: Bool

    init?(gate: FastQuantizedProjection, up: FastQuantizedProjection) {
        guard gemma4PairedGateUpPrefillEnabled
                || gemma4VerifyPairedGateUpPrefillBits,
              supportsGemma4FusedGateUp(gate: gate, up: up),
              gate.biases != nil,
              up.biases != nil
        else {
            return nil
        }
        self.gate = gate
        self.up = up
        self.verifyBits = gemma4VerifyPairedGateUpPrefillBits
    }

    func supports(_ input: MLXArray) -> Bool {
        input.dtype == .bfloat16
            && input.ndim == 3
            && input.dim(0) == 1
            && input.dim(1) > 1
            && input.dim(2) == 5_376
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(supports(input))
        guard let gateBiases = gate.biases, let upBiases = up.biases else {
            preconditionFailure("paired gate/up prefill requires affine biases")
        }
        let length = input.dim(1)
        let outputShape = [1, length, gate.weight.dim(0)]
        let candidate: MLXArray
        if length < 6 {
            candidate = gemma4PairedGateUpPrefillQMV(
                [
                    gate.weight, gate.scales, gateBiases,
                    up.weight, up.scales, upBiases, input,
                ],
                grid: (32, gate.weight.dim(0) / 2, length),
                threadGroup: (32, 2, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        } else if gemma4PairedGateUpPrefillNAXEnabled {
            let sequenceTiles = (length + 63) / 64
            let outputTiles = (gate.weight.dim(0) + 63) / 64
            candidate = gemma4PairedGateUpPrefillNAX(
                [
                    gate.weight, gate.scales, gateBiases,
                    up.weight, up.scales, upBiases, input, Int32(length),
                ],
                grid: (32, outputTiles * 4, sequenceTiles),
                threadGroup: (32, 4, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            let sequenceTiles = (length + 31) / 32
            let outputTiles = (gate.weight.dim(0) + 31) / 32
            candidate = gemma4PairedGateUpPrefillMPP(
                [
                    gate.weight, gate.scales, gateBiases,
                    up.weight, up.scales, upBiases, input, Int32(length),
                ],
                grid: (32, outputTiles * 4, sequenceTiles),
                threadGroup: (32, 4, 1),
                outputShapes: [outputShape],
                outputDTypes: [.bfloat16]
            )[0]
        }

        guard verifyBits else {
            return candidate
        }
        let gateOutput = gate(input)
        let upOutput = up(input)
        let reference = 0.5 * gateOutput * (
            1 + tanh(
                sqrt(2 / Float.pi)
                    * (gateOutput
                        + 0.044715 * gateOutput * gateOutput * gateOutput)
            )
        ) * upOutput
        let matches = arrayEqual(
            candidate.view(dtype: .uint16),
            reference.view(dtype: .uint16)
        )
        eval(matches)
        precondition(
            matches.item(Bool.self),
            "paired gate/up prefill differs from stock BF16 activation"
        )
        // Verifier mode deliberately propagates the stock tensor so full-model
        // checks isolate this boundary from downstream near ties.
        return reference
    }
}
