import Foundation
import MLX

/// One-dispatch decode attention input preparation: the indexed Q/K/V (or
/// full-attention Q/K) QMV projections and the fused RMSNorm/RoPE preparation
/// collapse into a single kernel per layer per decode step.
///
/// Bit-exactness is by construction, not by re-derivation:
///
/// - Each projection row is computed by one SIMD group with the promoted
///   standalone QKV kernel's exact lane-to-element mapping, sequential
///   21-block accumulation, `simd_sum` tree, and BF16 store (staged in
///   threadgroup memory instead of device memory).
/// - The RMS reduction, normalization, norm-weight product, and RoPE phases
///   replicate the promoted preparation kernel's arithmetic on the first
///   `headDim / 4` threads; the extra SIMD groups that exist for QMV
///   throughput contribute zero-valued `local_sums` entries exactly like the
///   lanes past the active SIMD count in the promoted kernel's fixed
///   32-entry reduction.
///
/// The removed work per sliding layer is one kernel launch plus the device
/// round-trip of the 16,384-element raw Q/K/V intermediate (write, then
/// re-read by the preparation kernel); full-attention layers remove the same
/// launch and an 18,432-element round-trip.
private func gemma4FusedQKVAttentionEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private func makeGemma4FusedQKVAttentionRMSKernel(
    name: String,
    headDim: Int,
    kvHeads: Int,
    hasSeparateV: Bool
) -> MLXFast.MLXFastKernel {
    precondition(
        (headDim == 256 && kvHeads == 16 && hasSeparateV)
            || (headDim == 512 && kvHeads == 4 && !hasSeparateV)
    )
    let inputNames = hasSeparateV
        ? [
            "q_weight", "q_indices", "q_lut",
            "k_weight", "k_indices", "k_lut",
            "v_weight", "v_indices", "v_lut", "x",
            "q_norm", "k_norm", "position", "rope_cosines", "rope_sines",
        ]
        : [
            "q_weight", "q_indices", "q_lut",
            "k_weight", "k_indices", "k_lut", "x",
            "q_norm", "k_norm", "position", "rope_cosines", "rope_sines",
        ]
    let projectionSelect = hasSeparateV
        ? """
            const device uint* weight = is_q
                ? q_weight
                : (is_k ? k_weight : v_weight);
            const device ushort* indices = is_q
                ? q_indices
                : (is_k ? k_indices : v_indices);
            const device uint* lut = is_q
                ? q_lut
                : (is_k ? k_lut : v_lut);
        """
        : """
            const device uint* weight = is_q ? q_weight : k_weight;
            const device ushort* indices = is_q ? q_indices : k_indices;
            const device uint* lut = is_q ? q_lut : k_lut;
        """
    return MLXFast.metalKernel(
        name: name,
        inputNames: inputNames,
        outputNames: ["queries", "combined_kv"],
        source: """
            constexpr uint kHeadDim = \(headDim);
            constexpr uint kQHeads = 32;
            constexpr uint kKVHeads = \(kvHeads);
            constexpr uint kKVSlabElements = kKVHeads * kHeadDim;
            constexpr bool kSharesFullKVReduction = \(hasSeparateV ? "false" : "true");
            constexpr uint kGroupsPerRow = 84;
            constexpr uint kWeightBytesPerRow = 2688;
            constexpr uint kRowsPerSIMD = 4;
            constexpr uint kSIMDSize = 32;
            constexpr uint kSIMDGroups = 32;
            constexpr uint kReads = 4;
            constexpr uint kRMSThreads = kHeadDim / kReads;
            constexpr uint kRMSSIMDGroups = kRMSThreads / kSIMDSize;
            constexpr uint kRowsPerSIMDGroup = kHeadDim / kSIMDGroups;
            constexpr uint kRowIterations = kRowsPerSIMDGroup / kRowsPerSIMD;

            const uint combined_row = threadgroup_position_in_grid.y;
            const bool is_q = combined_row < kQHeads;
            const bool is_k = !is_q && (
                kSharesFullKVReduction
                    || combined_row < kQHeads + kKVHeads);
            const uint head = is_q
                ? combined_row
                : (is_k ? combined_row - kQHeads
                        : combined_row - kQHeads - kKVHeads);

            \(projectionSelect)

            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const uint linear = simd * kSIMDSize + lane;

            threadgroup float inverse_mean[1];
            threadgroup float local_sums[kSIMDSize];
            threadgroup bfloat raw_row[kHeadDim];
            threadgroup bfloat normalized_row[kHeadDim];

            // Phase 1: this head's projection rows, one SIMD group per row
            // with the promoted QKV kernel's lane mapping, block order, and
            // simd_sum reduction.
            const uint head_row_base = head * kHeadDim;
            for (uint iteration = 0; iteration < kRowIterations; ++iteration) {
                const uint local_row =
                    simd * kRowsPerSIMDGroup + iteration * kRowsPerSIMD;
                const uint output_row = head_row_base + local_row;
                const device uchar* weight_bytes =
                    reinterpret_cast<const device uchar*>(weight)
                    + output_row * kWeightBytesPerRow
                    + lane * 4;
                const device ushort* row_indices =
                    indices + output_row * kGroupsPerRow + lane / 8;
                const device bfloat* input = x + lane * 8;

                float result[kRowsPerSIMD] = {0};
                for (int block = 0; block < 21; ++block) {
                    float values[8];
                    const float input_sum =
                        gemma4_fused_qkv_prep_load_values(input, values);

                    for (uint row = 0; row < kRowsPerSIMD; ++row) {
                        const device uchar* row_weight =
                            weight_bytes + row * kWeightBytesPerRow;
                        const ushort metadata_index =
                            row_indices[row * kGroupsPerRow];
                        const uint pair = lut[metadata_index];
                        result[row] += gemma4_fused_qkv_prep_qdot_4bit(
                            row_weight,
                            values,
                            gemma4_fused_qkv_prep_pair_scale(pair),
                            gemma4_fused_qkv_prep_pair_bias(pair),
                            input_sum);
                    }

                    weight_bytes += 128;
                    row_indices += 4;
                    input += 256;
                }

                for (uint row = 0; row < kRowsPerSIMD; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (lane == 0) {
                        raw_row[local_row + row] =
                            static_cast<bfloat>(result[row]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: RMS reduction on the first kRMSThreads threads,
            // replicating the promoted preparation kernel's partial-sum tree.
            float accumulator = 0;
            if (linear < kRMSThreads) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = raw_row[linear * kReads + index];
                    accumulator += value * value;
                }
            }
            accumulator = simd_sum(accumulator);

            if (simd == 0) {
                local_sums[lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0 && simd < kRMSSIMDGroups) {
                local_sums[simd] = accumulator;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                accumulator = simd_sum(local_sums[lane]);
                if (lane == 0) {
                    inverse_mean[0] = metal::precise::rsqrt(
                        accumulator / kHeadDim + 1.0e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: normalization and norm-weight product, staged for RoPE.
            const bool has_weight = is_q || is_k;
            const device bfloat* norm_weight = is_q ? q_norm : k_norm;
            device bfloat* output = is_q
                ? queries + head * kHeadDim
                : combined_kv
                    + (is_k ? 0 : kKVSlabElements)
                    + head * kHeadDim;
            if (linear < kRMSThreads) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = linear * kReads + index;
                    const bfloat normalized = static_cast<bfloat>(
                        raw_row[dimension] * inverse_mean[0]);
                    const bfloat weighted = has_weight
                        ? norm_weight[dimension] * normalized
                        : static_cast<bfloat>(1.0f) * normalized;
                    normalized_row[dimension] = weighted;
                    if (!has_weight) {
                        output[dimension] = weighted;
                    }
                    if (kSharesFullKVReduction && is_k) {
                        combined_kv[
                            kKVSlabElements + head * kHeadDim + dimension
                        ] = static_cast<bfloat>(1.0f) * normalized;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 4: RoPE from the precomputed tables, replicating the
            // promoted preparation kernel's pair mapping.
            if (has_weight && linear < kRMSThreads) {
                constexpr uint kPairs = kHeadDim / 2;
                for (uint pair = linear; pair < kPairs; pair += kRMSThreads) {
                    const uint rope_index =
                        static_cast<uint>(position) * kPairs + pair;
                    const float cosine = rope_cosines[rope_index];
                    const float sine = rope_sines[rope_index];
                    const float left = static_cast<float>(normalized_row[pair]);
                    const float right = static_cast<float>(
                        normalized_row[pair + kPairs]);
                    output[pair] = static_cast<bfloat>(
                        left * cosine - right * sine);
                    output[pair + kPairs] = static_cast<bfloat>(
                        left * sine + right * cosine);
                }
            }
            """,
        header: """
            using namespace metal;

            inline float gemma4_fused_qkv_prep_pair_scale(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair)));
            }

            inline float gemma4_fused_qkv_prep_pair_bias(uint pair) {
                return static_cast<float>(
                    as_type<bfloat>(static_cast<ushort>(pair >> 16)));
            }

            inline float gemma4_fused_qkv_prep_load_values(
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

            inline float gemma4_fused_qkv_prep_qdot_4bit(
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
}

private let gemma4FusedSlidingQKVAttentionRMS = makeGemma4FusedQKVAttentionRMSKernel(
    name: "gemma4_indexed_sliding_qkv_rms_rope_combined_kv_256_v1",
    headDim: 256,
    kvHeads: 16,
    hasSeparateV: true
)

private let gemma4FusedFullQKAttentionRMS = makeGemma4FusedQKVAttentionRMSKernel(
    name: "gemma4_indexed_full_qk_rms_rope_shared_combined_kv_512_v1",
    headDim: 512,
    kvHeads: 4,
    hasSeparateV: false
)

struct FusedQKVAttentionRMSPreparation: @unchecked Sendable {
    let isSliding: Bool
    let headDim: Int
    let kvHeads: Int
    private let slidingProjection: FusedSlidingQKVProjection?
    private let fullProjection: FusedFullQKProjection?
    private let preparation: FusedAttentionRMSPreparation
    private let enabled: Bool
    private let verifyBits: Bool

    init?(
        fusedQKV: FusedSlidingQKVProjection?,
        fusedQK: FusedFullQKProjection?,
        preparation: FusedAttentionRMSPreparation?
    ) {
        let enabled = gemma4FusedQKVAttentionEnvironmentFlag(
            "MLXFAST_FUSED_QKV_ATTENTION_RMS",
            default: true
        )
        let verifyBits = gemma4FusedQKVAttentionEnvironmentFlag(
            "MLXFAST_VERIFY_FUSED_QKV_ATTENTION_RMS_BITS",
            default: false
        )
        guard enabled || verifyBits, let preparation else { return nil }
        if preparation.isSliding {
            guard preparation.headDim == 256,
                  preparation.kvHeads == 16,
                  let fusedQKV
            else { return nil }
            self.slidingProjection = fusedQKV
            self.fullProjection = nil
        } else {
            guard preparation.headDim == 512,
                  preparation.kvHeads == 4,
                  let fusedQK
            else { return nil }
            self.slidingProjection = nil
            self.fullProjection = fusedQK
        }
        self.isSliding = preparation.isSliding
        self.headDim = preparation.headDim
        self.kvHeads = preparation.kvHeads
        self.preparation = preparation
        self.enabled = enabled
        self.verifyBits = verifyBits
    }

    func supports(offset: Int) -> Bool {
        preparation.supports(offset: offset)
    }

    /// Queries `[1,32,1,D]` plus the K/V-major combined slab `[2,1,Hkv,1,D]`,
    /// shaped identically to `FusedAttentionRMSPreparation.callCombined` so
    /// `Gemma4CombinedKVCache.updateCombined` consumes it unchanged.
    func callAsFunction(
        _ hidden: MLXArray,
        offset: Int
    ) -> (queries: MLXArray, combinedKV: MLXArray) {
        precondition(hidden.dtype == .bfloat16)
        precondition(hidden.shape == [1, 1, 5_376])
        precondition(supports(offset: offset))

        let position = preparation.positionViews[offset]
        let inputs: [MLXArray]
        let kernel: MLXFast.MLXFastKernel
        if let slidingProjection {
            kernel = gemma4FusedSlidingQKVAttentionRMS
            inputs = [
                slidingProjection.q.weight,
                slidingProjection.qMetadata.indices,
                slidingProjection.qMetadata.lut,
                slidingProjection.k.weight,
                slidingProjection.kMetadata.indices,
                slidingProjection.kMetadata.lut,
                slidingProjection.v.weight,
                slidingProjection.vMetadata.indices,
                slidingProjection.vMetadata.lut,
                hidden,
                preparation.qNormWeight,
                preparation.kNormWeight,
                position,
                preparation.ropeCosines,
                preparation.ropeSines,
            ]
        } else {
            kernel = gemma4FusedFullQKAttentionRMS
            inputs = [
                fullProjection!.q.weight,
                fullProjection!.qMetadata.indices,
                fullProjection!.qMetadata.lut,
                fullProjection!.k.weight,
                fullProjection!.kMetadata.indices,
                fullProjection!.kMetadata.lut,
                hidden,
                preparation.qNormWeight,
                preparation.kNormWeight,
                position,
                preparation.ropeCosines,
                preparation.ropeSines,
            ]
        }

        let rows = 32 + (isSliding ? 2 * kvHeads : kvHeads)
        let outputs = kernel(
            inputs,
            grid: (1_024, rows, 1),
            threadGroup: (1_024, 1, 1),
            outputShapes: [
                [1, 32, 1, headDim],
                [2, 1, kvHeads, 1, headDim],
            ],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        var queries = outputs[0]
        var combinedKV = outputs[1]

        if verifyBits {
            let reference: (queries: MLXArray, combinedKV: MLXArray)
            if let slidingProjection {
                let raw = slidingProjection(hidden)
                reference = preparation.callCombined(
                    rawQueries: raw.0,
                    rawKeys: raw.1,
                    rawValues: raw.2,
                    offset: offset
                )
            } else {
                let raw = fullProjection!(hidden)
                reference = preparation.callCombined(
                    rawQueries: raw.0,
                    rawKeys: raw.1,
                    rawValues: nil,
                    offset: offset
                )
            }
            let queriesMatch = arrayEqual(
                queries.view(dtype: .uint16),
                reference.queries.view(dtype: .uint16)
            )
            let combinedKVMatches = arrayEqual(
                combinedKV.view(dtype: .uint16),
                reference.combinedKV.view(dtype: .uint16)
            )
            eval(queriesMatch, combinedKVMatches)
            precondition(
                queriesMatch.item(Bool.self),
                "fused QKV attention preparation queries differ from the "
                    + "two-kernel path"
            )
            precondition(
                combinedKVMatches.item(Bool.self),
                "fused QKV attention preparation combined KV differs from "
                    + "the two-kernel path"
            )
            if !enabled {
                queries = reference.queries
                combinedKV = reference.combinedKV
            }
        }

        return (queries, combinedKV)
    }
}
