import Foundation
import MLX
import MLXNN

private func gemma4TokenIngressEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool = false
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4FusedTokenIngressFeatureEnabled =
    gemma4TokenIngressEnvironmentFlag(
        "DARKBLOOM_FUSED_TOKEN_INGRESS",
        default: true
    )

private let gemma4VerifyFusedTokenIngressBits =
    gemma4TokenIngressEnvironmentFlag("DARKBLOOM_VERIFY_FUSED_TOKEN_INGRESS_BITS")

@inline(__always)
func gemma4FusedTokenIngressEnabled() -> Bool {
    gemma4FusedTokenIngressFeatureEnabled
}

@inline(__always)
func gemma4FusedTokenIngressVerificationEnabled() -> Bool {
    gemma4VerifyFusedTokenIngressBits
}

private let gemma4FusedTokenIngressKernel = MLXFast.metalKernel(
    name: "gemma4_fused_token_ingress_262144x5376_v1",
    inputNames: [
        "embedding_weight", "embedding_scales", "embedding_biases",
        "input_ids", "embedding_scale", "input_norm_weight",
    ],
    outputNames: ["scaled_residual", "normalized_input"],
    source: """
        constexpr uint kWidth = 5376;
        constexpr uint kGroupsPerRow = 84;
        constexpr uint kWeightBytesPerRow = 2688;
        constexpr uint kReads = 4;
        constexpr uint kThreads = 1024;
        constexpr uint kSIMDSize = 32;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;
        const uint token = static_cast<uint>(input_ids[0]);

        const device uchar* row_weight =
            reinterpret_cast<const device uchar*>(embedding_weight)
            + static_cast<size_t>(token) * kWeightBytesPerRow;
        const device bfloat* row_scales =
            embedding_scales + static_cast<size_t>(token) * kGroupsPerRow;
        const device bfloat* row_biases =
            embedding_biases + static_cast<size_t>(token) * kGroupsPerRow;
        const bfloat scale = embedding_scale;

        threadgroup bfloat scaled_row[kWidth];
        threadgroup float inverse_mean[1];
        threadgroup float local_sums[kSIMDSize];

        // Preserve the two materialized BF16 boundaries in the stock path:
        // affine dequantization first, then the embedding-scale multiply.
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat dequantized = gemma4_ingress_dequantize(
                        row_weight, row_scales, row_biases, dimension);
                    const bfloat scaled = dequantized * scale;
                    scaled_row[dimension] = scaled;
                    scaled_residual[dimension] = scaled;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat dequantized = gemma4_ingress_dequantize(
                            row_weight, row_scales, row_biases, dimension);
                        const bfloat scaled = dequantized * scale;
                        scaled_row[dimension] = scaled;
                        scaled_residual[dimension] = scaled;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Clone pinned rms_looped<bfloat,4> for width 5376. Threads 0...319
        // consume a second four-value block after their first one.
        float accumulator = 0;
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = scaled_row[base + index];
                    accumulator += value * value;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    if (base + index < kWidth) {
                        const float value = scaled_row[base + index];
                        accumulator += value * value;
                    }
                }
            }
        }
        accumulator = simd_sum(accumulator);

        if (simd_group == 0) {
            local_sums[simd_lane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = accumulator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            accumulator = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                inverse_mean[0] = metal::precise::rsqrt(
                    accumulator / kWidth + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Match RMSNorm's BF16 unit-normalization boundary before the learned
        // BF16 input-layernorm weight multiply.
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat unit_normalized = static_cast<bfloat>(
                        scaled_row[dimension] * inverse_mean[0]);
                    normalized_input[dimension] =
                        input_norm_weight[dimension] * unit_normalized;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat unit_normalized = static_cast<bfloat>(
                            scaled_row[dimension] * inverse_mean[0]);
                        normalized_input[dimension] =
                            input_norm_weight[dimension] * unit_normalized;
                    }
                }
            }
        }
        """,
    header: """
        using namespace metal;

        inline bfloat gemma4_ingress_dequantize(
            const device uchar* weight,
            const device bfloat* scales,
            const device bfloat* biases,
            uint dimension
        ) {
            const uchar packed = weight[dimension >> 1];
            const uchar quantized = (dimension & 1) == 0
                ? (packed & 0x0f)
                : (packed >> 4);
            const uint group = dimension >> 6;
            return scales[group] * quantized + biases[group];
        }
        """,
    ensureRowContiguous: true
)

struct Gemma4FusedTokenIngressResult {
    let scaledResidual: MLXArray
    let normalizedInput: MLXArray
}

struct Gemma4FusedTokenIngress: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let embeddingScale: MLXArray
    let inputNormWeight: MLXArray

    init?(
        embedding: Embedding,
        embedScale: Float,
        inputNormWeight: MLXArray,
        eps: Float
    ) {
        guard let embedding = embedding as? QuantizedEmbedding,
              embedding.mode == .affine,
              embedding.groupSize == 64,
              embedding.bits == 4,
              embedding.weight.dtype == .uint32,
              embedding.weight.shape == [262_144, 672],
              embedding.scales.dtype == .bfloat16,
              embedding.scales.shape == [262_144, 84],
              let biases = embedding.biases,
              biases.dtype == .bfloat16,
              biases.shape == [262_144, 84],
              inputNormWeight.dtype == .bfloat16,
              inputNormWeight.shape == [5_376],
              eps == 1.0e-6
        else {
            return nil
        }
        self.weight = embedding.weight
        self.scales = embedding.scales
        self.biases = biases
        self.embeddingScale = MLXArray(embedScale, dtype: .bfloat16)
        self.inputNormWeight = inputNormWeight
    }

    func supports(_ inputIDs: MLXArray) -> Bool {
        inputIDs.dtype == .int32 && inputIDs.shape == [1, 1]
    }

    func callAsFunction(_ inputIDs: MLXArray) -> Gemma4FusedTokenIngressResult {
        precondition(supports(inputIDs))
        let outputs = gemma4FusedTokenIngressKernel(
            [weight, scales, biases, inputIDs, embeddingScale, inputNormWeight],
            grid: (1_024, 1, 1),
            threadGroup: (1_024, 1, 1),
            outputShapes: [[1, 1, 5_376], [1, 1, 5_376]],
            outputDTypes: [.bfloat16, .bfloat16]
        )
        return Gemma4FusedTokenIngressResult(
            scaledResidual: outputs[0],
            normalizedInput: outputs[1]
        )
    }

    func verifyRawBits(
        _ candidate: Gemma4FusedTokenIngressResult,
        stockScaledResidual: MLXArray,
        stockNormalizedInput: MLXArray
    ) {
        precondition(stockScaledResidual.dtype == .bfloat16)
        precondition(stockScaledResidual.shape == [1, 1, 5_376])
        precondition(stockNormalizedInput.dtype == .bfloat16)
        precondition(stockNormalizedInput.shape == [1, 1, 5_376])

        let residualMatches = arrayEqual(
            candidate.scaledResidual.view(dtype: .uint16),
            stockScaledResidual.view(dtype: .uint16)
        )
        let normalizedMatches = arrayEqual(
            candidate.normalizedInput.view(dtype: .uint16),
            stockNormalizedInput.view(dtype: .uint16)
        )
        eval(residualMatches, normalizedMatches)
        precondition(
            residualMatches.item(Bool.self),
            "fused token ingress scaled residual differs from stock embedding path"
        )
        precondition(
            normalizedMatches.item(Bool.self),
            "fused token ingress normalized input differs from stock first RMSNorm"
        )
    }
}
