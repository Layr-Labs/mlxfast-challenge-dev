import Foundation
import MLX

private func gemma4DecodeBoundaryEnvironmentFlag(
    _ name: String, default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

// Kept off until raw-bit and isolated timing qualification is complete.
private let gemma4DirectDecodeBoundaryRMSDefault = false
private let gemma4DirectDecodeBoundaryRMS = gemma4DecodeBoundaryEnvironmentFlag(
    "DARKBLOOM_DIRECT_DECODE_BOUNDARY_RMS",
    default: gemma4DirectDecodeBoundaryRMSDefault
)
private let gemma4VerifyDirectDecodeBoundaryRMSBits =
    gemma4DecodeBoundaryEnvironmentFlag(
        "DARKBLOOM_VERIFY_DIRECT_DECODE_BOUNDARY_RMS_BITS", default: false
    )

private func gemma4ReplacingFirst(
    _ source: String, _ target: String, with replacement: String
) -> String {
    guard let range = source.range(of: target) else {
        preconditionFailure("missing boundary kernel source fragment")
    }
    var result = source
    result.replaceSubrange(range, with: replacement)
    return result
}

private func gemma4ReplacingLast(
    _ source: String, _ target: String, with replacement: String
) -> String {
    guard let range = source.range(of: target, options: .backwards) else {
        preconditionFailure("missing boundary kernel source fragment")
    }
    var result = source
    result.replaceSubrange(range, with: replacement)
    return result
}

private func gemma4VerifyDecodeBoundaryOutputs(
    candidate: [MLXArray], reference: [MLXArray], kind: String
) {
    precondition(candidate.count == reference.count)
    var values = 0
    for (outputIndex, pair) in zip(candidate, reference).enumerated() {
        precondition(pair.0.dtype == .bfloat16 && pair.1.dtype == .bfloat16)
        precondition(pair.0.shape == pair.1.shape)
        values += pair.0.size
        let candidateBits = pair.0.view(dtype: .uint16)
        let referenceBits = pair.1.view(dtype: .uint16)
        let matches = arrayEqual(candidateBits, referenceBits)
        eval(matches)
        guard matches.item(Bool.self) else {
            let candidateValues = candidateBits.asArray(UInt16.self)
            let referenceValues = referenceBits.asArray(UInt16.self)
            let mismatch = zip(candidateValues, referenceValues).enumerated()
                .first { $0.element.0 != $0.element.1 }
            preconditionFailure(
                "direct decode boundary RMS raw BF16 mismatch kind=\(kind) "
                    + "output=\(outputIndex) index=\(mismatch?.offset ?? -1) "
                    + "candidate=\(mismatch?.element.0 ?? 0) "
                    + "reference=\(mismatch?.element.1 ?? 0)"
            )
        }
    }
    FileHandle.standardError.write(Data(
        "verify_direct_decode_boundary_rms kind=\(kind) arrays=\(candidate.count) values=\(values)\n".utf8
    ))
}

private let gemma4FusedAttentionToMLPBoundarySource = """
        constexpr uint kWidth = 5376;
        constexpr uint kReads = 4;
        constexpr uint kThreads = 1024;
        constexpr uint kSIMDSize = 32;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float inverse_mean[1];
        threadgroup float local_sums[kSIMDSize];
        threadgroup bfloat residual_row[kWidth];

        float accumulator = 0;
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = attention_output[base + index];
                    accumulator += value * value;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    if (base + index < kWidth) {
                        const float value = attention_output[base + index];
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

        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat unit_normalized = static_cast<bfloat>(
                        attention_output[dimension] * inverse_mean[0]);
                    const bfloat post_normalized =
                        post_attention_weight[dimension] * unit_normalized;
                    const bfloat combined =
                        residual[dimension] + post_normalized;
                    residual_row[dimension] = combined;
                    residual_output[dimension] = combined;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat unit_normalized = static_cast<bfloat>(
                            attention_output[dimension] * inverse_mean[0]);
                        const bfloat post_normalized =
                            post_attention_weight[dimension] * unit_normalized;
                        const bfloat combined =
                            residual[dimension] + post_normalized;
                        residual_row[dimension] = combined;
                        residual_output[dimension] = combined;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        accumulator = 0;
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = residual_row[base + index];
                    accumulator += value * value;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    if (base + index < kWidth) {
                        const float value = residual_row[base + index];
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

        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat unit_normalized = static_cast<bfloat>(
                        residual_row[dimension] * inverse_mean[0]);
                    normalized_output[dimension] =
                        pre_ffn_weight[dimension] * unit_normalized;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat unit_normalized = static_cast<bfloat>(
                            residual_row[dimension] * inverse_mean[0]);
                        normalized_output[dimension] =
                            pre_ffn_weight[dimension] * unit_normalized;
                    }
                }
            }
        }
        """

private let gemma4FusedAttentionToMLPBoundaryKernel = MLXFast.metalKernel(
    name: "gemma4_fused_attention_to_mlp_boundary_5376_v1",
    inputNames: [
        "attention_output", "residual", "post_attention_weight", "pre_ffn_weight",
    ],
    outputNames: ["residual_output", "normalized_output"],
    source: gemma4FusedAttentionToMLPBoundarySource,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

private let gemma4DirectAttentionToMLPBoundarySource: String = {
    var source = gemma4FusedAttentionToMLPBoundarySource
    source = source.replacingOccurrences(
        of: "threadgroup float inverse_mean[1];",
        with: "threadgroup float inverse_mean[2];"
    )
    source = source.replacingOccurrences(
        of: "        threadgroup bfloat residual_row[kWidth];\n", with: ""
    )
    source = source.replacingOccurrences(
        of: "                    residual_row[dimension] = combined;\n", with: ""
    )
    source = source.replacingOccurrences(
        of: "                            residual_row[dimension] = combined;\n", with: ""
    )
    let passBarrier = """
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        accumulator = 0;
    """
    source = gemma4ReplacingFirst(source, passBarrier, with: """
            }
        }

        accumulator = 0;
    """)
    let recreate = """
    const bfloat unit_normalized = static_cast<bfloat>(
                            attention_output[dimension] * inverse_mean[0]);
                        const bfloat post_normalized =
                            post_attention_weight[dimension] * unit_normalized;
                        const bfloat combined =
                            residual[dimension] + post_normalized;
    """
    source = source.replacingOccurrences(
        of: "const float value = residual_row[base + index];",
        with: """
    const uint dimension = base + index;
                        \(recreate)
                        const float value = combined;
    """
    )
    source = source.replacingOccurrences(
        of: "const float value = residual_row[base + index];",
        with: """
    const uint dimension = base + index;
                            \(recreate)
                            const float value = combined;
    """
    )
    // Preserve the first inverse while slot 1 receives the second reduction.
    source = gemma4ReplacingLast(
        source,
        "inverse_mean[0] = metal::precise::rsqrt(",
        with: "inverse_mean[1] = metal::precise::rsqrt("
    )
    source = source.replacingOccurrences(
        of: "const bfloat unit_normalized = static_cast<bfloat>(\n                        residual_row[dimension] * inverse_mean[0]);",
        with: recreate + """
                    const bfloat final_normalized = static_cast<bfloat>(
                        combined * inverse_mean[1]);
    """
    )
    source = source.replacingOccurrences(
        of: "pre_ffn_weight[dimension] * unit_normalized;",
        with: "pre_ffn_weight[dimension] * final_normalized;"
    )
    return source
}()

private let gemma4DirectAttentionToMLPBoundaryKernel = MLXFast.metalKernel(
    name: "gemma4_fused_attention_to_mlp_boundary_5376_direct_v2",
    inputNames: [
        "attention_output", "residual", "post_attention_weight", "pre_ffn_weight",
    ],
    outputNames: ["residual_output", "normalized_output"],
    source: gemma4DirectAttentionToMLPBoundarySource,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

struct FusedAttentionToMLPBoundary: @unchecked Sendable {
    let postAttentionWeight: MLXArray
    let preFFNWeight: MLXArray

    init?(
        postAttentionWeight: MLXArray,
        preFFNWeight: MLXArray,
        eps: Float
    ) {
        guard eps == 1.0e-6,
              postAttentionWeight.shape == [5376],
              preFFNWeight.shape == [5376],
              postAttentionWeight.dtype == .bfloat16,
              preFFNWeight.dtype == .bfloat16
        else {
            return nil
        }
        self.postAttentionWeight = postAttentionWeight
        self.preFFNWeight = preFFNWeight
    }

    func callAsFunction(
        attentionOutput: MLXArray,
        residual: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(attentionOutput.shape == [1, 1, 5376])
        precondition(residual.shape == [1, 1, 5376])
        precondition(attentionOutput.dtype == .bfloat16)
        precondition(residual.dtype == .bfloat16)
        let inputs = [attentionOutput, residual, postAttentionWeight, preFFNWeight]
        let candidate = (gemma4DirectDecodeBoundaryRMS
            || gemma4VerifyDirectDecodeBoundaryRMSBits)
            ? gemma4DirectAttentionToMLPBoundaryKernel(
                inputs, grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            ) : nil
        let reference = (!gemma4DirectDecodeBoundaryRMS
            || gemma4VerifyDirectDecodeBoundaryRMSBits)
            ? gemma4FusedAttentionToMLPBoundaryKernel(
                inputs, grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            ) : nil
        if let candidate, let reference {
            gemma4VerifyDecodeBoundaryOutputs(
                candidate: candidate, reference: reference, kind: "attention_to_mlp"
            )
        }
        // Verification deliberately propagates the promoted reference outputs.
        let outputs = gemma4VerifyDirectDecodeBoundaryRMSBits
            ? reference! : (gemma4DirectDecodeBoundaryRMS ? candidate! : reference!)
        return (outputs[0], outputs[1])
    }
}

private let gemma4FusedMLPToNextBoundarySource = """
        constexpr uint kWidth = 5376;
        constexpr uint kReads = 4;
        constexpr uint kThreads = 1024;
        constexpr uint kSIMDSize = 32;

        const uint thread_index = thread_position_in_threadgroup.x;
        const uint simd_lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float inverse_mean[1];
        threadgroup float local_sums[kSIMDSize];
        threadgroup bfloat hidden_row[kWidth];

        float accumulator = 0;
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = mlp_output[base + index];
                    accumulator += value * value;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    if (base + index < kWidth) {
                        const float value = mlp_output[base + index];
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

        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat unit_normalized = static_cast<bfloat>(
                        mlp_output[dimension] * inverse_mean[0]);
                    const bfloat post_normalized =
                        post_ffn_weight[dimension] * unit_normalized;
                    const bfloat combined =
                        residual[dimension] + post_normalized;
                    const bfloat scaled = combined * layer_scalar[0];
                    hidden_row[dimension] = scaled;
                    hidden_output[dimension] = scaled;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat unit_normalized = static_cast<bfloat>(
                            mlp_output[dimension] * inverse_mean[0]);
                        const bfloat post_normalized =
                            post_ffn_weight[dimension] * unit_normalized;
                        const bfloat combined =
                            residual[dimension] + post_normalized;
                        const bfloat scaled = combined * layer_scalar[0];
                        hidden_row[dimension] = scaled;
                        hidden_output[dimension] = scaled;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        accumulator = 0;
        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const float value = hidden_row[base + index];
                    accumulator += value * value;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    if (base + index < kWidth) {
                        const float value = hidden_row[base + index];
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

        for (uint row_offset = 0;
             row_offset < kWidth;
             row_offset += kThreads * kReads) {
            const uint base = row_offset + thread_index * kReads;
            if (base + kReads <= kWidth) {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    const bfloat unit_normalized = static_cast<bfloat>(
                        hidden_row[dimension] * inverse_mean[0]);
                    next_normalized_output[dimension] =
                        next_norm_weight[dimension] * unit_normalized;
                }
            } else {
                for (uint index = 0; index < kReads; ++index) {
                    const uint dimension = base + index;
                    if (dimension < kWidth) {
                        const bfloat unit_normalized = static_cast<bfloat>(
                            hidden_row[dimension] * inverse_mean[0]);
                        next_normalized_output[dimension] =
                            next_norm_weight[dimension] * unit_normalized;
                    }
                }
            }
        }
        """

private let gemma4FusedMLPToNextBoundaryKernel = MLXFast.metalKernel(
    name: "gemma4_fused_mlp_to_next_boundary_5376_v1",
    inputNames: [
        "mlp_output", "residual", "post_ffn_weight", "layer_scalar",
        "next_norm_weight",
    ],
    outputNames: ["hidden_output", "next_normalized_output"],
    source: gemma4FusedMLPToNextBoundarySource,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

private let gemma4DirectMLPToNextBoundarySource: String = {
    var source = gemma4FusedMLPToNextBoundarySource
    source = source.replacingOccurrences(
        of: "threadgroup float inverse_mean[1];",
        with: "threadgroup float inverse_mean[2];"
    )
    source = source.replacingOccurrences(
        of: "        threadgroup bfloat hidden_row[kWidth];\n", with: ""
    )
    source = source.replacingOccurrences(
        of: "                    hidden_row[dimension] = scaled;\n", with: ""
    )
    source = source.replacingOccurrences(
        of: "                            hidden_row[dimension] = scaled;\n", with: ""
    )
    let passBarrier = """
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        accumulator = 0;
    """
    source = gemma4ReplacingFirst(source, passBarrier, with: """
            }
        }

        accumulator = 0;
    """)
    let recreate = """
    const bfloat unit_normalized = static_cast<bfloat>(
                            mlp_output[dimension] * inverse_mean[0]);
                        const bfloat post_normalized =
                            post_ffn_weight[dimension] * unit_normalized;
                        const bfloat combined =
                            residual[dimension] + post_normalized;
                        const bfloat scaled = combined * layer_scalar[0];
    """
    source = source.replacingOccurrences(
        of: "const float value = hidden_row[base + index];",
        with: """
    const uint dimension = base + index;
                        \(recreate)
                        const float value = scaled;
    """
    )
    source = gemma4ReplacingLast(
        source,
        "inverse_mean[0] = metal::precise::rsqrt(",
        with: "inverse_mean[1] = metal::precise::rsqrt("
    )
    source = source.replacingOccurrences(
        of: "const bfloat unit_normalized = static_cast<bfloat>(\n                        hidden_row[dimension] * inverse_mean[0]);",
        with: recreate + """
                    const bfloat final_normalized = static_cast<bfloat>(
                        scaled * inverse_mean[1]);
    """
    )
    source = source.replacingOccurrences(
        of: "next_norm_weight[dimension] * unit_normalized;",
        with: "next_norm_weight[dimension] * final_normalized;"
    )
    return source
}()

private let gemma4DirectMLPToNextBoundaryKernel = MLXFast.metalKernel(
    name: "gemma4_fused_mlp_to_next_boundary_5376_direct_v2",
    inputNames: [
        "mlp_output", "residual", "post_ffn_weight", "layer_scalar",
        "next_norm_weight",
    ],
    outputNames: ["hidden_output", "next_normalized_output"],
    source: gemma4DirectMLPToNextBoundarySource,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

struct FusedMLPToNextBoundary: @unchecked Sendable {
    let postFFNWeight: MLXArray
    let layerScalar: MLXArray
    let nextNormWeight: MLXArray

    init?(
        postFFNWeight: MLXArray,
        layerScalar: MLXArray,
        nextNormWeight: MLXArray,
        eps: Float
    ) {
        guard eps == 1.0e-6,
              postFFNWeight.shape == [5376],
              layerScalar.shape == [1],
              nextNormWeight.shape == [5376],
              postFFNWeight.dtype == .bfloat16,
              layerScalar.dtype == .bfloat16,
              nextNormWeight.dtype == .bfloat16
        else {
            return nil
        }
        self.postFFNWeight = postFFNWeight
        self.layerScalar = layerScalar
        self.nextNormWeight = nextNormWeight
    }

    func callAsFunction(
        mlpOutput: MLXArray,
        residual: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(mlpOutput.shape == [1, 1, 5376])
        precondition(residual.shape == [1, 1, 5376])
        precondition(mlpOutput.dtype == .bfloat16)
        precondition(residual.dtype == .bfloat16)
        let inputs = [mlpOutput, residual, postFFNWeight, layerScalar, nextNormWeight]
        let candidate = (gemma4DirectDecodeBoundaryRMS
            || gemma4VerifyDirectDecodeBoundaryRMSBits)
            ? gemma4DirectMLPToNextBoundaryKernel(
                inputs, grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            ) : nil
        let reference = (!gemma4DirectDecodeBoundaryRMS
            || gemma4VerifyDirectDecodeBoundaryRMSBits)
            ? gemma4FusedMLPToNextBoundaryKernel(
                inputs, grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            ) : nil
        if let candidate, let reference {
            gemma4VerifyDecodeBoundaryOutputs(
                candidate: candidate, reference: reference, kind: "mlp_to_next"
            )
        }
        let outputs = gemma4VerifyDirectDecodeBoundaryRMSBits
            ? reference! : (gemma4DirectDecodeBoundaryRMS ? candidate! : reference!)
        return (outputs[0], outputs[1])
    }
}
