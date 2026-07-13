import Foundation
import MLX

private func gemma4BoundaryEnvironmentFlag(
    _ name: String,
    default defaultValue: Bool
) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return defaultValue
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

private let gemma4StreamSecondRMSBoundariesDefault = true

private func gemma4RecordStreamSecondRMSVerification(_ boundary: String) {
    FileHandle.standardError.write(Data(
        "verify_stream_second_rms_boundary kind=\(boundary) arrays=2 values=10752\n"
            .utf8
    ))
}

/// Derives a second-RMS streaming specialization from the raw-verified
/// reference source. Every thread accumulates the square of the same BF16
/// value, in the same dimension order, while it authors the public first
/// output. This removes only the redundant threadgroup row store/barrier/read.
func gemma4StreamSecondRMSBoundarySource(
    _ reference: String,
    stagedRow: String,
    stagedValue: String
) -> String {
    var source = reference

    let declaration = "threadgroup bfloat \(stagedRow)[kWidth];"
    precondition(source.components(separatedBy: declaration).count == 2)
    source = source.replacingOccurrences(
        of: declaration,
        with: """
        bfloat streamed_values[8];
                uint streamed_index = 0;
                float second_accumulator = 0;
        """
    )

    let stagedStore = "\(stagedRow)[dimension] = \(stagedValue);"
    precondition(source.components(separatedBy: stagedStore).count == 3)
    source = source.replacingOccurrences(
        of: stagedStore,
        with: """
        streamed_values[streamed_index++] = \(stagedValue);
                            const float second_value = \(stagedValue);
                            second_accumulator += second_value * second_value;
        """
    )

    let secondPassStart = "\nthreadgroup_barrier(mem_flags::mem_threadgroup);"
        + "\n\naccumulator = 0;"
    guard let start = source.range(of: secondPassStart) else {
        preconditionFailure("missing staged second-RMS pass")
    }
    let reductionMarker = "\naccumulator = simd_sum(accumulator);"
    guard let end = source.range(
        of: reductionMarker,
        range: start.upperBound..<source.endIndex
    ) else {
        preconditionFailure("missing second-RMS reduction")
    }
    source.replaceSubrange(
        start.lowerBound..<end.lowerBound,
        with: "\naccumulator = second_accumulator;"
    )

    let stagedRead = "\(stagedRow)[dimension]"
    precondition(source.components(separatedBy: stagedRead).count == 3)
    source = source.replacingOccurrences(
        of: stagedRead,
        with: "streamed_values[streamed_index++]"
    )
    let finalLoop = "\nfor (uint row_offset = 0;"
    guard let finalLoopStart = source.range(of: finalLoop, options: .backwards) else {
        preconditionFailure("missing final normalized-output loop")
    }
    source.insert(
        contentsOf: "\nstreamed_index = 0;",
        at: finalLoopStart.lowerBound
    )
    precondition(!source.contains(stagedRow))
    return source
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

private let gemma4StreamSecondRMSAttentionToMLPBoundaryKernel =
    MLXFast.metalKernel(
    name: "gemma4_stream_second_rms_attention_to_mlp_boundary_5376_v1",
    inputNames: [
        "attention_output", "residual", "post_attention_weight", "pre_ffn_weight",
    ],
    outputNames: ["residual_output", "normalized_output"],
    source: gemma4StreamSecondRMSBoundarySource(
        gemma4FusedAttentionToMLPBoundarySource,
        stagedRow: "residual_row",
        stagedValue: "combined"
    ),
    header: "using namespace metal;",
    ensureRowContiguous: true
)

struct FusedAttentionToMLPBoundary: @unchecked Sendable {
    let postAttentionWeight: MLXArray
    let preFFNWeight: MLXArray
    private let streamSecondRMS: Bool
    private let verifyStreamSecondRMSBits: Bool

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
        self.streamSecondRMS = gemma4BoundaryEnvironmentFlag(
            "DARKBLOOM_STREAM_SECOND_RMS_BOUNDARIES",
            default: gemma4StreamSecondRMSBoundariesDefault
        )
        self.verifyStreamSecondRMSBits = gemma4BoundaryEnvironmentFlag(
            "DARKBLOOM_VERIFY_STREAM_SECOND_RMS_BOUNDARIES_BITS",
            default: false
        )
    }

    func callAsFunction(
        attentionOutput: MLXArray,
        residual: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(attentionOutput.shape == [1, 1, 5376])
        precondition(residual.shape == [1, 1, 5376])
        precondition(attentionOutput.dtype == .bfloat16)
        precondition(residual.dtype == .bfloat16)
        let inputs = [
            attentionOutput, residual, postAttentionWeight, preFFNWeight,
        ]
        let candidate = streamSecondRMS || verifyStreamSecondRMSBits
            ? gemma4StreamSecondRMSAttentionToMLPBoundaryKernel(
                inputs,
                grid: (1024, 1, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            )
            : nil
        let reference = !streamSecondRMS || verifyStreamSecondRMSBits
            ? gemma4FusedAttentionToMLPBoundaryKernel(
                inputs,
                grid: (1024, 1, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            )
            : nil
        if verifyStreamSecondRMSBits,
           let candidate,
           let reference
        {
            for (candidateOutput, referenceOutput) in zip(candidate, reference) {
                let matches = arrayEqual(
                    candidateOutput.view(dtype: .uint16),
                    referenceOutput.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "streamed second-RMS attention boundary differs"
                )
            }
            gemma4RecordStreamSecondRMSVerification("attention_to_mlp")
        }
        let outputs = streamSecondRMS
            ? (candidate ?? reference)
            : (reference ?? candidate)
        guard let outputs else {
            preconditionFailure("attention-to-MLP boundary kernel was not selected")
        }
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

private let gemma4StreamSecondRMSMLPToNextBoundaryKernel =
    MLXFast.metalKernel(
    name: "gemma4_stream_second_rms_mlp_to_next_boundary_5376_v1",
    inputNames: [
        "mlp_output", "residual", "post_ffn_weight", "layer_scalar",
        "next_norm_weight",
    ],
    outputNames: ["hidden_output", "next_normalized_output"],
    source: gemma4StreamSecondRMSBoundarySource(
        gemma4FusedMLPToNextBoundarySource,
        stagedRow: "hidden_row",
        stagedValue: "scaled"
    ),
    header: "using namespace metal;",
    ensureRowContiguous: true
)

struct FusedMLPToNextBoundary: @unchecked Sendable {
    let postFFNWeight: MLXArray
    let layerScalar: MLXArray
    let nextNormWeight: MLXArray
    private let streamSecondRMS: Bool
    private let verifyStreamSecondRMSBits: Bool

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
        self.streamSecondRMS = gemma4BoundaryEnvironmentFlag(
            "DARKBLOOM_STREAM_SECOND_RMS_BOUNDARIES",
            default: gemma4StreamSecondRMSBoundariesDefault
        )
        self.verifyStreamSecondRMSBits = gemma4BoundaryEnvironmentFlag(
            "DARKBLOOM_VERIFY_STREAM_SECOND_RMS_BOUNDARIES_BITS",
            default: false
        )
    }

    func callAsFunction(
        mlpOutput: MLXArray,
        residual: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(mlpOutput.shape == [1, 1, 5376])
        precondition(residual.shape == [1, 1, 5376])
        precondition(mlpOutput.dtype == .bfloat16)
        precondition(residual.dtype == .bfloat16)
        let inputs = [
            mlpOutput, residual, postFFNWeight, layerScalar, nextNormWeight,
        ]
        let candidate = streamSecondRMS || verifyStreamSecondRMSBits
            ? gemma4StreamSecondRMSMLPToNextBoundaryKernel(
                inputs,
                grid: (1024, 1, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            )
            : nil
        let reference = !streamSecondRMS || verifyStreamSecondRMSBits
            ? gemma4FusedMLPToNextBoundaryKernel(
                inputs,
                grid: (1024, 1, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[1, 1, 5376], [1, 1, 5376]],
                outputDTypes: [.bfloat16, .bfloat16]
            )
            : nil
        if verifyStreamSecondRMSBits,
           let candidate,
           let reference
        {
            for (candidateOutput, referenceOutput) in zip(candidate, reference) {
                let matches = arrayEqual(
                    candidateOutput.view(dtype: .uint16),
                    referenceOutput.view(dtype: .uint16)
                )
                eval(matches)
                precondition(
                    matches.item(Bool.self),
                    "streamed second-RMS MLP boundary differs"
                )
            }
            gemma4RecordStreamSecondRMSVerification("mlp_to_next")
        }
        let outputs = streamSecondRMS
            ? (candidate ?? reference)
            : (reference ?? candidate)
        guard let outputs else {
            preconditionFailure("MLP-to-next boundary kernel was not selected")
        }
        return (outputs[0], outputs[1])
    }
}
