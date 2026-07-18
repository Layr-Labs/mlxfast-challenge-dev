import Foundation
import MLX
import MLXFast

// Fused prefill residual-boundary epilogues.
//
// At prefill (L > 1) each transformer layer runs two eager residual-boundary
// chains on [M, 5376] bf16 activations:
//
//   post-attention: out = residual + rmsNorm(attnOut, postAttnW)
//   post-FFN:       out = (residual + rmsNorm(mlpOut, postFfnW)) * layerScalar
//
// Stock execution is two kernel passes per boundary: the rms_looped reduction
// kernel (axis 5376 > RMS_LOOPED_LIMIT 4096) plus an elementwise add (or a
// compiled add+mul) pass. Each pass reads and writes the full 5.5 MB tensor
// at M=512, so a boundary moves ~27.5 MB where one fused pass moves 16.5 MB.
//
// This file replaces each boundary with ONE custom kernel: the verbatim
// rms_looped reduction (same 1024-thread threadgroup, 4 reads/thread,
// ascending-index float accumulation, simd_sum tree over 32 simd groups,
// metal::precise::rsqrt) with the residual add (and layer-scalar multiply)
// appended as elementwise epilogue terms in the store phase. The reduction is
// single-pass and epilogue-only -- unlike the decode boundary pair in
// Gemma4FusedLayerBoundary.swift there is no second norm and no threadgroup
// row round-trip, so the kernel sustains stock rms_norm bandwidth.
//
// BIT-EXACTNESS CONTRACT
// ----------------------
// The reduction replicates, statement for statement, the stock
// rms_looped<bfloat16_t, 4> kernel
// (Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm.metal)
// at the stock dispatch geometry (threadgroup size =
// maxTotalThreadsPerThreadgroup = 1024, N_READS = 4, one threadgroup per row),
// so every row's sum-of-squares is the identical float accumulation tree.
// The epilogue then reproduces the eager rounding sequence node-for-node:
//
//   t1 = rnd(x * inv_mean)              // stock rms store cast
//   t2 = rnd(w * t1)                    // stock rms output (bf16 mul)
//   t3 = rnd(residual + t2)             // eager add's bf16 add
//   out = rnd(t3 * layerScalar)         // (post-FFN only) bf16 mul
//
// Each step is an explicitly-typed bfloat16_t temporary, so the Metal bfloat
// arithmetic (compute in f32, round to bf16 per op) matches the eager graph
// (and the compiled add+mul tail, whose codegen declares every tape temporary
// with the node's dtype) exactly; no reassociation or contraction crosses the
// typed roundings. Verified raw-bit equal against the stock kernels and
// against the real eager/compiled MLX graphs (Tests/MLXFastTests/
// Gemma4BoundaryEpilogueTests.swift, GPU-gated).
//
// Engagement is restricted to the scored layout: hidden size 5376, bf16
// activations and weights, eps == 1e-6, contiguous norm weight, L > 1. The
// M=64 last-layer tail prune uses the same boundary at M=64 and is supported
// (any M >= 1 maps one threadgroup per row; the residual may be a
// row-contiguous slice). The decode path (L == 1) never routes here.

/// Default-on engagement switch. `DARKBLOOM_PREFILL_BOUNDARY_EPILOGUE=0`
/// rolls both boundaries back to the stock rmsNorm + elementwise passes.
let gemma4PrefillBoundaryEpilogueEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_BOUNDARY_EPILOGUE"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Debug-only raw-bit verifier: every fused dispatch additionally computes the
/// stock eager reference and precondition-fails on any bit difference.
/// `DARKBLOOM_VERIFY_PREFILL_BOUNDARY_EPILOGUE_BITS=1` enables.
let gemma4VerifyPrefillBoundaryEpilogueBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_PREFILL_BOUNDARY_EPILOGUE_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Reduction prologue shared by both boundary kernels: the verbatim
/// rms_looped accumulation and cross-thread reduction tree. `x` is the row
/// being normalized (already offset to the threadgroup's row). Internal (not
/// private) so the contract tests can pin its structure.
let gemma4BoundaryEpilogueReduction = """
    constexpr uint kWidth = 5376;
    constexpr uint kReads = 4;
    constexpr uint kThreads = 1024;
    constexpr uint kSIMDSize = 32;

    const uint thread_index = thread_position_in_threadgroup.x;
    const uint simd_lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const size_t row_offset = size_t(threadgroup_position_in_grid.y) * kWidth;

    const device bfloat* x_row = x + row_offset;
    const device bfloat* res_row = residual + row_offset;
    device bfloat* out_row = output + row_offset;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[kSIMDSize];

    float acc = 0;
    for (uint r = 0; r < kWidth; r += kThreads * kReads) {
        const uint base = r + thread_index * kReads;
        if (base + kReads <= kWidth) {
            for (uint index = 0; index < kReads; ++index) {
                const float value = x_row[base + index];
                acc += value * value;
            }
        } else {
            for (uint index = 0; index < kReads; ++index) {
                if (base + index < kWidth) {
                    const float value = x_row[base + index];
                    acc += value * value;
                }
            }
        }
    }
    acc = simd_sum(acc);

    if (simd_group == 0) {
        local_sums[simd_lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        local_sums[simd_group] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        acc = simd_sum(local_sums[simd_lane]);
        if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(
                acc / kWidth + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    """

/// Post-attention store epilogue: emit `residual + rmsNorm(x)` in the store
/// phase of the same loop the stock kernel uses. Internal for test pinning.
let gemma4BoundaryEpilogueResidualStore = """

    for (uint r = 0; r < kWidth; r += kThreads * kReads) {
        const uint base = r + thread_index * kReads;
        if (base + kReads <= kWidth) {
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = base + index;
                const bfloat unit = static_cast<bfloat>(
                    x_row[dimension] * local_inv_mean[0]);
                const bfloat normed = weight[dimension] * unit;
                out_row[dimension] = res_row[dimension] + normed;
            }
        } else {
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = base + index;
                if (dimension < kWidth) {
                    const bfloat unit = static_cast<bfloat>(
                        x_row[dimension] * local_inv_mean[0]);
                    const bfloat normed = weight[dimension] * unit;
                    out_row[dimension] = res_row[dimension] + normed;
                }
            }
        }
    }
    """

/// Post-FFN store epilogue: emit `(residual + rmsNorm(x)) * layerScalar`.
/// Internal for test pinning.
let gemma4BoundaryEpilogueResidualScalarStore = """

    for (uint r = 0; r < kWidth; r += kThreads * kReads) {
        const uint base = r + thread_index * kReads;
        if (base + kReads <= kWidth) {
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = base + index;
                const bfloat unit = static_cast<bfloat>(
                    x_row[dimension] * local_inv_mean[0]);
                const bfloat normed = weight[dimension] * unit;
                const bfloat combined = res_row[dimension] + normed;
                out_row[dimension] = combined * layer_scalar[0];
            }
        } else {
            for (uint index = 0; index < kReads; ++index) {
                const uint dimension = base + index;
                if (dimension < kWidth) {
                    const bfloat unit = static_cast<bfloat>(
                        x_row[dimension] * local_inv_mean[0]);
                    const bfloat normed = weight[dimension] * unit;
                    const bfloat combined = res_row[dimension] + normed;
                    out_row[dimension] = combined * layer_scalar[0];
                }
            }
        }
    }
    """

/// Post-attention boundary kernel: output = residual + rmsNorm(x, weight).
private let gemma4PrefillBoundaryResidualKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_boundary_residual_5376_v1",
    inputNames: ["x", "weight", "residual"],
    outputNames: ["output"],
    source: gemma4BoundaryEpilogueReduction + gemma4BoundaryEpilogueResidualStore,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

/// Post-FFN boundary kernel: output = (residual + rmsNorm(x, weight)) *
/// layerScalar.
private let gemma4PrefillBoundaryResidualScalarKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_boundary_residual_scalar_5376_v1",
    inputNames: ["x", "weight", "residual", "layer_scalar"],
    outputNames: ["output"],
    source: gemma4BoundaryEpilogueReduction + gemma4BoundaryEpilogueResidualScalarStore,
    header: "using namespace metal;",
    ensureRowContiguous: true
)

/// One prefill residual boundary (post-attention or post-FFN) backed by the
/// fused single-reduction epilogue kernels above.
struct Gemma4PrefillBoundaryEpilogue: @unchecked Sendable {
    let weight: MLXArray
    let layerScalar: MLXArray?

    /// - Parameters:
    ///   - weight: the boundary's norm weight (`[5376]` contiguous bf16).
    ///   - layerScalar: post-FFN layer scalar (`[1]` bf16); nil selects the
    ///     post-attention `residual + rmsNorm` form.
    ///   - eps: must be the Gemma 4 value 1e-6 (hard-coded in the kernels).
    init?(weight: MLXArray, layerScalar: MLXArray? = nil, eps: Float) {
        // The loaded norm weights are contiguous 1-D bf16 rows (same
        // assumption the decode boundary kernels make).
        guard eps == 1.0e-6,
              weight.shape == [5376],
              weight.dtype == .bfloat16
        else {
            return nil
        }
        if let layerScalar {
            guard layerScalar.shape == [1],
                  layerScalar.dtype == .bfloat16
            else {
                return nil
            }
        }
        self.weight = weight
        self.layerScalar = layerScalar
    }

    /// Returns `residual + rmsNorm(x, weight)` (post-attention form) or
    /// `(residual + rmsNorm(x, weight)) * layerScalar` (post-FFN form), or
    /// nil when the shapes/dtypes are outside the scored layout (caller falls
    /// back to the stock chain). `x` and `residual` must share one shape with
    /// last dimension 5376; any leading dimensions collapse into rows.
    func callAsFunction(x: MLXArray, residual: MLXArray) -> MLXArray? {
        guard x.dtype == .bfloat16,
              residual.dtype == .bfloat16,
              x.shape == residual.shape,
              x.ndim >= 2,
              x.dim(x.ndim - 1) == 5376
        else {
            return nil
        }
        let rows = x.shape.dropLast().reduce(1, *)
        guard rows >= 1 else { return nil }
        let x2 = x.reshaped([rows, 5376])
        let residual2 = residual.reshaped([rows, 5376])
        let fused: MLXArray
        if let layerScalar {
            fused = gemma4PrefillBoundaryResidualScalarKernel(
                [x2, weight, residual2, layerScalar],
                grid: (1024, rows, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[rows, 5376]],
                outputDTypes: [.bfloat16]
            )[0]
        } else {
            fused = gemma4PrefillBoundaryResidualKernel(
                [x2, weight, residual2],
                grid: (1024, rows, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [[rows, 5376]],
                outputDTypes: [.bfloat16]
            )[0]
        }
        let result = fused.reshaped(x.shape)
        if gemma4VerifyPrefillBoundaryEpilogueBits {
            var reference = residual + MLXFast.rmsNorm(x, weight: weight, eps: 1.0e-6)
            if let layerScalar {
                reference = reference * layerScalar
            }
            let matches = arrayEqual(
                result.view(dtype: .uint16),
                reference.view(dtype: .uint16)
            )
            eval(matches)
            precondition(
                matches.item(Bool.self),
                "prefill boundary epilogue differs from stock residual chain"
            )
        }
        return result
    }
}
