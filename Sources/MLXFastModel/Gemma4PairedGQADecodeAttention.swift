import Foundation
import MLX

private func gemma4PairedGQAFlag(_ name: String, default value: Bool) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else { return value }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

let gemma4PairedSlidingGQADecodeAttentionEnabled = gemma4PairedGQAFlag(
    "DARKBLOOM_PAIRED_SLIDING_GQA_SDPA", default: true)
let gemma4VerifyPairedSlidingGQADecodeAttentionBits = gemma4PairedGQAFlag(
    "DARKBLOOM_VERIFY_PAIRED_SLIDING_GQA_SDPA_BITS", default: false)

/// Host-side observability for tests. No device value is read on the production path.
struct Gemma4PairedGQADecodeAttentionCounters: Sendable {
    let candidateCalls: Int
    let activeN: Int
}

private let gemma4PairedGQACounterLock = NSLock()
private nonisolated(unsafe) var gemma4PairedGQACalls = 0
private nonisolated(unsafe) var gemma4PairedGQALastN = 0

func gemma4PairedGQADecodeAttentionCounters() -> Gemma4PairedGQADecodeAttentionCounters {
    gemma4PairedGQACounterLock.lock()
    defer { gemma4PairedGQACounterLock.unlock() }
    return .init(candidateCalls: gemma4PairedGQACalls, activeN: gemma4PairedGQALastN)
}

func gemma4ResetPairedGQADecodeAttentionCounters() {
    gemma4PairedGQACounterLock.lock()
    gemma4PairedGQACalls = 0
    gemma4PairedGQALastN = 0
    gemma4PairedGQACounterLock.unlock()
}

private let gemma4PairedSlidingGQADecodeKernel = MLXFast.metalKernel(
    name: "gemma4_paired_sliding_gqa_decode_sdpa_bf16_d256_v1",
    inputNames: ["queries", "keys", "values"],
    outputNames: ["output"],
    source: """
        constexpr uint BN = 32;
        constexpr uint BD = 32;
        constexpr uint kReads = 8;

        const uint kv_head = threadgroup_position_in_grid.x;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const uint N = keys_shape[2];

        thread float q[2][kReads];
        thread float o[2][kReads];
        threadgroup float outputs[BN * BD];
        threadgroup float max_scores[BN];
        threadgroup float sum_exp_scores[BN];

        for (uint head = 0; head < 2; ++head) {
            const uint query_head = 2 * kv_head + head;
            const device bfloat* query = queries
                + static_cast<int64_t>(query_head) * queries_strides[1]
                + static_cast<int64_t>(simd_lid * kReads) * queries_strides[3];
            for (uint j = 0; j < kReads; ++j) {
                q[head][j] = static_cast<float>(
                    query[static_cast<int64_t>(j) * queries_strides[3]]);
                o[head][j] = 0.0f;
            }
        }

        float max_score[2] = {
            metal::numeric_limits<float>::lowest(),
            metal::numeric_limits<float>::lowest()
        };
        float sum_exp_score[2] = {0.0f, 0.0f};

        for (uint i = simd_gid; i < N; i += BN) {
            const device bfloat* key = keys
                + static_cast<int64_t>(kv_head) * keys_strides[1]
                + static_cast<int64_t>(i) * keys_strides[2]
                + static_cast<int64_t>(simd_lid * kReads) * keys_strides[3];
            const device bfloat* value = values
                + static_cast<int64_t>(kv_head) * values_strides[1]
                + static_cast<int64_t>(i) * values_strides[2]
                + static_cast<int64_t>(simd_lid * kReads) * values_strides[3];
            float k[kReads];
            float v[kReads];
            for (uint j = 0; j < kReads; ++j) {
                k[j] = static_cast<float>(
                    key[static_cast<int64_t>(j) * keys_strides[3]]);
                v[j] = static_cast<float>(
                    value[static_cast<int64_t>(j) * values_strides[3]]);
            }

            for (uint head = 0; head < 2; ++head) {
                float score = 0.0f;
                for (uint j = 0; j < kReads; ++j) {
                    score += q[head][j] * k[j];
                }
                score = simd_sum(score);
                float new_max = max(max_score[head], score);
                float factor = fast::exp(max_score[head] - new_max);
                float exp_score = fast::exp(score - new_max);
                max_score[head] = new_max;
                sum_exp_score[head] = sum_exp_score[head] * factor + exp_score;
                for (uint j = 0; j < kReads; ++j) {
                    o[head][j] = o[head][j] * factor + exp_score * v[j];
                }
            }
        }

        // This is the stock sdpa_vector cross-SIMD reduction, performed once
        // for each independent query-head state.
        for (uint head = 0; head < 2; ++head) {
            if (simd_lid == 0) {
                max_scores[simd_gid] = max_score[head];
                sum_exp_scores[simd_gid] = sum_exp_score[head];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score[head] = max_scores[simd_lid];
            float new_max = simd_max(max_score[head]);
            float factor = fast::exp(max_score[head] - new_max);
            sum_exp_score[head] = simd_sum(
                sum_exp_scores[simd_lid] * factor);

            for (uint j = 0; j < kReads; ++j) {
                outputs[simd_lid * BD + simd_gid] = o[head][j];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[head][j] = simd_sum(
                    outputs[simd_gid * BD + simd_lid] * factor);
                o[head][j] = sum_exp_score[head] == 0.0f
                    ? o[head][j] : o[head][j] / sum_exp_score[head];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (simd_lid == 0) {
                const uint query_head = 2 * kv_head + head;
                device bfloat* destination = output
                    + static_cast<int64_t>(query_head) * 256
                    + static_cast<int64_t>(simd_gid * kReads);
                for (uint j = 0; j < kReads; ++j) {
                    destination[j] = static_cast<bfloat>(o[head][j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """,
    header: "#include <metal_stdlib>\n",
    ensureRowContiguous: false
)

func supportsGemma4PairedSlidingGQADecodeAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    scale: Float, hasMask: Bool
) -> Bool {
    guard !hasMask, scale == 1.0,
          queries.dtype == .bfloat16, keys.dtype == .bfloat16,
          values.dtype == .bfloat16,
          queries.shape == [1, 32, 1, 256],
          keys.ndim == 4, values.ndim == 4,
          keys.shape == values.shape,
          keys.dim(0) == 1, keys.dim(1) == 16,
          keys.dim(2) > 0, keys.dim(2) <= 1024, keys.dim(3) == 256
    else { return false }
    return true
}

/// Returns nil for every shape or mode outside the exact decode specialization.
func gemma4PairedSlidingGQADecodeAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray,
    scale: Float = 1.0, hasMask: Bool = false
) -> MLXArray? {
    guard supportsGemma4PairedSlidingGQADecodeAttention(
        queries: queries, keys: keys, values: values,
        scale: scale, hasMask: hasMask
    ) else { return nil }

    gemma4PairedGQACounterLock.lock()
    gemma4PairedGQACalls += 1
    gemma4PairedGQALastN = keys.dim(2)
    gemma4PairedGQACounterLock.unlock()

    // MLX custom-kernel grid dimensions count threads, rather than groups:
    // (512, 32, 1) / (32, 32, 1) is exactly 16 one-per-KV-head groups.
    return gemma4PairedSlidingGQADecodeKernel(
        [queries, keys, values],
        grid: (512, 32, 1),
        threadGroup: (32, 32, 1),
        outputShapes: [[1, 32, 1, 256]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Test hook returning both paths without changing production propagation.
func gemma4PairedSlidingGQADecodeAttentionOutputs(
    queries: MLXArray, keys: MLXArray, values: MLXArray
) -> (candidate: MLXArray, reference: MLXArray)? {
    guard let candidate = gemma4PairedSlidingGQADecodeAttention(
        queries: queries, keys: keys, values: values
    ) else { return nil }
    let reference = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: 1.0, mask: .none)
    return (candidate, reference)
}
