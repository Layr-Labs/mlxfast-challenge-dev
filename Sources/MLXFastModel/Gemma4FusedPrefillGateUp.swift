import Foundation
import MLX
import MLXNN

private func gemma4PrefillGateUpFlag(_ name: String, default value: Bool) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment[name] else { return value }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

/// Dual affine-QMM prefill kernel.  The arithmetic body is the pinned MLX
/// qmm_t_nax_tgp_impl body, with its device epilogue redirected to a local
/// BF16 tile.  The gate tile survives while the identical body is run for up;
/// the weight staging tile is then reused for the rounded up tile.
private let gemma4FusedPrefillGateUpNAX = MLXFast.metalKernel(
    name: "gemma4_fused_prefill_gate_up_nax_bf16_5376_21504_v1",
    inputNames: [
        "gate_weight", "gate_scales", "gate_biases",
        "up_weight", "up_scales", "up_biases", "x", "row_count",
    ],
    outputNames: ["activated"],
    source: """
        constexpr int K = 5376;
        constexpr int N = 21504;
        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 64;
        constexpr int BKP = BK + 16 / sizeof(bfloat);
        const int M = row_count[0];
        threadgroup bfloat weights_tile[BN * BKP];
        threadgroup bfloat gate_tile[BM * BN];

        gemma4_prefill_qmm_local(
            gate_weight, gate_scales, gate_biases, x, gate_tile,
            weights_tile, M, threadgroup_position_in_grid,
            thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemma4_prefill_qmm_local(
            up_weight, up_scales, up_biases, x, weights_tile,
            weights_tile, M, threadgroup_position_in_grid,
            thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_thread = thread_index_in_threadgroup;
        const uint valid_rows = min(BM, M - int(threadgroup_position_in_grid.y) * BM);
        const uint output_base =
            (threadgroup_position_in_grid.y * BM) * N
            + threadgroup_position_in_grid.x * BN;
        for (uint index = local_thread; index < valid_rows * BN; index += 128) {
            const uint row = index / BN;
            const uint col = index - row * BN;
            const bfloat gate = gate_tile[index];
            const bfloat up = weights_tile[index];
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
            const bfloat gelu = scaled * shifted;
            activated[output_base + row * N + col] = gelu * up;
        }
        """,
    header: """
        #include "mlx/backend/metal/kernels/quantized_nax.h"

        METAL_FUNC void gemma4_prefill_qmm_local(
            const device uint32_t* w,
            const device bfloat* scales,
            const device bfloat* biases,
            const device bfloat* x,
            threadgroup bfloat* result_tile,
            threadgroup bfloat* Ws,
            int M,
            uint3 tid,
            uint lid,
            uint simd_gid,
            uint simd_lid
        ) {
          constexpr int K = 5376;
          constexpr int N = 21504;
          constexpr int BM = 64;
          constexpr int BK = 64;
          constexpr int BN = 64;
          constexpr int WM = 2;
          constexpr int WN = 2;
          constexpr int BK_padded = BK + 16 / sizeof(bfloat);
          using loader_w_t = QuantizedBlockLoader<
              bfloat, BN, BK, BK_padded, 1, WM * WN * 32, 64, 4>;

          const int K_w = K / 2;
          const int K_g = K / 64;
          const int y_row = tid.y * BM;
          const int y_col = tid.x * BN;
          auto wl = (const device uint8_t*)w;
          x += y_row * static_cast<int64_t>(K);
          wl += y_col * K_w;
          scales += y_col * K_g;
          biases += y_col * K_g;
          loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

          constexpr short SM = BM / WM;
          constexpr short SN = BN / WN;
          constexpr short SK = 32;
          constexpr short TM = SM / 16;
          constexpr short TN = SN / 16;
          constexpr short TK = SK / 16;
          const short tm = SM * (simd_gid / WN);
          const short tn = SN * (simd_gid % WN);
          const short sgp_sm = min(int(SM), M - (y_row + tm));
          const bool is_unaligned_sm = sgp_sm != SM;
          NAXTile<float, TM, TN> Dtile;
          Dtile.clear();
          x += tm * K;

          dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
            for (int k = 0; k < K; k += BK) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              loader_w.load_unsafe();
              threadgroup_barrier(mem_flags::mem_threadgroup);
              STEEL_PRAGMA_NO_UNROLL
              for (int kk1 = 0; kk1 < BK; kk1 += SK) {
                NAXTile<bfloat, TM, TK> Atile;
                NAXTile<bfloat, TN, TK> Btile;
                volatile int compiler_barrier;
                if constexpr (kAlignedM.value) {
                  Atile.load(x + kk1, K);
                } else {
                  Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
                }
                Btile.template load<bfloat, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
                tile_matmad_nax(
                    Dtile, Atile, metal::bool_constant<false>{},
                    Btile, metal::bool_constant<true>{});
                (void)compiler_barrier;
              }
              x += BK;
              loader_w.next();
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if constexpr (kAlignedM.value) {
              Dtile.store(result_tile + tm * BN + tn, BN);
            } else {
              Dtile.store_safe(
                  result_tile + tm * BN + tn, BN, short2(SN, sgp_sm));
            }
          });
        }
        """,
    ensureRowContiguous: true
)

struct FusedPrefillGateUpProjection: @unchecked Sendable {
    let gate: FastQuantizedProjection
    let up: FastQuantizedProjection
    private let enabled: Bool
    private let verifyBits: Bool

    init(gate: FastQuantizedProjection, up: FastQuantizedProjection) {
        self.gate = gate
        self.up = up
        self.enabled = gemma4PrefillGateUpFlag(
            "DARKBLOOM_FUSED_PREFILL_GATE_UP_NAX", default: true)
        self.verifyBits = gemma4PrefillGateUpFlag(
            "DARKBLOOM_VERIFY_FUSED_PREFILL_GATE_UP_NAX_BITS", default: false)
    }

    private var supportsWeights: Bool {
        gate.bits == 4 && up.bits == 4
            && gate.groupSize == 64 && up.groupSize == 64
            && gate.weight.shape == [21_504, 672]
            && up.weight.shape == gate.weight.shape
            && gate.weight.dtype == .uint32 && up.weight.dtype == .uint32
            && gate.scales.shape == [21_504, 84]
            && up.scales.shape == gate.scales.shape
            && gate.scales.dtype == .bfloat16 && up.scales.dtype == .bfloat16
            && gate.biases?.shape == gate.scales.shape
            && up.biases?.shape == up.scales.shape
            && gate.biases?.dtype == .bfloat16 && up.biases?.dtype == .bfloat16
    }

    func supports(_ input: MLXArray) -> Bool {
        // MLX's QMV/QMM crossover is 6 rows in the pinned backend.  Keep all
        // singleton and very-small decode-like calls on the stock route.
        enabled && supportsWeights && input.dtype == .bfloat16
            && input.ndim == 3 && input.dim(0) == 1
            && input.dim(1) >= 6 && input.dim(2) == 5_376
    }

    func callAsFunction(
        _ input: MLXArray,
        reference: @Sendable (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard supports(input), let gateBiases = gate.biases,
              let upBiases = up.biases else { return reference(input) }
        var outputShape = input.shape
        outputShape[2] = 21_504
        let rows = input.dim(1)
        let candidate = gemma4FusedPrefillGateUpNAX(
            [gate.weight, gate.scales, gateBiases,
             up.weight, up.scales, upBiases, input, MLXArray(Int32(rows))],
            grid: (32 * (21_504 / 64), 2 * ((rows + 63) / 64), 2),
            threadGroup: (32, 2, 2),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
        if verifyBits {
            let stock = reference(input)
            let matches = arrayEqual(
                candidate.view(dtype: .uint16), stock.view(dtype: .uint16))
            eval(matches)
            precondition(matches.item(Bool.self),
                "fused prefill gate/up NAX differs from stock activation")
            return stock
        }
        return candidate
    }
}
