import Foundation
import MLX
import MLXFastCore

/// Deliberate, GPU-loaded decode slowdown for ranked-floor validation.
///
/// This is an HONEST slowdown, not a bypass: it performs real, extra GPU
/// matmul work on every single-token decode step and then DISCARDS the
/// result, so measured decode seconds-per-token increase while model outputs
/// stay bit-identical. Unlike the sleep-based `measuredDecodeDelayMilliseconds`
/// hook (which idles the GPU and trips the timing telemetry gate), this keeps
/// the GPU fully loaded, so `measure-job` accepts the measurement and the
/// ranked speedup-floor arithmetic is exercised on a genuinely regressed
/// candidate. The work runs uniformly on every M=1 decode (correctness and the
/// timed benchmark alike); it is not special-cased to timed workers and it can
/// only make the score worse, never better.
enum Gemma4DecodeBallast {
    /// Number of throwaway matmul iterations per decode step. Sized to push the
    /// measured decode speedup clearly below the 0.95 ranked floor while keeping
    /// GPU utilization pegged for the timing telemetry gate.
    private static let iterations = 8
    private static let dimension = 4096

    static func applyDuringMeasurement() {
        let a = zeros([dimension, dimension], dtype: .bfloat16)
        let b = zeros([dimension, dimension], dtype: .bfloat16)
        var acc = a
        for _ in 0..<iterations {
            acc = matmul(acc, b)
            eval(acc)
        }
    }
}
