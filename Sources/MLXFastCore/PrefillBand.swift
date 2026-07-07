import Foundation

/// Robust acceptance band for a set of prefill seconds-per-token samples.
///
/// Prefill is a single cold forward and is intrinsically noisy (see
/// docs/thermal-variance-investigation.md). On a stable machine it clusters at a
/// floor, but an individual measurement can spike slow (transient contention) or
/// dip suspiciously fast. This drops the single worst (slowest) sample, averages
/// the rest into a robust estimate `B`, and rejects the whole measurement if any
/// retained sample drifts outside an asymmetric band around `B`:
///
///   - more than `upTolerance` ABOVE `B`  -> a real slowdown survived even after
///     dropping one outlier => the environment is too unstable to trust => FAIL.
///   - more than `downTolerance` BELOW `B` -> a suspiciously fast ("lucky")
///     reading => reject so it cannot be variance-harvested => FAIL.
///
/// Only the single SLOWEST sample is dropped; a lucky-fast sample is deliberately
/// NOT dropped, so it drags `B` down and trips the band (fail), which is the
/// intended "got lucky -> fail" behavior.
///
/// SOUNDNESS: the samples MUST come from DISTINCT prompts (same 512-token shape),
/// never repeats of one prompt in the same worker process. K identical charged
/// forwards would let submitted model code memoize one and serve the rest
/// instantly; the drop-slowest step would then discard the one real forward and
/// `B` would collapse toward zero -- the band cannot catch that, because all the
/// memoized samples agree with each other. Distinct prompts make every forward
/// real. See benchmark-window-freeze.md invariant #2 ("never repeat an identical
/// charged forward in the same worker process").
public struct PrefillBandResult: Equatable {
    /// `B`: the outlier-dropped mean, i.e. the accepted prefill seconds-per-token.
    public let accepted: Double
    public let passed: Bool
    /// Empty when `passed`; otherwise a human-readable failure reason.
    public let reason: String

    public init(accepted: Double, passed: Bool, reason: String) {
        self.accepted = accepted
        self.passed = passed
        self.reason = reason
    }
}

public enum PrefillBand {
    public static func evaluate(
        samples: [Double],
        upTolerance: Double = MLXFastConstants.prefillBandUpTolerance,
        downTolerance: Double = MLXFastConstants.prefillBandDownTolerance
    ) -> PrefillBandResult {
        // Need at least 3 so that after dropping one outlier >= 2 remain to average.
        guard samples.count >= 3 else {
            return PrefillBandResult(
                accepted: .nan, passed: false,
                reason: "prefill band needs >= 3 samples, got \(samples.count)"
            )
        }
        guard samples.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            return PrefillBandResult(
                accepted: .nan, passed: false,
                reason: "prefill samples must be finite and positive"
            )
        }
        // Drop exactly one: the single largest (slowest) sample.
        var rest = samples
        if let slowest = rest.indices.max(by: { rest[$0] < rest[$1] }) {
            rest.remove(at: slowest)
        }
        let B = rest.reduce(0.0, +) / Double(rest.count)
        let hi = B * (1.0 + upTolerance)
        let lo = B * (1.0 - downTolerance)
        for sample in rest {
            if sample > hi {
                return PrefillBandResult(
                    accepted: B, passed: false,
                    reason: "prefill sample \(sample) exceeds +\(upTolerance * 100)% band "
                        + "(> \(hi)): real slowdown after dropping one outlier"
                )
            }
            if sample < lo {
                return PrefillBandResult(
                    accepted: B, passed: false,
                    reason: "prefill sample \(sample) below -\(downTolerance * 100)% band "
                        + "(< \(lo)): suspiciously lucky-fast reading"
                )
            }
        }
        return PrefillBandResult(accepted: B, passed: true, reason: "")
    }
}
