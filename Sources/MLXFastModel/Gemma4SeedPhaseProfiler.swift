import Foundation
import MLX

/// Qualification-only timing support for the first zero-offset L512 seed.
/// Both switches are read once at process start; production never enters the
/// timed path and does not perform environment lookups in a layer loop.
enum Gemma4SeedProfileMode {
    case disabled, phases, control

    static let processStart: Self = {
        let environment = ProcessInfo.processInfo.environment
        let phases = environment["DARKBLOOM_PROFILE_L512_SEED_PHASES"] == "1"
        let control = environment["DARKBLOOM_PROFILE_L512_SEED_CONTROL"] == "1"
        precondition(!(phases && control), "L512 profiler modes are mutually exclusive")
        return phases ? .phases : (control ? .control : .disabled)
    }()
}

enum Gemma4SeedPhase: String, CaseIterable {
    case inputNorm = "input_norm"
    case qkvQMM = "qkv_qmm"
    case attentionPrepareCache = "attention_prepare_cache"
    case slidingSDPA = "sliding_sdpa"
    case fullSDPA = "full_sdpa"
    case oProj = "o_proj"
    case postAttentionBoundary = "post_attention_boundary"
    case gateQMM = "gate_qmm"
    case upQMM = "up_qmm"
    case activation
    case downQMM = "down_qmm"
    case postMLPBoundary = "post_mlp_boundary"
    case finalNormHead = "final_norm_head"
}

private struct Gemma4SeedPhaseSample {
    let layer: Int
    let sliding: Bool
    let phase: Gemma4SeedPhase
    let nanoseconds: UInt64
}

/// Intentionally lock-based: this diagnostic is qualification-only and each
/// interval includes GPU completion, so bookkeeping is outside the interval.
final class Gemma4SeedPhaseProfiler: @unchecked Sendable {
    static let shared = Gemma4SeedPhaseProfiler()
    static var enabled: Bool { Gemma4SeedProfileMode.processStart != .disabled }
    static var phasesEnabled: Bool { Gemma4SeedProfileMode.processStart == .phases }

    private let lock = NSLock()
    private var samples: [Gemma4SeedPhaseSample] = []
    private var nextLayer = 0
    private var emitted = false
    private var seedStart: ContinuousClock.Instant?

    private init() {}

    func beginSeed() {
        lock.lock(); defer { lock.unlock() }
        guard !emitted, seedStart == nil else { return }
        nextLayer = 0
        samples.removeAll(keepingCapacity: true)
        seedStart = ContinuousClock.now
    }

    func allocateLayer() -> Int {
        lock.lock(); defer { lock.unlock() }
        let result = nextLayer
        nextLayer += 1
        return result
    }

    @inline(never)
    func measure(_ phase: Gemma4SeedPhase, layer: Int, sliding: Bool,
                 construct: () -> [MLXArray]) -> [MLXArray] {
        let start = ContinuousClock.now
        let outputs = construct()
        eval(outputs)
        let elapsed = start.duration(to: .now)
        let components = elapsed.components
        let ns = UInt64(max(0, components.seconds)) * 1_000_000_000
            + UInt64(max(0, components.attoseconds / 1_000_000_000))
        lock.lock()
        samples.append(.init(layer: layer, sliding: sliding,
                             phase: phase, nanoseconds: ns))
        lock.unlock()
        return outputs
    }

    func finishSeed(finalOutput: MLXArray) {
        lock.lock()
        guard !emitted, let start = seedStart else { lock.unlock(); return }
        lock.unlock()

        if Gemma4SeedProfileMode.processStart == .control {
            eval(finalOutput)
        }
        let total = start.duration(to: .now)
        let c = total.components
        let totalNS = UInt64(max(0, c.seconds)) * 1_000_000_000
            + UInt64(max(0, c.attoseconds / 1_000_000_000))

        lock.lock(); defer { lock.unlock() }
        guard !emitted else { return }
        emitted = true
        var record: [String: Any] = [
            "record": "gemma4_l512_seed_phases",
            "mode": Gemma4SeedProfileMode.processStart == .phases ? "profile" : "control",
            "total_nanoseconds": totalNS,
        ]
        if Gemma4SeedProfileMode.processStart == .phases {
            record["summaries"] = summaries(samples)
            record["samples"] = samples.map {
                ["layerIndex": $0.layer, "attention": $0.sliding ? "sliding" : "full",
                 "phase": $0.phase.rawValue, "nanoseconds": $0.nanoseconds] as [String: Any]
            }
        }
        if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
           let line = String(data: data, encoding: .utf8) {
            print(line)
        }
    }

    private func summaries(_ source: [Gemma4SeedPhaseSample]) -> [[String: Any]] {
        func summary(_ values: [UInt64], phase: String, attention: String) -> [String: Any]? {
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            func percentile(_ p: Double) -> UInt64 {
                sorted[Int((Double(sorted.count - 1) * p).rounded())]
            }
            return ["phase": phase, "attention": attention, "count": sorted.count,
                    "sum_nanoseconds": sorted.reduce(0, +),
                    "median_nanoseconds": percentile(0.5),
                    "p25_nanoseconds": percentile(0.25),
                    "p75_nanoseconds": percentile(0.75)]
        }
        var result: [[String: Any]] = []
        for phase in Gemma4SeedPhase.allCases {
            for kind in ["all", "sliding", "full"] {
                let values = source.filter {
                    $0.phase == phase && (kind == "all" || ($0.sliding ? "sliding" : "full") == kind)
                }.map(\.nanoseconds)
                if let item = summary(values, phase: phase.rawValue, attention: kind) { result.append(item) }
            }
        }
        return result
    }
}
