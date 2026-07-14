import Foundation
import MLX
import MLXLMCommon

/// Qualification-only profiler for the production 512-token seed graph.
/// The immutable mode is read once at process start; `.off` has no evaluation,
/// locking, timing, or output on the model call path.
enum Gemma4SeedProfileMode: String {
    case off
    case control
    case coarse
    case detailSliding = "detail-sliding"
    case detailFull = "detail-full"

    static let processMode: Self = {
        guard let value = ProcessInfo.processInfo.environment["DARKBLOOM_SEED_PROFILE"]?
            .lowercased()
        else { return .off }
        return Self(rawValue: value) ?? .off
    }()
}

final class Gemma4SeedProfileRun {
    typealias Checkpoint = (_ name: String, _ roots: [MLXArray]) -> Void

    let mode: Gemma4SeedProfileMode
    private let lock = NSLock()
    private var phases: [String: UInt64] = [:]
    private var cuts = 0
    private let started = ContinuousClock.now

    private static let armLock = NSLock()
    nonisolated(unsafe) private static var armed = false

    private init(mode: Gemma4SeedProfileMode) {
        self.mode = mode
    }

    static func armIfQualifying(inputs: MLXArray, cache: [KVCache]?) -> Gemma4SeedProfileRun? {
        let mode = Gemma4SeedProfileMode.processMode
        guard mode != .off, inputs.shape == [1, 512] else { return nil }
        if let cache, !cache.allSatisfy({ $0.offset == 0 }) { return nil }

        armLock.lock()
        defer { armLock.unlock() }
        guard !armed else { return nil }
        armed = true
        return Gemma4SeedProfileRun(mode: mode)
    }

    func evaluateCheckpoint(name: String, roots: [MLXArray]) {
        guard !roots.isEmpty else { return }
        let start = ContinuousClock.now
        eval(roots)
        let elapsed = Self.nanoseconds(start.duration(to: .now))
        lock.lock()
        phases[name, default: 0] += elapsed
        cuts += 1
        lock.unlock()
    }

    func finish() {
        let total = Self.nanoseconds(started.duration(to: .now))
        lock.lock()
        let phaseSnapshot = phases
        let cutSnapshot = cuts
        lock.unlock()

        var object: [String: Any] = [
            "mode": mode.rawValue,
            "eval_cuts": cutSnapshot,
            "total_wall_ns": total,
            "phase_ns": phaseSnapshot,
        ]
        if mode == .detailSliding,
           let attention = phaseSnapshot["attention_front_sdpa_output"],
           let mlp = phaseSnapshot["mlp_boundaries"]
        {
            object["extrapolated_ns"] = [
                "sliding_attention_x50": attention.multipliedReportingOverflow(by: 50).partialValue,
                "sliding_mlp_boundaries_x50": mlp.multipliedReportingOverflow(by: 50).partialValue,
            ]
        } else if mode == .detailFull,
                  let attention = phaseSnapshot["attention_front_sdpa_output"],
                  let mlp = phaseSnapshot["mlp_boundaries"]
        {
            object["extrapolated_ns"] = [
                "full_attention_x10": attention.multipliedReportingOverflow(by: 10).partialValue,
                "full_mlp_boundaries_x10": mlp.multipliedReportingOverflow(by: 10).partialValue,
            ]
        }
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8)
        else { return }
        FileHandle.standardError.write(Data("DARKBLOOM_SEED_PROFILE_JSON=\(json)\n".utf8))
    }

    private static func nanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        let seconds = components.seconds > 0 ? UInt64(components.seconds) : 0
        let attoseconds = components.attoseconds > 0 ? UInt64(components.attoseconds) : 0
        let (whole, overflow) = seconds.multipliedReportingOverflow(by: 1_000_000_000)
        if overflow { return UInt64.max }
        return whole.addingReportingOverflow(attoseconds / 1_000_000_000).partialValue
    }
}
