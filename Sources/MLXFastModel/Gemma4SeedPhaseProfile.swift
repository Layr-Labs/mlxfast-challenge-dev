import Foundation
import MLX
import MLXLMCommon

// Temporary compile-time selector. Change this literal and perform a fresh
// release build for each profiling campaign; never select it from runtime
// input or the environment.
enum Gemma4SeedProfileMode: String {
    case control
    case coarse
    case detailSliding
    case detailFull
}

private let gemma4SeedProfileMode: Gemma4SeedProfileMode = .detailFull

struct Gemma4SeedProfileRun {
    let nonce: String
    let startNanoseconds: UInt64

    init() {
        nonce = UUID().uuidString
        startNanoseconds = DispatchTime.now().uptimeNanoseconds
    }
}

private final class Gemma4SeedProfileWriter: @unchecked Sendable {
    static let shared = Gemma4SeedProfileWriter()
    private let lock = NSLock()

    func write(_ fields: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(fields),
              let data = try? JSONSerialization.data(withJSONObject: fields),
              var line = String(data: data, encoding: .utf8)
        else { return }
        line.append("\n")
        lock.lock()
        defer { lock.unlock() }
        FileHandle.standardError.write(Data(line.utf8))
    }
}

@inline(__always)
func gemma4SeedProfileCurrentMode() -> Gemma4SeedProfileMode {
    gemma4SeedProfileMode
}

@inline(__always)
func gemma4SeedProfileIsDetailLayer(_ layer: Int) -> Bool {
    switch gemma4SeedProfileMode {
    case .detailSliding: return layer == 58
    case .detailFull: return layer == 59
    default: return false
    }
}

func gemma4SeedProfileCacheRoots(_ cache: KVCache?) -> [MLXArray] {
    cache?.innerState() ?? []
}

private func gemma4SeedProfileShape(_ array: MLXArray) -> [Int] {
    array.shape
}

/// Synchronize roots at an existing arithmetic boundary, then immediately
/// repeat the same eval to expose fixed/no-op checkpoint tax in the log.
func gemma4SeedProfileCheckpoint(
    run: Gemma4SeedProfileRun,
    boundary: String,
    layer: Int?,
    active: [MLXArray],
    cache: [MLXArray] = []
) {
    let before = DispatchTime.now().uptimeNanoseconds
    eval(active + cache)
    let after = DispatchTime.now().uptimeNanoseconds
    eval(active + cache)
    let repeated = DispatchTime.now().uptimeNanoseconds
    Gemma4SeedProfileWriter.shared.write([
        "mode": gemma4SeedProfileMode.rawValue,
        "boundary": boundary,
        "layer": layer ?? -1,
        "elapsed_nanoseconds": after - run.startNanoseconds,
        "checkpoint_nanoseconds": after - before,
        "repeated_eval_nanoseconds": repeated - after,
        "active_shapes": active.map(gemma4SeedProfileShape),
        "cache_shapes": cache.map(gemma4SeedProfileShape),
        "run_nonce": run.nonce,
    ])
}

/// Control mode deliberately performs no eval: it measures only CPU-side
/// recorder/graph-construction overhead and leaves completion to the harness.
func gemma4SeedProfileControlRecord(
    run: Gemma4SeedProfileRun,
    boundary: String,
    layer: Int?,
    active: [MLXArray],
    cache: [MLXArray] = []
) {
    Gemma4SeedProfileWriter.shared.write([
        "mode": gemma4SeedProfileMode.rawValue,
        "boundary": boundary,
        "layer": layer ?? -1,
        "elapsed_nanoseconds": DispatchTime.now().uptimeNanoseconds - run.startNanoseconds,
        "checkpoint_nanoseconds": 0,
        "repeated_eval_nanoseconds": 0,
        "active_shapes": active.map(gemma4SeedProfileShape),
        "cache_shapes": cache.map(gemma4SeedProfileShape),
        "run_nonce": run.nonce,
    ])
}
