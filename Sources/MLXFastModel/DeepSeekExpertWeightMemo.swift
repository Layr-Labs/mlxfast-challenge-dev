import Foundation
import MLX

/// Bounded, input-independent memo for decoded expert linear weights on the
/// `preferStaged == false` path (decode and one-token gates).
///
/// `DeepSeekWeightLoader.expertLinearWeight` rebuilds a fresh
/// `DeepSeekLinearWeight` on every call: it re-derives the quant mode from the
/// packed shape + scales dtype, constructs three `MLXArray`s from the source
/// bytes via `bridge.makeArray`, and reshapes them. On the decode hot path the
/// same (layerIndex, expertIndex, projection) triple is revisited across
/// consecutive tokens, so this memo skips that repeat reconstruction.
///
/// Bit-identical by construction: `linearWeight` is a pure function of
/// `(baseName, expectedShape, tensor, scalesTensor, biasesTensor)`, and on the
/// `preferStaged == false` path those sources are process-stable (resident
/// scales are loader-lifetime, pinned codes are loader-lifetime, and bank
/// bytes are returned as retained `Data`). The produced `MLXArray`s own their
/// own materialized buffers (the `MLXArray(Data, ...)` initializer copies the
/// bytes), so a memoized `DeepSeekLinearWeight` is self-contained — a later
/// `ExpertSlotBank` eviction of the source slice cannot change the array's
/// contents, it only changes the metrics counters recorded on a re-read.
///
/// Strictly gated to `preferStaged == false`: staged prefill bytes are freed by
/// `ExpertLayerStager.releaseLayer` after the layer is consumed, so a memo that
/// crossed the release would point at freed/staged buffers. Memoizing the
/// *result* weight (not the source tensor) sidesteps that, but the gate is kept
/// as a belt-and-suspenders guard against staging interactions.
///
/// Memory: each `DeepSeekLinearWeight` view is ~4MiB codes + ~0.25MiB scales of
/// MLX buffer. Decode's routed-expert working set is small and stable, so the
/// live entry count is bounded by the model's expert reuse; a FIFO cap is a
/// backstop against pathological growth on long sessions.
public final class DeepSeekExpertWeightMemo {
    private var entries: [String: DeepSeekLinearWeight] = [:]
    private var insertionOrder: [String] = []
    private let cap: Int

    public init(cap: Int = 4096) {
        self.cap = max(1, cap)
    }

    /// Returns the memoized weight for `key` if present, else `nil`. Lookup
    /// does not mutate the cache.
    public func weight(forKey key: String) -> DeepSeekLinearWeight? {
        entries[key]
    }

    /// Stores `weight` for `key`, evicting the oldest entry if the cap is
    /// exceeded. Idempotent if `key` is already present (the existing entry is
    /// kept; insertion order is unchanged).
    public func setWeight(_ weight: DeepSeekLinearWeight, forKey key: String) {
        if entries[key] != nil {
            return
        }
        entries[key] = weight
        insertionOrder.append(key)
        if insertionOrder.count > cap {
            let evictedKey = insertionOrder.removeFirst()
            entries[evictedKey] = nil
        }
    }

    public var count: Int { entries.count }
}
