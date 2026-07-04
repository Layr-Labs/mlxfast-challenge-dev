import Foundation

/// Bounded per-layer LRU over fully-assembled routed-expert weight triples,
/// engaged only on the 1-token decode path for dynamic (non-hash) layers.
///
/// Consecutive decode tokens re-route a large fraction of the previous
/// token's experts per layer, and the frontier rebuilds each one from
/// absolute zero every step: advisory read, side-bank pread, page-cache ->
/// Data memcpy, Data -> Metal memcpy, MLXArray construction, and the lazy
/// quant-assembly graph nodes. A hit here hands the graph the EXACT
/// DeepSeekMLPWeights built by the identical frontier constructors from the
/// identical bytes on an earlier step — same MLXArray objects, same Metal
/// buffers — so values are unchanged by construction; only which buffers
/// stay alive across steps differs. Host `Data` is never retained: the
/// cached weights hold only the GPU-side arrays (~12.6 MB/expert), keeping
/// the worst-case footprint to capacity x dynamicLayers x 12.6 MB.
///
/// MLXArray is not Sendable; all model execution happens on one thread and
/// the lock makes the bookkeeping itself race-free, so the unchecked
/// conformance is sound for how the runtime uses this cache (mirrors
/// LockedCache).
public final class DecodeExpertWeightCache: @unchecked Sendable {
    private struct Entry {
        let weights: DeepSeekMLPWeights
        var lastUse: UInt64
    }

    private let capacityPerLayer: Int
    private let lock = NSLock()
    private var entriesByLayer: [Int: [Int: Entry]] = [:]
    private var clock: UInt64 = 0

    public init(capacityPerLayer: Int) {
        self.capacityPerLayer = max(1, capacityPerLayer)
    }

    /// Returns the cached weight triples for the given routed experts,
    /// touching each hit's LRU stamp. Misses are simply absent from the
    /// result; callers build them through the untouched frontier pipeline
    /// and insert them afterwards.
    public func hits(layerIndex: Int, expertIndices: [Int]) -> [Int: DeepSeekMLPWeights] {
        lock.lock()
        defer { lock.unlock() }
        guard var layer = entriesByLayer[layerIndex], !layer.isEmpty else {
            return [:]
        }
        var found: [Int: DeepSeekMLPWeights] = [:]
        for expertIndex in expertIndices where found[expertIndex] == nil {
            guard var entry = layer[expertIndex] else {
                continue
            }
            clock += 1
            entry.lastUse = clock
            layer[expertIndex] = entry
            found[expertIndex] = entry.weights
        }
        if !found.isEmpty {
            entriesByLayer[layerIndex] = layer
        }
        return found
    }

    /// Inserts a freshly built weight triple, evicting the least-recently
    /// used entry beyond the per-layer capacity by dropping its references.
    public func insert(layerIndex: Int, expertIndex: Int, weights: DeepSeekMLPWeights) {
        lock.lock()
        defer { lock.unlock() }
        clock += 1
        var layer = entriesByLayer[layerIndex] ?? [:]
        layer[expertIndex] = Entry(weights: weights, lastUse: clock)
        while layer.count > capacityPerLayer {
            guard let victim = layer.min(by: { $0.value.lastUse < $1.value.lastUse })?.key else {
                break
            }
            layer.removeValue(forKey: victim)
        }
        entriesByLayer[layerIndex] = layer
    }
}
