import Foundation
import MLX

/// Bounded per-layer LRU over fully-assembled routed-expert weight triples,
/// engaged only on the 1-token decode path for dynamic (non-hash) layers,
/// with a HARD global byte budget enforced by exact byte accounting and
/// admission control.
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
/// cached weights hold only the GPU-side arrays (~12.6 MB/expert).
///
/// Footprint discipline (both enforced, whichever binds first):
///   - per-layer LRU capacity (retention discipline; a *global* LRU would
///     be defeated by the cyclic 40-layer decode scan, which inserts ~3.1 GB
///     between successive visits to the same layer), and
///   - a global byte budget: exact per-entry byte accounting with admission
///     control whenever the total would exceed the budget, so the cache can
///     never wire more than `byteBudget` bytes of GPU buffers.
///
/// MLXArray is not Sendable; all model execution happens on one thread and
/// the lock makes the bookkeeping itself race-free, so the unchecked
/// conformance is sound for how the runtime uses this cache (mirrors
/// LockedCache).
public final class DecodeExpertWeightCache: @unchecked Sendable {
    private struct Entry {
        let weights: DeepSeekMLPWeights
        let bytes: Int
        var lastUse: UInt64
    }

    private let capacityPerLayer: Int
    private let byteBudget: Int
    private let lock = NSLock()
    private var entriesByLayer: [Int: [Int: Entry]] = [:]
    private var clock: UInt64 = 0
    private var totalBytes: Int = 0

    public init(capacityPerLayer: Int, byteBudget: Int) {
        self.capacityPerLayer = max(1, capacityPerLayer)
        self.byteBudget = byteBudget
    }

    /// Exact GPU-side footprint of one cached weight triple: sum of
    /// elementCount x dtype byte size over every MLXArray the triple holds
    /// (codes, scales, and biases when present). Pure host arithmetic on
    /// array metadata — never forces an eval. If a scales array is a view
    /// into the resident scales store this over-counts slightly, which is
    /// the conservative direction for a hard budget.
    private static func entryBytes(_ weights: DeepSeekMLPWeights) -> Int {
        var bytes = 0
        for projection in [weights.gate, weights.up, weights.down] {
            bytes += projection.weight.size * projection.weight.itemSize
            if let scales = projection.scales {
                bytes += scales.size * scales.itemSize
            }
            if let biases = projection.biases {
                bytes += biases.size * biases.itemSize
            }
        }
        return bytes
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
    /// used entry beyond the per-layer capacity by dropping its references,
    /// then dropping the fresh entry itself if the total retained bytes
    /// would exceed the hard budget (admission control).
    public func insert(layerIndex: Int, expertIndex: Int, weights: DeepSeekMLPWeights) {
        lock.lock()
        defer { lock.unlock() }
        clock += 1
        let bytes = Self.entryBytes(weights)
        var layer = entriesByLayer[layerIndex] ?? [:]
        if let replaced = layer[expertIndex] {
            totalBytes -= replaced.bytes
        }
        layer[expertIndex] = Entry(weights: weights, bytes: bytes, lastUse: clock)
        totalBytes += bytes
        while layer.count > capacityPerLayer {
            guard let victim = layer.min(by: { $0.value.lastUse < $1.value.lastUse })?.key else {
                break
            }
            if let evicted = layer.removeValue(forKey: victim) {
                totalBytes -= evicted.bytes
            }
        }
        entriesByLayer[layerIndex] = layer
        // Hard global budget, enforced as ADMISSION CONTROL: if retaining
        // this fresh entry would leave the cache over budget (the per-layer
        // eviction above already freed a slot when the layer was full, so
        // this only triggers while a layer is still growing at the budget
        // frontier), drop the just-inserted entry instead of evicting an
        // older one. Evicting the globally-oldest entry is provably useless
        // under the decode loop's cyclic 40-layer scan — the oldest entry is
        // exactly the one the NEXT layer visit would have hit, so global-LRU
        // eviction removes each entry just before its reuse (measured: 0%
        // hits). Admission control instead freezes the cache at whichever
        // layers filled first (~112 of 120 slots); those layers keep exact
        // per-layer LRU semantics and the remaining layers stay uncached on
        // the untouched frontier path. Host-side integer bookkeeping only;
        // no MLX graph state is touched.
        if totalBytes > byteBudget {
            if let dropped = entriesByLayer[layerIndex]?.removeValue(forKey: expertIndex) {
                totalBytes -= dropped.bytes
            }
            if entriesByLayer[layerIndex]?.isEmpty == true {
                entriesByLayer.removeValue(forKey: layerIndex)
            }
        }
    }
}
