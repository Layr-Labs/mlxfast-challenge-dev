import Foundation
import MLX
import MLXFastCore

/// Bounded cross-step cache of assembled routed-expert weights for the
/// 1-token decode path.
///
/// Decode re-routes every token, and expert selection is heavily skewed
/// toward a per-layer hot set (measured on the public fixture: the 32
/// most-selected of 256 experts cover ~85% of selections per layer), but the
/// streaming path re-reads each selected expert's code slices from SSD and
/// rebuilds their MLXArrays every step. This cache keeps hot experts'
/// assembled `DeepSeekMLPWeights` (the exact arrays the per-expert matmuls
/// consume) resident across steps, so a repeat selection skips the SSD read
/// and Data->Metal copy entirely.
///
/// Policy: LFU with periodic decay and frequency-based admission. The decode
/// working set cycles through more unique experts per step (~250) than a
/// small cache can hold, which defeats plain LRU (cyclic scans evict entries
/// right before reuse); frequency admission keeps the stable hot set pinned
/// while decay lets it drift with the sequence. Frequencies are tracked for
/// every (layer, expert) key — 43x256 counters — so admission needs no
/// probabilistic sketch.
///
/// Value identity: entries are keyed by (layer, expert) and expert weights
/// are immutable for the process lifetime, so a hit feeds the exact same
/// bytes — and the exact same kernels, at the exact same shapes — as the
/// streamed path. This is input-independent weight caching (like the pinned
/// hash-layer codes), not input-keyed memoization: hits depend only on which
/// experts the model routes to, never on prompt content matching a previous
/// request.
///
/// Memory: the budget is sized against the OFFICIAL 48 GB runner's remaining
/// headroom next to the resident scales (~8 GiB), pinned hash-layer codes
/// (~6.4 GiB), staging buffers, and MLX allocator peak — do not raise it
/// because a larger local machine has room. Cached arrays live in the MLX
/// allocator, so they are visible in the reported peak-memory diagnostics.
public final class DecodeExpertWeightCache {
    public struct Key: Hashable {
        public let layerIndex: Int
        public let expertIndex: Int

        public init(layerIndex: Int, expertIndex: Int) {
            self.layerIndex = layerIndex
            self.expertIndex = expertIndex
        }
    }

    private struct Entry {
        let weights: DeepSeekMLPWeights
        let byteCount: Int
    }

    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    private var frequencies: [Key: Int] = [:]
    private var residentBytes = 0
    private var accessesSinceDecay = 0
    public let budgetBytes: Int
    private(set) public var hits: UInt64 = 0
    private(set) public var misses: UInt64 = 0

    /// Halve all frequency counters this often (in accesses) so the hot set
    /// can drift with the decoded sequence instead of fossilizing.
    private static let decayInterval = 2000

    /// Budget for the official 48 GB runner leaves OS headroom after the MLX
    /// allocator peak (~10 GiB), resident scales (~8 GiB), pinned hash-layer
    /// codes (~6.4 GiB), and staging buffers (~6 GiB). Smaller machines get a
    /// reduced budget so the mechanism still engages without threatening
    /// local memory.
    public static func defaultBudgetBytes(
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        physicalMemory >= (40 << 30) ? (6 << 30) : (4 << 30)
    }

    public init(budgetBytes: Int = DecodeExpertWeightCache.defaultBudgetBytes()) {
        self.budgetBytes = max(0, budgetBytes)
    }

    /// Records the access (for admission frequency) and returns the cached
    /// weights on a hit.
    public func cachedWeights(layerIndex: Int, expertIndex: Int) -> DeepSeekMLPWeights? {
        guard budgetBytes > 0 else {
            return nil
        }
        let key = Key(layerIndex: layerIndex, expertIndex: expertIndex)
        lock.lock()
        defer { lock.unlock() }
        recordAccessLocked(key)
        guard let entry = entries[key] else {
            misses += 1
            return nil
        }
        hits += 1
        return entry.weights
    }

    /// Inserts freshly assembled weights, admitting them only while under
    /// budget or when their access frequency beats the coldest resident
    /// entry's (TinyLFU-style admission, exact counters).
    public func insert(
        layerIndex: Int,
        expertIndex: Int,
        weights: DeepSeekMLPWeights,
        byteCount: Int
    ) {
        guard budgetBytes > 0, byteCount > 0, byteCount < budgetBytes else {
            return
        }
        let key = Key(layerIndex: layerIndex, expertIndex: expertIndex)
        lock.lock()
        defer { lock.unlock() }
        if let existing = entries.removeValue(forKey: key) {
            residentBytes -= existing.byteCount
        }
        let frequency = frequencies[key, default: 0]
        while residentBytes + byteCount > budgetBytes {
            guard let victim = entries.min(by: {
                frequencies[$0.key, default: 0] < frequencies[$1.key, default: 0]
            }) else {
                break
            }
            guard frequency > frequencies[victim.key, default: 0] else {
                return
            }
            entries.removeValue(forKey: victim.key)
            residentBytes -= victim.value.byteCount
        }
        guard residentBytes + byteCount <= budgetBytes else {
            return
        }
        entries[key] = Entry(weights: weights, byteCount: byteCount)
        residentBytes += byteCount
    }

    private func recordAccessLocked(_ key: Key) {
        frequencies[key, default: 0] += 1
        accessesSinceDecay += 1
        if accessesSinceDecay >= Self.decayInterval {
            accessesSinceDecay = 0
            for (key, value) in frequencies {
                frequencies[key] = value >> 1
            }
        }
    }

    /// Resident byte estimate for one expert's assembled weights, from the
    /// linear weights' packed shapes. Counts codes plus scales/biases
    /// companions.
    public static func residentByteCount(for weights: DeepSeekMLPWeights) -> Int {
        [weights.gate, weights.up, weights.down].reduce(0) { total, linear in
            var bytes = linear.weight.shape.reduce(1, *) * linear.weight.dtype.size
            if let scales = linear.scales {
                bytes += scales.shape.reduce(1, *) * scales.dtype.size
            }
            if let biases = linear.biases {
                bytes += biases.shape.reduce(1, *) * biases.dtype.size
            }
            return total + bytes
        }
    }
}

/// Process-wide registry so every DeepSeekWeightLoader for the same manifest
/// shares ONE cache instance (the harness keeps two loaders alive at once;
/// see ResidentExpertStoreRegistry for the same pattern and rationale).
public enum DecodeExpertWeightCacheRegistry {
    private static let caches = LockedCache<String, DecodeExpertWeightCache>()

    public static func cache(manifestPath: String) -> DecodeExpertWeightCache {
        caches.value(for: manifestPath) {
            DecodeExpertWeightCache()
        }
    }
}
