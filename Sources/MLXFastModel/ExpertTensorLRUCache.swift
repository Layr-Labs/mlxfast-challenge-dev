import Foundation
import MLXFastCore

/// Cross-step LRU cache of materialized expert tensors for the parallel
/// streaming path.
///
/// The trusted `ExpertSlotBank` couples its byte cache to its (serial) read
/// path, so a pool of parallel reader banks either fragments the LRU across
/// shards (losing hits to imbalance) or bypasses caching entirely. This cache
/// restores the exact policy of the single-bank baseline — one global LRU
/// over the same slice keys with the same default capacity (768 tensors,
/// ~3.2 GiB of mxfp4 codes) — in front of the parallel pool.
///
/// Accounting stays honest: every miss is read through a metric-sharing
/// `ExpertSlotBank`, so `expert_bytes_read` counts exactly the bytes that
/// really left disk/page cache; a hit here simply issues no read at all,
/// which is the same observable effect the trusted bank's own cache hit has
/// on the bandwidth diagnostic. Memory use replaces (rather than adds to)
/// the single-bank cache budget, because the pool banks and the loader's
/// fallback bank see almost no traffic on this path.
final class ExpertTensorLRUCache {
    private struct Entry {
        let tensor: MaterializedTensor
        var lastUse: UInt64
    }

    private let capacity: Int
    private var entries: [String: Entry] = [:]
    private var clock: UInt64 = 0

    init(capacity: Int = ExpertStreamingConfig.defaultTensorCacheCapacity) {
        self.capacity = max(0, capacity)
    }

    func lookup(_ key: String) -> MaterializedTensor? {
        guard capacity > 0, var entry = entries[key] else {
            return nil
        }
        clock += 1
        entry.lastUse = clock
        entries[key] = entry
        return entry.tensor
    }

    func insert(_ key: String, _ tensor: MaterializedTensor) {
        guard capacity > 0 else {
            return
        }
        clock += 1
        entries[key] = Entry(tensor: tensor, lastUse: clock)
        while entries.count > capacity {
            var oldestKey: String?
            var oldestUse = UInt64.max
            for (candidateKey, entry) in entries where entry.lastUse < oldestUse {
                oldestUse = entry.lastUse
                oldestKey = candidateKey
            }
            guard let oldestKey else {
                break
            }
            entries.removeValue(forKey: oldestKey)
        }
    }
}
