import Foundation

/// O(1) LRU cache of fully materialized routed-expert weights, keyed by
/// (layer, expert). A routing hit returns the resident `DeepSeekMLPWeights`
/// (MLXArrays already in unified memory) directly, skipping the SSD pread, the
/// per-slice `Data` -> `MLXArray` copy (`makeArray`), the reshape, and
/// `DeepSeekLinearWeight` reconstruction.
///
/// This complements (and supersedes) the raw-byte LRU in `ExpertSlotBank`:
/// caching bytes still forces a ~12.75 MiB/expert host->device copy on every
/// reuse, and at high hit rate those copies (~3 GiB/token) dominate decode
/// latency and inflate transient memory. Caching the built arrays turns the
/// bandwidth win from a larger cache into a decode/prefill latency win as well,
/// and removes the transient duplication that otherwise pressures memory.
///
/// Reference type so it is shared across value-type `DeepSeekWeightLoader`
/// copies. Single-threaded use (the MoE forward loop is serial); no locking.
public final class ExpertWeightCache {
    public struct Key: Hashable {
        public let layer: Int
        public let expert: Int
        public init(layer: Int, expert: Int) {
            self.layer = layer
            self.expert = expert
        }
    }

    private final class Node {
        let key: Key
        var value: DeepSeekMLPWeights
        var prev: Node?
        var next: Node?
        init(key: Key, value: DeepSeekMLPWeights) {
            self.key = key
            self.value = value
        }
    }

    public let capacity: Int
    private var map: [Key: Node] = [:]
    private var head: Node?  // least recently used
    private var tail: Node?  // most recently used

    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0
    public private(set) var evictions: Int = 0
    public private(set) var peakCount: Int = 0

    public init(capacity: Int) {
        self.capacity = max(0, capacity)
    }

    public var count: Int { map.count }

    public func value(for key: Key) -> DeepSeekMLPWeights? {
        guard let node = map[key] else {
            misses += 1
            return nil
        }
        hits += 1
        moveToTail(node)
        return node.value
    }

    public func insert(_ value: DeepSeekMLPWeights, for key: Key) {
        guard capacity > 0 else {
            return
        }
        if let existing = map[key] {
            existing.value = value
            moveToTail(existing)
            return
        }
        let node = Node(key: key, value: value)
        map[key] = node
        appendToTail(node)
        while map.count > capacity, let evicted = head {
            removeNode(evicted)
            map.removeValue(forKey: evicted.key)
            evictions += 1
        }
        if map.count > peakCount {
            peakCount = map.count
        }
    }

    private func appendToTail(_ node: Node) {
        node.prev = tail
        node.next = nil
        tail?.next = node
        tail = node
        if head == nil {
            head = node
        }
    }

    private func removeNode(_ node: Node) {
        let prev = node.prev
        let next = node.next
        prev?.next = next
        next?.prev = prev
        if head === node {
            head = next
        }
        if tail === node {
            tail = prev
        }
        node.prev = nil
        node.next = nil
    }

    private func moveToTail(_ node: Node) {
        guard tail !== node else {
            return
        }
        removeNode(node)
        appendToTail(node)
    }
}
