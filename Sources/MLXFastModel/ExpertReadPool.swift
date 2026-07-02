import Foundation
import MLXFastCore

/// Concurrent expert tensor reads for decode-shaped MoE dispatch.
///
/// A decode step must stream every routed expert's quantized tensors from the
/// reference checkpoint before its MoE matmuls can run. Reading them one after
/// another on the model thread leaves the SSD at queue depth 1, so the layer
/// pays (tensor count) x (single-read latency). This pool fans the same reads
/// out across a small fixed set of worker slots so the drive sees a deep queue
/// and the layer pays roughly (total bytes / aggregate bandwidth) instead.
///
/// Byte accounting stays honest by construction: every read goes through an
/// `ExpertSlotBank` that shares the loader's `ExpertStreamingMetrics`, which
/// is internally lock-protected. Requests are partitioned across banks by a
/// stable hash of the tensor cache key, and each bank is touched by exactly
/// one in-flight task per batch, so the non-locking bank state needs no
/// extra synchronization. Cross-step caching lives in the loader's
/// `ExpertTensorLRUCache` in front of this pool (same keys and capacity as
/// the single trusted bank cache this path replaces). No data is fabricated
/// and no read is hidden: a request that fails here is simply absent from
/// the result and the caller falls back to the loader's main bank,
/// reproducing the original behavior (and its error reporting) exactly.
public final class ExpertReadPool {
    public struct Request {
        public let recordName: String
        /// First-axis slice index into a stacked `[experts, ...]` record, or
        /// nil to read the whole record.
        public let expertIndex: Int?

        public init(recordName: String, expertIndex: Int?) {
            self.recordName = recordName
            self.expertIndex = expertIndex
        }

        /// Matches the trusted bank's slice cache-key spelling so callers can
        /// look results up with the same key they would pass to the bank.
        public var cacheKey: String {
            if let expertIndex {
                return "\(recordName)[\(expertIndex)]"
            }
            return recordName
        }
    }

    private let banks: [ExpertSlotBank]
    private let queue = DispatchQueue(
        label: "mlxfast.expert.readpool",
        qos: .userInitiated,
        attributes: .concurrent
    )

    public init?(
        manifestPath: String,
        metrics: ExpertStreamingMetrics?,
        workers: Int = ExpertReadPool.defaultWorkerCount
    ) {
        let workerCount = max(1, workers)
        var banks: [ExpertSlotBank] = []
        banks.reserveCapacity(workerCount)
        for _ in 0..<workerCount {
            // Capacity 0: caching lives in the loader's cross-step
            // ExpertTensorLRUCache in front of this pool, so these banks are
            // pure metric-sharing readers (capacity-0 banks never mutate LRU
            // state, keeping per-bank single-task confinement sound).
            guard let bank = try? ExpertSlotBank(
                manifestPath: manifestPath,
                capacity: 0,
                metrics: metrics
            ) else {
                return nil
            }
            banks.append(bank)
        }
        self.banks = banks
    }

    /// Sized for the official 12-vCPU runner: enough in-flight preads to keep
    /// the SSD busy without oversubscribing cores that the GPU driver and
    /// model thread need.
    public static let defaultWorkerCount = 8

    /// Stable per-tensor bank assignment (Swift's `hashValue` is seeded per
    /// process launch, which is fine: assignment only needs to be stable
    /// within one process for the LRU caches to see repeat traffic).
    private func bankIndex(for request: Request) -> Int {
        var hasher = Hasher()
        hasher.combine(request.recordName)
        hasher.combine(request.expertIndex ?? -1)
        let remainder = hasher.finalize() % banks.count
        return remainder < 0 ? remainder + banks.count : remainder
    }

    /// Reads every request concurrently and returns the successfully
    /// materialized tensors keyed by `Request.cacheKey`. Failed requests are
    /// simply absent; callers fall back to the loader's main-bank path, which
    /// reproduces the original single-threaded behavior and error reporting.
    public func materializedTensors(_ requests: [Request]) -> [String: MaterializedTensor] {
        guard !requests.isEmpty else {
            return [:]
        }
        var partitions = [[Request]](repeating: [], count: banks.count)
        for request in requests {
            partitions[bankIndex(for: request)].append(request)
        }

        let mergeLock = NSLock()
        var merged: [String: MaterializedTensor] = [:]
        merged.reserveCapacity(requests.count)
        let group = DispatchGroup()
        for (bankIdx, partition) in partitions.enumerated() where !partition.isEmpty {
            group.enter()
            let bank = banks[bankIdx]
            queue.async {
                defer { group.leave() }
                var loaded: [(String, MaterializedTensor)] = []
                loaded.reserveCapacity(partition.count)
                for item in partition {
                    let tensor: MaterializedTensor?
                    if let expertIndex = item.expertIndex {
                        tensor = try? bank.materializedTensor(
                            named: item.recordName,
                            firstAxisIndex: expertIndex
                        )
                    } else {
                        tensor = try? bank.materializedTensor(named: item.recordName)
                    }
                    if let tensor {
                        loaded.append((item.cacheKey, tensor))
                    }
                }
                mergeLock.lock()
                for (key, tensor) in loaded {
                    merged[key] = tensor
                }
                mergeLock.unlock()
            }
        }
        group.wait()
        return merged
    }
}
