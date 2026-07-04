import Foundation
import MLXFastCore

/// Whole-stacked-tensor staging for prefill-shaped expert consumption.
///
/// A long prefill touches essentially every routed expert in every layer, so
/// reading a layer's experts as ~770 individual 0.5-4.5 MB slice preads wastes
/// the SSD: each pays open/fstat/pread/close and arrives in routing order
/// rather than file order. This stager instead reads each stacked expert code
/// tensor (~1 GiB) as ONE sequential pread on a background thread, one layer
/// ahead of consumption, and the consumer builds per-expert weights from
/// no-copy Data views into the staged buffer.
///
/// Byte identity: the trusted bank's firstAxisIndex slice read returns
/// bytes[byteOffset + i*(byteLength/firstDim), +byteLength/firstDim); a
/// whole-tensor read returns [byteOffset, byteOffset+byteLength), so the CPU
/// sub-range is the same file bytes by arithmetic identity. Weights built
/// from staged views therefore feed the exact bytes — and the exact same
/// kernels — as the streamed path.
///
/// Reads go through a dedicated capacity-0 ExpertSlotBank sharing the
/// loader's metrics: no LRU is mutated (capacity-0 banks never insert, which
/// also makes them safe to use from this background thread) and every staged
/// byte is recorded honestly on the same counters the benchmark reports.
public final class ExpertLayerStager {
    public struct LayerPlan {
        public let layerIndex: Int
        public let recordNames: [String]

        public init(layerIndex: Int, recordNames: [String]) {
            self.layerIndex = layerIndex
            self.recordNames = recordNames
        }
    }

    private let sideBank: ExpertSlotBank
    private let queue = DispatchQueue(label: "mlxfast.expert.stager", qos: .userInitiated)
    private let condition = NSCondition()
    // All three guarded by `condition`.
    private var stagedBytesByRecordName: [String: Data] = [:]
    private var recordNamesByLayer: [Int: [String]] = [:]
    private var pendingLayers: Set<Int> = []
    private var failedLayers: Set<Int> = []

    public init?(manifestPath: String, metrics: ExpertStreamingMetrics?) {
        guard let bank = try? ExpertSlotBank(
            manifestPath: manifestPath,
            capacity: 0,
            metrics: metrics
        ) else {
            return nil
        }
        self.sideBank = bank
    }

    /// Enqueues background staging for a layer. Non-blocking, so callers can
    /// schedule before their routing sync and let the sequential reads overlap
    /// GPU work. Duplicate schedules are ignored.
    public func schedule(_ plan: LayerPlan) {
        guard !plan.recordNames.isEmpty else {
            return
        }
        condition.lock()
        if !failedLayers.contains(plan.layerIndex) {
            scheduleLocked(plan)
        }
        condition.unlock()
    }

    /// Blocks until a scheduled layer is staged. Returns false when staging
    /// failed (or was never scheduled) — callers must fall back to the
    /// per-slice streaming path, which reproduces today's behavior exactly.
    public func waitForLayer(_ layerIndex: Int) -> Bool {
        condition.lock()
        while pendingLayers.contains(layerIndex) {
            condition.wait()
        }
        let isStaged = recordNamesByLayer[layerIndex] != nil
        failedLayers.remove(layerIndex)
        condition.unlock()
        return isStaged
    }

    /// Whole-tensor bytes for a staged record, or nil when not staged.
    public func stagedBytes(recordName: String) -> Data? {
        condition.lock()
        defer { condition.unlock() }
        return stagedBytesByRecordName[recordName]
    }

    /// Frees a layer's staged byte buffers while keeping the layer marked as
    /// staged (waitForLayer keeps returning true). For callers that already
    /// copied the bytes into derived arrays, the staged Data is dead weight
    /// until releaseLayer; dropping it early keeps ~3 GiB out of the peak
    /// while the next layer stages.
    public func releaseStagedBytesKeepingLayer(_ layerIndex: Int) {
        condition.lock()
        if let names = recordNamesByLayer[layerIndex] {
            for name in names {
                stagedBytesByRecordName.removeValue(forKey: name)
            }
        }
        condition.unlock()
    }

    /// Frees a consumed layer's staged buffers.
    public func releaseLayer(_ layerIndex: Int) {
        condition.lock()
        if let names = recordNamesByLayer.removeValue(forKey: layerIndex) {
            for name in names {
                stagedBytesByRecordName.removeValue(forKey: name)
            }
        }
        condition.unlock()
    }

    private final class StagerFailureFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var failed = false
        var value: Bool {
            lock.lock(); defer { lock.unlock() }; return failed
        }
        func set() {
            lock.lock(); failed = true; lock.unlock()
        }
    }

    // Workers write disjoint byte ranges of preallocated buffers, so the
    // aliasing is disjoint and the unchecked Sendable conformance is sound.
    private struct StagerBufferSink: @unchecked Sendable {
        let buffers: [UnsafeMutableRawPointer]
    }

    private func scheduleLocked(_ plan: LayerPlan) {
        guard
            recordNamesByLayer[plan.layerIndex] == nil,
            !pendingLayers.contains(plan.layerIndex)
        else {
            return
        }
        pendingLayers.insert(plan.layerIndex)
        queue.async { [self] in
            // Read each ~1 GiB projection tensor as MANY concurrent
            // first-axis slice reads through the trusted side bank instead of
            // one whole-tensor pread per record. Rationale: the official
            // runner's disk throughput scales with queue depth (serial slice
            // reads measured ~0.8 GB/s, 18-wide concurrent decode reads
            // several times that), so 3 sequential streams underdrive it,
            // while local Apple SSDs saturate either way. Each worker copies
            // its slice into the assembled buffer at the bank's own slice
            // offset (byteLength / firstDimension), so the assembled bytes
            // are identical to the whole-tensor read byte-for-byte, and
            // every byte is metered by the same counters. Records that are
            // not cleanly sliceable fall back to one whole-tensor read.
            let names = plan.recordNames
            struct SliceTask {
                let recordIndex: Int
                let sliceIndex: Int  // -1 => whole-tensor read
            }
            var buffers: [UnsafeMutableRawPointer] = []
            var byteLengths: [Int] = []
            var sliceLengths: [Int] = []
            var tasks: [SliceTask] = []
            var planFailed = false
            for (recordIndex, name) in names.enumerated() {
                guard let record = sideBank.record(named: name), record.byteLength > 0 else {
                    planFailed = true
                    break
                }
                buffers.append(
                    UnsafeMutableRawPointer.allocate(byteCount: record.byteLength, alignment: 16)
                )
                byteLengths.append(record.byteLength)
                if let firstDimension = record.shape.first,
                   firstDimension > 1,
                   record.shape.count >= 2,
                   record.byteLength % firstDimension == 0 {
                    sliceLengths.append(record.byteLength / firstDimension)
                    for sliceIndex in 0..<firstDimension {
                        tasks.append(SliceTask(recordIndex: recordIndex, sliceIndex: sliceIndex))
                    }
                } else {
                    sliceLengths.append(record.byteLength)
                    tasks.append(SliceTask(recordIndex: recordIndex, sliceIndex: -1))
                }
            }
            let failed = StagerFailureFlag()
            if planFailed {
                failed.set()
            } else {
                let sink = StagerBufferSink(buffers: buffers)
                DispatchQueue.concurrentPerform(iterations: tasks.count) { taskIndex in
                    let task = tasks[taskIndex]
                    let name = names[task.recordIndex]
                    let tensor: MaterializedTensor?
                    let offset: Int
                    let expected: Int
                    if task.sliceIndex >= 0 {
                        tensor = try? sideBank.materializedTensor(
                            named: name,
                            firstAxisIndex: task.sliceIndex
                        )
                        offset = task.sliceIndex * sliceLengths[task.recordIndex]
                        expected = sliceLengths[task.recordIndex]
                    } else {
                        tensor = try? sideBank.materializedTensor(named: name)
                        offset = 0
                        expected = byteLengths[task.recordIndex]
                    }
                    guard let tensor, tensor.bytes.count == expected else {
                        failed.set()
                        return
                    }
                    tensor.bytes.withUnsafeBytes { source in
                        guard let base = source.baseAddress else {
                            failed.set()
                            return
                        }
                        sink.buffers[task.recordIndex]
                            .advanced(by: offset)
                            .copyMemory(from: base, byteCount: expected)
                    }
                }
            }
            var loaded: [String: Data] = [:]
            let succeeded = !failed.value && !planFailed
            if succeeded {
                for (index, name) in names.enumerated() {
                    loaded[name] = Data(
                        bytesNoCopy: buffers[index],
                        count: byteLengths[index],
                        deallocator: .custom { pointer, _ in pointer.deallocate() }
                    )
                }
            } else {
                for buffer in buffers {
                    buffer.deallocate()
                }
            }
            condition.lock()
            pendingLayers.remove(plan.layerIndex)
            if succeeded {
                recordNamesByLayer[plan.layerIndex] = plan.recordNames
                for (name, bytes) in loaded {
                    stagedBytesByRecordName[name] = bytes
                }
            } else {
                failedLayers.insert(plan.layerIndex)
            }
            condition.broadcast()
            condition.unlock()
        }
    }
}
