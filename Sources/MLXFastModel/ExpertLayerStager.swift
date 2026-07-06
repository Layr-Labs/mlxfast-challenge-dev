import Foundation
import MLXFastCore

/// Whole-layer staging for prefill-shaped expert consumption.
///
/// A long prefill touches most routed experts in every layer, so reading a
/// layer's experts as ~770 individual 0.5-4.5 MB slice preads wastes the SSD:
/// each pays open/fstat/pread/close and arrives in routing order rather than
/// file order. This stager instead reads each stacked expert tensor as a
/// small number of large sequential preads on a background thread, one layer
/// ahead of consumption, and the consumer builds per-expert weights from
/// no-copy Data views into the staged buffers.
///
/// Stacked CODE tensors with span-alias manifest records are read as
/// file-order 32-expert spans (one trusted first-axis slice pread per span,
/// ~1/8 of the tensor each). Everything else (scales/biases companions, code
/// tensors without aliases) is read whole, exactly as before. Once a
/// forward's routing indices are known, `restrict` lets the in-flight layer
/// skip the not-yet-read spans' UNUSED experts: remaining spans are replaced
/// by their used experts' maximal runs, each read as aligned power-of-two
/// blocks through the matching span alias. Skipped experts feed no kernel —
/// the prefill consumers iterate only routed experts — so consumed values
/// are unchanged.
///
/// Byte identity: the trusted bank's firstAxisIndex slice read returns
/// bytes[byteOffset + i*(byteLength/firstDim), +byteLength/firstDim). A span
/// alias with shape [E/g, g*rows, cols] shares the base record's
/// shard/offset/length, so its index-j slice is exactly the file bytes of
/// experts [j*g, (j+1)*g) — the same bytes as g consecutive base-record
/// slices or the matching whole-tensor subrange, by arithmetic identity.
/// Weights built from staged views therefore feed the exact bytes — and the
/// exact same kernels — as the streamed path.
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

    /// The contiguous staged bytes of experts
    /// [expertStart, expertStart+expertCount) of one stacked code record.
    public struct StagedSpan {
        public let expertStart: Int
        public let expertCount: Int
        public let bytes: Data
    }

    /// One staged record: either the whole tensor's bytes or an ascending,
    /// disjoint list of expert spans (span-read code tensors).
    private enum StagedRecord {
        case whole(Data)
        case spans([StagedSpan])
    }

    /// File-order read granularity for span-eligible code records; matches
    /// the largest emitted `.mlxspan` alias (256 experts -> 8 preads/record).
    private static let spanExpertCount = 32
    private static let spanAliasGroups = [32, 16, 8, 4, 2]

    private let sideBank: ExpertSlotBank
    private let queue = DispatchQueue(label: "mlxfast.expert.stager", qos: .userInitiated)
    private let condition = NSCondition()
    // All guarded by `condition`.
    private var stagedRecordsByName: [String: StagedRecord] = [:]
    private var recordNamesByLayer: [Int: [String]] = [:]
    private var pendingLayers: Set<Int> = []
    private var failedLayers: Set<Int> = []
    // Per-layer used-expert restriction for the layer's IN-FLIGHT stage job,
    // set from the forward's already-synced routing. Only spans not yet
    // started honor it; cleared when the job completes or the layer is
    // released, so cross-forward captures (whose routing is unknown) always
    // stage complete layers.
    private var restrictedUsedExpertsByLayer: [Int: Set<Int>] = [:]
    // Bumped by releaseAllStagedLayers; staging jobs scheduled under an older
    // generation bail at start and discard at store, so stale cross-forward
    // captures cannot spill work into the decode window.
    private var generation = 0

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

    /// Restricts the layer's in-flight stage job to the given used experts:
    /// spans that have not started reading skip every other expert. No-op
    /// when the layer's job already finished (or was never scheduled), so a
    /// later unrestricted capture of the same layer is unaffected. Callers
    /// pass THIS forward's exact routed-expert set, so no consumed expert is
    /// ever skipped.
    public func restrict(layerIndex: Int, usedExperts: Set<Int>) {
        guard !usedExperts.isEmpty else {
            return
        }
        condition.lock()
        if pendingLayers.contains(layerIndex) {
            restrictedUsedExpertsByLayer[layerIndex] = usedExperts
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

    /// Whole-tensor bytes for a staged record, or nil when not staged (or
    /// staged as spans).
    public func stagedBytes(recordName: String) -> Data? {
        condition.lock()
        defer { condition.unlock() }
        guard case .whole(let bytes)? = stagedRecordsByName[recordName] else {
            return nil
        }
        return bytes
    }

    /// Ascending, disjoint staged expert spans for a span-read record, or nil
    /// when not staged (or staged whole).
    public func stagedSpans(recordName: String) -> [StagedSpan]? {
        condition.lock()
        defer { condition.unlock() }
        guard case .spans(let spans)? = stagedRecordsByName[recordName] else {
            return nil
        }
        return spans
    }

    /// Frees every staged layer and cancels queued stage jobs. Called at
    /// one-token decode entry: decode never consumes staged data, so anything
    /// staged there is a stale cross-forward capture.
    public func releaseAllStagedLayers() {
        condition.lock()
        generation += 1
        recordNamesByLayer.removeAll()
        stagedRecordsByName.removeAll()
        pendingLayers.removeAll()
        restrictedUsedExpertsByLayer.removeAll()
        condition.broadcast()
        condition.unlock()
    }

    /// Frees a consumed layer's staged buffers.
    public func releaseLayer(_ layerIndex: Int) {
        condition.lock()
        restrictedUsedExpertsByLayer.removeValue(forKey: layerIndex)
        if let names = recordNamesByLayer.removeValue(forKey: layerIndex) {
            for name in names {
                stagedRecordsByName.removeValue(forKey: name)
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

    private struct StagerResultSink: @unchecked Sendable {
        let buffer: UnsafeMutableBufferPointer<StagedRecord?>
    }

    private func scheduleLocked(_ plan: LayerPlan) {
        guard
            recordNamesByLayer[plan.layerIndex] == nil,
            !pendingLayers.contains(plan.layerIndex)
        else {
            return
        }
        pendingLayers.insert(plan.layerIndex)
        let scheduledGeneration = generation
        queue.async { [self] in
            condition.lock()
            let stale = scheduledGeneration != generation
            condition.unlock()
            if stale {
                return
            }
            // Read the layer's projection tensors concurrently instead of one
            // after another. The side bank is capacity 0, so it never mutates
            // its cache/LRU and each read is an independent open/pread/close
            // through the trusted read path (metrics are NSLock-guarded);
            // concurrent reads are therefore race-free and stage
            // byte-identical data — only the order/overlap of the reads
            // changes, so consumers see the exact same bytes.
            let names = plan.recordNames
            var results = [StagedRecord?](repeating: nil, count: names.count)
            let failed = StagerFailureFlag()
            results.withUnsafeMutableBufferPointer { buffer in
                let sink = StagerResultSink(buffer: buffer)
                DispatchQueue.concurrentPerform(iterations: names.count) { index in
                    // Per-record cancellation check: a cancelled cross-forward
                    // capture (decode entry bumped the generation) stops
                    // between records instead of finishing the whole ~3.2 GB
                    // layer, bounding in-window churn to one record's read.
                    // Live stage jobs never observe a bumped generation, so
                    // the consumed path is unchanged.
                    condition.lock()
                    let cancelled = scheduledGeneration != generation
                    condition.unlock()
                    if cancelled {
                        failed.set()
                        return
                    }
                    if let staged = stageRecord(
                        named: names[index],
                        layerIndex: plan.layerIndex,
                        scheduledGeneration: scheduledGeneration
                    ) {
                        sink.buffer[index] = staged
                    } else {
                        failed.set()
                    }
                }
            }
            var loaded: [String: StagedRecord] = [:]
            var succeeded = !failed.value
            if succeeded {
                for (index, name) in names.enumerated() {
                    guard let staged = results[index] else {
                        succeeded = false
                        break
                    }
                    loaded[name] = staged
                }
            }
            condition.lock()
            pendingLayers.remove(plan.layerIndex)
            restrictedUsedExpertsByLayer.removeValue(forKey: plan.layerIndex)
            if scheduledGeneration != generation {
                // Cancelled mid-read: discard rather than store stale bytes.
            } else if succeeded {
                recordNamesByLayer[plan.layerIndex] = plan.recordNames
                for (name, staged) in loaded {
                    stagedRecordsByName[name] = staged
                }
            } else {
                failedLayers.insert(plan.layerIndex)
            }
            condition.broadcast()
            condition.unlock()
        }
    }

    /// Stages one record: span-eligible code tensors as file-order expert
    /// spans (honoring a mid-flight restriction), everything else whole.
    private func stageRecord(
        named name: String,
        layerIndex: Int,
        scheduledGeneration: Int
    ) -> StagedRecord? {
        guard
            let record = sideBank.record(named: name),
            record.dtype == "U32",
            record.shape.count == 3,
            let expertCount = record.shape.first,
            expertCount > 0,
            expertCount % Self.spanExpertCount == 0,
            record.byteLength % expertCount == 0,
            sideBank.record(named: Self.spanAliasName(name, Self.spanExpertCount)) != nil
        else {
            // Whole-tensor read: byte-identical to the pre-span stager.
            guard let tensor = try? sideBank.materializedTensor(named: name) else {
                return nil
            }
            return .whole(tensor.bytes)
        }

        var spans: [StagedSpan] = []
        spans.reserveCapacity(expertCount / Self.spanExpertCount)
        for spanIndex in 0..<(expertCount / Self.spanExpertCount) {
            condition.lock()
            let cancelled = scheduledGeneration != generation
            let restriction = restrictedUsedExpertsByLayer[layerIndex]
            condition.unlock()
            if cancelled {
                return nil
            }
            let start = spanIndex * Self.spanExpertCount
            let end = start + Self.spanExpertCount
            if let restriction {
                // Replace this not-yet-read span with its used experts'
                // maximal runs (the sentinel iteration flushes a run ending
                // at the span boundary).
                var runStart = -1
                for expert in start...end {
                    let used = expert < end && restriction.contains(expert)
                    if used {
                        if runStart < 0 {
                            runStart = expert
                        }
                    } else if runStart >= 0 {
                        guard appendRunSpans(
                            recordName: name,
                            runStart: runStart,
                            runEnd: expert,
                            into: &spans
                        ) else {
                            return nil
                        }
                        runStart = -1
                    }
                }
            } else {
                guard let bytes = readExpertBlock(recordName: name, start: start, count: Self.spanExpertCount) else {
                    return nil
                }
                spans.append(StagedSpan(expertStart: start, expertCount: Self.spanExpertCount, bytes: bytes))
            }
        }
        return .spans(spans)
    }

    /// Reads the used-run [runStart, runEnd) as aligned power-of-two blocks
    /// (largest alias first), appending one span per block.
    private func appendRunSpans(
        recordName: String,
        runStart: Int,
        runEnd: Int,
        into spans: inout [StagedSpan]
    ) -> Bool {
        var position = runStart
        while position < runEnd {
            var group = Self.spanExpertCount
            while group > 1, position % group != 0 || position + group > runEnd {
                group /= 2
            }
            guard let bytes = readExpertBlock(recordName: recordName, start: position, count: group) else {
                return false
            }
            spans.append(StagedSpan(expertStart: position, expertCount: group, bytes: bytes))
            position += group
        }
        return true
    }

    /// One trusted pread of experts [start, start+count): count>1 goes
    /// through the matching span alias's first-axis slice, count==1 through
    /// the base record's — the same file bytes either way (see the class
    /// doc's arithmetic identity). A missing alias splits into two aligned
    /// halves.
    private func readExpertBlock(recordName: String, start: Int, count: Int) -> Data? {
        if count == 1 {
            return (try? sideBank.materializedTensor(named: recordName, firstAxisIndex: start))?.bytes
        }
        let alias = Self.spanAliasName(recordName, count)
        guard sideBank.record(named: alias) != nil else {
            let half = count / 2
            guard
                half >= 1,
                let low = readExpertBlock(recordName: recordName, start: start, count: half),
                let high = readExpertBlock(recordName: recordName, start: start + half, count: half)
            else {
                return nil
            }
            return low + high
        }
        return (try? sideBank.materializedTensor(named: alias, firstAxisIndex: start / count))?.bytes
    }

    private static func spanAliasName(_ recordName: String, _ group: Int) -> String {
        "\(recordName).mlxspan\(group)"
    }
}
