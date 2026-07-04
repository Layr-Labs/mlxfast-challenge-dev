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
    /// Describes an optional gate+up fusion for a staged layer: the stager's
    /// background thread interleaves the two stacked code tensors (and their
    /// RAM-resident scales) per expert into ONE `[E, 2*rows, packed]` host
    /// buffer, so the batched prefill path runs one first-projection QMM per
    /// expert instead of two. Pure byte relocation: expert e's gate slice is
    /// the identical file bytes at a different host offset.
    public struct FusedFirstProjectionPlan {
        public let gateName: String
        public let upName: String
        public let gateScalesName: String
        public let upScalesName: String
        public let expertCount: Int

        public init(
            gateName: String,
            upName: String,
            gateScalesName: String,
            upScalesName: String,
            expertCount: Int
        ) {
            self.gateName = gateName
            self.upName = upName
            self.gateScalesName = gateScalesName
            self.upScalesName = upScalesName
            self.expertCount = expertCount
        }
    }

    public struct LayerPlan {
        public let layerIndex: Int
        public let recordNames: [String]
        public let fusedFirstProjection: FusedFirstProjectionPlan?

        public init(
            layerIndex: Int,
            recordNames: [String],
            fusedFirstProjection: FusedFirstProjectionPlan? = nil
        ) {
            self.layerIndex = layerIndex
            self.recordNames = recordNames
            self.fusedFirstProjection = fusedFirstProjection
        }
    }

    /// A fused gate+up staged layer buffer. `codes`/`scales` hold, per expert
    /// e, gate_e's rows immediately followed by up_e's rows — the exact bytes
    /// the separate staged buffers held, re-ordered on the host. The separate
    /// gate/up whole buffers are dropped after fusion, so steady staged RAM is
    /// unchanged.
    private struct FusedLayerEntry {
        let plan: FusedFirstProjectionPlan
        let codes: Data
        let scales: Data
        // Per-expert byte length of ONE projection's code rows (gate == up).
        let codeSliceBytes: Int
    }

    private let sideBank: ExpertSlotBank
    private let residentScales: ResidentExpertTensors?
    private let queue = DispatchQueue(label: "mlxfast.expert.stager", qos: .userInitiated)
    private let condition = NSCondition()
    // All guarded by `condition`.
    private var stagedBytesByRecordName: [String: Data] = [:]
    private var recordNamesByLayer: [Int: [String]] = [:]
    private var pendingLayers: Set<Int> = []
    private var failedLayers: Set<Int> = []
    private var fusedByLayer: [Int: FusedLayerEntry] = [:]
    // recordName -> (layerIndex, isUp) for serving per-expert fallback slices
    // out of the fused buffer.
    private var fusedCodeRecordNames: [String: (layerIndex: Int, isUp: Bool)] = [:]

    public init?(
        manifestPath: String,
        metrics: ExpertStreamingMetrics?,
        residentScales: ResidentExpertTensors? = nil
    ) {
        guard let bank = try? ExpertSlotBank(
            manifestPath: manifestPath,
            capacity: 0,
            metrics: metrics
        ) else {
            return nil
        }
        self.sideBank = bank
        self.residentScales = residentScales
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

    /// Whole-tensor bytes for a staged record, or nil when not staged. For a
    /// fused layer's gate/up code records this returns nil (their contiguous
    /// whole-tensor bytes no longer exist); the fused accessors below serve
    /// those layers.
    public func stagedBytes(recordName: String) -> Data? {
        condition.lock()
        defer { condition.unlock() }
        return stagedBytesByRecordName[recordName]
    }

    /// The fused `[E, 2*rows, packed]` gate+up code/scales buffers for a
    /// staged layer, or nil when the layer was not staged with fusion.
    public func fusedFirstProjection(layerIndex: Int) -> (codes: Data, scales: Data, expertCount: Int)? {
        condition.lock()
        defer { condition.unlock() }
        guard let entry = fusedByLayer[layerIndex] else {
            return nil
        }
        return (entry.codes, entry.scales, entry.plan.expertCount)
    }

    /// Per-expert code slice for a fused layer's gate or up record: the same
    /// bytes the separate staged buffer's `byteLength / firstDim` slice held,
    /// served as a no-copy view at the fused offset. nil for non-fused records
    /// (callers then use `stagedBytes` exactly as before).
    public func fusedSliceBytes(recordName: String, expertIndex: Int) -> Data? {
        condition.lock()
        defer { condition.unlock() }
        guard
            let location = fusedCodeRecordNames[recordName],
            let entry = fusedByLayer[location.layerIndex],
            expertIndex >= 0,
            expertIndex < entry.plan.expertCount
        else {
            return nil
        }
        let slice = entry.codeSliceBytes
        let start = entry.codes.startIndex + expertIndex * 2 * slice + (location.isUp ? slice : 0)
        return entry.codes[start..<(start + slice)]
    }

    /// Frees a consumed layer's staged buffers.
    public func releaseLayer(_ layerIndex: Int) {
        condition.lock()
        if let names = recordNamesByLayer.removeValue(forKey: layerIndex) {
            for name in names {
                stagedBytesByRecordName.removeValue(forKey: name)
            }
        }
        if let fused = fusedByLayer.removeValue(forKey: layerIndex) {
            fusedCodeRecordNames.removeValue(forKey: fused.plan.gateName)
            fusedCodeRecordNames.removeValue(forKey: fused.plan.upName)
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
        let buffer: UnsafeMutableBufferPointer<Data?>
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
            // Read the layer's ~1 GiB projection tensors concurrently instead
            // of one after another. The side bank is capacity 0, so it never
            // mutates its cache/LRU and each read is an independent
            // open/pread/close through the trusted read path (metrics are
            // NSLock-guarded); concurrent reads are therefore race-free and
            // stage byte-identical data — only the order/overlap of the reads
            // changes, so consumers see the exact same bytes.
            let names = plan.recordNames
            var results = [Data?](repeating: nil, count: names.count)
            let failed = StagerFailureFlag()
            results.withUnsafeMutableBufferPointer { buffer in
                let sink = StagerResultSink(buffer: buffer)
                DispatchQueue.concurrentPerform(iterations: names.count) { index in
                    if let tensor = try? sideBank.materializedTensor(named: names[index]) {
                        sink.buffer[index] = tensor.bytes
                    } else {
                        failed.set()
                    }
                }
            }
            var loaded: [String: Data] = [:]
            var succeeded = !failed.value
            if succeeded {
                for (index, name) in names.enumerated() {
                    guard let bytes = results[index] else {
                        succeeded = false
                        break
                    }
                    loaded[name] = bytes
                }
            }
            // Optional gate+up fusion: interleave the two projections' bytes
            // per expert on this background thread (overlapped with GPU work),
            // then DROP the separate whole buffers so steady staged RAM is
            // unchanged. Strictly best-effort: any mismatch leaves the layer
            // staged exactly as before.
            var fusedEntry: FusedLayerEntry?
            if succeeded, let fusion = plan.fusedFirstProjection {
                fusedEntry = buildFusedEntry(fusion: fusion, loaded: loaded)
                if fusedEntry != nil {
                    loaded.removeValue(forKey: fusion.gateName)
                    loaded.removeValue(forKey: fusion.upName)
                }
            }
            condition.lock()
            pendingLayers.remove(plan.layerIndex)
            if succeeded {
                recordNamesByLayer[plan.layerIndex] = plan.recordNames
                for (name, bytes) in loaded {
                    stagedBytesByRecordName[name] = bytes
                }
                if let fusedEntry {
                    fusedByLayer[plan.layerIndex] = fusedEntry
                    fusedCodeRecordNames[fusedEntry.plan.gateName] = (plan.layerIndex, false)
                    fusedCodeRecordNames[fusedEntry.plan.upName] = (plan.layerIndex, true)
                }
            } else {
                failedLayers.insert(plan.layerIndex)
            }
            condition.broadcast()
            condition.unlock()
        }
    }

    /// Assembles the fused gate+up buffers for one staged layer. Byte
    /// identity: for each expert e, copies gate's slice
    /// [e*slice, (e+1)*slice) then up's slice — the same per-expert ranges the
    /// bank's firstAxisIndex slice arithmetic defines — into consecutive fused
    /// offsets. Values are never touched; only their host location changes.
    private func buildFusedEntry(
        fusion: FusedFirstProjectionPlan,
        loaded: [String: Data]
    ) -> FusedLayerEntry? {
        guard
            let gateCodes = loaded[fusion.gateName],
            let upCodes = loaded[fusion.upName],
            let fusedCodes = Self.interleavedPerExpert(
                gateCodes, upCodes, expertCount: fusion.expertCount
            ),
            let gateScales = residentScales?.materializedTensor(
                named: fusion.gateScalesName, firstAxisIndex: nil
            )?.bytes,
            let upScales = residentScales?.materializedTensor(
                named: fusion.upScalesName, firstAxisIndex: nil
            )?.bytes,
            let fusedScales = Self.interleavedPerExpert(
                gateScales, upScales, expertCount: fusion.expertCount
            )
        else {
            return nil
        }
        return FusedLayerEntry(
            plan: fusion,
            codes: fusedCodes,
            scales: fusedScales,
            codeSliceBytes: gateCodes.count / fusion.expertCount
        )
    }

    /// fused[e] = a[e] || b[e] for e in 0..<expertCount, where x[e] is the
    /// e-th first-axis slice of length (x.count / expertCount). Pure memcpy.
    private static func interleavedPerExpert(_ a: Data, _ b: Data, expertCount: Int) -> Data? {
        guard
            expertCount > 0,
            a.count == b.count,
            a.count > 0,
            a.count % expertCount == 0
        else {
            return nil
        }
        let slice = a.count / expertCount
        var fused = Data(count: 2 * a.count)
        fused.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
            guard let destinationBase = destination.baseAddress else { return }
            a.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                guard let sourceBase = source.baseAddress else { return }
                for expert in 0..<expertCount {
                    destinationBase.advanced(by: expert * 2 * slice)
                        .copyMemory(from: sourceBase.advanced(by: expert * slice), byteCount: slice)
                }
            }
            b.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                guard let sourceBase = source.baseAddress else { return }
                for expert in 0..<expertCount {
                    destinationBase.advanced(by: expert * 2 * slice + slice)
                        .copyMemory(from: sourceBase.advanced(by: expert * slice), byteCount: slice)
                }
            }
        }
        return fused
    }
}
