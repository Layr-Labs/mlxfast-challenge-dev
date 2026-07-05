import Foundation
import MLXFastCore

/// One-layer-ahead background construction of the batched prefill path's
/// three stacked projection MLXArrays.
///
/// `DeepSeekRoutedExperts.batchedResidentForward` otherwise pays the whole
/// layer's stacked-projection build — ~3.2 GB of Data->Metal copies through
/// `stackedExpertProjection` — synchronously on the compute thread after
/// `waitForLayer` returns, even though the layer's staged BYTES typically
/// landed well before the consumer arrived (the stager reads one layer
/// ahead). This class converts those bytes into the stacked arrays on a
/// dedicated serial queue as soon as staging completes, then atomically swaps
/// the staged Data for the built arrays (bytes out, arrays in: net-zero
/// steady-state footprint) so the consumer claims ready-made arrays instead
/// of copying inline.
///
/// Bit-identity: the prebuild job invokes the SAME `stackedExpertProjection`
/// function over the SAME staged bytes with the SAME expected shapes the
/// consumer uses today, producing the same MLXArrays the compute thread would
/// have built moments later. Only WHO/WHEN constructs the arrays moves; no
/// kernel, shape, order, or reduction changes anywhere. Background MLXArray
/// construction is an established pattern on this code path (decode-stream
/// prebuilds, concurrent staged-slice builds).
///
/// Lifecycle: arrays live exactly as long as the staged Data they replace —
/// built after staging completes, consumed by the very next forward of that
/// layer, and cleared wholesale at one-token decode entry alongside
/// `releaseAllStagedLayers` (stale jobs bail on the bumped stager
/// generation). Any prebuild failure removes the entry and leaves the staged
/// Data untouched, so the consumer reproduces the existing inline build
/// exactly.
final class StackedProjectionPrebuilder {
    /// Identity token for an in-flight job so a stale job never removes or
    /// publishes over an entry installed by a newer schedule of the same
    /// layer.
    private final class InFlightToken {}

    private enum Entry {
        case inFlight(InFlightToken)
        case ready([DeepSeekWeightLoader.StackedExpertProjection])
    }

    private let stager: ExpertLayerStager
    // Dedicated serial queue — NEVER the stager's read queue, so SSD reads
    // for later layers are never blocked behind a memcpy.
    private let queue = DispatchQueue(
        label: "mlxfast.stacked.prebuild",
        qos: .userInitiated
    )
    private let condition = NSCondition()
    // Guarded by `condition`.
    private var byLayer: [Int: Entry] = [:]

    init(stager: ExpertLayerStager) {
        self.stager = stager
    }

    /// Enqueues a background build of the layer's three stacked projections.
    /// Non-blocking; duplicate schedules are ignored. The job waits for the
    /// layer's staging, builds via the loader's existing
    /// `stackedExpertProjection`, and on success atomically swaps the staged
    /// bytes for the built arrays. On any failure the entry is removed and
    /// the staged Data is left untouched.
    func schedule(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        loader: DeepSeekWeightLoader
    ) {
        let token = InFlightToken()
        condition.lock()
        if byLayer[layerIndex] != nil {
            condition.unlock()
            return
        }
        byLayer[layerIndex] = .inFlight(token)
        condition.unlock()

        let scheduledGeneration = stager.currentGeneration
        queue.async { [self] in
            // Decode entry cancelled everything scheduled before it; bail
            // before touching the stager so no work spills into the decode
            // window.
            if stager.currentGeneration != scheduledGeneration {
                resolve(layerIndex: layerIndex, token: token, triple: nil)
                return
            }
            guard stager.waitForLayerStaged(layerIndex) else {
                resolve(layerIndex: layerIndex, token: token, triple: nil)
                return
            }
            // Identical construction to the consumer's inline path: same
            // 3-way concurrentPerform, same function, same expected shapes.
            var projections = [DeepSeekWeightLoader.StackedExpertProjection?](
                repeating: nil,
                count: 3
            )
            let expectedShapes: [[Int]] = [
                [intermediateSize, hiddenSize],
                [intermediateSize, hiddenSize],
                [hiddenSize, intermediateSize],
            ]
            let kinds: [DeepSeekExpertProjection] = [.gate, .up, .down]
            projections.withUnsafeMutableBufferPointer { buffer in
                let sink = PrebuildProjectionSink(buffer: buffer)
                DispatchQueue.concurrentPerform(iterations: 3) { index in
                    sink.buffer[index] = loader.stackedExpertProjection(
                        layerIndex: layerIndex,
                        projection: kinds[index],
                        expectedShape: expectedShapes[index]
                    )
                }
            }
            guard let gate = projections[0], let up = projections[1],
                  let down = projections[2]
            else {
                resolve(layerIndex: layerIndex, token: token, triple: nil)
                return
            }
            resolve(
                layerIndex: layerIndex,
                token: token,
                triple: [gate, up, down],
                scheduledGeneration: scheduledGeneration
            )
        }
    }

    /// Publishes or clears a job's outcome. Publication is atomic with the
    /// staged-byte release: consumers see either (arrays, no bytes) or
    /// (bytes, no arrays), never neither.
    private func resolve(
        layerIndex: Int,
        token: InFlightToken,
        triple: [DeepSeekWeightLoader.StackedExpertProjection]?,
        scheduledGeneration: Int = -1
    ) {
        condition.lock()
        defer {
            condition.broadcast()
            condition.unlock()
        }
        guard case .inFlight(let installed)? = byLayer[layerIndex],
              installed === token
        else {
            // releaseAll cleared us, or a newer schedule owns the slot.
            return
        }
        if let triple,
           stager.releaseLayerBytesKeepingMarker(
               layerIndex,
               ifGeneration: scheduledGeneration
           )
        {
            byLayer[layerIndex] = .ready(triple)
        } else {
            // Failure (or cancelled mid-build): drop the entry and leave the
            // staged bytes untouched so the consumer falls back to the
            // existing inline build.
            byLayer.removeValue(forKey: layerIndex)
        }
    }

    /// Claims the prebuilt triple for a layer: returns it when ready
    /// (removing it), waits when in-flight (never longer than the consumer
    /// building inline — the job runs the identical build), and returns nil
    /// immediately when absent.
    func claim(layerIndex: Int) -> [DeepSeekWeightLoader.StackedExpertProjection]? {
        condition.lock()
        defer { condition.unlock() }
        while true {
            switch byLayer[layerIndex] {
            case .none:
                return nil
            case .ready(let triple):
                byLayer.removeValue(forKey: layerIndex)
                return triple
            case .inFlight:
                condition.wait()
            }
        }
    }

    /// Drops every prebuilt array reference. Called at one-token decode entry
    /// (alongside `releaseAllStagedLayers`) so no stacked arrays survive into
    /// decode; in-flight jobs bail on the bumped stager generation and clear
    /// themselves via the token check.
    func releaseAll() {
        condition.lock()
        byLayer.removeAll()
        condition.broadcast()
        condition.unlock()
    }
}

// Lets concurrentPerform write the three stacked projections into distinct
// slots from worker threads; each index is written by exactly one iteration.
private struct PrebuildProjectionSink: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<DeepSeekWeightLoader.StackedExpertProjection?>
}
