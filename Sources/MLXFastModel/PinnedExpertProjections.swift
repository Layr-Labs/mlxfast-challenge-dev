import Foundation
import MLX
import MLXFastCore

/// GPU-materialized stacked projections for the RAM-pinned hash layers.
///
/// The previous pinned store kept the hash layers' stacked code tensors as
/// host `Data` (~9 GiB) and paid, on EVERY consumption, a Data->Metal copy:
/// ~150 MB per decode step (rebuilt per-expert slice arrays) and ~3 GiB per
/// prefill forward (whole stacked arrays). Unified memory makes that copy
/// pure overhead — the bytes were already resident. This store materializes
/// each pinned layer's three stacked projections ONCE as evaluated MLXArrays
/// (codes as U32 leaves, scales GPU-decoded from the packed overlay to the
/// byte-identical U8 tensor) and serves:
///
/// - decode steps: zero-copy first-axis views (`weight[e]`, `scales[e]`) — no
///   per-step copies, reads, or prebuild scheduling at all;
/// - the batched prefill path: the whole stacked arrays directly.
///
/// Per-expert QMMs over first-axis views of a stacked array compute the same
/// per-row quantized dot products over the same bytes as freshly copied slice
/// arrays — the batched prefill path has always relied on exactly this.
///
/// Loads happen once in the (untimed) loader constructor through a dedicated
/// capacity-0 ExpertSlotBank sharing the loader's metrics, so every byte is
/// recorded as honest misses/bytes on the same counters the benchmark
/// reports. The host copies are dropped after the arrays are evaluated: this
/// REPLACES the old host store rather than duplicating it (~9.6 GiB total
/// either way on the 48 GB runner).
public final class PinnedExpertProjections {
    public struct Projection {
        /// The stacked code record name this projection serves.
        public let name: String
        /// Materialized `[experts, rows, packedU32]` codes.
        public let weight: MLXArray
        /// Materialized `[experts, rows, groups]` U8 e8m0 scales.
        public let scales: MLXArray
        /// Per-expert logical shape (`[rows, logicalInput]`).
        public let logicalShape: [Int]
        public let expertCount: Int
        public let groupSize: Int
        public let bits: Int
        public let mode: QuantizationMode
    }

    private let projectionsByName: [String: Projection]

    public var pinnedProjectionCount: Int {
        projectionsByName.count
    }

    public func projection(named name: String) -> Projection? {
        projectionsByName[name]
    }

    /// Builds the store for the first `hashLayerCount` layers. Returns nil
    /// when any piece is missing or inconsistent — callers then keep the
    /// streaming/staging paths, which reproduce pre-pinning behavior exactly.
    public init?(
        manifestPath: String,
        hashLayerCount: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        metrics: ExpertStreamingMetrics?,
        residentScales: ResidentExpertTensors?
    ) {
        guard
            hashLayerCount > 0,
            hiddenSize > 0,
            intermediateSize > 0,
            let bank = try? ExpertSlotBank(
                manifestPath: manifestPath,
                capacity: 0,
                metrics: metrics
            )
        else {
            return nil
        }

        let bridge = MLXArrayTensorBridge()
        let projections: [(DeepSeekExpertProjection, [Int])] = [
            (.gate, [intermediateSize, hiddenSize]),
            (.up, [intermediateSize, hiddenSize]),
            (.down, [hiddenSize, intermediateSize]),
        ]

        var built: [String: Projection] = [:]
        var arraysToEvaluate: [MLXArray] = []
        for layerIndex in 0..<hashLayerCount {
            for (projection, expectedShape) in projections {
                let candidates = DeepSeekWeightNames.routedExpert(
                    layerIndex: layerIndex,
                    expertIndex: 0,
                    projection: projection
                )
                guard
                    let candidate = candidates.first(where: { bank.record(named: $0) != nil }),
                    let record = bank.record(named: candidate),
                    record.dtype == "U32",
                    record.shape.count == 3,
                    let expertCount = record.shape.first,
                    expertCount > 0,
                    let packedInput = record.shape.last,
                    packedInput > 0,
                    record.shape[1] == expectedShape[0],
                    let logicalInput = expectedShape.last,
                    (packedInput * 32) % logicalInput == 0
                else {
                    return nil
                }
                let bits = packedInput * 32 / logicalInput
                guard [2, 4, 8].contains(bits) else {
                    return nil
                }

                // Biases would need affine handling this store does not model;
                // this checkpoint has none. Bail so callers keep staging.
                let biasesName = Self.companionName(for: candidate, suffix: "biases")
                guard bank.record(named: biasesName) == nil else {
                    return nil
                }

                let scalesName = Self.companionName(for: candidate, suffix: "scales")
                guard
                    let scalesRecord = bank.record(named: scalesName),
                    scalesRecord.dtype == "U8",
                    scalesRecord.shape.count == 3,
                    scalesRecord.shape.first == expertCount,
                    scalesRecord.shape[1] == expectedShape[0],
                    let scaleGroups = scalesRecord.shape.last,
                    scaleGroups > 0,
                    logicalInput % scaleGroups == 0
                else {
                    return nil
                }

                // Codes: one metered whole-tensor read, copied into a Metal
                // buffer; the host Data drops at scope exit.
                guard
                    let codesTensor = try? bank.materializedTensor(named: candidate),
                    let weightArray = try? bridge.makeArray(from: codesTensor)
                else {
                    return nil
                }

                // Scales: GPU palette decode from the packed overlay when
                // available, otherwise the resident raw bytes, otherwise one
                // metered bank read. All byte-identical U8.
                let scalesArray: MLXArray
                if let view = residentScales?.packedScaleView(named: scalesName, firstAxisIndex: nil),
                   let decoded = DeepSeekScaleDecode.scalesArray(from: view) {
                    scalesArray = decoded
                } else if let resident = residentScales?.materializedTensor(named: scalesName, firstAxisIndex: nil),
                          let array = try? bridge.makeArray(from: resident) {
                    scalesArray = array
                } else if let tensor = try? bank.materializedTensor(named: scalesName),
                          let array = try? bridge.makeArray(from: tensor) {
                    scalesArray = array
                } else {
                    return nil
                }

                built[candidate] = Projection(
                    name: candidate,
                    weight: weightArray,
                    scales: scalesArray,
                    logicalShape: expectedShape,
                    expertCount: expertCount,
                    groupSize: logicalInput / scaleGroups,
                    bits: bits,
                    mode: .mxfp4
                )
                arraysToEvaluate.append(weightArray)
                arraysToEvaluate.append(scalesArray)
            }
        }
        guard !built.isEmpty else {
            return nil
        }
        // Materialize everything once, in the untimed constructor: leaf code
        // buffers are already data-backed; the scales decode graphs execute
        // here so steady-state consumers only ever see constant buffers.
        eval(arraysToEvaluate)
        self.projectionsByName = built
    }

    private static func companionName(for weightName: String, suffix: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".\(suffix)"
        }
        return "\(weightName).\(suffix)"
    }
}
