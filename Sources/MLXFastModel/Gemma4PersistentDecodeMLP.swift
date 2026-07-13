import Foundation
import MLX

/// Qualification result for the proposed grid-synchronized decode MLP.
///
/// `MLXFast.metalKernel` deliberately exposes an array primitive rather than
/// its compiled `MTLComputePipelineState`. Consequently clients cannot query
/// `maxTotalThreadsPerThreadgroup`, static threadgroup memory, or the device's
/// `maxTotalThreadsPerThreadgroup`/resident-threadgroup bound for the compiled
/// function. Without those values there is no way to prove that every group in
/// a grid-wide atomic barrier is concurrently resident. Launching such a
/// kernel would permit a scheduled group to wait for an unscheduled group and
/// deadlock the command queue.
struct PersistentDecodeMLPQualification: Sendable {
    let isQualified: Bool
    let detail: String

    static let unavailable = PersistentDecodeMLPQualification(
        isQualified: false,
        detail: "MLXFast.metalKernel does not expose compiled-pipeline occupancy; "
            + "a fully resident grid cannot be proved"
    )
}

/// Safe, fail-closed representation of the persistent decode MLP experiment.
///
/// The requested primitive is intentionally not authored as an unbounded
/// spin-barrier kernel. The exact-shape and metadata checks remain here so a
/// future MLX API that supplies a guaranteed resident-grid bound can enable the
/// implementation without weakening routing contracts. Production routing
/// must test `qualification.isQualified`, which is false with the current API.
struct PersistentDecodeMLP: @unchecked Sendable {
    let qualification: PersistentDecodeMLPQualification

    init?(
        gate: FastQuantizedProjection,
        up: FastQuantizedProjection,
        down: FastQuantizedProjection,
        gateMetadata: IndexedAffineMetadata,
        upMetadata: IndexedAffineMetadata,
        downMetadata: IndexedAffineMetadata
    ) {
        guard supportsGemma4FusedGateUp(gate: gate, up: up),
              supportsGemma4IndexedDown(
                  projection: down,
                  metadata: downMetadata
              ),
              gateMetadata.indices.dtype == .uint16,
              gateMetadata.indices.shape == [21_504, 84],
              upMetadata.indices.dtype == .uint16,
              upMetadata.indices.shape == [21_504, 84]
        else {
            return nil
        }
        self.qualification = .unavailable
    }

    var isQualified: Bool { qualification.isQualified }
}
