import Foundation
import MLX

struct CombinedGateUpPrefillWeights: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// One transform-authored affine QMM for multi-token gate/up projection. The
/// output split is view-only and preserves the BF16 boundary before GELU.
struct CombinedGateUpPrefillProjection: @unchecked Sendable {
    private let combined: FastQuantizedProjection
    private let gate: FastQuantizedProjection
    private let up: FastQuantizedProjection
    private let gateWidth: Int
    private let enabled: Bool
    private let verifyBits: Bool

    init?(
        weights: CombinedGateUpPrefillWeights,
        gate: FastQuantizedProjection,
        up: FastQuantizedProjection
    ) {
        guard weights.weight.dtype == .uint32,
              weights.weight.shape == [43_008, 672],
              weights.scales.dtype == .bfloat16,
              weights.scales.shape == [43_008, 84],
              weights.biases.dtype == .bfloat16,
              weights.biases.shape == [43_008, 84],
              gate.groupSize == 64, up.groupSize == 64,
              gate.bits == 4, up.bits == 4,
              gate.weight.shape == [21_504, 672],
              up.weight.shape == [21_504, 672],
              gate.scales.shape == [21_504, 84],
              up.scales.shape == [21_504, 84]
        else { return nil }
        self.combined = FastQuantizedProjection(
            weight: weights.weight,
            scales: weights.scales,
            biases: weights.biases,
            groupSize: 64,
            bits: 4
        )
        self.gate = gate
        self.up = up
        self.gateWidth = 21_504
        let environment = ProcessInfo.processInfo.environment
        self.enabled = !["0", "false", "no", "off"].contains(
            environment["DARKBLOOM_COMBINED_GATE_UP_PREFILL"]?.lowercased() ?? "1")
        self.verifyBits = ["1", "true", "yes", "on"].contains(
            environment["DARKBLOOM_VERIFY_COMBINED_GATE_UP_PREFILL_BITS"]?.lowercased() ?? "0")
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dtype == .bfloat16)
        precondition(input.ndim == 3 && input.dim(0) == 1 && input.dim(1) > 1)
        precondition(input.dim(2) == 5_376)
        let projected = combined(input)
        let candidateGate = projected[.ellipsis, 0..<gateWidth]
        let candidateUp = projected[.ellipsis, gateWidth..<(gateWidth * 2)]
        if verifyBits {
            let referenceGate = gate(input)
            let referenceUp = up(input)
            let gateMatch = arrayEqual(
                candidateGate.view(dtype: .uint16), referenceGate.view(dtype: .uint16))
            let upMatch = arrayEqual(
                candidateUp.view(dtype: .uint16), referenceUp.view(dtype: .uint16))
            eval(gateMatch, upMatch)
            precondition(
                gateMatch.item(Bool.self) && upMatch.item(Bool.self),
                "combined gate/up prefill differs from separate stock QMMs")
            if !enabled { return (referenceGate, referenceUp) }
        }
        return enabled ? (candidateGate, candidateUp) : (gate(input), up(input))
    }
}
