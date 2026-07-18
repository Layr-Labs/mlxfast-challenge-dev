import Foundation
import MLX

private func gemma4PairedPrefillEnvironmentFlag(_ name: String) -> Bool? {
    guard let raw = ProcessInfo.processInfo.environment[name] else {
        return nil
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}

/// Mirrors the pinned MLX NAX availability contract. The paired projection
/// relies on a narrow epilogue in the runtime-JIT NAX QMM, so falling through
/// to an older GPU's non-NAX kernel would produce the ordinary wide matrix.
private let gemma4PairedGateUpPrefillNAXAvailable: Bool = {
    guard #available(macOS 26.2, *) else { return false }
    let architecture = Array(GPU.deviceInfo().architecture)
    guard architecture.count >= 3,
          let tens = architecture[architecture.count - 3].wholeNumberValue,
          let ones = architecture[architecture.count - 2].wholeNumberValue
    else {
        return false
    }
    let generation = tens * 10 + ones
    return generation >= (architecture.last == "p" ? 18 : 17)
}()

private let gemma4VerifyPairedGateUpPrefillBits =
    gemma4PairedPrefillEnvironmentFlag(
        "DARKBLOOM_VERIFY_PAIRED_GATE_UP_PREFILL_BITS"
    ) ?? false

private let gemma4PairedGateUpPrefillEnabled: Bool = {
    guard gemma4PairedGateUpPrefillNAXAvailable else { return false }
    if let explicitlyEnabled = gemma4PairedPrefillEnvironmentFlag(
        "DARKBLOOM_PAIRED_GATE_UP_PREFILL"
    ) {
        return explicitlyEnabled
    }
    // The layout adds 7.27 GiB to the full profile. Keep the ranked 128 GiB
    // route on by default without pushing 64 GiB NAX Macs beyond the startup
    // envelope; an explicit environment choice still wins.
    return ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30)
}()

/// Full-profile prefill projection backed by the stock 64x64x64 NAX affine
/// QMM. Parameter rows alternate gate/up, so its guarded register epilogue
/// can materialize only the exact BF16 GELU(gate)*up tensor.
struct PairedGateUpPrefillProjection: @unchecked Sendable {
    private static let inputWidth = 5_376
    private static let outputWidth = 21_504
    private static let pairedOutputWidth = 43_008
    private static let minimumQMMRows = 12

    private let paired: FastQuantizedProjection
    private let gate: FastQuantizedProjection
    private let up: FastQuantizedProjection

    init?(gate: FastQuantizedProjection, up: FastQuantizedProjection) {
        guard gemma4PairedGateUpPrefillEnabled,
              supportsGemma4FusedGateUp(gate: gate, up: up),
              let gateBiases = gate.biases,
              let upBiases = up.biases
        else {
            return nil
        }

        // Stacking on axis one produces [g0,u0,g1,u1,...] after the reshape.
        // These input-independent layouts are retained only on 96 GiB+ by
        // default and are materialized before scored inference.
        let weight = stacked([gate.weight, up.weight], axis: 1)
            .reshaped(Self.pairedOutputWidth, Self.inputWidth / 8)
        let scales = stacked([gate.scales, up.scales], axis: 1)
            .reshaped(Self.pairedOutputWidth, Self.inputWidth / gate.groupSize)
        let biases = stacked([gateBiases, upBiases], axis: 1)
            .reshaped(Self.pairedOutputWidth, Self.inputWidth / gate.groupSize)
        eval(weight, scales, biases)

        self.paired = FastQuantizedProjection(
            weight: weight,
            scales: scales,
            biases: biases,
            groupSize: gate.groupSize,
            bits: gate.bits
        )
        self.gate = gate
        self.up = up
    }

    func supports(_ input: MLXArray) -> Bool {
        input.dtype == .bfloat16
            && input.ndim == 3
            && input.dim(0) == 1
            && input.dim(1) >= Self.minimumQMMRows
            && input.dim(2) == Self.inputWidth
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(supports(input))
        let rows = input.dim(0) * input.dim(1)
        let activatedCount = rows * Self.outputWidth

        // The NAX epilogue densely writes this prefix with a 21,504-element
        // row stride. Flattening before slicing therefore stays row-contiguous
        // for the following down projection; a per-wide-row slice would not.
        let wide = paired(input)
        let activated = wide.reshaped(-1)[0..<activatedCount]
            .reshaped(input.dim(0), input.dim(1), Self.outputWidth)

        guard gemma4VerifyPairedGateUpPrefillBits else {
            return activated
        }

        let gateOutput = gate(input)
        let upOutput = up(input)
        let reference = 0.5 * gateOutput * (
            1 + tanh(
                sqrt(2 / Float.pi)
                    * (gateOutput
                        + 0.044715 * gateOutput * gateOutput * gateOutput)
            )
        ) * upOutput
        let matches = arrayEqual(
            activated.view(dtype: .uint16),
            reference.view(dtype: .uint16)
        )
        eval(matches)
        precondition(
            matches.item(Bool.self),
            "paired NAX gate/up prefill differs from stock BF16 activation"
        )
        return reference
    }
}
