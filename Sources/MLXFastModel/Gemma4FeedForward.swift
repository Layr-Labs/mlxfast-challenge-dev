import MLX

public struct Gemma4MLPWeights {
    /// Fused gate+up projection: the gate output rows first, then the up output
    /// rows. A single quantized matmul replaces the two separate gate/up
    /// projections; its output is split back at `intermediateSize`. Because the
    /// affine 4-bit matmul is independent per output row, the fused-then-split
    /// result is bit-identical to running gate and up separately (see
    /// `Gemma4LinearWeight.concatenatedRows`).
    public let gateUp: Gemma4LinearWeight
    public let down: Gemma4LinearWeight
    /// Output-row split point: gate = fused[..<intermediateSize],
    /// up = fused[intermediateSize ..< 2 * intermediateSize].
    public let intermediateSize: Int

    /// Builds the weights from the checkpoint's separate gate/up/down
    /// projections, fusing gate and up once at (untimed) construction time.
    public init(gate: Gemma4LinearWeight, up: Gemma4LinearWeight, down: Gemma4LinearWeight) {
        self.gateUp = Gemma4LinearWeight.concatenatedRows([gate, up])
        self.down = down
        self.intermediateSize = gate.logicalShape[0]
    }

    /// Direct initializer for an already-fused gate/up weight.
    public init(gateUp: Gemma4LinearWeight, down: Gemma4LinearWeight, intermediateSize: Int) {
        self.gateUp = gateUp
        self.down = down
        self.intermediateSize = intermediateSize
    }
}

/// Gemma 4's gated MLP: `down(gelu_tanh(gate(x)) * up(x))`.
///
/// Rerun probe (2026-07-08): identical optimization to submission f9ddc17
/// (scored 1.0164 pre-migration), rebased onto the vanilla mlx-swift-lm base.
public enum Gemma4MLP {
    public static func forward(_ x: MLXArray, weights: Gemma4MLPWeights) -> MLXArray {
        let fused = Gemma4Ops.linear(x, weights.gateUp)
        let intermediate = weights.intermediateSize
        let gate = fused[.ellipsis, 0 ..< intermediate]
        let up = fused[.ellipsis, intermediate ..< (2 * intermediate)]
        let hidden = Gemma4Ops.geluTanh(gate) * up
        return Gemma4Ops.linear(hidden, weights.down)
    }
}
