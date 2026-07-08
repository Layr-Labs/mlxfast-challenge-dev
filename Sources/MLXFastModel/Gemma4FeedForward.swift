import MLX

public struct Gemma4MLPWeights {
    public let gate: Gemma4LinearWeight
    public let up: Gemma4LinearWeight
    public let down: Gemma4LinearWeight
    /// Fused gate+up projection, combining both into a single quantized matmul.
    /// When non-nil, `forward` uses this instead of separate gate/up calls.
    public let fusedGateUp: Gemma4LinearWeight?

    public init(
        gate: Gemma4LinearWeight,
        up: Gemma4LinearWeight,
        down: Gemma4LinearWeight,
        fusedGateUp: Gemma4LinearWeight? = nil
    ) {
        self.gate = gate
        self.up = up
        self.down = down
        self.fusedGateUp = fusedGateUp
    }
}

/// Gemma 4's gated MLP: `down(gelu_tanh(gate(x)) * up(x))`.
public enum Gemma4MLP {
    public static func forward(_ x: MLXArray, weights: Gemma4MLPWeights) -> MLXArray {
        let gate: MLXArray
        let up: MLXArray

        if let fused = weights.fusedGateUp {
            let fusedOut = Gemma4Ops.linear(x, fused)
            let mid = fusedOut.shape[fusedOut.shape.count - 1] / 2
            gate = fusedOut[.ellipsis, 0..<mid]
            up = fusedOut[.ellipsis, mid...]
        } else {
            gate = Gemma4Ops.linear(x, weights.gate)
            up = Gemma4Ops.linear(x, weights.up)
        }

        let hidden = Gemma4Ops.geluTanh(gate) * up
        return Gemma4Ops.linear(hidden, weights.down)
    }
}
