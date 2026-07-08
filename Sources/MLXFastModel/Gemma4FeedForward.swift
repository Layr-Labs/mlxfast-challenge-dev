import MLX

public struct Gemma4MLPWeights {
    public let gate: Gemma4LinearWeight
    public let up: Gemma4LinearWeight
    public let down: Gemma4LinearWeight

    public init(gate: Gemma4LinearWeight, up: Gemma4LinearWeight, down: Gemma4LinearWeight) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

/// Gemma 4's gated MLP: `down(gelu_tanh(gate(x)) * up(x))`.
public enum Gemma4MLP {
    /// Context for fusing the down-projection with the post-FF RMSNorm.
    /// Set by `Gemma4Block.forward` before calling the feed-forward closure;
    /// consumed and cleared by `forward` so the norm is applied immediately
    /// after the down-projection without materializing the intermediate result.
    nonisolated(unsafe) public static var postNormContext: (weight: MLXArray, eps: Double)? = nil

    public static func forward(_ x: MLXArray, weights: Gemma4MLPWeights) -> MLXArray {
        let gate = Gemma4Ops.linear(x, weights.gate)
        let up = Gemma4Ops.linear(x, weights.up)
        let hidden = Gemma4Ops.geluTanh(gate) * up
        let projected = Gemma4Ops.linear(hidden, weights.down)
        if let ctx = Self.postNormContext {
            Self.postNormContext = nil
            return Gemma4Ops.rmsNorm(projected, weight: ctx.weight, eps: ctx.eps)
        }
        return projected
    }
}
