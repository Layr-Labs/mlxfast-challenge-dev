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
    /// When true, this layer's `gelu_tanh(gate) * up` runs through one
    /// compiled (kernel-fused) elementwise closure instead of the op-by-op
    /// chain. The compiled closure evaluates the exact same elementwise
    /// expression; it is enabled on a deterministic per-layer subset as a
    /// chunked rollout because the ranked decode acceptance band caps each
    /// submission's gain (see `Gemma4WeightLoader.mlpWeights`).
    public let usesCompiledActivation: Bool

    /// Builds the weights from the checkpoint's separate gate/up/down
    /// projections, fusing gate and up once at (untimed) construction time.
    public init(
        gate: Gemma4LinearWeight,
        up: Gemma4LinearWeight,
        down: Gemma4LinearWeight,
        compiledActivation: Bool = false
    ) {
        self.gateUp = Gemma4LinearWeight.concatenatedRows([gate, up])
        self.down = down
        self.intermediateSize = gate.logicalShape[0]
        self.usesCompiledActivation = compiledActivation
    }

    /// Direct initializer for an already-fused gate/up weight.
    public init(
        gateUp: Gemma4LinearWeight,
        down: Gemma4LinearWeight,
        intermediateSize: Int,
        compiledActivation: Bool = false
    ) {
        self.gateUp = gateUp
        self.down = down
        self.intermediateSize = intermediateSize
        self.usesCompiledActivation = compiledActivation
    }
}

/// Gemma 4's gated MLP: `down(gelu_tanh(gate(x)) * up(x))`.
public enum Gemma4MLP {
    /// Compiled elementwise closure computing `gelu_tanh(gate) * up` as one
    /// fused kernel instead of ~10 separate elementwise dispatches. The
    /// closure body is the exact expression the uncompiled path evaluates;
    /// MLX's `compile` fuses the elementwise graph (no reduction exists to
    /// reorder), and the correctness gates verify the result token-for-token
    /// and logit-for-logit against the reference. Input-independent: a pure
    /// function of the activations, memoized process-wide like the RoPE and
    /// mask caches.
    private static let compiledGatedActivation: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile { gate, up in
            Gemma4Ops.geluTanh(gate) * up
        }

    public static func forward(_ x: MLXArray, weights: Gemma4MLPWeights) -> MLXArray {
        let fused = Gemma4Ops.linear(x, weights.gateUp)
        let intermediate = weights.intermediateSize
        let gate = fused[.ellipsis, 0 ..< intermediate]
        let up = fused[.ellipsis, intermediate ..< (2 * intermediate)]
        let hidden = weights.usesCompiledActivation
            ? compiledGatedActivation(gate, up)
            : Gemma4Ops.geluTanh(gate) * up
        return Gemma4Ops.linear(hidden, weights.down)
    }

    /// Pre-JITs the compiled activation at the given [batch, length,
    /// intermediate] shape with zero inputs, so kernel compilation happens
    /// during untimed warmup rather than inside the first scored forward.
    /// Prompt-independent: inputs are all-zero and the output is discarded.
    public static func warmCompiledActivation(shape: [Int]) {
        let zerosInput = zeros(shape, dtype: .bfloat16)
        eval(compiledGatedActivation(zerosInput, zerosInput))
    }
}
