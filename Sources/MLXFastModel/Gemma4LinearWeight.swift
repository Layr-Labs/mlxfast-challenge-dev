import MLX

/// A dense linear weight, either quantized (affine 4-bit, U32 codes + BF16
/// scales/biases -- matching the Gemma 4 4-bit checkpoint) or plain.
public struct Gemma4LinearWeight {
    public let weight: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let logicalShape: [Int]
    public let groupSize: Int
    public let bits: Int

    public init(_ weight: MLXArray) {
        self.init(
            weight: weight,
            scales: nil,
            biases: nil,
            logicalShape: weight.shape,
            groupSize: 0,
            bits: 0
        )
    }

    public init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        logicalShape: [Int],
        groupSize: Int,
        bits: Int
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.logicalShape = logicalShape
        self.groupSize = groupSize
        self.bits = bits
    }

    public var isQuantized: Bool {
        scales != nil
    }

    public var shape: [Int] {
        logicalShape
    }
}

extension Gemma4LinearWeight {
    /// Lazy view of a contiguous output-row range of this weight (the inverse
    /// of `concatenatedRows`). Row slicing commutes with the row-independent
    /// (quantized) matmul, so `linear(x, w.rowSlice(r))` equals the matching
    /// output columns of `linear(x, w)`. Off the hot path; used by
    /// compatibility accessors.
    func rowSlice(_ range: Range<Int>) -> Gemma4LinearWeight {
        guard range != 0..<logicalShape[0] else { return self }
        let slicedWeight = weight[range.lowerBound..<range.upperBound]
        guard let scales else {
            return Gemma4LinearWeight(
                weight: slicedWeight,
                scales: nil,
                biases: nil,
                logicalShape: [range.count, logicalShape[1]],
                groupSize: 0,
                bits: 0
            )
        }
        return Gemma4LinearWeight(
            weight: slicedWeight,
            scales: scales[range.lowerBound..<range.upperBound],
            biases: biases.map { $0[range.lowerBound..<range.upperBound] },
            logicalShape: [range.count, logicalShape[1]],
            groupSize: groupSize,
            bits: bits
        )
    }

    /// Fuse several linear weights into one by concatenating along the
    /// output-row (first logical) axis. Every part must share the same input
    /// feature count and, when quantized, the same group size and bit width.
    ///
    /// Affine 4-bit (and plain) matmul is independent per output row: the
    /// packed codes, `scales`, and `biases` are grouped along the *input* axis,
    /// so stacking parts along the output axis simply relocates each part's rows
    /// intact. Therefore `linear(x, fused)` equals the row-wise concatenation of
    /// `linear(x, part)` for every part, and slicing that output along the last
    /// axis at the original row boundaries reproduces each part bit-for-bit.
    /// This lets the gated MLP issue one quantized matmul instead of two,
    /// cutting a decode-time kernel dispatch per layer with no change to results.
    ///
    /// The fused arrays are evaluated eagerly so the source part arrays can be
    /// released at (untimed) construction time and are never re-concatenated on
    /// a scored forward.
    static func concatenatedRows(_ parts: [Gemma4LinearWeight]) -> Gemma4LinearWeight {
        precondition(!parts.isEmpty, "concatenatedRows requires at least one weight")
        guard parts.count > 1 else { return parts[0] }

        let quantizedCount = parts.lazy.filter { $0.scales != nil }.count
        precondition(
            quantizedCount == 0 || quantizedCount == parts.count,
            "cannot fuse a mix of quantized and plain linear weights"
        )

        let inFeatures = parts[0].logicalShape[1]
        let outFeatures = parts.reduce(0) { $0 + $1.logicalShape[0] }
        let fusedWeight = concatenated(parts.map { $0.weight }, axis: 0)

        guard quantizedCount == parts.count else {
            eval(fusedWeight)
            return Gemma4LinearWeight(
                weight: fusedWeight,
                scales: nil,
                biases: nil,
                logicalShape: [outFeatures, inFeatures],
                groupSize: 0,
                bits: 0
            )
        }

        let fusedScales = concatenated(parts.map { $0.scales! }, axis: 0)
        let fusedBiases: MLXArray? = parts.allSatisfy { $0.biases != nil }
            ? concatenated(parts.map { $0.biases! }, axis: 0)
            : nil
        if let fusedBiases {
            eval(fusedWeight, fusedScales, fusedBiases)
        } else {
            eval(fusedWeight, fusedScales)
        }
        return Gemma4LinearWeight(
            weight: fusedWeight,
            scales: fusedScales,
            biases: fusedBiases,
            logicalShape: [outFeatures, inFeatures],
            groupSize: parts[0].groupSize,
            bits: parts[0].bits
        )
    }
}
