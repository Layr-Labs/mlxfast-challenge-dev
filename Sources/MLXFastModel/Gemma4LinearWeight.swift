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

    /// Fuse two quantized linear weights along the output (row) dimension.
    /// Both must have the same input features, group size, and bit width.
    /// The weight codes, scales, and biases are concatenated along axis 0.
    public static func fused(rowwise a: Gemma4LinearWeight, _ b: Gemma4LinearWeight) -> Gemma4LinearWeight {
        precondition(a.logicalShape[1] == b.logicalShape[1], "input features must match")
        precondition(a.groupSize == b.groupSize, "group sizes must match")
        precondition(a.bits == b.bits, "bit widths must match")

        let fusedWeight = concatenated([a.weight, b.weight], axis: 0)
        let fusedScales: MLXArray?
        if let aScales = a.scales, let bScales = b.scales {
            fusedScales = concatenated([aScales, bScales], axis: 0)
        } else {
            fusedScales = nil
        }
        let fusedBiases: MLXArray?
        if let aBiases = a.biases, let bBiases = b.biases {
            fusedBiases = concatenated([aBiases, bBiases], axis: 0)
        } else {
            fusedBiases = nil
        }

        return Gemma4LinearWeight(
            weight: fusedWeight,
            scales: fusedScales,
            biases: fusedBiases,
            logicalShape: [a.logicalShape[0] + b.logicalShape[0], a.logicalShape[1]],
            groupSize: a.groupSize,
            bits: a.bits
        )
    }
}
