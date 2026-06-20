import MLX

public struct DeepSeekLinearWeight {
    public let weight: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let logicalShape: [Int]
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(_ weight: MLXArray) {
        self.init(
            weight: weight,
            scales: nil,
            biases: nil,
            logicalShape: weight.shape,
            groupSize: 0,
            bits: 0,
            mode: .affine
        )
    }

    public init(
        weight: MLXArray,
        scales: MLXArray?,
        biases: MLXArray?,
        logicalShape: [Int],
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.logicalShape = logicalShape
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    public var isQuantized: Bool {
        scales != nil
    }

    public var shape: [Int] {
        logicalShape
    }

    public func rows(_ rowRange: Range<Int>, logicalShape: [Int]? = nil) -> DeepSeekLinearWeight {
        if let scales {
            return DeepSeekLinearWeight(
                weight: weight[rowRange, 0...],
                scales: scales[rowRange, 0...],
                biases: biases.map { $0[rowRange, 0...] },
                logicalShape: logicalShape ?? [rowRange.count, self.logicalShape.last ?? 0],
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }
        return DeepSeekLinearWeight(
            weight: weight[rowRange, 0...],
            scales: nil,
            biases: nil,
            logicalShape: logicalShape ?? [rowRange.count, self.logicalShape.last ?? 0],
            groupSize: 0,
            bits: 0,
            mode: .affine
        )
    }
}
