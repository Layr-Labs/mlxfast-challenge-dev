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

    public static func concatenatingRows(_ lhs: DeepSeekLinearWeight, _ rhs: DeepSeekLinearWeight) -> DeepSeekLinearWeight? {
        guard lhs.isQuantized == rhs.isQuantized,
              lhs.logicalShape.count == 2,
              rhs.logicalShape.count == 2,
              lhs.logicalShape[1] == rhs.logicalShape[1],
              lhs.groupSize == rhs.groupSize,
              lhs.bits == rhs.bits,
              lhs.mode == rhs.mode
        else {
            return nil
        }
        if let lhsScales = lhs.scales, let rhsScales = rhs.scales {
            let biases: MLXArray?
            switch (lhs.biases, rhs.biases) {
            case let (lhsBiases?, rhsBiases?):
                biases = concatenated([lhsBiases, rhsBiases], axis: 0)
            case (nil, nil):
                biases = nil
            default:
                return nil
            }
            return DeepSeekLinearWeight(
                weight: concatenated([lhs.weight, rhs.weight], axis: 0),
                scales: concatenated([lhsScales, rhsScales], axis: 0),
                biases: biases,
                logicalShape: [lhs.logicalShape[0] + rhs.logicalShape[0], lhs.logicalShape[1]],
                groupSize: lhs.groupSize,
                bits: lhs.bits,
                mode: lhs.mode
            )
        }
        guard lhs.scales == nil, rhs.scales == nil else {
            return nil
        }
        return DeepSeekLinearWeight(
            weight: concatenated([lhs.weight, rhs.weight], axis: 0),
            scales: nil,
            biases: nil,
            logicalShape: [lhs.logicalShape[0] + rhs.logicalShape[0], lhs.logicalShape[1]],
            groupSize: 0,
            bits: 0,
            mode: .affine
        )
    }
}
