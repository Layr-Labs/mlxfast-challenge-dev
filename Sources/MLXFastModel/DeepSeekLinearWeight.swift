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

    public func concatenatingRows(with other: DeepSeekLinearWeight) -> DeepSeekLinearWeight? {
        guard logicalShape.count == 2,
              other.logicalShape.count == 2,
              logicalShape[1] == other.logicalShape[1],
              weight.shape.count == 2,
              other.weight.shape.count == 2,
              weight.shape[1] == other.weight.shape[1],
              groupSize == other.groupSize,
              bits == other.bits,
              mode == other.mode
        else {
            return nil
        }

        switch (scales, other.scales) {
        case (nil, nil):
            break
        case let (lhs?, rhs?):
            guard lhs.shape.count == 2,
                  rhs.shape.count == 2,
                  lhs.shape[1] == rhs.shape[1]
            else {
                return nil
            }
        default:
            return nil
        }

        switch (biases, other.biases) {
        case (nil, nil):
            break
        case let (lhs?, rhs?):
            guard lhs.shape.count == 2,
                  rhs.shape.count == 2,
                  lhs.shape[1] == rhs.shape[1]
            else {
                return nil
            }
        default:
            return nil
        }

        return DeepSeekLinearWeight(
            weight: concatenated([weight, other.weight], axis: 0),
            scales: zipOptional(scales, other.scales).map { concatenated([$0, $1], axis: 0) },
            biases: zipOptional(biases, other.biases).map { concatenated([$0, $1], axis: 0) },
            logicalShape: [logicalShape[0] + other.logicalShape[0], logicalShape[1]],
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
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

private func zipOptional<T>(_ lhs: T?, _ rhs: T?) -> (T, T)? {
    guard let lhs, let rhs else {
        return nil
    }
    return (lhs, rhs)
}
