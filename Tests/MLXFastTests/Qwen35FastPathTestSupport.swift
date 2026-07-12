import Foundation
import MLX
import Testing

func qwen35MLXTestsEnabled() -> Bool {
    ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"]
        == "1"
}

func qwen35Identity(
    rows: Int,
    columns: Int,
    scale: Float = 1
) -> MLXArray {
    var values = Array(repeating: Float(0), count: rows * columns)
    for index in 0..<min(rows, columns) {
        values[index * columns + index] = scale
    }
    return MLXArray(values, [rows, columns])
}

func qwen35RepeatedIdentity(
    repeats: Int,
    dimensions: Int,
    scale: Float = 1
) -> MLXArray {
    var values = Array(
        repeating: Float(0),
        count: repeats * dimensions * dimensions
    )
    for block in 0..<repeats {
        for index in 0..<dimensions {
            let row = block * dimensions + index
            values[row * dimensions + index] = scale
        }
    }
    return MLXArray(
        values,
        [repeats * dimensions, dimensions]
    )
}

func qwen35DeterministicMatrix(
    rows: Int,
    columns: Int,
    scale: Float = 0.01
) -> MLXArray {
    let values = (0..<(rows * columns)).map { index in
        Float((index % 17) - 8) * scale
    }
    return MLXArray(values, [rows, columns])
}

func qwen35DepthwiseCurrentTokenKernel(
    channels: Int,
    kernelSize: Int
) -> MLXArray {
    var values = Array(
        repeating: Float(0),
        count: channels * kernelSize
    )
    for channel in 0..<channels {
        values[channel * kernelSize + kernelSize - 1] = 1
    }
    return MLXArray(values, [channels, kernelSize, 1])
}

func qwen35MaximumAbsoluteDifference(
    _ left: MLXArray,
    _ right: MLXArray
) -> Float {
    eval(left, right)
    let leftValues = left.asArray(Float.self)
    let rightValues = right.asArray(Float.self)
    #expect(!leftValues.isEmpty)
    #expect(leftValues.count == rightValues.count)
    guard !leftValues.isEmpty,
          leftValues.count == rightValues.count
    else {
        return .infinity
    }
    return zip(leftValues, rightValues)
        .map { abs($0 - $1) }
        .max() ?? 0
}
