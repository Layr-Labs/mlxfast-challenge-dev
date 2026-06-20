import Foundation
import MLX
@testable import MLXFastModel
import Testing

@Test
func deepSeekOpsRunWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let ids = MLXArray([Int32(2), Int32(0)], [2])
    let embeddingWeight = MLXArray((1...6).map { Float($0) }, [3, 2])
    let embedded = DeepSeekOps.embedding(inputIDs: ids, weight: embeddingWeight)
    #expect(embedded.shape == [2, 2])
    #expect(embedded.asArray(Float.self) == [5, 6, 1, 2])

    let input = MLXArray([Float(2), Float(3)], [1, 2])
    let weight = MLXArray([Float(1), Float(10), Float(2), Float(20)], [2, 2])
    let bias = MLXArray([Float(1), Float(-1)], [2])
    let projected = DeepSeekOps.linear(input: input, weight: weight, bias: bias)
    #expect(projected.shape == [1, 2])
    #expect(projected.asArray(Float.self) == [33, 63])

    let groupedInput = MLXArray([Float(1), 2, 3, 4], [1, 2, 1, 2])
    let groupedWeight = MLXArray(
        [
            Float(10), 20,
            30, 40,
            50, 60,
            70, 80,
        ],
        [2, 2, 2]
    )
    let grouped = DeepSeekOps.multiLinear(input: groupedInput, weight: groupedWeight)
    #expect(grouped.shape == [1, 2, 1, 2])
    #expect(grouped.asArray(Float.self) == [50, 110, 390, 530])

    let denseQuantizedWeight = MLXArray((0..<128).map { Float($0) / 128 }, [2, 64])
    let (packedWeight, scales, quantBiases) = quantized(
        denseQuantizedWeight,
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let quantizedWeight = DeepSeekLinearWeight(
        weight: packedWeight,
        scales: scales,
        biases: quantBiases,
        logicalShape: [2, 64],
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let quantizedProjected = DeepSeekOps.linear(
        input: MLXArray(Array(repeating: Float(1), count: 64), [1, 64]),
        weight: quantizedWeight
    )
    #expect(quantizedProjected.shape == [1, 2])

    let (mxfp4Packed, mxfp4Scales, mxfp4Biases) = quantized(
        denseQuantizedWeight,
        groupSize: 32,
        bits: 4,
        mode: .mxfp4
    )
    #expect(mxfp4Biases == nil)
    let mxfp4Weight = DeepSeekLinearWeight(
        weight: mxfp4Packed,
        scales: mxfp4Scales,
        biases: nil,
        logicalShape: [2, 64],
        groupSize: 32,
        bits: 4,
        mode: .mxfp4
    )
    let mxfp4Projected = DeepSeekOps.linear(
        input: MLXArray(Array(repeating: Float(1), count: 64), [1, 64]),
        weight: mxfp4Weight
    )
    #expect(mxfp4Projected.shape == [1, 2])

    let quantizedEmbedding = DeepSeekOps.embedding(
        inputIDs: MLXArray([Int32(1)], [1]),
        weight: quantizedWeight
    )
    #expect(quantizedEmbedding.shape == [1, 64])

    let groupedDenseWeight = MLXArray((0..<256).map { Float($0) / 256 }, [4, 64])
    let (groupedPacked, groupedScales, groupedBiases) = quantized(
        groupedDenseWeight,
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let groupedQuantizedWeight = DeepSeekLinearWeight(
        weight: groupedPacked,
        scales: groupedScales,
        biases: groupedBiases,
        logicalShape: [2, 2, 64],
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let groupedQuantized = try DeepSeekOps.multiLinear(
        input: MLXArray(Array(repeating: Float(1), count: 128), [1, 2, 1, 64]),
        weight: groupedQuantizedWeight
    )
    #expect(groupedQuantized.shape == [1, 2, 1, 2])

    let gate = MLXArray([Float(0), Float(1)], [2])
    let up = MLXArray([Float(2), Float(3)], [2])
    let swiglu = DeepSeekOps.limitedSwiGLU(gate: gate, up: up, limit: 0)
        .asArray(Float.self)
    #expect(abs(swiglu[0] - 0) < 1e-6)
    #expect(abs(swiglu[1] - 2.1931758) < 1e-5)

    let norm = DeepSeekOps.rmsNorm(
        input: MLXArray([Float(3), Float(4)], [1, 2]),
        weight: MLXArray([Float(1), Float(1)], [2]),
        eps: 0
    ).asArray(Float.self)
    #expect(abs(norm[0] - 0.84852815) < 1e-5)
    #expect(abs(norm[1] - 1.1313709) < 1e-5)
}
