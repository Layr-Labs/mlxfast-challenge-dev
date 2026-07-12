import MLX
import MLXNN
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func qwen35OpsCoverDenseQuantizedEmbeddingLinearAndRMSNorm() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let ids = MLXArray([Int32(2), 0], [2])
    let embedding = Qwen35LinearWeight(
        MLXArray((1...6).map(Float.init), [3, 2])
    )
    #expect(
        Qwen35Ops.embedding(inputIDs: ids, weight: embedding)
            .asArray(Float.self) == [5, 6, 1, 2]
    )

    let input = MLXArray([Float(2), 3], [1, 2])
    let dense = Qwen35LinearWeight(
        MLXArray([Float(1), 10, 2, 20], [2, 2])
    )
    #expect(
        Qwen35Ops.linear(input, dense).asArray(Float.self)
            == [32, 64]
    )

    let source = MLXArray(
        (0..<128).map { Float($0) / 128 },
        [2, 64]
    )
    let (packed, scales, biases) = quantized(
        source,
        groupSize: 64,
        bits: 4,
        mode: .affine
    )
    let quantizedWeight = try Qwen35LinearWeight(
        weight: packed,
        scales: scales,
        biases: biases,
        logicalShape: [2, 64],
        groupSize: 64,
        bits: 4
    )
    #expect(
        Qwen35Ops.linear(
            MLXArray(Array(repeating: Float(1), count: 64), [1, 64]),
            quantizedWeight
        ).shape == [1, 2]
    )
    #expect(
        Qwen35Ops.embedding(
            inputIDs: MLXArray([Int32(1), 0], [1, 2]),
            weight: quantizedWeight
        ).shape == [1, 2, 64]
    )
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35LinearWeight(
            weight: packed,
            scales: scales,
            biases: nil,
            logicalShape: [2, 64],
            groupSize: 64,
            bits: 4
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35LinearWeight(
            weight: packed,
            scales: MLXArray.zeros([2, 2]),
            biases: MLXArray.zeros([2, 2]),
            logicalShape: [2, 64],
            groupSize: 64,
            bits: 4
        )
    }

    let normInput = MLXArray([Float(3), 4], [1, 2])
    let normWeight = MLXArray.ones([2])
    let actualNorm = Qwen35Ops.rmsNorm(
        normInput,
        weight: normWeight,
        eps: 1e-6
    )
    let referenceNorm = MLXFast.rmsNorm(
        normInput,
        weight: normWeight,
        eps: 1e-6
    )
    #expect(
        qwen35MaximumAbsoluteDifference(actualNorm, referenceNorm)
            < 1e-6
    )

    let gate = MLXArray([Float(1), -1], [1, 2])
    let actualGated = Qwen35Ops.preciseGatedRMSNorm(
        normInput,
        gate: gate,
        weight: normWeight,
        eps: 1e-6
    )
    let referenceGated = (
        silu(gate.asType(.float32))
            * referenceNorm.asType(.float32)
    ).asType(normInput.dtype)
    #expect(
        qwen35MaximumAbsoluteDifference(actualGated, referenceGated)
            < 1e-6
    )
}
