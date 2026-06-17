import Foundation
import MLX
@testable import MLXFastDeepSeek
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
