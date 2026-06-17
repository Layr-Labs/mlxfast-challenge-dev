import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekMLPRunsDenseSwiGLUPathWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let weights = DeepSeekMLPWeights(
        gate: MLXArray([Float(1), 0, 0, 1], [2, 2]),
        up: MLXArray([Float(2), 0, 0, 3], [2, 2]),
        down: MLXArray([Float(1), 1], [1, 2])
    )
    let output = DeepSeekMLP.forward(
        MLXArray([Float(1), 2], [1, 2]),
        weights: weights,
        swigluLimit: 0
    )

    #expect(output.shape == [1, 1])
    let value = output.item(Float.self)
    #expect(abs(value - 12.031682) < 1e-5)
}

@Test
func deepSeekMoECombinesRoutedAndSharedOutputsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let routed = MLXArray(
        [
            Float(1), 2,
            10, 20,
            100, 200,
        ],
        [1, 1, 3, 2]
    )
    let routeWeights = MLXArray([Float(0.2), 0.3, 0.5], [1, 1, 3])
    let shared = MLXArray([Float(7), 11], [1, 1, 2])

    let output = DeepSeekMoE.combine(
        routedExpertOutput: routed,
        routeWeights: routeWeights,
        sharedExpertOutput: shared
    )

    #expect(output.shape == [1, 1, 2])
    let values = output.asArray(Float.self)
    #expect(abs(values[0] - 60.2) < 1e-5)
    #expect(abs(values[1] - 117.4) < 1e-5)
}
