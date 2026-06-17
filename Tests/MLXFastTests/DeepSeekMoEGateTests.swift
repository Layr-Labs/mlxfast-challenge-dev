import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekMoEGateRoutesTopKWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let hidden = MLXArray([Float(1), Float(2)], [1, 1, 2])
    let weight = MLXArray(
        [
            Float(1), Float(0),
            Float(0), Float(1),
            Float(3), Float(0),
            Float(0), Float(3),
        ],
        [4, 2]
    )
    let bias = MLXArray([Float(0), Float(0), Float(0), Float(0)], [4])

    let result = try DeepSeekMoEGate.route(
        hidden: hidden,
        weight: weight,
        correctionBias: bias,
        topK: 2,
        routedScalingFactor: 1.5,
        normTopKProb: true,
        scoring: .sigmoid
    )

    let selected = Set(result.indices.asArray(Int32.self))
    #expect(selected == Set([2, 3]))
    #expect(abs(result.weights.sum().item(Float.self) - 1.5) < 1e-5)
}

@Test
func deepSeekMoEGateUsesHashRoutingWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let hidden = MLXArray([Float(1), Float(2)], [1, 1, 2])
    let weight = MLXArray(
        [
            Float(1), Float(0),
            Float(0), Float(1),
            Float(3), Float(0),
            Float(0), Float(3),
        ],
        [4, 2]
    )
    let inputIDs = MLXArray([Int32(2)], [1, 1])
    let tokenToExpert = MLXArray(
        [
            Int32(0), Int32(1),
            Int32(1), Int32(2),
            Int32(3), Int32(1),
        ],
        [3, 2]
    )

    let result = try DeepSeekMoEGate.route(
        hidden: hidden,
        inputIDs: inputIDs,
        weight: weight,
        tokenToExpert: tokenToExpert,
        topK: 2,
        routedScalingFactor: 1.0,
        normTopKProb: false,
        scoring: .sigmoid
    )

    #expect(result.indices.asArray(Int32.self) == [3, 1])
    #expect(result.weights.shape == [1, 1, 2])
}
