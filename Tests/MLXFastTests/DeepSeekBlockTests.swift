import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekBlockRunsHyperConnectionNormAndCallbacksWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let hc = DeepSeekHyperConnectionWeights(
        fn: zeros([8, 4], dtype: .float32),
        base: zeros([8], dtype: .float32),
        scale: ones([3], dtype: .float32)
    )
    let weights = DeepSeekBlockWeights(
        attentionNorm: ones([2], dtype: .float32),
        feedForwardNorm: ones([2], dtype: .float32),
        attentionHyperConnection: hc,
        feedForwardHyperConnection: hc
    )
    let spec = DeepSeekBlockSpec(
        hcMult: 2,
        hcSinkhornIters: 2,
        hcEps: 0,
        rmsNormEps: 0
    )

    var capturedAttentionInput: MLXArray?
    var capturedFeedForwardInput: MLXArray?
    let output = try DeepSeekBlock.forward(
        hidden: MLXArray([Float(2), 4, 6, 8], [1, 1, 2, 2]),
        weights: weights,
        spec: spec,
        attention: { normalized in
            capturedAttentionInput = normalized
            return zeros([1, 1, 2], dtype: .float32)
        },
        feedForward: { normalized in
            capturedFeedForwardInput = normalized
            return zeros([1, 1, 2], dtype: .float32)
        }
    )

    #expect(output.shape == [1, 1, 2, 2])
    #expect(output.asArray(Float.self) == [4, 6, 4, 6])

    let attentionInput = try #require(capturedAttentionInput).asArray(Float.self)
    let feedForwardInput = try #require(capturedFeedForwardInput).asArray(Float.self)
    #expect(abs(attentionInput[0] - 0.78446454) < 1e-5)
    #expect(abs(attentionInput[1] - 1.1766968) < 1e-5)
    #expect(abs(feedForwardInput[0] - 0.78446454) < 1e-5)
    #expect(abs(feedForwardInput[1] - 1.1766968) < 1e-5)
}
