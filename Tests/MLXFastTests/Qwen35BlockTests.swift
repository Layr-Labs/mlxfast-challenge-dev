import MLX
import MLXLMCommon
import MLXNN
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func qwen35BlockUsesTwoPreNormResidualFormulaForFullAttention() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let hiddenSize = 128
    let attentionSpec = try Qwen35AttentionSpec(
        hiddenSize: hiddenSize,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDimension: 128,
        rmsNormEps: 1e-6,
        ropeSpec: Qwen35RopeSpec(
            theta: 10_000_000,
            type: "default",
            partialRotaryFactor: 0.25,
            mropeInterleaved: true,
            mropeSection: [6, 5, 5]
        )
    )
    let zeroAttention = Qwen35AttentionWeights(
        queryProjection: Qwen35LinearWeight(
            MLXArray.zeros([
                attentionSpec.queryAndGateProjectionSize,
                hiddenSize,
            ])
        ),
        keyProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        valueProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        outputProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, attentionSpec.querySize])
        ),
        queryNorm: MLXArray.ones([128]),
        keyNorm: MLXArray.ones([128])
    )
    let identity = Qwen35LinearWeight(
        qwen35Identity(rows: hiddenSize, columns: hiddenSize)
    )
    let block = Qwen35BlockWeights(
        inputLayerNorm: MLXArray.ones([hiddenSize]),
        postAttentionLayerNorm: MLXArray.ones([hiddenSize]),
        mixer: .full(weights: zeroAttention, spec: attentionSpec),
        mlp: Qwen35MLPWeights(
            gateProjection: identity,
            upProjection: identity,
            downProjection: identity
        )
    )
    let cache = Qwen35BlockCache(fullAttention: KVCacheSimple())
    let input = qwen35DeterministicMatrix(
        rows: 2,
        columns: hiddenSize,
        scale: 0.02
    ).reshaped(1, 2, hiddenSize)

    #expect(throws: MLXFastError.self) {
        _ = try Qwen35Block.forward(
            input,
            weights: block,
            rmsNormEps: 1e-6,
            attentionMask: .causal,
            cache: nil
        )
    }
    let actual = try Qwen35Block.forward(
        input,
        weights: block,
        rmsNormEps: 1e-6,
        attentionMask: .causal,
        cache: cache
    )
    let normalized = Qwen35Ops.rmsNorm(
        input,
        weight: block.postAttentionLayerNorm,
        eps: 1e-6
    )
    let expected = input + silu(normalized) * normalized
    let gap = qwen35MaximumAbsoluteDifference(actual, expected)

    #expect(cache.fullAttention?.offset == 2)
    #expect(cache.gatedDelta == nil)
    #expect(gap < 2e-5)
}

@Test
func qwen35BlockDispatchesLinearLayerToGatedDeltaState() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let spec = try tinyQwen35GatedDeltaSpec()
    let linearWeights = tinyQwen35GatedDeltaWeights(spec: spec)
    let zero = Qwen35LinearWeight(
        MLXArray.zeros([spec.hiddenSize, spec.hiddenSize])
    )
    let block = Qwen35BlockWeights(
        inputLayerNorm: MLXArray.ones([spec.hiddenSize]),
        postAttentionLayerNorm: MLXArray.ones([spec.hiddenSize]),
        mixer: .linear(weights: linearWeights, spec: spec),
        mlp: Qwen35MLPWeights(
            gateProjection: zero,
            upProjection: zero,
            downProjection: zero
        )
    )
    let cache = Qwen35BlockCache()
    let input = qwen35DeterministicMatrix(
        rows: 1,
        columns: spec.hiddenSize,
        scale: 0.01
    ).reshaped(1, 1, spec.hiddenSize)

    let output = try Qwen35Block.forward(
        input,
        weights: block,
        rmsNormEps: 1e-6,
        cache: cache
    )

    #expect(output.shape == input.shape)
    #expect(cache.fullAttention == nil)
    #expect(cache.gatedDelta?.convolution.shape == [1, 3, 384])
    #expect(cache.gatedDelta?.recurrent.dtype == .float32)
}
