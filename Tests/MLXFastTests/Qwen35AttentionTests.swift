import MLX
import MLXLMCommon
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func qwen35PinnedFullAttentionDimensionsMatchReference() throws {
    let spec = try Qwen35AttentionSpec(
        hiddenSize: 5_120,
        numAttentionHeads: 24,
        numKeyValueHeads: 4,
        headDimension: 256,
        rmsNormEps: 1e-6,
        ropeSpec: Qwen35RopeSpec(
            theta: 10_000_000,
            type: "default",
            partialRotaryFactor: 0.25,
            mropeInterleaved: true,
            mropeSection: [11, 11, 10]
        )
    )

    #expect(spec.querySize == 6_144)
    #expect(spec.queryAndGateProjectionSize == 12_288)
    #expect(spec.keyValueSize == 1_024)
    #expect(abs(spec.scale - 0.0625) < 1e-7)
    #expect(spec.rope.rotaryDimensions == 64)
}

@Test
func qwen35AttentionSpecRejectsDerivedDimensionOverflow() {
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35AttentionSpec(
            hiddenSize: 8,
            numAttentionHeads: 2,
            numKeyValueHeads: 1,
            headDimension: Int.max,
            rmsNormEps: 1e-6,
            ropeSpec: Qwen35RopeSpec(
                theta: 10_000_000,
                type: "default",
                partialRotaryFactor: 0.25,
                mropeInterleaved: true,
                mropeSection: [1, 1, 1]
            )
        )
    }
}

@Test
func qwen35FullAttentionPrefillMatchesCachedTokenSteps() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let spec = try tinyQwen35AttentionSpec()
    let weights = tinyQwen35AttentionWeights(spec: spec)
    let input = MLXArray(
        (0..<(3 * spec.hiddenSize)).map {
            Float(($0 % 13) - 6) / 16
        },
        [1, 3, spec.hiddenSize]
    )

    let prefill = try Qwen35FullAttention.forward(
        input,
        weights: weights,
        spec: spec,
        mask: .causal
    )
    let cache = KVCacheSimple()
    var stepOutputs: [MLXArray] = []
    for position in 0..<3 {
        let step = input[
            0...,
            position..<(position + 1),
            0...
        ]
        stepOutputs.append(
            try Qwen35FullAttention.forward(
                step,
                weights: weights,
                spec: spec,
                mask: .none,
                cache: cache,
                positionOffset: position
            )
        )
    }
    let stepped = concatenated(stepOutputs, axis: 1)
    let gap = qwen35MaximumAbsoluteDifference(prefill, stepped)

    #expect(prefill.shape == [1, 3, spec.hiddenSize])
    #expect(cache.offset == 3)
    #expect(gap < 2e-4)
}

private func tinyQwen35AttentionSpec() throws -> Qwen35AttentionSpec {
    try Qwen35AttentionSpec(
        hiddenSize: 16,
        numAttentionHeads: 2,
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
}

private func tinyQwen35AttentionWeights(
    spec: Qwen35AttentionSpec
) -> Qwen35AttentionWeights {
    Qwen35AttentionWeights(
        queryProjection: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: spec.queryAndGateProjectionSize,
                columns: spec.hiddenSize,
                scale: 0.002
            )
        ),
        keyProjection: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: spec.keyValueSize,
                columns: spec.hiddenSize,
                scale: 0.003
            )
        ),
        valueProjection: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: spec.keyValueSize,
                columns: spec.hiddenSize,
                scale: 0.004
            )
        ),
        outputProjection: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: spec.hiddenSize,
                columns: spec.querySize,
                scale: 0.002
            )
        ),
        queryNorm: MLXArray.ones([spec.headDimension]),
        keyNorm: MLXArray.ones([spec.headDimension])
    )
}
