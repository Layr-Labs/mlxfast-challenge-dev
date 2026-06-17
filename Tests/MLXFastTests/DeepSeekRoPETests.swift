import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekRoPEBuildsDefaultFrequencies() throws {
    let freqs = try DeepSeekRoPE.makeBaseFrequencies(
        rotaryDimensions: 8,
        base: 10_000.0
    )

    assertClose(freqs[0], 1.0)
    assertClose(freqs[1], 10.0)
    assertClose(freqs[2], 100.0)
    assertClose(freqs[3], 1_000.0)
}

@Test
func deepSeekRoPEBuildsYarnFrequencies() throws {
    let freqs = try DeepSeekRoPE.makeBaseFrequencies(
        rotaryDimensions: 8,
        base: 10_000.0,
        scaling: DeepSeekRopeScaling(
            type: "deepseek_yarn",
            ropeType: nil,
            factor: 4.0,
            originalMaxPositionEmbeddings: 4_096,
            betaFast: 32.0,
            betaSlow: 1.0
        )
    )

    #expect(freqs.count == 4)
    assertClose(freqs[0], 1.0)
    #expect(freqs[3] > 1_000.0)
    #expect(freqs[3] <= 4_000.0)
}

@Test
func deepSeekRoPEAddsNoPositionPrefixAndInverseFrequencies() throws {
    let rope = try DeepSeekRoPE(
        rotaryDimensions: 4,
        base: 10_000.0,
        maxPositionEmbeddings: 1_024,
        freqScale: 2
    )

    let freqs = try rope.frequencies(headDimension: 8)
    #expect(freqs.count == 4)
    #expect(freqs[0].isInfinite)
    #expect(freqs[1].isInfinite)
    assertClose(freqs[2], 0.5)
    assertClose(freqs[3], 50.0)

    let inverse = try rope.frequencies(headDimension: 8, inverse: true)
    #expect(inverse[0].isInfinite)
    assertClose(inverse[2], -0.5)
    assertClose(inverse[3], -50.0)
}

@Test
func deepSeekRoPEAppliesWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let rope = try DeepSeekRoPE(
        rotaryDimensions: 4,
        base: 10_000.0,
        maxPositionEmbeddings: 1_024
    )
    let input = MLXArray((0..<8).map { Float($0) }, [1, 1, 1, 8])
    let output = try rope.applied(to: input, offset: 0)

    #expect(output.shape == [1, 1, 1, 8])
    #expect(output.asArray(Float.self) == input.asArray(Float.self))
}

private func assertClose(_ actual: Float, _ expected: Float, tolerance: Float = 1e-5) {
    #expect(abs(actual - expected) <= tolerance)
}
