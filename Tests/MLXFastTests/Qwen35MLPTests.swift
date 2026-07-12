import MLX
import MLXNN
@testable import MLXFastModel
import Testing

@Test
func qwen35MLPMatchesPinnedDenseSiluSwiGLUFormula() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let identity = Qwen35LinearWeight(
        qwen35Identity(rows: 3, columns: 3)
    )
    let weights = Qwen35MLPWeights(
        gateProjection: identity,
        upProjection: identity,
        downProjection: identity
    )
    let input = MLXArray([Float(-1), 0.5, 2], [1, 1, 3])

    let actual = Qwen35MLP.forward(input, weights: weights)
    let expected = silu(input) * input
    #expect(actual.shape == [1, 1, 3])
    #expect(
        qwen35MaximumAbsoluteDifference(actual, expected) < 1e-6
    )
}
