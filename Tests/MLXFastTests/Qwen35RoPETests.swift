import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

private let pinnedQwen35RopeSpec = Qwen35RopeSpec(
    theta: 10_000_000,
    type: "default",
    partialRotaryFactor: 0.25,
    mropeInterleaved: true,
    mropeSection: [11, 11, 10]
)

@Test
func qwen35RoPEPinsThetaAndFirst64Of256Dimensions() throws {
    let rope = try Qwen35RoPE(
        headDimensions: 256,
        spec: pinnedQwen35RopeSpec
    )
    #expect(rope.theta == 10_000_000)
    #expect(rope.rotaryDimensions == 64)
    #expect(rope.mropeSection == [11, 11, 10])

    #expect(throws: MLXFastError.self) {
        _ = try Qwen35RoPE(
            headDimensions: 256,
            spec: Qwen35RopeSpec(
                theta: 10_000_000,
                type: "default",
                partialRotaryFactor: 0.25,
                mropeInterleaved: true,
                mropeSection: [10, 10, 10]
            )
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35RoPE(
            headDimensions: 256,
            spec: Qwen35RopeSpec(
                theta: 10_000_000,
                type: "default",
                partialRotaryFactor: 0.25,
                mropeInterleaved: true,
                mropeSection: [Int.max, Int.max, Int.max]
            )
        )
    }
}

@Test
func qwen35TextMRoPEReducesToPinnedPartialRoPE() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let rope = try Qwen35RoPE(
        headDimensions: 256,
        spec: pinnedQwen35RopeSpec
    )
    let input = MLXArray(
        (0..<(3 * 256)).map { Float(($0 % 31) - 15) / 16 },
        [1, 1, 3, 256]
    )
    let actual = rope.applied(to: input, offset: 7)
    let reference = MLXFast.RoPE(
        input,
        dimensions: 64,
        traditional: false,
        base: 10_000_000,
        scale: 1,
        offset: 7
    )

    #expect(actual.shape == input.shape)
    #expect(
        qwen35MaximumAbsoluteDifference(actual, reference) < 1e-6
    )
    #expect(
        qwen35MaximumAbsoluteDifference(
            actual[.ellipsis, 64...],
            input[.ellipsis, 64...]
        ) < 1e-6
    )
}
