import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekHyperConnectionCollapseExpandAndHeadWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let x = MLXArray([Float(2), 4, 6, 8], [1, 1, 2, 2])
    let fn = zeros([8, 4], dtype: .float32)
    let base = zeros([8], dtype: .float32)
    let scale = ones([3], dtype: .float32)

    let output = try DeepSeekHyperConnection.collapse(
        x,
        fn: fn,
        base: base,
        scale: scale,
        hcMult: 2,
        sinkhornIters: 2,
        eps: 0,
        normEps: 0
    )

    #expect(output.collapsed.shape == [1, 1, 2])
    assertArrayClose(output.collapsed.asArray(Float.self), [4, 6])
    assertArrayClose(output.post.asArray(Float.self), [1, 1])
    assertArrayClose(output.combination.asArray(Float.self), [0.5, 0.5, 0.5, 0.5])

    let expanded = DeepSeekHyperConnection.expand(
        MLXArray([Float(10), 20], [1, 1, 2]),
        residual: x,
        post: output.post,
        combination: output.combination
    )
    #expect(expanded.shape == [1, 1, 2, 2])
    assertArrayClose(expanded.asArray(Float.self), [14, 26, 14, 26])

    let head = try DeepSeekHyperConnection.head(
        x,
        fn: zeros([2, 4], dtype: .float32),
        base: zeros([2], dtype: .float32),
        scale: ones([1], dtype: .float32),
        hcMult: 2,
        eps: 0,
        normEps: 0
    )
    assertArrayClose(head.asArray(Float.self), [4, 6])
}

private func assertArrayClose(
    _ actual: [Float],
    _ expected: [Float],
    tolerance: Float = 1e-5
) {
    #expect(actual.count == expected.count)
    for (lhs, rhs) in zip(actual, expected) {
        #expect(abs(lhs - rhs) <= tolerance)
    }
}
