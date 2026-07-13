import MLX

/// Exact three-row fallback built from the promoted exact-two kernel. The third
/// row is duplicated only inside the dense projection and its duplicate output
/// is discarded; no fake token reaches attention or cache state.
extension FusedSlidingQKVProjection {
    func exactThreeVector(_ input: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
        let first = exactTwoVector(input[0..<2, 0...])
        let tailInput = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVector(tailInput)
        return (
            concatenated([first.queries, tail.queries[0..<1, 0...]], axis: 0),
            concatenated([first.keys, tail.keys[0..<1, 0...]], axis: 0),
            concatenated([first.values, tail.values[0..<1, 0...]], axis: 0)
        )
    }
}

extension FusedFullQKProjection {
    func exactThreeVector(_ input: MLXArray) -> (queries: MLXArray, keys: MLXArray) {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
        let first = exactTwoVector(input[0..<2, 0...])
        let tailInput = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVector(tailInput)
        return (
            concatenated([first.queries, tail.queries[0..<1, 0...]], axis: 0),
            concatenated([first.keys, tail.keys[0..<1, 0...]], axis: 0)
        )
    }
}
