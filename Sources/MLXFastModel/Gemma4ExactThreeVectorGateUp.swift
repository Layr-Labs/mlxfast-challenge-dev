import MLX

extension FusedGateUpProjection {
    func exactThreeVectorActivated(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
        let first = exactTwoVectorActivated(input[0..<2, 0...])
        let duplicated = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVectorActivated(duplicated)
        return concatenated([first, tail[0..<1, 0...]], axis: 0)
    }
}
