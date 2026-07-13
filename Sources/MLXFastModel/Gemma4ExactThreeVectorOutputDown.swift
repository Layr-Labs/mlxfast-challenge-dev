import MLX

extension IndexedOutputProjection {
    func exactThreeVector(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [3, inputWidth])
        let first = exactTwoVector(input[0..<2, 0...])
        let duplicated = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVector(duplicated)
        return concatenated([first, tail[0..<1, 0...]], axis: 0)
    }
}

extension IndexedDownProjection {
    func exactThreeVector(_ input: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 21_504])
        let first = exactTwoVector(input[0..<2, 0...])
        let duplicated = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVector(duplicated)
        return concatenated([first, tail[0..<1, 0...]], axis: 0)
    }
}
