import MLX

extension Gemma4TiedVocabularyHead {
    func exactThreeVectorPacked13Softcapped(_ input: MLXArray, cap: MLXArray) -> MLXArray {
        precondition(input.dtype == .bfloat16 && input.shape == [3, 5_376])
        let first = exactTwoVectorPacked13Softcapped(input[0..<2, 0...], cap: cap)
        let duplicated = concatenated([input[2..<3, 0...], input[2..<3, 0...]], axis: 0)
        let tail = exactTwoVectorPacked13Softcapped(duplicated, cap: cap)
        return concatenated([first, tail[0..<1, 0...]], axis: 0)
    }
}
