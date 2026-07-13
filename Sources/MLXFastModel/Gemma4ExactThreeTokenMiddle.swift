import MLX

/// Safe production path: all three queries use the same serialized one-query
/// SDPA shape as ordinary decode. No q=3 batched sliding route is enabled.
func gemma4ExactThreeTokenAttention(
    queries: MLXArray,
    keysBeforeDrafts: MLXArray,
    valuesBeforeDrafts: MLXArray,
    keysWithDrafts: MLXArray,
    valuesWithDrafts: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.ndim == 4 && queries.dim(2) == 3)
    precondition(keysWithDrafts.dim(2) == keysBeforeDrafts.dim(2) + 2)
    let middleEnd = keysWithDrafts.dim(2) - 1
    let keysThroughFirst = keysWithDrafts[0..., 0..., 0..<middleEnd, 0...]
    let valuesThroughFirst = valuesWithDrafts[0..., 0..., 0..<middleEnd, 0...]
    let a0 = MLXFast.scaledDotProductAttention(
        queries: queries[0..., 0..., 0..<1, 0...], keys: keysBeforeDrafts,
        values: valuesBeforeDrafts, scale: scale, mask: .none)
    let a1 = MLXFast.scaledDotProductAttention(
        queries: queries[0..., 0..., 1..<2, 0...], keys: keysThroughFirst,
        values: valuesThroughFirst, scale: scale, mask: .none)
    let a2 = MLXFast.scaledDotProductAttention(
        queries: queries[0..., 0..., 2..<3, 0...], keys: keysWithDrafts,
        values: valuesWithDrafts, scale: scale, mask: .none)
    return concatenated([a0, a1, a2], axis: 2)
}
