import MLX
import MLXFastCore

/// Stock-operation full-attention graph for Gemma 4's exact L512 prefill
/// shape. Flattening repeat and query-row dimensions turns each KV head into
/// one large-M GEMM while preserving every matrix reduction and head order.
func gemma4KVHeadMajorFullPrefillAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    scale: Float
) -> MLXArray {
    precondition(queries.dtype == .bfloat16)
    precondition(keys.dtype == .bfloat16 && values.dtype == .bfloat16)
    precondition(queries.shape == [1, 32, 512, 512])
    precondition(keys.shape == [1, 4, 512, 512])
    precondition(values.shape == [1, 4, 512, 512])
    precondition(scale == 1.0)

    let qGrouped = queries.reshaped(1, 4, 8, 512, 512)
    let qFlat = qGrouped.reshaped(1, 4, 4096, 512)
    let kT = keys.swappedAxes(-1, -2)
    let scoresFlat = matmul(qFlat, kT)
    var scores = scoresFlat.reshaped(1, 4, 8, 512, 512)

    func applyMask(_ maskArray: MLXArray) {
        var broadcastMask = maskArray
        if broadcastMask.ndim == 4 && broadcastMask.dim(0) == 1 {
            broadcastMask = expandedDimensions(broadcastMask, axis: 2)
        }
        if broadcastMask.dtype == .bool {
            // Largest-magnitude finite negative BF16, represented exactly as
            // Float32 before MLX applies the score dtype.
            let lowestBF16 = MLXArray(
                Float(bitPattern: 0xff7f_0000), dtype: scores.dtype)
            scores = MLX.where(broadcastMask, scores, lowestBF16)
        } else {
            scores = scores + broadcastMask
        }
    }

    switch mask {
    case .none:
        break
    case .causal:
        let rows = MLXArray(0..<512)
        let columns = MLXArray(0..<512)
        let causal = greaterEqual(
            expandedDimensions(rows, axis: -1),
            expandedDimensions(columns, axis: -2))
        applyMask(causal)
    case .array(let maskArray):
        applyMask(maskArray)
    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            applyMask(maskArray)
        }
    }

    var probs = softmax(scores.asType(.float32), axis: -1, precise: true)
    probs = MLX.where(probs .!= probs, MLXArray(Float(0)), probs)
    let probsBF16 = probs.asType(scores.dtype)
    let probsFlat = probsBF16.reshaped(1, 4, 4096, 512)
    let outputFlat = matmul(probsFlat, values)
    return outputFlat
        .reshaped(1, 4, 8, 512, 512)
        .reshaped(1, 32, 512, 512)
}
