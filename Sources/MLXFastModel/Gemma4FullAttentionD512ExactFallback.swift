import Foundation
import MLX

func supportsGemma4FullAttentionD512ExactFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> Bool {
    guard case .none = mask else { return false }
    return scale == 1.0
        && queries.dtype == .bfloat16
        && keys.dtype == .bfloat16
        && values.dtype == .bfloat16
        && queries.shape == [1, 32, 1, 512]
        && keys.ndim == 4
        && values.ndim == 4
        && keys.dim(0) == 1
        && keys.dim(1) == 4
        && keys.dim(2) > 0
        && keys.dim(3) == 512
        && values.shape == keys.shape
}

/// Exact-shape spelling of the pinned MLX D=512 fallback graph.
///
/// The GQA reshapes/expansions below are stride-only views. QK, precise
/// softmax, and PV remain the stock MLX primitives, preserving both BF16
/// materialization boundaries and the device-selected reduction order. The
/// only compute operation omitted is BF16 `queries * 1`, which is a bitwise
/// identity for the finite BF16 activations on the model inference path.
func gemma4FullAttentionD512ExactFallback(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray
) -> MLXArray {
    precondition(supportsGemma4FullAttentionD512ExactFallback(
        queries: queries,
        keys: keys,
        values: values,
        scale: 1.0,
        mask: .none
    ))

    let groupedQueries = queries.reshaped([1, 4, 8, 1, 512])
    let groupedKeys = expandedDimensions(keys, axis: 2)
    let groupedValues = expandedDimensions(values, axis: 2)
    let scores = matmul(
        groupedQueries,
        groupedKeys.swappedAxes(-1, -2)
    )
    let probabilities = softmax(scores, axis: -1, precise: true)
    return matmul(probabilities, groupedValues)
        .reshaped([1, 32, 1, 512])
}

/// Diagnostic-only raw BF16 verifier. Both graphs are evaluated and compared
/// as UInt16 payloads; the first mismatch terminates without a tolerance path.
func verifyGemma4FullAttentionD512ExactFallbackBits(
    candidate: MLXArray,
    reference: MLXArray
) {
    eval(candidate, reference)
    let candidateBits = candidate.view(dtype: .uint16).asArray(UInt16.self)
    let referenceBits = reference.view(dtype: .uint16).asArray(UInt16.self)
    precondition(candidateBits.count == referenceBits.count)

    if let firstMismatch = candidateBits.indices.first(where: {
        candidateBits[$0] != referenceBits[$0]
    }) {
        let candidatePayload = String(
            format: "0x%04x", candidateBits[firstMismatch])
        let referencePayload = String(
            format: "0x%04x", referenceBits[firstMismatch])
        preconditionFailure(
            "D512 exact fallback differs from stock at raw UInt16 index "
                + "\(firstMismatch): candidate=\(candidatePayload), "
                + "reference=\(referencePayload)"
        )
    }
}
