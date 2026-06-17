import Foundation
import MLX

public enum DeepSeekOps {
    public static func embedding(inputIDs: MLXArray, weight: MLXArray) -> MLXArray {
        weight[inputIDs]
    }

    public static func linear(input: MLXArray, weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        if let bias {
            return addMM(bias, input, weight.T)
        }
        return matmul(input, weight.T)
    }

    public static func multiLinear(input: MLXArray, weight: MLXArray, transpose: Bool = true) -> MLXArray {
        if transpose {
            return matmul(input, weight.swappedAxes(-1, -2))
        }
        return matmul(input, weight)
    }

    public static func rmsNorm(input: MLXArray, weight: MLXArray, eps: Double) -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight, eps: Float(eps))
    }

    public static func silu(_ input: MLXArray) -> MLXArray {
        input * sigmoid(input)
    }

    public static func limitedSwiGLU(gate: MLXArray, up: MLXArray, limit: Double) -> MLXArray {
        guard limit > 0 else {
            return silu(gate) * up
        }
        let cappedGate = minimum(gate, Float(limit))
        let clippedUp = clip(up, min: Float(-limit), max: Float(limit))
        return silu(cappedGate) * clippedUp
    }
}
