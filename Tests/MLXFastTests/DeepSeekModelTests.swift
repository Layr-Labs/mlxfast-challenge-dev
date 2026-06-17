import Foundation
import MLX
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekModelInitialHiddenBroadcastsEmbeddingWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekModelSpec(
        vocabSize: 3,
        hiddenSize: 2,
        numHiddenLayers: 0,
        compressRatios: [],
        slidingWindow: 4,
        hcMult: 2,
        hcSinkhornIters: 1,
        hcEps: 0,
        rmsNormEps: 0
    )
    let inputIDs = MLXArray([Int32(2), Int32(0)], [1, 2])
    let embedding = MLXArray((1...6).map { Float($0) }, [3, 2])

    let hidden = try DeepSeekModel.initialHidden(
        inputIDs: inputIDs,
        embedding: embedding,
        spec: spec
    )

    #expect(hidden.shape == [1, 2, 2, 2])
    #expect(hidden.asArray(Float.self) == [5, 6, 5, 6, 1, 2, 1, 2])
}

@Test
func deepSeekModelTopLevelLogitsRunInjectedLayersWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let spec = DeepSeekModelSpec(
        vocabSize: 3,
        hiddenSize: 2,
        numHiddenLayers: 1,
        compressRatios: [0],
        slidingWindow: 4,
        hcMult: 2,
        hcSinkhornIters: 1,
        hcEps: 0,
        rmsNormEps: 0
    )
    let weights = DeepSeekModelWeights(
        embedTokens: MLXArray((1...6).map { Float($0) }, [3, 2]),
        finalNorm: ones([2], dtype: .float32),
        headHyperConnection: DeepSeekHeadHyperConnectionWeights(
            fn: zeros([2, 4], dtype: .float32),
            base: zeros([2], dtype: .float32),
            scale: ones([1], dtype: .float32)
        ),
        lmHead: MLXArray([Float(1), 0, 0, 1, 1, 1], [3, 2])
    )
    let inputIDs = MLXArray([Int32(1)], [1, 1])
    var seenLayers: [Int] = []

    let logits = try DeepSeekModel.logits(
        inputIDs: inputIDs,
        weights: weights,
        spec: spec
    ) { layerIndex, hidden in
        seenLayers.append(layerIndex)
        return hidden
    }

    #expect(seenLayers == [0])
    #expect(logits.shape == [1, 1, 3])
    let values = logits.asArray(Float.self)
    #expect(abs(values[0] - 0.84852815) < 1e-5)
    #expect(abs(values[1] - 1.1313709) < 1e-5)
    #expect(abs(values[2] - 1.979899) < 1e-5)
}
