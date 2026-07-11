import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

func gemma4LastTokenRange(sequenceLength: Int) -> Range<Int>? {
    sequenceLength > 1 ? (sequenceLength - 1)..<sequenceLength : nil
}

func gemma4LastTokenHidden(_ hidden: MLXArray) -> MLXArray {
    guard let range = gemma4LastTokenRange(sequenceLength: hidden.dim(1)) else {
        return hidden
    }
    return hidden[0..., range, 0...]
}

/// Scored Gemma 4 runtime model: loads the pinned library trunk modules, then
/// executes a custom high-performance path with fused compiled MLPs and a
/// last-token-only vocabulary head while keeping eager active-length attention.
public final class Gemma4RuntimeModel: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: Gemma4TextModelInner

    public let configuration: Gemma4TextConfiguration
    private var fastEngine: Gemma4FastEngine?

    public init(_ config: Gemma4TextConfiguration) {
        self.configuration = config
        self._model.wrappedValue = Gemma4TextModelInner(config)
        super.init()
    }

    /// Build the fused-MLP engine after weights have been loaded and quantized.
    public func prepareFastEngine() throws {
        fastEngine = try Gemma4FastEngine(model: self)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        if let fastEngine {
            return fastEngine(inputs, cache: cache)
        }
        // Fallback before prepareFastEngine(): library trunk + last-token head.
        let fullHidden = model(inputs, cache: cache)
        let hidden = gemma4LastTokenHidden(fullHidden)
        let logits = model.embedTokens.asLinear(hidden)
        return tanh(logits / MLXArray(configuration.finalLogitSoftcapping))
            * MLXArray(configuration.finalLogitSoftcapping)
    }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func newCache(parameters _: GenerateParameters?) -> [KVCache] {
        let firstSharedLayer = configuration.numHiddenLayers - configuration.numKvSharedLayers
        return (0..<firstSharedLayer).map { layerIndex in
            if configuration.layerTypes[layerIndex] == Gemma4LayerType.full.rawValue {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
            }
        }
    }
}
