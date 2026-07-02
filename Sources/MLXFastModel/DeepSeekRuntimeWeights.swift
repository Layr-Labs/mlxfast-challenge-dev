import Foundation

public final class DeepSeekRuntimeWeightCache {
    public let loader: DeepSeekWeightLoader
    public let config: DeepSeekConfig
    public let modelSpec: DeepSeekModelSpec
    public let blockSpec: DeepSeekBlockSpec

    private var cachedModelWeights: DeepSeekModelWeights?
    private var cachedBlockWeights: [DeepSeekBlockWeights?]
    private var cachedLocalAttentionWeights: [DeepSeekLocalAttentionWeights?]
    private var cachedCompressedAttentionWeights: [DeepSeekCompressedAttentionWeights?]
    private var cachedMoEWeights: [DeepSeekMoEWeights?]
    private var cachedMoESpecs: [DeepSeekMoESpec?]
    private var cachedLocalAttentionSpecs: [DeepSeekLocalAttentionSpec?]
    private var cachedCompressedAttentionSpecs: [DeepSeekCompressedAttentionSpec?]

    public init(loader: DeepSeekWeightLoader, config: DeepSeekConfig) {
        self.loader = loader
        self.config = config
        self.modelSpec = DeepSeekModelSpec(config: config)
        self.blockSpec = DeepSeekBlockSpec(config: config)
        self.cachedBlockWeights = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedLocalAttentionWeights = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedCompressedAttentionWeights = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedMoEWeights = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedMoESpecs = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedLocalAttentionSpecs = Array(repeating: nil, count: config.numHiddenLayers)
        self.cachedCompressedAttentionSpecs = Array(repeating: nil, count: config.numHiddenLayers)
    }

    public func modelWeights() throws -> DeepSeekModelWeights {
        if let cachedModelWeights {
            return cachedModelWeights
        }
        let weights = try loader.modelWeights(config: config)
        cachedModelWeights = weights
        return weights
    }

    public func blockWeights(layerIndex: Int) throws -> DeepSeekBlockWeights {
        if let weights = cachedBlockWeights[layerIndex] {
            return weights
        }
        let weights = try loader.blockWeights(layerIndex: layerIndex, config: config)
        cachedBlockWeights[layerIndex] = weights
        return weights
    }

    public func localAttentionWeights(layerIndex: Int) throws -> DeepSeekLocalAttentionWeights {
        if let weights = cachedLocalAttentionWeights[layerIndex] {
            return weights
        }
        let weights = try loader.localAttentionWeights(layerIndex: layerIndex, config: config)
        cachedLocalAttentionWeights[layerIndex] = weights
        return weights
    }

    public func compressedAttentionWeights(layerIndex: Int) throws -> DeepSeekCompressedAttentionWeights {
        if let weights = cachedCompressedAttentionWeights[layerIndex] {
            return weights
        }
        let weights = try loader.compressedAttentionWeights(layerIndex: layerIndex, config: config)
        cachedCompressedAttentionWeights[layerIndex] = weights
        return weights
    }

    public func moeWeights(layerIndex: Int) throws -> DeepSeekMoEWeights {
        if let weights = cachedMoEWeights[layerIndex] {
            return weights
        }
        let weights = try loader.moeWeights(layerIndex: layerIndex, config: config)
        cachedMoEWeights[layerIndex] = weights
        return weights
    }

    public func moeSpec(layerIndex: Int) throws -> DeepSeekMoESpec {
        if let spec = cachedMoESpecs[layerIndex] {
            return spec
        }
        let spec = try DeepSeekMoESpec(layerIndex: layerIndex, config: config)
        cachedMoESpecs[layerIndex] = spec
        return spec
    }

    public func localAttentionSpec(layerIndex: Int) -> DeepSeekLocalAttentionSpec {
        precondition(cachedLocalAttentionSpecs.indices.contains(layerIndex), "DeepSeek layer index out of range")
        if let spec = cachedLocalAttentionSpecs[layerIndex] {
            return spec
        }
        let spec = DeepSeekLocalAttentionSpec(config: config)
        cachedLocalAttentionSpecs[layerIndex] = spec
        return spec
    }

    public func compressedAttentionSpec(layerIndex: Int) -> DeepSeekCompressedAttentionSpec {
        precondition(cachedCompressedAttentionSpecs.indices.contains(layerIndex), "DeepSeek layer index out of range")
        if let spec = cachedCompressedAttentionSpecs[layerIndex] {
            return spec
        }
        let spec = DeepSeekCompressedAttentionSpec(config: config, layerIndex: layerIndex)
        cachedCompressedAttentionSpecs[layerIndex] = spec
        return spec
    }
}
