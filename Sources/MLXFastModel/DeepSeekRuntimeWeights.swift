import Foundation

public final class DeepSeekRuntimeWeightCache {
    public let loader: DeepSeekWeightLoader
    public let config: DeepSeekConfig

    private var cachedModelWeights: DeepSeekModelWeights?
    private var cachedBlockWeights: [Int: DeepSeekBlockWeights] = [:]
    private var cachedLocalAttentionWeights: [Int: DeepSeekLocalAttentionWeights] = [:]
    private var cachedCompressedAttentionWeights: [Int: DeepSeekCompressedAttentionWeights] = [:]
    private var cachedMoEWeights: [Int: DeepSeekMoEWeights] = [:]
    private var cachedBlockSpec: DeepSeekBlockSpec?
    private var cachedLocalAttentionSpec: DeepSeekLocalAttentionSpec?
    private var cachedCompressedAttentionSpecs: [Int: DeepSeekCompressedAttentionSpec] = [:]
    private var cachedMoESpecs: [Int: DeepSeekMoESpec] = [:]

    public init(loader: DeepSeekWeightLoader, config: DeepSeekConfig) {
        self.loader = loader
        self.config = config
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

    public func blockSpec() -> DeepSeekBlockSpec {
        if let cachedBlockSpec {
            return cachedBlockSpec
        }
        let spec = DeepSeekBlockSpec(config: config)
        cachedBlockSpec = spec
        return spec
    }

    public func localAttentionSpec() -> DeepSeekLocalAttentionSpec {
        if let cachedLocalAttentionSpec {
            return cachedLocalAttentionSpec
        }
        let spec = DeepSeekLocalAttentionSpec(config: config)
        cachedLocalAttentionSpec = spec
        return spec
    }

    public func compressedAttentionSpec(layerIndex: Int) -> DeepSeekCompressedAttentionSpec {
        if let spec = cachedCompressedAttentionSpecs[layerIndex] {
            return spec
        }
        let spec = DeepSeekCompressedAttentionSpec(config: config, layerIndex: layerIndex)
        cachedCompressedAttentionSpecs[layerIndex] = spec
        return spec
    }

    public func moeSpec(layerIndex: Int) throws -> DeepSeekMoESpec {
        if let spec = cachedMoESpecs[layerIndex] {
            return spec
        }
        let spec = try DeepSeekMoESpec(layerIndex: layerIndex, config: config)
        cachedMoESpecs[layerIndex] = spec
        return spec
    }
}
