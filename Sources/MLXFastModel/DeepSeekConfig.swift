import Foundation
import MLXFastCore

public struct DeepSeekConfig: Equatable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let sharedExperts: Int
    public let routedExperts: Int
    public let routedScalingFactor: Double
    public let qLoraRank: Int
    public let qkRopeHeadDim: Int
    public let expertsPerToken: Int
    public let normTopkProb: Bool
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Double
    public let ropeTheta: Double
    public let ropeScaling: DeepSeekRopeScaling?
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let headDim: Int
    public let scoringFunc: String
    public let compressRatios: [Int]
    public let compressRopeTheta: Double
    public let hcMult: Int
    public let hcSinkhornIters: Int
    public let hcEps: Double
    public let numHashLayers: Int
    public let swigluLimit: Double
    public let slidingWindow: Int
    public let outputGroups: Int
    public let outputLoraRank: Int
    public let indexHeads: Int
    public let indexHeadDim: Int
    public let indexTopk: Int
    public let tieWordEmbeddings: Bool
    public let topkMethod: String

    public static func load(from weightsPath: String) throws -> DeepSeekConfig {
        let path = URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json")
        try requireFile(path.path, description: "transformed weights config")

        let data = try Data(contentsOf: path)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("config.json must be a JSON object")
        }

        let textConfig = root["text_config"] as? [String: Any]
        let numHiddenLayers = try intField(
            "num_hidden_layers",
            root: root,
            nested: textConfig
        )
        let compressRatios = try normalizeCompressRatios(
            intArrayField(
                "compress_ratios",
                root: root,
                nested: textConfig,
                defaultValue: DeepSeekConfig.defaultCompressRatios(layerCount: numHiddenLayers)
            ),
            layerCount: numHiddenLayers
        )
        let config = DeepSeekConfig(
            modelType: try stringField("model_type", root: root, nested: textConfig, defaultValue: "deepseek_v4"),
            vocabSize: try intField("vocab_size", root: root, nested: textConfig),
            hiddenSize: try intField("hidden_size", root: root, nested: textConfig, defaultValue: MLXFastConstants.hiddenSize),
            intermediateSize: try intField("intermediate_size", root: root, nested: textConfig, defaultValue: MLXFastConstants.intermediateSize),
            moeIntermediateSize: try intField("moe_intermediate_size", root: root, nested: textConfig, defaultValue: MLXFastConstants.moeIntermediateSize),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try intField("num_attention_heads", root: root, nested: textConfig, defaultValue: MLXFastConstants.attentionHeads),
            numKeyValueHeads: try intField("num_key_value_heads", root: root, nested: textConfig, defaultValue: MLXFastConstants.keyValueHeads),
            sharedExperts: try intField("n_shared_experts", root: root, nested: textConfig, defaultValue: 1),
            routedExperts: try intField("n_routed_experts", root: root, nested: textConfig),
            routedScalingFactor: try doubleField("routed_scaling_factor", root: root, nested: textConfig, defaultValue: 1.5),
            qLoraRank: try intField("q_lora_rank", root: root, nested: textConfig, defaultValue: 1_024),
            qkRopeHeadDim: try intField("qk_rope_head_dim", root: root, nested: textConfig, defaultValue: 64),
            expertsPerToken: try intField("num_experts_per_tok", root: root, nested: textConfig),
            normTopkProb: try boolField("norm_topk_prob", root: root, nested: textConfig, defaultValue: true),
            hiddenAct: try stringField("hidden_act", root: root, nested: textConfig, defaultValue: "silu"),
            maxPositionEmbeddings: try intField("max_position_embeddings", root: root, nested: textConfig, defaultValue: 1_048_576),
            rmsNormEps: try doubleField("rms_norm_eps", root: root, nested: textConfig, defaultValue: 1e-6),
            ropeTheta: try doubleField("rope_theta", root: root, nested: textConfig, defaultValue: 10_000.0),
            ropeScaling: try DeepSeekRopeScaling.loadOptional(root: root, nested: textConfig),
            attentionBias: try boolField("attention_bias", root: root, nested: textConfig, defaultValue: false),
            attentionDropout: try doubleField("attention_dropout", root: root, nested: textConfig, defaultValue: 0.0),
            headDim: try intField("head_dim", root: root, nested: textConfig, defaultValue: 512),
            scoringFunc: try stringField("scoring_func", root: root, nested: textConfig, defaultValue: "sqrtsoftplus"),
            compressRatios: compressRatios,
            compressRopeTheta: try doubleField("compress_rope_theta", root: root, nested: textConfig, defaultValue: 160_000.0),
            hcMult: try intField("hc_mult", root: root, nested: textConfig, defaultValue: 4),
            hcSinkhornIters: try intField("hc_sinkhorn_iters", root: root, nested: textConfig, defaultValue: 20),
            hcEps: try doubleField("hc_eps", root: root, nested: textConfig, defaultValue: 1e-6),
            numHashLayers: try intField("num_hash_layers", root: root, nested: textConfig, defaultValue: 3),
            swigluLimit: try doubleField("swiglu_limit", root: root, nested: textConfig, defaultValue: 10.0),
            slidingWindow: try intField("sliding_window", root: root, nested: textConfig, defaultValue: 128),
            outputGroups: try intField("o_groups", root: root, nested: textConfig, defaultValue: 8),
            outputLoraRank: try intField("o_lora_rank", root: root, nested: textConfig, defaultValue: 1_024),
            indexHeads: try intField("index_n_heads", root: root, nested: textConfig, defaultValue: 64),
            indexHeadDim: try intField("index_head_dim", root: root, nested: textConfig, defaultValue: 128),
            indexTopk: try intField("index_topk", root: root, nested: textConfig, defaultValue: 512),
            tieWordEmbeddings: try boolField("tie_word_embeddings", root: root, nested: textConfig, defaultValue: false),
            topkMethod: try stringField("topk_method", root: root, nested: textConfig, defaultValue: "noaux_tc")
        )
        try config.validateFrozenInvariants()
        return config
    }

    public static func defaultCompressRatios(layerCount: Int) -> [Int] {
        guard layerCount > 0 else {
            return []
        }
        return (0..<layerCount).map { layerIndex in
            layerIndex < 2 ? 0 : (layerIndex % 2 == 0 ? 4 : 128)
        }
    }

    public func validateFrozenInvariants() throws {
        let expected: [(String, Int, Int)] = [
            ("vocab_size", vocabSize, MLXFastConstants.vocabSize),
            ("hidden_size", hiddenSize, MLXFastConstants.hiddenSize),
            ("intermediate_size", intermediateSize, MLXFastConstants.intermediateSize),
            ("moe_intermediate_size", moeIntermediateSize, MLXFastConstants.moeIntermediateSize),
            ("num_hidden_layers", numHiddenLayers, MLXFastConstants.numHiddenLayers),
            ("num_attention_heads", numAttentionHeads, MLXFastConstants.attentionHeads),
            ("num_key_value_heads", numKeyValueHeads, MLXFastConstants.keyValueHeads),
            ("n_routed_experts", routedExperts, MLXFastConstants.routedExperts),
            ("num_experts_per_tok", expertsPerToken, MLXFastConstants.expertsPerToken),
        ]
        let errors = expected.compactMap { name, actual, expected in
            actual == expected ? nil : "\(name)=\(actual) expected \(expected)"
        }
        let badRatios = compressRatios.filter { ![0, 4, 128].contains($0) }
        let ratioErrors = [
            compressRatios.count == numHiddenLayers
                ? nil
                : "compress_ratios count=\(compressRatios.count) expected \(numHiddenLayers)",
            badRatios.isEmpty ? nil : "compress_ratios contains unsupported values \(badRatios)",
        ].compactMap { $0 }
        let allErrors = errors + ratioErrors
        if !allErrors.isEmpty {
            throw MLXFastError.invalidInput(
                "DeepSeek V4 Flash config invariant check failed: \(allErrors.joined(separator: ", "))"
            )
        }
    }
}

public struct DeepSeekRopeScaling: Equatable {
    public let type: String?
    public let ropeType: String?
    public let factor: Double?
    public let originalMaxPositionEmbeddings: Int?
    public let betaFast: Double?
    public let betaSlow: Double?

    public static func loadOptional(
        root: [String: Any],
        nested: [String: Any]?
    ) throws -> DeepSeekRopeScaling? {
        guard let raw = fieldValue("rope_scaling", root: root, nested: nested) else {
            return nil
        }
        guard !(raw is NSNull) else {
            return nil
        }
        guard let object = raw as? [String: Any] else {
            throw MLXFastError.invalidInput("config field rope_scaling must be an object or null")
        }
        return DeepSeekRopeScaling(
            type: try optionalString(object["type"], field: "rope_scaling.type"),
            ropeType: try optionalString(object["rope_type"], field: "rope_scaling.rope_type"),
            factor: try optionalDouble(object["factor"], field: "rope_scaling.factor"),
            originalMaxPositionEmbeddings: try optionalInt(
                object["original_max_position_embeddings"],
                field: "rope_scaling.original_max_position_embeddings"
            ),
            betaFast: try optionalDouble(object["beta_fast"], field: "rope_scaling.beta_fast"),
            betaSlow: try optionalDouble(object["beta_slow"], field: "rope_scaling.beta_slow")
        )
    }
}

private func intField(
    _ key: String,
    root: [String: Any],
    nested: [String: Any]?,
    defaultValue: Int? = nil
) throws -> Int {
    if let value = fieldValue(key, root: root, nested: nested) {
        return try parseInt(value, field: key)
    }
    if let defaultValue {
        return defaultValue
    }
    throw MLXFastError.invalidInput("config.json missing required field \(key)")
}

private func normalizeCompressRatios(_ ratios: [Int], layerCount: Int) throws -> [Int] {
    guard layerCount >= 0 else {
        throw MLXFastError.invalidInput("num_hidden_layers must be non-negative")
    }
    if ratios.count == layerCount {
        return ratios
    }
    if ratios.count == layerCount + 1, ratios.last == 0 {
        return Array(ratios.prefix(layerCount))
    }
    throw MLXFastError.invalidInput(
        "compress_ratios count=\(ratios.count) expected \(layerCount)"
    )
}

private func intArrayField(
    _ key: String,
    root: [String: Any],
    nested: [String: Any]?,
    defaultValue: [Int]
) throws -> [Int] {
    guard let value = fieldValue(key, root: root, nested: nested) else {
        return defaultValue
    }
    guard let values = value as? [Any] else {
        throw MLXFastError.invalidInput("config field \(key) must be an integer array")
    }
    return try values.enumerated().map { index, value in
        try parseInt(value, field: "\(key)[\(index)]")
    }
}

private func doubleField(
    _ key: String,
    root: [String: Any],
    nested: [String: Any]?,
    defaultValue: Double
) throws -> Double {
    guard let value = fieldValue(key, root: root, nested: nested) else {
        return defaultValue
    }
    return try parseDouble(value, field: key)
}

private func boolField(
    _ key: String,
    root: [String: Any],
    nested: [String: Any]?,
    defaultValue: Bool
) throws -> Bool {
    guard let value = fieldValue(key, root: root, nested: nested) else {
        return defaultValue
    }
    guard let bool = value as? Bool else {
        throw MLXFastError.invalidInput("config field \(key) must be a boolean")
    }
    return bool
}

private func stringField(
    _ key: String,
    root: [String: Any],
    nested: [String: Any]?,
    defaultValue: String
) throws -> String {
    guard let value = fieldValue(key, root: root, nested: nested) else {
        return defaultValue
    }
    guard let string = value as? String else {
        throw MLXFastError.invalidInput("config field \(key) must be a string")
    }
    return string
}

private func fieldValue(_ key: String, root: [String: Any], nested: [String: Any]?) -> Any? {
    if let value = root[key] {
        return value is NSNull ? nil : value
    }
    if let value = nested?[key] {
        return value is NSNull ? nil : value
    }
    return nil
}

private func parseInt(_ value: Any, field: String) throws -> Int {
    if let int = value as? Int {
        return int
    }
    if let number = value as? NSNumber {
        return number.intValue
    }
    throw MLXFastError.invalidInput("config field \(field) must be an integer")
}

private func parseDouble(_ value: Any, field: String) throws -> Double {
    if let double = value as? Double {
        return double
    }
    if let int = value as? Int {
        return Double(int)
    }
    if let number = value as? NSNumber {
        return number.doubleValue
    }
    throw MLXFastError.invalidInput("config field \(field) must be a number")
}

private func optionalInt(_ value: Any?, field: String) throws -> Int? {
    guard let value, !(value is NSNull) else {
        return nil
    }
    return try parseInt(value, field: field)
}

private func optionalDouble(_ value: Any?, field: String) throws -> Double? {
    guard let value, !(value is NSNull) else {
        return nil
    }
    return try parseDouble(value, field: field)
}

private func optionalString(_ value: Any?, field: String) throws -> String? {
    guard let value, !(value is NSNull) else {
        return nil
    }
    guard let string = value as? String else {
        throw MLXFastError.invalidInput("config field \(field) must be a string")
    }
    return string
}
