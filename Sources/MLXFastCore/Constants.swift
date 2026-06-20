public enum MLXFastConstants {
    public static let referenceModelName = "DeepSeek-V4-Flash-4bit"
    public static let defaultReferencePath = "reference_weights/DeepSeek-V4-Flash-4bit"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultScorePath = "score.json"

    public static let vocabSize = 129_280
    public static let hiddenSize = 4_096
    public static let intermediateSize = 18_432
    public static let moeIntermediateSize = 2_048
    public static let numHiddenLayers = 43
    public static let attentionHeads = 64
    public static let keyValueHeads = 1
    public static let routedExperts = 256
    public static let expertsPerToken = 6
    public static let correctnessSteps = 256
    public static let benchmarkPrefillPromptTokens = 512
    public static let benchmarkDecodeSteps = 512
    public static let benchmarkDecodeSeedTokens = 32
    public static let benchmarkPrefillWarmupRuns = 1
    public static let benchmarkPrefillTimedRuns = 2
}
