public enum MLXFastConstants {
    public static let referenceModelName = "DeepSeek-V4-Flash-4bit"
    public static let defaultReferencePath = "reference_weights/DeepSeek-V4-Flash-4bit"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_512.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_256.json"
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
    public static let correctnessPromptTokens = 512
    // Keep the gate and timed decode windows long enough to exercise cache and
    // expert-routing behavior while staying within the target benchmark budget.
    public static let correctnessSteps = 256
    public static let quickCorrectnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 64
    public static let correctnessGPQACaseCount = 9
    // Cross-machine greedy decode can drift after the first answer token even
    // with pinned Swift/MLX. Keep hidden GPQA behavior gates broad across cases
    // and shallow per case so local M-series and official Blacksmith runs agree.
    public static let correctnessGPQAMaxNewTokens = 1
    // Semantic judging uses short hidden GPQA answers as a pass/fail backstop
    // for optimizations that preserve the exact prefix but damage answer sense.
    public static let semanticGPQACaseCount = 9
    public static let semanticGPQAMaxNewTokens = 10
    public static let semanticGPQAMinPassCount = 8
    public static let benchmarkPrefillPromptTokens = 512
    public static let benchmarkDecodeSteps = 256
    public static let quickBenchmarkDecodeSteps = 64
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    public static let benchmarkDecodeSeedTokens = 512
    public static let benchmarkPrefillWarmupRuns = 1
    public static let benchmarkPrefillTimedRuns = 1
    // Official baseline measured on the Blacksmith runner for the current
    // 512-token prefill / 256-token decode benchmark oracle. The ranked score is
    // normalized to this baseline; raw RAM, bandwidth, and read metrics remain
    // audit fields instead of primary score factors.
    public static let officialBaselinePrefillSecondsPerToken = 0.1417240929375
    public static let officialBaselineDecodeSecondsPerToken = 3.018321923023438
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let defaultMaxTransformedWeightsBytes = 50 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024
}
