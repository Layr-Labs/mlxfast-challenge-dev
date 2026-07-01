public enum MLXFastConstants {
    public static let referenceModelName = "DeepSeek-V4-Flash-4bit"
    public static let defaultReferencePath = "reference_weights/DeepSeek-V4-Flash-4bit"
    public static let defaultReferenceCachePath = ".cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_512.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_256.json"
    public static let defaultPublicLocalSubmitGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_1024.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

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
    // Keep the public gate long enough to catch broad decode regressions while
    // leaving budget for the hidden GPQA behavior checks in the official job.
    public static let correctnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 64
    public static let correctnessGPQACaseCount = 5
    // Cross-machine greedy decode can drift after the first answer token even
    // with pinned Swift/MLX. Exact GPQA behavior accepts the stable first-token
    // prefix, while the short continuation feeds the private semantic judge.
    public static let correctnessGPQAMaxNewTokens = 10
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // hard gate for optimizations that preserve the exact prefix but damage
    // answer sense. Five cases keeps the full GitHub job near the 30-minute
    // budget; baseline DeepSeek currently establishes a 3/5 threshold.
    public static let semanticGPQACaseCount = 5
    public static let semanticGPQAMaxNewTokens = 10
    public static let semanticGPQAMinPassCount = 3
    public static let benchmarkPrefillPromptTokens = 512
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
    public static let localIterateBenchmarkDecodeSteps = 16
    // Local submit uses a longer public fixture so the Yukon pre-submit hook
    // exercises one continuous decode trajectory for about ten minutes instead
    // of repeating the short local-iterate correctness window.
    public static let localSubmitBenchmarkDecodeSteps = 1023
    public static let localSubmitBenchmarkRepeats = 1
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    public static let benchmarkDecodeSeedTokens = 512
    // Correctness and hidden GPQA already run substantial prefills before the
    // timed benchmark. Avoid a separate prefill warmup so the full GitHub job
    // stays within the 30-minute target while keeping one measured prefill run.
    public static let benchmarkPrefillWarmupRuns = 0
    public static let benchmarkPrefillTimedRuns = 1
    // Official baseline measured on the Blacksmith runner for this model. After
    // changing timed windows, run one trusted baseline validation before using
    // scores for the public leaderboard. Raw RAM, bandwidth, and read metrics
    // remain audit fields instead of primary score factors.
    public static let officialBaselinePrefillSecondsPerToken = 0.17330563175390626
    public static let officialBaselineDecodeSecondsPerToken = 4.220506571617188
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let scorePrefillSpeedupFloor = 0.95
    public static let scoreDecodeSpeedupFloor = 0.95
    public static let defaultMaxTransformedWeightsBytes = 25 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024
}
