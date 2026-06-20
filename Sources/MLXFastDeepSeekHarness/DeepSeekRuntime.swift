import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastDeepSeek

public struct CorrectnessOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String

    public init(weightsPath: String, goldenPath: String) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
    }
}

public struct CorrectnessTraceOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let caseName: String?
    public let step: Int
    public let topK: Int

    public init(
        weightsPath: String,
        goldenPath: String,
        caseName: String? = nil,
        step: Int,
        topK: Int = 8
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.caseName = caseName
        self.step = step
        self.topK = topK
    }
}

public struct CorrectnessTraceLogit: Codable, Equatable {
    public let token: Int
    public let logit: Double
}

public struct CorrectnessTraceReport: Codable, Equatable {
    public let caseName: String
    public let step: Int
    public let promptTokenCount: Int
    public let expectedToken: Int
    public let actualToken: Int
    public let matchedPrefixSteps: Int
    public let generatedPrefix: [Int]
    public let actualTokenLogit: Double
    public let expectedTokenLogit: Double
    public let actualExpectedLogitDelta: Double
    public let expectedTokenRank: Int
    public let topLogitMargin: Double?
    public let topLogits: [CorrectnessTraceLogit]
    public let goldenHash: String

    enum CodingKeys: String, CodingKey {
        case caseName = "case_name"
        case step
        case promptTokenCount = "prompt_token_count"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case matchedPrefixSteps = "matched_prefix_steps"
        case generatedPrefix = "generated_prefix"
        case actualTokenLogit = "actual_token_logit"
        case expectedTokenLogit = "expected_token_logit"
        case actualExpectedLogitDelta = "actual_expected_logit_delta"
        case expectedTokenRank = "expected_token_rank"
        case topLogitMargin = "top_logit_margin"
        case topLogits = "top_logits"
        case goldenHash = "golden_hash"
    }
}

public struct CorrectnessReport: Codable, Equatable {
    public let passed: Bool
    public let checkedSteps: Int
    public let caseCount: Int
    public let expertCacheHits: UInt64
    public let expertCacheMisses: UInt64
    public let expertCacheEvictions: UInt64
    public let expertBytesRead: UInt64
    public let expertReadSeconds: Double
    public let expertPeakCachedTensors: UInt64
    public let expertHitRate: Double
    public let firstFailingCase: String?
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?
    public let goldenHash: String
    public let error: String

    enum CodingKeys: String, CodingKey {
        case passed
        case checkedSteps = "checked_steps"
        case caseCount = "case_count"
        case expertCacheHits = "expert_cache_hits"
        case expertCacheMisses = "expert_cache_misses"
        case expertCacheEvictions = "expert_cache_evictions"
        case expertBytesRead = "expert_bytes_read"
        case expertReadSeconds = "expert_read_seconds"
        case expertPeakCachedTensors = "expert_peak_cached_tensors"
        case expertHitRate = "expert_hit_rate"
        case firstFailingCase = "first_failing_case"
        case firstFailingStep = "first_failing_step"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case goldenHash = "golden_hash"
        case error
    }

    public init(
        passed: Bool,
        checkedSteps: Int,
        caseCount: Int,
        expertCacheHits: UInt64 = 0,
        expertCacheMisses: UInt64 = 0,
        expertCacheEvictions: UInt64 = 0,
        expertBytesRead: UInt64 = 0,
        expertReadSeconds: Double = 0,
        expertPeakCachedTensors: UInt64 = 0,
        expertHitRate: Double = 0,
        firstFailingCase: String?,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        goldenHash: String,
        error: String
    ) {
        self.passed = passed
        self.checkedSteps = checkedSteps
        self.caseCount = caseCount
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.expertCacheEvictions = expertCacheEvictions
        self.expertBytesRead = expertBytesRead
        self.expertReadSeconds = expertReadSeconds
        self.expertPeakCachedTensors = expertPeakCachedTensors
        self.expertHitRate = expertHitRate
        self.firstFailingCase = firstFailingCase
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
        self.goldenHash = goldenHash
        self.error = error
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(passed, forKey: .passed)
        try container.encode(checkedSteps, forKey: .checkedSteps)
        try container.encode(caseCount, forKey: .caseCount)
        try container.encode(expertCacheHits, forKey: .expertCacheHits)
        try container.encode(expertCacheMisses, forKey: .expertCacheMisses)
        try container.encode(expertCacheEvictions, forKey: .expertCacheEvictions)
        try container.encode(expertBytesRead, forKey: .expertBytesRead)
        try container.encode(expertReadSeconds, forKey: .expertReadSeconds)
        try container.encode(expertPeakCachedTensors, forKey: .expertPeakCachedTensors)
        try container.encode(expertHitRate, forKey: .expertHitRate)
        if let firstFailingCase {
            try container.encode(firstFailingCase, forKey: .firstFailingCase)
        } else {
            try container.encodeNil(forKey: .firstFailingCase)
        }
        if let firstFailingStep {
            try container.encode(firstFailingStep, forKey: .firstFailingStep)
        } else {
            try container.encodeNil(forKey: .firstFailingStep)
        }
        if let expectedToken {
            try container.encode(expectedToken, forKey: .expectedToken)
        } else {
            try container.encodeNil(forKey: .expectedToken)
        }
        if let actualToken {
            try container.encode(actualToken, forKey: .actualToken)
        } else {
            try container.encodeNil(forKey: .actualToken)
        }
        try container.encode(goldenHash, forKey: .goldenHash)
        try container.encode(error, forKey: .error)
    }

    public var expertStreamingStats: ExpertStreamingStats {
        ExpertStreamingStats(
            cacheHits: expertCacheHits,
            cacheMisses: expertCacheMisses,
            cacheEvictions: expertCacheEvictions,
            bytesRead: expertBytesRead,
            readSeconds: expertReadSeconds,
            peakCachedTensors: expertPeakCachedTensors
        )
    }
}

public struct BenchmarkOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String

    public init(weightsPath: String, goldenPath: String) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
    }
}

public struct GoldenGenerationOptions: Equatable {
    public let weightsPath: String
    public let promptManifest: GoldenPromptManifest
    public let progressIntervalSteps: Int

    public init(
        weightsPath: String,
        promptManifest: GoldenPromptManifest,
        progressIntervalSteps: Int = 0
    ) {
        self.weightsPath = weightsPath
        self.promptManifest = promptManifest
        self.progressIntervalSteps = progressIntervalSteps
    }
}

private struct BenchmarkTokenMismatchError: Error, CustomStringConvertible {
    let label: String
    let step: Int?
    let expectedToken: Int?
    let actualToken: Int?

    init(comparison: BenchmarkTokenComparison) {
        self.label = comparison.label
        self.step = comparison.step
        self.expectedToken = comparison.expectedToken
        self.actualToken = comparison.actualToken
    }

    var description: String {
        var message = "\(label) mismatch"
        if let step {
            message += " at step \(step)"
        }
        message += ": expected \(expectedToken.map { String($0) } ?? "nil"), actual \(actualToken.map { String($0) } ?? "nil")"
        return message
    }
}

public enum DeepSeekRuntime {
    public static func generateGolden(_ options: GoldenGenerationOptions) throws -> GoldenDocument {
        let config = try DeepSeekConfig.load(from: options.weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: options.weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: false)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let progress = makeGoldenProgressReporter(
            intervalSteps: options.progressIntervalSteps,
            startedAt: startedAt
        )
        progress(
            "start cases=\(options.promptManifest.cases.count) correctness_steps=\(MLXFastConstants.correctnessSteps) benchmark_decode_steps=\(MLXFastConstants.benchmarkDecodeSteps)"
        )

        let cases = try options.promptManifest.cases.map { promptCase in
            progress(
                "case \(promptCase.name) start prompt_tokens=\(promptCase.promptTokens.count)"
            )
            return GoldenCase(
                name: promptCase.name,
                promptTokens: promptCase.promptTokens,
                expectedTokens: try generateGreedyCached(
                    promptTokens: promptCase.promptTokens,
                    steps: MLXFastConstants.correctnessSteps,
                    weightCache: weightCache,
                    progressIntervalSteps: options.progressIntervalSteps,
                    progress: { step, total in
                        progress("case \(promptCase.name) generated \(step)/\(total) tokens")
                    }
                )
            )
        }
        progress("benchmark oracle start prompt_tokens=\(options.promptManifest.benchmark.promptTokens.count)")
        let benchmark = try generateBenchmarkGolden(
            promptTokens: options.promptManifest.benchmark.promptTokens,
            weightCache: weightCache,
            progressIntervalSteps: options.progressIntervalSteps,
            progress: { step, total in
                progress("benchmark oracle generated \(step)/\(total) decode tokens")
            }
        )
        progress("complete")
        return GoldenDocument(cases: cases, benchmark: benchmark)
    }

    public static func runCorrectness(_ options: CorrectnessOptions) throws -> CorrectnessReport {
        var loadedGolden: GoldenFixture?
        var loader: DeepSeekWeightLoader?
        do {
            let golden = try loadGoldenFixture(from: options.goldenPath)
            loadedGolden = golden
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            let runtimeLoader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
            )
            loader = runtimeLoader
            let weightCache = DeepSeekRuntimeWeightCache(loader: runtimeLoader, config: config)
            return runCorrectness(
                cases: golden.cases,
                weightCache: weightCache,
                goldenHash: golden.sha256
            )
        } catch {
            return failedCorrectnessReport(
                checkedSteps: 0,
                caseCount: loadedGolden?.cases.count ?? 0,
                goldenHash: loadedGolden?.sha256 ?? "",
                expertStats: expertStats(from: loader),
                error: "\(error)"
            )
        }
    }

    public static func traceCorrectness(_ options: CorrectnessTraceOptions) throws -> CorrectnessTraceReport {
        let golden = try loadGoldenFixture(from: options.goldenPath)
        let selectedCase: GoldenCase
        if let caseName = options.caseName, !caseName.isEmpty {
            guard let match = golden.cases.first(where: { $0.name == caseName }) else {
                throw MLXFastError.invalidInput("correctness golden does not contain case \(caseName)")
            }
            selectedCase = match
        } else {
            guard let first = golden.cases.first else {
                throw MLXFastError.invalidInput("correctness golden contains no cases")
            }
            selectedCase = first
        }

        let config = try DeepSeekConfig.load(from: options.weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: options.weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        return try traceGreedyCached(
            testCase: selectedCase,
            step: options.step,
            topK: options.topK,
            weightCache: weightCache,
            goldenHash: golden.sha256
        )
    }

    public static func benchmark(_ options: BenchmarkOptions) -> ScorePayload {
        let benchmarkStart = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: benchmarkStart)
        var correctnessReport: CorrectnessReport?
        var benchmarkLoader: DeepSeekWeightLoader?
        var transformedWeightsDigest: DirectoryDigest?
        var preflightSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedBenchmarkSeconds = 0.0

        progress(
            "start correctness_steps=\(MLXFastConstants.correctnessSteps) "
                + "benchmark_decode_steps=\(MLXFastConstants.benchmarkDecodeSteps)"
        )

        func makeFailedScore(
            error: String,
            correctness: CorrectnessReport?,
            passedCorrectness: Bool,
            expertStats explicitExpertStats: ExpertStreamingStats? = nil,
            firstFailingCase explicitFirstFailingCase: String? = nil,
            firstFailingStep explicitFirstFailingStep: Int? = nil,
            expectedToken explicitExpectedToken: Int? = nil,
            actualToken explicitActualToken: Int? = nil,
            weightsDigest: DirectoryDigest? = nil
        ) -> ScorePayload {
            progress("failed passed_correctness=\(passedCorrectness) error=\(redactedProgressError(error))")
            return failedScore(
                error: error,
                correctness: correctness,
                passedCorrectness: passedCorrectness,
                expertStats: explicitExpertStats,
                firstFailingCase: explicitFirstFailingCase,
                firstFailingStep: explicitFirstFailingStep,
                expectedToken: explicitExpectedToken,
                actualToken: explicitActualToken,
                weightsDigest: weightsDigest,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: currentResidentMemoryGB()
            )
        }

        do {
            progress("preflight start")
            let preflightStart = DispatchTime.now().uptimeNanoseconds
            _ = try BenchmarkPreflight.check(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            preflightSeconds = secondsSince(preflightStart)
            progress("preflight complete seconds=\(formatSeconds(preflightSeconds))")
            progress("weights digest start")
            transformedWeightsDigest = try directoryDigest(
                rootPath: options.weightsPath,
                ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
            )
            if let transformedWeightsDigest {
                progress(
                    "weights digest complete files=\(transformedWeightsDigest.fileCount) "
                        + "bytes=\(transformedWeightsDigest.byteCount)"
                )
            }
            progress("golden load start")
            let golden = try loadGoldenFixture(from: options.goldenPath)
            progress(
                "golden load complete cases=\(golden.cases.count) "
                    + "benchmark_oracle=\(golden.benchmark == nil ? "missing" : "present")"
            )
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            progress("correctness loader start")
            let correctnessLoader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
            )
            let correctnessCache = DeepSeekRuntimeWeightCache(loader: correctnessLoader, config: config)
            let correctnessStart = DispatchTime.now().uptimeNanoseconds
            progress("correctness start cases=\(golden.cases.count)")
            let correctness = runCorrectness(
                cases: golden.cases,
                weightCache: correctnessCache,
                goldenHash: golden.sha256,
                progress: progress
            )
            correctnessSeconds = secondsSince(correctnessStart)
            correctnessReport = correctness
            progress(
                "correctness complete passed=\(correctness.passed) "
                    + "checked_steps=\(correctness.checkedSteps) "
                    + "seconds=\(formatSeconds(correctnessSeconds))"
            )
            guard correctness.passed else {
                return makeFailedScore(
                    error: correctness.error.isEmpty ? "correctness gate failed" : correctness.error,
                    correctness: correctness,
                    passedCorrectness: false,
                    weightsDigest: transformedWeightsDigest
                )
            }

            let runtimeBenchmarkLoader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
            )
            benchmarkLoader = runtimeBenchmarkLoader
            let benchmarkCache = DeepSeekRuntimeWeightCache(loader: runtimeBenchmarkLoader, config: config)
            guard let benchmarkGolden = golden.benchmark else {
                throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
            }
            let promptPlan = try BenchmarkPrompt.plan(from: benchmarkGolden)
            progress(
                "benchmark oracle ready prefill_tokens=\(promptPlan.prefillTokens.count) "
                    + "decode_seed_tokens=\(promptPlan.decodeSeedTokens.count) "
                    + "decode_tokens=\(promptPlan.expectedDecodeTokens.count)"
            )
            let idleGBPerSecond = measureMactopIdleGBPerSecond(progress: progress)

            Memory.peakMemory = 0
            let timedBenchmarkStart = DispatchTime.now().uptimeNanoseconds
            progress("timed benchmark start")
            let prefillSecondsPerToken = try measurePrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                expectedToken: promptPlan.expectedPrefillToken,
                weightCache: benchmarkCache,
                progress: progress
            )
            let decode = try measureDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                expectedTokens: promptPlan.expectedDecodeTokens,
                weightCache: benchmarkCache,
                idleGBPerSecond: idleGBPerSecond,
                progress: progress
            )
            timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
            let peakRamGB = Double(Memory.peakMemory) / Double(1 << 30)
            let score = peakRamGB
                * decode.bandwidthGBPerToken
                * decode.secondsPerToken
                * prefillSecondsPerToken
            let expertStats = expertStats(from: runtimeBenchmarkLoader)

            guard score.isFinite, score >= 0 else {
                return makeFailedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: true,
                    expertStats: expertStats,
                    weightsDigest: transformedWeightsDigest
                )
            }
            progress(
                "complete score=\(formatDouble(score)) "
                    + "wall_seconds=\(formatSeconds(secondsSince(benchmarkStart))) "
                    + "timed_seconds=\(formatSeconds(timedBenchmarkSeconds))"
            )

            return ScorePayload(
                score: score,
                passed: true,
                metrics: ScoreMetrics(
                    peakRamGB: peakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
                    benchmarkWallSeconds: secondsSince(benchmarkStart),
                    preflightSeconds: preflightSeconds,
                    correctnessSeconds: correctnessSeconds,
                    timedBenchmarkSeconds: timedBenchmarkSeconds,
                    processResidentMemoryGB: currentResidentMemoryGB(),
                    passedCorrectness: true,
                    numLayers: config.numHiddenLayers,
                    checkedSteps: correctness.checkedSteps,
                    caseCount: correctness.caseCount,
                    expertCacheHits: expertStats.cacheHits,
                    expertCacheMisses: expertStats.cacheMisses,
                    expertCacheEvictions: expertStats.cacheEvictions,
                    expertBytesRead: expertStats.bytesRead,
                    expertReadSeconds: expertStats.readSeconds,
                    expertPeakCachedTensors: expertStats.peakCachedTensors,
                    expertHitRate: expertStats.hitRate,
                    firstFailingLayer: nil,
                    firstFailingCase: nil,
                    firstFailingStep: nil,
                    expectedToken: nil,
                    actualToken: nil,
                    maxAbsDiff: 0,
                    goldenHash: correctness.goldenHash,
                    bandwidthSource: decode.bandwidthSource,
                    error: "",
                    commit: commitIdentifier(),
                    timestamp: ISO8601DateFormatter().string(from: Date()),
                    harnessHash: harnessHash(),
                    weightsHash: transformedWeightsDigest?.sha256 ?? "",
                    weightsByteCount: transformedWeightsDigest?.byteCount ?? 0,
                    weightsFileCount: transformedWeightsDigest?.fileCount ?? 0,
                    runtime: "swift"
                )
            )
        } catch let mismatch as BenchmarkTokenMismatchError {
            return makeFailedScore(
                error: mismatch.description,
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader),
                firstFailingCase: "benchmark",
                firstFailingStep: mismatch.step,
                expectedToken: mismatch.expectedToken,
                actualToken: mismatch.actualToken,
                weightsDigest: transformedWeightsDigest
            )
        } catch {
            return makeFailedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader),
                weightsDigest: transformedWeightsDigest
            )
        }
    }

    private static func runCorrectness(
        cases: [GoldenCase],
        weightCache: DeepSeekRuntimeWeightCache,
        goldenHash: String,
        progress: ((String) -> Void)? = nil
    ) -> CorrectnessReport {
        var checkedSteps = 0
        var currentCase: GoldenCase?
        do {
            for (caseIndex, testCase) in cases.enumerated() {
                currentCase = testCase
                let caseLabel = "\(caseIndex + 1)/\(cases.count)"
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let comparison = try compareGreedyCached(
                    testCase: testCase,
                    weightCache: weightCache,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness case \(caseLabel) generated \(step)/\(total) tokens")
                    }
                )
                if !comparison.passed {
                    progress?("correctness case \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    let expertStats = expertStats(from: weightCache)
                    return CorrectnessReport(
                        passed: false,
                        checkedSteps: checkedSteps + comparison.checkedSteps,
                        caseCount: cases.count,
                        expertCacheHits: expertStats.cacheHits,
                        expertCacheMisses: expertStats.cacheMisses,
                        expertCacheEvictions: expertStats.cacheEvictions,
                        expertBytesRead: expertStats.bytesRead,
                        expertReadSeconds: expertStats.readSeconds,
                        expertPeakCachedTensors: expertStats.peakCachedTensors,
                        expertHitRate: expertStats.hitRate,
                        firstFailingCase: testCase.name,
                        firstFailingStep: comparison.firstFailingStep,
                        expectedToken: comparison.expectedToken,
                        actualToken: comparison.actualToken,
                        goldenHash: goldenHash,
                        error: "generated token mismatch"
                    )
                }
                progress?("correctness case \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
                checkedSteps += comparison.checkedSteps
            }
        } catch {
            progress?("correctness error=\(redactedProgressError("\(error)"))")
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: cases.count,
                firstFailingCase: currentCase?.name,
                goldenHash: goldenHash,
                expertStats: expertStats(from: weightCache),
                error: "\(error)"
            )
        }

        let expertStats = expertStats(from: weightCache)
        return CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: cases.count,
            expertCacheHits: expertStats.cacheHits,
            expertCacheMisses: expertStats.cacheMisses,
            expertCacheEvictions: expertStats.cacheEvictions,
            expertBytesRead: expertStats.bytesRead,
            expertReadSeconds: expertStats.readSeconds,
            expertPeakCachedTensors: expertStats.peakCachedTensors,
            expertHitRate: expertStats.hitRate,
            firstFailingCase: nil,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: goldenHash,
            error: ""
        )
    }

    private struct DecodeMeasurement {
        let secondsPerToken: Double
        let bandwidthGBPerToken: Double
        let bandwidthSource: String
    }

    private static func measureMactopIdleGBPerSecond(progress: ((String) -> Void)? = nil) -> Double? {
        do {
            progress?("mactop idle measurement start")
            let idleSamples = try MactopSession.measureIdleSamples()
            guard !idleSamples.isEmpty else {
                throw MLXFastError.invalidInput("mactop idle measurement produced no samples")
            }
            let idleGBPerSecond = idleSamples.reduce(0, +) / Double(idleSamples.count)
            progress?(
                "mactop idle measurement complete samples=\(idleSamples.count) "
                    + "idle_gb_per_second=\(formatDouble(idleGBPerSecond))"
            )
            return idleGBPerSecond
        } catch {
            progress?(
                "mactop idle measurement unavailable; using expert streaming byte fallback "
                    + "error=\(redactedProgressError("\(error)"))"
            )
            return nil
        }
    }

    private static func measurePrefillSecondsPerToken(
        promptTokens: [Int],
        expectedToken: Int,
        weightCache: DeepSeekRuntimeWeightCache,
        progress: ((String) -> Void)? = nil
    ) throws -> Double {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill prompt must not be empty")
        }

        let totalRuns = MLXFastConstants.benchmarkPrefillWarmupRuns
            + MLXFastConstants.benchmarkPrefillTimedRuns
        var timedElapsed: [Double] = []
        timedElapsed.reserveCapacity(MLXFastConstants.benchmarkPrefillTimedRuns)

        for runIndex in 0..<totalRuns {
            let runLabel = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns ? "warmup" : "timed"
            let runOrdinal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? runIndex + 1
                : runIndex - MLXFastConstants.benchmarkPrefillWarmupRuns + 1
            let runTotal = runIndex < MLXFastConstants.benchmarkPrefillWarmupRuns
                ? MLXFastConstants.benchmarkPrefillWarmupRuns
                : MLXFastConstants.benchmarkPrefillTimedRuns
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) start "
                    + "prompt_tokens=\(promptTokens.count)"
            )
            let cache = DeepSeekModelCache(config: weightCache.config)
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray(promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            eval(logits)
            let token = try DeepSeekCorrectness.greedyToken(from: logits)
            try requireBenchmarkMatch(
                BenchmarkOutputValidator.comparePrefillToken(
                    expectedToken: expectedToken,
                    actualToken: token
                )
            )
            let elapsed = secondsSince(start)
            Memory.clearCache()
            progress?(
                "prefill \(runLabel) \(runOrdinal)/\(runTotal) complete "
                    + "seconds=\(formatSeconds(elapsed))"
            )

            if runIndex >= MLXFastConstants.benchmarkPrefillWarmupRuns {
                timedElapsed.append(elapsed)
            }
        }

        guard !timedElapsed.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill needs at least one timed run")
        }
        let meanElapsed = timedElapsed.reduce(0, +) / Double(timedElapsed.count)
        let secondsPerToken = meanElapsed / Double(promptTokens.count)
        progress?("prefill complete seconds_per_token=\(formatDouble(secondsPerToken))")
        return secondsPerToken
    }

    private static func measureDecode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        weightCache: DeepSeekRuntimeWeightCache,
        idleGBPerSecond: Double?,
        progress: ((String) -> Void)? = nil
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        let timingPlan = try DecodeTimingPlan(
            seedTokenCount: seedTokens.count,
            decodeSteps: MLXFastConstants.benchmarkDecodeSteps
        )

        progress?("decode warmup start seed_tokens=\(seedTokens.count)")
        let warmupCache = DeepSeekModelCache(config: weightCache.config)
        let warmupLogits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: warmupCache,
            positionOffset: 0
        )
        _ = try DeepSeekCorrectness.greedyToken(from: warmupLogits)
        Memory.clearCache()
        progress?("decode warmup complete")

        progress?("decode seed prefill start seed_tokens=\(seedTokens.count)")
        let cache = DeepSeekModelCache(config: weightCache.config)
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: expectedSeedToken,
                actualToken: token
            )
        )
        cache.materializeCachedState()
        progress?("decode seed prefill complete")

        var actualTokens: [Int] = []
        actualTokens.reserveCapacity(timingPlan.decodeSteps)
        let metricsBeforeDecode = weightCache.loader.expertStreamingMetrics?.snapshot()
        let session: MactopSession?
        if idleGBPerSecond != nil {
            do {
                session = try MactopSession.start()
                progress?("decode mactop measurement start")
            } catch {
                progress?(
                    "decode mactop measurement unavailable; using expert streaming byte fallback "
                        + "error=\(redactedProgressError("\(error)"))"
                )
                session = nil
            }
        } else {
            session = nil
        }
        let start = DispatchTime.now().uptimeNanoseconds
        do {
            progress?("decode measured start tokens=\(timingPlan.decodeSteps)")
            for decodedStep in 0..<timingPlan.decodeSteps {
                logits = try DeepSeekModel.logits(
                    inputIDs: inputIDsArray([token]),
                    weightCache: weightCache,
                    cache: cache,
                    positionOffset: try timingPlan.positionOffset(forDecodedStep: decodedStep)
                )
                token = try DeepSeekCorrectness.greedyToken(from: logits)
                actualTokens.append(token)
                let expectedToken = expectedTokens[decodedStep]
                if token != expectedToken {
                    throw BenchmarkTokenMismatchError(
                        comparison: BenchmarkTokenComparison(
                            passed: false,
                            label: "benchmark decode token",
                            step: decodedStep,
                            expectedToken: expectedToken,
                            actualToken: token
                        )
                    )
                }
                reportProgress(
                    step: decodedStep + 1,
                    total: timingPlan.decodeSteps,
                    intervalSteps: 64,
                    progress: { step, total in
                        progress?("decode measured generated \(step)/\(total) tokens")
                    }
                )
            }

            try requireBenchmarkMatch(
                BenchmarkOutputValidator.compareDecodeTokens(
                    expectedTokens: expectedTokens,
                    actualTokens: actualTokens
                )
            )

            let elapsed = secondsSince(start)
            let bandwidth: (gbPerToken: Double, source: String)
            if let session, let idleGBPerSecond {
                do {
                    let samples = try session.stop()
                    bandwidth = (
                        try MactopBandwidth.gigabytesPerToken(
                            samples: samples,
                            idleGBPerSecond: idleGBPerSecond,
                            decodeElapsedSeconds: elapsed,
                            decodedTokens: timingPlan.decodeSteps
                        ),
                        "mactop_hardware"
                    )
                } catch {
                    progress?(
                        "decode mactop samples unavailable; using expert streaming byte fallback "
                            + "error=\(redactedProgressError("\(error)"))"
                    )
                    bandwidth = try expertStreamingBandwidthGBPerToken(
                        before: metricsBeforeDecode,
                        after: weightCache.loader.expertStreamingMetrics?.snapshot(),
                        decodedTokens: timingPlan.decodeSteps
                    )
                }
            } else {
                bandwidth = try expertStreamingBandwidthGBPerToken(
                    before: metricsBeforeDecode,
                    after: weightCache.loader.expertStreamingMetrics?.snapshot(),
                    decodedTokens: timingPlan.decodeSteps
                )
            }
            progress?(
                "decode measured complete seconds=\(formatSeconds(elapsed)) "
                    + "seconds_per_token=\(formatDouble(elapsed / Double(timingPlan.decodeSteps))) "
                    + "bandwidth_gb_per_token=\(formatDouble(bandwidth.gbPerToken)) "
                    + "bandwidth_source=\(bandwidth.source)"
            )
            return DecodeMeasurement(
                secondsPerToken: elapsed / Double(timingPlan.decodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source
            )
        } catch {
            _ = try? session?.stop()
            throw error
        }
    }

    private static func expertStreamingBandwidthGBPerToken(
        before: ExpertStreamingMetrics.Snapshot?,
        after: ExpertStreamingMetrics.Snapshot?,
        decodedTokens: Int
    ) throws -> (gbPerToken: Double, source: String) {
        guard decodedTokens > 0 else {
            throw MLXFastError.invalidInput("benchmark decode steps must be positive")
        }
        guard let after else {
            throw MLXFastError.invalidInput("expert streaming metrics unavailable for bandwidth fallback")
        }
        let beforeBytes = before?.bytesRead ?? 0
        let bytesRead = after.bytesRead >= beforeBytes ? after.bytesRead - beforeBytes : after.bytesRead
        guard bytesRead > 0 else {
            throw MLXFastError.invalidInput("expert streaming bandwidth fallback observed no decoded expert reads")
        }
        return (
            Double(bytesRead) / Double(1 << 30) / Double(decodedTokens),
            "expert_streaming_reads"
        )
    }

    private static func expertStats(from weightCache: DeepSeekRuntimeWeightCache) -> ExpertStreamingStats {
        expertStats(from: weightCache.loader)
    }

    private static func expertStats(from loader: DeepSeekWeightLoader?) -> ExpertStreamingStats {
        loader?.expertStreamingMetrics?.snapshot().stats ?? .zero
    }

    private static func failedScore(
        error: String,
        correctness: CorrectnessReport?,
        passedCorrectness: Bool,
        expertStats explicitExpertStats: ExpertStreamingStats? = nil,
        firstFailingCase explicitFirstFailingCase: String? = nil,
        firstFailingStep explicitFirstFailingStep: Int? = nil,
        expectedToken explicitExpectedToken: Int? = nil,
        actualToken explicitActualToken: Int? = nil,
        weightsDigest: DirectoryDigest? = nil,
        benchmarkWallSeconds: Double = 0,
        preflightSeconds: Double = 0,
        correctnessSeconds: Double = 0,
        timedBenchmarkSeconds: Double = 0,
        processResidentMemoryGB: Double = 0
    ) -> ScorePayload {
        let expertStats = explicitExpertStats ?? correctness?.expertStreamingStats ?? .zero
        return ScorePayload(
            score: nil,
            passed: false,
            metrics: ScoreMetrics(
                peakRamGB: 0,
                bandwidthGBPerToken: 0,
                decodeSecondsPerToken: 0,
                prefillSecondsPerToken: 0,
                benchmarkWallSeconds: benchmarkWallSeconds,
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: processResidentMemoryGB,
                passedCorrectness: passedCorrectness,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: correctness?.checkedSteps ?? 0,
                caseCount: correctness?.caseCount ?? 0,
                expertCacheHits: expertStats.cacheHits,
                expertCacheMisses: expertStats.cacheMisses,
                expertCacheEvictions: expertStats.cacheEvictions,
                expertBytesRead: expertStats.bytesRead,
                expertReadSeconds: expertStats.readSeconds,
                expertPeakCachedTensors: expertStats.peakCachedTensors,
                expertHitRate: expertStats.hitRate,
                firstFailingLayer: nil,
                firstFailingCase: explicitFirstFailingCase ?? correctness?.firstFailingCase,
                firstFailingStep: explicitFirstFailingStep ?? correctness?.firstFailingStep,
                expectedToken: explicitExpectedToken ?? correctness?.expectedToken,
                actualToken: explicitActualToken ?? correctness?.actualToken,
                maxAbsDiff: 0,
                goldenHash: correctness?.goldenHash ?? "",
                bandwidthSource: "",
                error: error,
                commit: commitIdentifier(),
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash(),
                weightsHash: weightsDigest?.sha256 ?? "",
                weightsByteCount: weightsDigest?.byteCount ?? 0,
                weightsFileCount: weightsDigest?.fileCount ?? 0,
                runtime: "swift"
            )
        )
    }

    private static func currentResidentMemoryGB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        guard result == KERN_SUCCESS else {
            return 0
        }
        return Double(info.resident_size) / Double(1 << 30)
    }

    private static func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    }

    private static func makeGoldenProgressReporter(
        intervalSteps: Int,
        startedAt: UInt64
    ) -> (String) -> Void {
        guard intervalSteps > 0 else {
            return { _ in }
        }
        return { message in
            let elapsed = formatSeconds(secondsSince(startedAt))
            fputs("mlxfast: make-golden elapsed=\(elapsed)s \(message)\n", stderr)
            fflush(stderr)
        }
    }

    private static func makeBenchmarkProgressReporter(startedAt: UInt64) -> (String) -> Void {
        { message in
            let elapsed = formatSeconds(secondsSince(startedAt))
            fputs("mlxfast: benchmark elapsed=\(elapsed)s \(message)\n", stderr)
            fflush(stderr)
        }
    }

    private static func formatSeconds(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private static func formatDouble(_ value: Double) -> String {
        String(format: "%.6f", value)
    }

    private static func singleLine(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
    }

    private static func redactedProgressError(_ value: String) -> String {
        let line = singleLine(value)
        if line.range(of: "expected", options: .caseInsensitive) != nil
            || line.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return line
    }

    private static func reportProgress(
        step: Int,
        total: Int,
        intervalSteps: Int,
        progress: ((Int, Int) -> Void)?
    ) {
        guard let progress, intervalSteps > 0 else {
            return
        }
        if step == 1 || step == total || step.isMultiple(of: intervalSteps) {
            progress(step, total)
        }
    }

    private static func commitIdentifier() -> String {
        (try? runProcess("/usr/bin/git", arguments: ["rev-parse", "--short", "HEAD"])) ?? ""
    }

    private static func harnessHash() -> String {
        let roots = [
            "Package.swift",
            "Sources",
            "Tests",
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools",
            "README.md",
            "CHALLENGE.md",
        ]
        var files: [String] = []
        for root in roots {
            let url = URL(fileURLWithPath: root)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory) else {
                continue
            }
            if isDirectory.boolValue {
                guard let enumerator = FileManager.default.enumerator(
                    at: url,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                ) else {
                    continue
                }
                for case let fileURL as URL in enumerator {
                    let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    if values?.isRegularFile == true {
                        files.append(fileURL.path)
                    }
                }
            } else {
                files.append(url.path)
            }
        }

        var hasher = SHA256()
        for path in files.sorted() {
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                continue
            }
            hasher.update(data: Data(path.utf8))
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private struct DirectoryDigest: Equatable {
        let fileCount: Int
        let byteCount: Int
        let sha256: String
    }

    private static func directoryDigest(
        rootPath: String,
        ignoredRelativePaths: Set<String>
    ) throws -> DirectoryDigest {
        let root = URL(fileURLWithPath: rootPath).standardizedFileURL
        let rootPrefix = root.path + "/"
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("directory not found at \(root.path)")
        }

        var files: [(relativePath: String, url: URL)] = []
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let path = standardized.path
            guard path.hasPrefix(rootPrefix) else {
                throw MLXFastError.invalidInput("path escaped digest root: \(path)")
            }
            let relativePath = String(path.dropFirst(rootPrefix.count))
            if ignoredRelativePaths.contains(relativePath) {
                continue
            }

            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("directory digest rejects symlink \(relativePath)")
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput("directory digest rejects non-regular file \(relativePath)")
            }
            files.append((relativePath: relativePath, url: standardized))
        }

        var treeHasher = SHA256()
        var byteCount = 0
        for file in files.sorted(by: { $0.relativePath < $1.relativePath }) {
            let size = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: file.url.path),
                path: file.url.path
            )
            guard byteCount <= Int.max - size else {
                throw MLXFastError.invalidInput("directory digest byte count exceeds Int range")
            }
            byteCount += size
            let digest = try fileDigest(file.url)
            treeHasher.update(data: Data(file.relativePath.utf8))
            treeHasher.update(data: Data([0]))
            treeHasher.update(data: Data(digest))
            treeHasher.update(data: Data([0]))
        }

        return DirectoryDigest(
            fileCount: files.count,
            byteCount: byteCount,
            sha256: treeHasher.finalize().map { String(format: "%02x", $0) }.joined()
        )
    }

    private static func fileDigest(_ url: URL) throws -> SHA256.Digest {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        let chunkSize = 8 * 1024 * 1024
        while true {
            let data = handle.readData(ofLength: chunkSize)
            if data.isEmpty {
                return hasher.finalize()
            }
            hasher.update(data: data)
        }
    }

    private static func runProcess(_ executable: String, arguments: [String]) throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return ""
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private static func compareGreedyCached(
        testCase: GoldenCase,
        weightCache: DeepSeekRuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> CorrectnessTokenComparison {
        let steps = MLXFastConstants.correctnessSteps
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard testCase.expectedTokens.count == steps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need exactly \(steps)"
            )
        }

        let config = weightCache.config
        let cache = DeepSeekModelCache(config: config)

        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        var generated: [Int] = []
        generated.reserveCapacity(steps)

        for step in 0..<steps {
            generated.append(token)
            let comparison = DeepSeekCorrectness.compareTokens(
                expected: testCase.expectedTokens,
                actual: generated,
                steps: step + 1
            )
            if !comparison.passed {
                return comparison
            }
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )

            if step == steps - 1 {
                break
            }

            let positionOffset = testCase.promptTokens.count + step
            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([token]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: positionOffset
            )
            token = try DeepSeekCorrectness.greedyToken(from: logits)
        }

        return DeepSeekCorrectness.compareTokens(
            expected: testCase.expectedTokens,
            actual: generated,
            steps: steps
        )
    }

    private static func traceGreedyCached(
        testCase: GoldenCase,
        step: Int,
        topK: Int,
        weightCache: DeepSeekRuntimeWeightCache,
        goldenHash: String
    ) throws -> CorrectnessTraceReport {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard step >= 0, step < testCase.expectedTokens.count else {
            throw MLXFastError.invalidInput(
                "trace step \(step) is outside expected token range 0..<\(testCase.expectedTokens.count)"
            )
        }
        guard topK > 0 else {
            throw MLXFastError.invalidInput("trace topK must be positive")
        }

        let cache = DeepSeekModelCache(config: weightCache.config)
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        var generated: [Int] = []
        generated.reserveCapacity(step + 1)

        for currentStep in 0...step {
            generated.append(token)
            if currentStep == step {
                return try traceReport(
                    logits: logits,
                    testCase: testCase,
                    step: step,
                    topK: topK,
                    generated: generated,
                    goldenHash: goldenHash
                )
            }

            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([token]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: testCase.promptTokens.count + currentStep
            )
            token = try DeepSeekCorrectness.greedyToken(from: logits)
        }

        throw MLXFastError.invalidInput("trace failed to reach step \(step)")
    }

    private static func traceReport(
        logits: MLXArray,
        testCase: GoldenCase,
        step: Int,
        topK: Int,
        generated: [Int],
        goldenHash: String
    ) throws -> CorrectnessTraceReport {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("trace logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        let last = rows[-1]
        eval(last)
        let values = last.asArray(Float.self).map(Double.init)
        guard values.count == vocabSize else {
            throw MLXFastError.invalidInput(
                "trace logits materialized \(values.count) values, expected \(vocabSize)"
            )
        }

        let expectedToken = testCase.expectedTokens[step]
        let actualToken = generated[step]
        guard expectedToken >= 0, expectedToken < values.count else {
            throw MLXFastError.invalidInput("expected token \(expectedToken) is outside vocab size \(values.count)")
        }
        guard actualToken >= 0, actualToken < values.count else {
            throw MLXFastError.invalidInput("actual token \(actualToken) is outside vocab size \(values.count)")
        }

        let sortedIndices = values.indices.sorted {
            let lhs = values[$0]
            let rhs = values[$1]
            return lhs == rhs ? $0 < $1 : lhs > rhs
        }
        let requestedTopK = min(topK, sortedIndices.count)
        let topLogits = sortedIndices.prefix(requestedTopK).map {
            CorrectnessTraceLogit(token: $0, logit: values[$0])
        }
        let expectedRank = (sortedIndices.firstIndex(of: expectedToken) ?? sortedIndices.count - 1) + 1
        let topMargin: Double?
        if sortedIndices.count >= 2 {
            topMargin = values[sortedIndices[0]] - values[sortedIndices[1]]
        } else {
            topMargin = nil
        }
        let matchedPrefixSteps = zip(generated, testCase.expectedTokens)
            .prefix { pair in pair.0 == pair.1 }
            .count

        return CorrectnessTraceReport(
            caseName: testCase.name,
            step: step,
            promptTokenCount: testCase.promptTokens.count,
            expectedToken: expectedToken,
            actualToken: actualToken,
            matchedPrefixSteps: matchedPrefixSteps,
            generatedPrefix: generated,
            actualTokenLogit: values[actualToken],
            expectedTokenLogit: values[expectedToken],
            actualExpectedLogitDelta: values[actualToken] - values[expectedToken],
            expectedTokenRank: expectedRank,
            topLogitMargin: topMargin,
            topLogits: topLogits,
            goldenHash: goldenHash
        )
    }

    private static func generateGreedyCached(
        promptTokens: [Int],
        steps: Int,
        weightCache: DeepSeekRuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard steps >= 0 else {
            throw MLXFastError.invalidInput("greedy correctness steps must be non-negative")
        }

        let cache = DeepSeekModelCache(config: weightCache.config)
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        var generated: [Int] = []
        generated.reserveCapacity(steps)

        for step in 0..<steps {
            generated.append(token)
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )
            if step == steps - 1 {
                break
            }
            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([token]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: promptTokens.count + step
            )
            token = try DeepSeekCorrectness.greedyToken(from: logits)
        }
        return generated
    }

    private static func generateBenchmarkGolden(
        promptTokens: [Int],
        weightCache: DeepSeekRuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> BenchmarkGolden {
        guard promptTokens.count >= MLXFastConstants.benchmarkPrefillPromptTokens else {
            throw MLXFastError.invalidInput(
                "benchmark.prompt_tokens has \(promptTokens.count) tokens; need at least \(MLXFastConstants.benchmarkPrefillPromptTokens)"
            )
        }
        let prefillTokens = Array(promptTokens.prefix(MLXFastConstants.benchmarkPrefillPromptTokens))
        let expectedPrefillToken = try firstGreedyToken(
            promptTokens: prefillTokens,
            weightCache: weightCache
        )
        let seedTokens = Array(promptTokens.prefix(MLXFastConstants.benchmarkDecodeSeedTokens))
        let seedCache = DeepSeekModelCache(config: weightCache.config)
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: seedCache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        let expectedSeedToken = token

        var decodeTokens: [Int] = []
        decodeTokens.reserveCapacity(MLXFastConstants.benchmarkDecodeSteps)
        let timingPlan = try DecodeTimingPlan(
            seedTokenCount: seedTokens.count,
            decodeSteps: MLXFastConstants.benchmarkDecodeSteps
        )
        for decodedStep in 0..<timingPlan.decodeSteps {
            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([token]),
                weightCache: weightCache,
                cache: seedCache,
                positionOffset: try timingPlan.positionOffset(forDecodedStep: decodedStep)
            )
            token = try DeepSeekCorrectness.greedyToken(from: logits)
            decodeTokens.append(token)
            reportProgress(
                step: decodedStep + 1,
                total: timingPlan.decodeSteps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )
        }

        return BenchmarkGolden(
            prefillPromptTokens: prefillTokens,
            expectedPrefillToken: expectedPrefillToken,
            decodeSeedTokens: seedTokens,
            expectedDecodeSeedToken: expectedSeedToken,
            expectedDecodeTokens: decodeTokens
        )
    }

    private static func firstGreedyToken(
        promptTokens: [Int],
        weightCache: DeepSeekRuntimeWeightCache
    ) throws -> Int {
        let cache = DeepSeekModelCache(config: weightCache.config)
        let logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        return try DeepSeekCorrectness.greedyToken(from: logits)
    }

    private static func requireBenchmarkMatch(_ comparison: BenchmarkTokenComparison) throws {
        guard comparison.passed else {
            throw BenchmarkTokenMismatchError(comparison: comparison)
        }
    }

    private static func inputIDsArray(_ tokens: [Int]) throws -> MLXArray {
        guard !tokens.isEmpty else {
            throw MLXFastError.invalidInput("input token array must not be empty")
        }
        let values = try tokens.enumerated().map { index, token -> Int32 in
            guard token >= 0, token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput(
                    "input token[\(index)]=\(token) is outside DeepSeek vocab range 0..<\(MLXFastConstants.vocabSize)"
                )
            }
            return Int32(token)
        }
        return MLXArray(values, [1, values.count])
    }

    private static func failedCorrectnessReport(
        checkedSteps: Int,
        caseCount: Int = 0,
        firstFailingCase: String? = nil,
        goldenHash: String = "",
        expertStats: ExpertStreamingStats = .zero,
        error: String
    ) -> CorrectnessReport {
        CorrectnessReport(
            passed: false,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: expertStats.cacheHits,
            expertCacheMisses: expertStats.cacheMisses,
            expertCacheEvictions: expertStats.cacheEvictions,
            expertBytesRead: expertStats.bytesRead,
            expertReadSeconds: expertStats.readSeconds,
            expertPeakCachedTensors: expertStats.peakCachedTensors,
            expertHitRate: expertStats.hitRate,
            firstFailingCase: firstFailingCase,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: goldenHash,
            error: error
        )
    }
}

struct DecodeTimingPlan: Equatable {
    let seedTokenCount: Int
    let decodeSteps: Int

    init(seedTokenCount: Int, decodeSteps: Int) throws {
        guard seedTokenCount > 0 else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard decodeSteps > 0 else {
            throw MLXFastError.invalidInput("benchmark decode steps must be positive")
        }
        self.seedTokenCount = seedTokenCount
        self.decodeSteps = decodeSteps
    }

    func positionOffset(forDecodedStep step: Int) throws -> Int {
        guard step >= 0 && step < decodeSteps else {
            throw MLXFastError.invalidInput(
                "decode step \(step) is outside benchmark range 0..<\(decodeSteps)"
            )
        }
        return seedTokenCount + step
    }
}
