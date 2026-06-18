import Foundation
import CryptoKit
import MLX
import MLXFastCore

public struct CorrectnessOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String

    public init(weightsPath: String, goldenPath: String) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
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

    public init(weightsPath: String, promptManifest: GoldenPromptManifest) {
        self.weightsPath = weightsPath
        self.promptManifest = promptManifest
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

        let cases = try options.promptManifest.cases.map { promptCase in
            GoldenCase(
                name: promptCase.name,
                promptTokens: promptCase.promptTokens,
                expectedTokens: try generateGreedyCached(
                    promptTokens: promptCase.promptTokens,
                    steps: MLXFastConstants.correctnessSteps,
                    weightCache: weightCache
                )
            )
        }
        let benchmark = try generateBenchmarkGolden(
            promptTokens: options.promptManifest.benchmark.promptTokens,
            weightCache: weightCache
        )
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

    public static func benchmark(_ options: BenchmarkOptions) -> ScorePayload {
        var correctnessReport: CorrectnessReport?
        var benchmarkLoader: DeepSeekWeightLoader?
        do {
            _ = try BenchmarkPreflight.check(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            let golden = try loadGoldenFixture(from: options.goldenPath)
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            let correctnessLoader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
            )
            let correctnessCache = DeepSeekRuntimeWeightCache(loader: correctnessLoader, config: config)
            let correctness = runCorrectness(
                cases: golden.cases,
                weightCache: correctnessCache,
                goldenHash: golden.sha256
            )
            correctnessReport = correctness
            guard correctness.passed else {
                return failedScore(
                    error: correctness.error.isEmpty ? "correctness gate failed" : correctness.error,
                    correctness: correctness,
                    passedCorrectness: false
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
            let idleSamples = try MactopSession.measureIdleSamples()
            guard !idleSamples.isEmpty else {
                throw MLXFastError.invalidInput("mactop idle measurement produced no samples")
            }
            let idleGBPerSecond = idleSamples.reduce(0, +) / Double(idleSamples.count)

            Memory.peakMemory = 0
            let prefillSecondsPerToken = try measurePrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                expectedToken: promptPlan.expectedPrefillToken,
                weightCache: benchmarkCache
            )
            let decode = try measureDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                expectedTokens: promptPlan.expectedDecodeTokens,
                weightCache: benchmarkCache,
                idleGBPerSecond: idleGBPerSecond
            )
            let peakRamGB = Double(Memory.peakMemory) / Double(1 << 30)
            let score = peakRamGB
                * decode.bandwidthGBPerToken
                * decode.secondsPerToken
                * prefillSecondsPerToken
            let expertStats = expertStats(from: runtimeBenchmarkLoader)

            guard score.isFinite, score >= 0 else {
                return failedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: true,
                    expertStats: expertStats
                )
            }

            return ScorePayload(
                score: score,
                passed: true,
                metrics: ScoreMetrics(
                    peakRamGB: peakRamGB,
                    bandwidthGBPerToken: decode.bandwidthGBPerToken,
                    decodeSecondsPerToken: decode.secondsPerToken,
                    prefillSecondsPerToken: prefillSecondsPerToken,
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
                    bandwidthSource: "mactop_hardware",
                    error: "",
                    commit: commitIdentifier(),
                    timestamp: ISO8601DateFormatter().string(from: Date()),
                    harnessHash: harnessHash(),
                    runtime: "swift"
                )
            )
        } catch let mismatch as BenchmarkTokenMismatchError {
            return failedScore(
                error: mismatch.description,
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader),
                firstFailingCase: "benchmark",
                firstFailingStep: mismatch.step,
                expectedToken: mismatch.expectedToken,
                actualToken: mismatch.actualToken
            )
        } catch {
            return failedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                expertStats: expertStats(from: benchmarkLoader)
            )
        }
    }

    private static func runCorrectness(
        cases: [GoldenCase],
        weightCache: DeepSeekRuntimeWeightCache,
        goldenHash: String
    ) -> CorrectnessReport {
        var checkedSteps = 0
        var currentCase: GoldenCase?
        do {
            for testCase in cases {
                currentCase = testCase
                let comparison = try compareGreedyCached(testCase: testCase, weightCache: weightCache)
                if !comparison.passed {
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
                checkedSteps += comparison.checkedSteps
            }
        } catch {
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
    }

    private static func measurePrefillSecondsPerToken(
        promptTokens: [Int],
        expectedToken: Int,
        weightCache: DeepSeekRuntimeWeightCache
    ) throws -> Double {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill prompt must not be empty")
        }

        let totalRuns = MLXFastConstants.benchmarkPrefillWarmupRuns
            + MLXFastConstants.benchmarkPrefillTimedRuns
        var timedElapsed: [Double] = []
        timedElapsed.reserveCapacity(MLXFastConstants.benchmarkPrefillTimedRuns)

        for runIndex in 0..<totalRuns {
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

            if runIndex >= MLXFastConstants.benchmarkPrefillWarmupRuns {
                timedElapsed.append(elapsed)
            }
        }

        guard !timedElapsed.isEmpty else {
            throw MLXFastError.invalidInput("benchmark prefill needs at least one timed run")
        }
        let meanElapsed = timedElapsed.reduce(0, +) / Double(timedElapsed.count)
        return meanElapsed / Double(promptTokens.count)
    }

    private static func measureDecode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        weightCache: DeepSeekRuntimeWeightCache,
        idleGBPerSecond: Double
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        let timingPlan = try DecodeTimingPlan(
            seedTokenCount: seedTokens.count,
            decodeSteps: MLXFastConstants.benchmarkDecodeSteps
        )

        let warmupCache = DeepSeekModelCache(config: weightCache.config)
        let warmupLogits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: warmupCache,
            positionOffset: 0
        )
        _ = try DeepSeekCorrectness.greedyToken(from: warmupLogits)
        Memory.clearCache()

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

        var actualTokens: [Int] = []
        actualTokens.reserveCapacity(timingPlan.decodeSteps)
        let session = try MactopSession.start()
        let start = DispatchTime.now().uptimeNanoseconds
        do {
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
            }

            try requireBenchmarkMatch(
                BenchmarkOutputValidator.compareDecodeTokens(
                    expectedTokens: expectedTokens,
                    actualTokens: actualTokens
                )
            )

            let elapsed = secondsSince(start)
            let samples = try session.stop()
            let bandwidth = try MactopBandwidth.gigabytesPerToken(
                samples: samples,
                idleGBPerSecond: idleGBPerSecond,
                decodeElapsedSeconds: elapsed,
                decodedTokens: timingPlan.decodeSteps
            )
            return DecodeMeasurement(
                secondsPerToken: elapsed / Double(timingPlan.decodeSteps),
                bandwidthGBPerToken: bandwidth
            )
        } catch {
            _ = try? session.stop()
            throw error
        }
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
        actualToken explicitActualToken: Int? = nil
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
                runtime: "swift"
            )
        )
    }

    private static func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
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
        weightCache: DeepSeekRuntimeWeightCache
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

        for step in 0..<steps {
            let expectedToken = testCase.expectedTokens[step]
            if token != expectedToken {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: token
                )
            }

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

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: steps,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    private static func generateGreedyCached(
        promptTokens: [Int],
        steps: Int,
        weightCache: DeepSeekRuntimeWeightCache
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
        weightCache: DeepSeekRuntimeWeightCache
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
