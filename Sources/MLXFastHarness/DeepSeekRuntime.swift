import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel

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
    public let correctnessSteps: Int
    public let benchmarkDecodeSteps: Int

    public init(
        weightsPath: String,
        goldenPath: String,
        correctnessSteps: Int = MLXFastConstants.correctnessSteps,
        benchmarkDecodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.correctnessSteps = correctnessSteps
        self.benchmarkDecodeSteps = benchmarkDecodeSteps
    }
}

public struct RuntimeWorkerOptions: Equatable {
    public let executablePath: String
    public let sandboxProfilePath: String?

    public init(executablePath: String, sandboxProfilePath: String? = nil) {
        self.executablePath = executablePath
        self.sandboxProfilePath = sandboxProfilePath
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

    public static func runCorrectness(
        _ options: CorrectnessOptions,
        worker: RuntimeWorkerOptions? = nil
    ) throws -> CorrectnessReport {
        if let worker {
            return runCorrectnessWithWorker(options, worker: worker)
        }

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

    private static func runCorrectnessWithWorker(
        _ options: CorrectnessOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) -> CorrectnessReport {
        var loadedGolden: GoldenFixture?
        var lastExpertStats = ExpertStreamingStats.zero
        var checkedSteps = 0
        var currentCase: GoldenCase?
        do {
            try requireRegularFile(options.weightsPath + "/config.json", description: "transformed config")
            try requireRegularFile(options.goldenPath, description: "correctness golden file")
            let golden = try loadGoldenFixture(from: options.goldenPath)
            loadedGolden = golden
            let worker = try RuntimeWorkerClient(
                options: workerOptions,
                weightsPath: options.weightsPath
            )
            defer {
                worker.close()
            }

            for testCase in golden.cases {
                currentCase = testCase
                let result = try compareTeacherForcedWithWorker(
                    testCase: testCase,
                    worker: worker
                )
                lastExpertStats = result.expertStats
                let comparison = result.comparison
                checkedSteps += comparison.checkedSteps
                if !comparison.passed {
                    return CorrectnessReport(
                        passed: false,
                        checkedSteps: checkedSteps,
                        caseCount: golden.cases.count,
                        expertCacheHits: lastExpertStats.cacheHits,
                        expertCacheMisses: lastExpertStats.cacheMisses,
                        expertCacheEvictions: lastExpertStats.cacheEvictions,
                        expertBytesRead: lastExpertStats.bytesRead,
                        expertReadSeconds: lastExpertStats.readSeconds,
                        expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
                        expertHitRate: lastExpertStats.hitRate,
                        firstFailingCase: testCase.name,
                        firstFailingStep: comparison.firstFailingStep,
                        expectedToken: comparison.expectedToken,
                        actualToken: comparison.actualToken,
                        goldenHash: golden.sha256,
                        error: "teacher-forced token mismatch"
                    )
                }
            }

            return CorrectnessReport(
                passed: true,
                checkedSteps: checkedSteps,
                caseCount: golden.cases.count,
                expertCacheHits: lastExpertStats.cacheHits,
                expertCacheMisses: lastExpertStats.cacheMisses,
                expertCacheEvictions: lastExpertStats.cacheEvictions,
                expertBytesRead: lastExpertStats.bytesRead,
                expertReadSeconds: lastExpertStats.readSeconds,
                expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
                expertHitRate: lastExpertStats.hitRate,
                firstFailingCase: nil,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil,
                goldenHash: golden.sha256,
                error: ""
            )
        } catch {
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: loadedGolden?.cases.count ?? 0,
                firstFailingCase: currentCase?.name,
                goldenHash: loadedGolden?.sha256 ?? "",
                expertStats: lastExpertStats,
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

    public static func runWorker(weightsPath: String) throws {
        let config = try DeepSeekConfig.load(from: weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        var state = RuntimeWorkerState()

        while let line = readLine(strippingNewline: true) {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                response = try handleWorkerRequest(request, weightCache: weightCache, state: &state)
            } catch {
                response = RuntimeWorkerResponse(id: -1, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([0x0a]))
            fflush(stdout)
        }
    }

    private static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        weightCache: DeepSeekRuntimeWeightCache,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        switch request.kind {
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            let tokens = try generateGreedyCached(
                promptTokens: promptTokens,
                steps: steps,
                weightCache: weightCache
            )
            return RuntimeWorkerResponse(
                id: request.id,
                ok: true,
                tokens: tokens,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "correctness_begin":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing prompt_tokens")
            }
            let cache = DeepSeekModelCache(config: weightCache.config)
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray(promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            let token = try DeepSeekCorrectness.greedyToken(from: logits)
            state.correctnessCache = cache
            state.correctnessPromptTokenCount = promptTokens.count
            state.correctnessStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                ok: true,
                token: token,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "correctness_step":
            guard let previousToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing token")
            }
            guard let cache = state.correctnessCache else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness step before begin")
            }
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([previousToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: state.correctnessPromptTokenCount + state.correctnessStep
            )
            let token = try DeepSeekCorrectness.greedyToken(from: logits)
            state.correctnessStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                ok: true,
                token: token,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "prefill":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker prefill request missing prompt_tokens")
            }
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
            let elapsed = secondsSince(start)
            Memory.clearCache()
            return RuntimeWorkerResponse(
                id: request.id,
                ok: true,
                token: token,
                seconds: elapsed,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "decode":
            guard let seedTokens = request.seedTokens, let decodeSteps = request.decodeSteps else {
                throw MLXFastError.invalidInput("runtime worker decode request missing seed_tokens or decode_steps")
            }
            guard let expectedSeedToken = request.expectedSeedToken,
                  let expectedTokens = request.expectedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode request missing expected benchmark tokens")
            }
            guard expectedTokens.count >= decodeSteps else {
                throw MLXFastError.invalidInput(
                    "runtime worker decode oracle has \(expectedTokens.count) tokens; need at least \(decodeSteps)"
                )
            }
            let validationDelayMS = try request.validationDelayMilliseconds
                ?? submissionValidationDelayMilliseconds()
            guard validationDelayMS >= 0 else {
                throw MLXFastError.invalidInput("runtime worker validation delay must be non-negative")
            }
            let timingPlan = try DecodeTimingPlan(seedTokenCount: seedTokens.count, decodeSteps: decodeSteps)

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
            let seedToken = token
            cache.materializeCachedState()

            var actualTokens: [Int] = []
            actualTokens.reserveCapacity(timingPlan.decodeSteps)
            let metricsBeforeDecode = weightCache.loader.expertStreamingMetrics?.snapshot()
            let start = DispatchTime.now().uptimeNanoseconds
            for decodedStep in 0..<timingPlan.decodeSteps {
                let inputToken = decodedStep == 0 ? expectedSeedToken : expectedTokens[decodedStep - 1]
                logits = try DeepSeekModel.logits(
                    inputIDs: inputIDsArray([inputToken]),
                    weightCache: weightCache,
                    cache: cache,
                    positionOffset: try timingPlan.positionOffset(forDecodedStep: decodedStep)
                )
                token = try DeepSeekCorrectness.greedyToken(from: logits)
                actualTokens.append(token)
                if validationDelayMS > 0 {
                    Thread.sleep(forTimeInterval: Double(validationDelayMS) / 1_000.0)
                }
            }
            let elapsed = secondsSince(start)
            let bandwidth = try expertStreamingBandwidthGBPerToken(
                before: metricsBeforeDecode,
                after: weightCache.loader.expertStreamingMetrics?.snapshot(),
                decodedTokens: timingPlan.decodeSteps
            )
            return RuntimeWorkerResponse(
                id: request.id,
                ok: true,
                seedToken: seedToken,
                tokens: actualTokens,
                seconds: elapsed,
                secondsPerToken: elapsed / Double(timingPlan.decodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
    }

    public static func benchmark(
        _ options: BenchmarkOptions,
        worker: RuntimeWorkerOptions? = nil
    ) -> ScorePayload {
        if let worker {
            return benchmarkWithWorker(options, worker: worker)
        }

        let benchmarkStart = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: benchmarkStart)
        var correctnessReport: CorrectnessReport?
        var benchmarkLoader: DeepSeekWeightLoader?
        var transformedWeightsDigest: DirectoryDigest?
        var preflightSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedBenchmarkSeconds = 0.0

        progress(
            "start correctness_steps=\(options.correctnessSteps) "
                + "benchmark_decode_steps=\(options.benchmarkDecodeSteps)"
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
            try validateBenchmarkOptions(options)
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
                steps: options.correctnessSteps,
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
                    + "decode_tokens=\(options.benchmarkDecodeSteps)"
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
                decodeSteps: options.benchmarkDecodeSteps,
                weightCache: benchmarkCache,
                idleGBPerSecond: idleGBPerSecond,
                progress: progress
            )
            timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
            let peakRamGB = Double(Memory.peakMemory) / Double(1 << 30)
            let score = BenchmarkScore.score(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken
            )
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

            return passedScore(
                score: score,
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                numLayers: config.numHiddenLayers,
                correctness: correctness,
                expertStats: expertStats,
                bandwidthSource: decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest
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

    private static func validateBenchmarkOptions(_ options: BenchmarkOptions) throws {
        guard options.correctnessSteps > 0 else {
            throw MLXFastError.invalidInput("benchmark correctness steps must be positive")
        }
        guard options.correctnessSteps <= MLXFastConstants.correctnessSteps else {
            throw MLXFastError.invalidInput(
                "benchmark correctness steps \(options.correctnessSteps) exceeds golden length \(MLXFastConstants.correctnessSteps)"
            )
        }
        guard options.benchmarkDecodeSteps > 0 else {
            throw MLXFastError.invalidInput("benchmark decode steps must be positive")
        }
        guard options.benchmarkDecodeSteps <= MLXFastConstants.benchmarkDecodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode steps \(options.benchmarkDecodeSteps) exceeds oracle length \(MLXFastConstants.benchmarkDecodeSteps)"
            )
        }
    }

    private static func benchmarkWithWorker(
        _ options: BenchmarkOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) -> ScorePayload {
        let benchmarkStart = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: benchmarkStart)
        var correctnessReport: CorrectnessReport?
        var transformedWeightsDigest: DirectoryDigest?
        var preflightSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedBenchmarkSeconds = 0.0
        var lastExpertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0

        progress(
            "start correctness_steps=\(options.correctnessSteps) "
                + "benchmark_decode_steps=\(options.benchmarkDecodeSteps)"
        )

        func makeFailedScore(
            error: String,
            correctness: CorrectnessReport?,
            passedCorrectness: Bool,
            firstFailingCase explicitFirstFailingCase: String? = nil,
            firstFailingStep explicitFirstFailingStep: Int? = nil,
            expectedToken explicitExpectedToken: Int? = nil,
            actualToken explicitActualToken: Int? = nil
        ) -> ScorePayload {
            progress("failed passed_correctness=\(passedCorrectness) error=\(redactedProgressError(error))")
            return failedScore(
                error: error,
                correctness: correctness,
                passedCorrectness: passedCorrectness,
                expertStats: lastExpertStats,
                firstFailingCase: explicitFirstFailingCase,
                firstFailingStep: explicitFirstFailingStep,
                expectedToken: explicitExpectedToken,
                actualToken: explicitActualToken,
                weightsDigest: transformedWeightsDigest,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: currentResidentMemoryGB()
            )
        }

        do {
            try validateBenchmarkOptions(options)
            progress("preflight start")
            let preflightStart = DispatchTime.now().uptimeNanoseconds
            try checkWorkerBenchmarkInputs(
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
                try enforceTransformedWeightsByteLimit(transformedWeightsDigest.byteCount)
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

            progress("runtime worker start")
            let worker = try RuntimeWorkerClient(
                options: workerOptions,
                weightsPath: options.weightsPath
            )
            defer {
                worker.close()
            }

            let correctnessStart = DispatchTime.now().uptimeNanoseconds
            progress("correctness start cases=\(golden.cases.count)")
            var checkedSteps = 0
            var firstFailingCase: String?
            var firstFailingComparison: CorrectnessTokenComparison?
            for (caseIndex, testCase) in golden.cases.enumerated() {
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
                progress("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let result = try compareTeacherForcedWithWorker(
                    testCase: testCase,
                    worker: worker,
                    steps: options.correctnessSteps,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress("correctness case \(caseLabel) checked \(step)/\(total) tokens")
                    }
                )
                lastExpertStats = result.expertStats
                peakRamGB = max(peakRamGB, result.peakRamGB)
                let comparison = result.comparison
                checkedSteps += comparison.checkedSteps
                if !comparison.passed {
                    firstFailingCase = testCase.name
                    firstFailingComparison = comparison
                    progress("correctness case \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    break
                }
                progress("correctness case \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
            }
            correctnessSeconds = secondsSince(correctnessStart)
            let correctness = CorrectnessReport(
                passed: firstFailingComparison == nil,
                checkedSteps: checkedSteps,
                caseCount: golden.cases.count,
                expertCacheHits: lastExpertStats.cacheHits,
                expertCacheMisses: lastExpertStats.cacheMisses,
                expertCacheEvictions: lastExpertStats.cacheEvictions,
                expertBytesRead: lastExpertStats.bytesRead,
                expertReadSeconds: lastExpertStats.readSeconds,
                expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
                expertHitRate: lastExpertStats.hitRate,
                firstFailingCase: firstFailingCase,
                firstFailingStep: firstFailingComparison?.firstFailingStep,
                expectedToken: firstFailingComparison?.expectedToken,
                actualToken: firstFailingComparison?.actualToken,
                goldenHash: golden.sha256,
                error: firstFailingComparison == nil ? "" : "teacher-forced token mismatch"
            )
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
                    passedCorrectness: false
                )
            }

            guard let benchmarkGolden = golden.benchmark else {
                throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
            }
            let promptPlan = try BenchmarkPrompt.plan(from: benchmarkGolden)
            progress(
                "benchmark oracle ready prefill_tokens=\(promptPlan.prefillTokens.count) "
                    + "decode_seed_tokens=\(promptPlan.decodeSeedTokens.count) "
                    + "decode_tokens=\(options.benchmarkDecodeSteps)"
            )
            progress("mactop idle measurement skipped; runtime worker uses expert streaming byte fallback")

            let timedBenchmarkStart = DispatchTime.now().uptimeNanoseconds
            progress("timed benchmark start")
            let prefillSecondsPerToken = try measureWorkerPrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                expectedToken: promptPlan.expectedPrefillToken,
                worker: worker,
                progress: progress,
                peakRamGB: &peakRamGB,
                expertStats: &lastExpertStats
            )
            let decode = try measureWorkerDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                expectedTokens: promptPlan.expectedDecodeTokens,
                decodeSteps: options.benchmarkDecodeSteps,
                worker: worker,
                progress: progress,
                peakRamGB: &peakRamGB,
                expertStats: &lastExpertStats
            )
            timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
            let score = BenchmarkScore.score(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken
            )

            guard score.isFinite, score >= 0 else {
                return makeFailedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: true
                )
            }
            progress(
                "complete score=\(formatDouble(score)) "
                    + "wall_seconds=\(formatSeconds(secondsSince(benchmarkStart))) "
                    + "timed_seconds=\(formatSeconds(timedBenchmarkSeconds))"
            )

            return passedScore(
                score: score,
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: decode.bandwidthGBPerToken,
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                benchmarkWallSeconds: secondsSince(benchmarkStart),
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                numLayers: MLXFastConstants.numHiddenLayers,
                correctness: correctness,
                expertStats: lastExpertStats,
                bandwidthSource: decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest
            )
        } catch let mismatch as BenchmarkTokenMismatchError {
            return makeFailedScore(
                error: mismatch.description,
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true,
                firstFailingCase: "benchmark",
                firstFailingStep: mismatch.step,
                expectedToken: mismatch.expectedToken,
                actualToken: mismatch.actualToken
            )
        } catch {
            return makeFailedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true
            )
        }
    }

    private static func runCorrectness(
        cases: [GoldenCase],
        weightCache: DeepSeekRuntimeWeightCache,
        goldenHash: String,
        steps: Int = MLXFastConstants.correctnessSteps,
        progress: ((String) -> Void)? = nil
    ) -> CorrectnessReport {
        var checkedSteps = 0
        var currentCase: GoldenCase?
        do {
            for (caseIndex, testCase) in cases.enumerated() {
                currentCase = testCase
                let caseLabel = "\(caseIndex + 1)/\(cases.count)"
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let comparison = try compareTeacherForcedCached(
                    testCase: testCase,
                    weightCache: weightCache,
                    steps: steps,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness case \(caseLabel) checked \(step)/\(total) tokens")
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
                        error: "teacher-forced token mismatch"
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

    private static func measureWorkerPrefillSecondsPerToken(
        promptTokens: [Int],
        expectedToken: Int,
        worker: RuntimeWorkerClient,
        progress: ((String) -> Void)? = nil,
        peakRamGB: inout Double,
        expertStats: inout ExpertStreamingStats
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
            let response = try worker.prefill(promptTokens: promptTokens)
            expertStats = response.expertStats ?? expertStats
            peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
            guard let token = response.token, let elapsed = response.seconds else {
                throw MLXFastError.invalidInput("runtime worker prefill response missing token or seconds")
            }
            try requireBenchmarkMatch(
                BenchmarkOutputValidator.comparePrefillToken(
                    expectedToken: expectedToken,
                    actualToken: token
                )
            )
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
        decodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps,
        weightCache: DeepSeekRuntimeWeightCache,
        idleGBPerSecond: Double?,
        progress: ((String) -> Void)? = nil
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard expectedTokens.count >= decodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode oracle has \(expectedTokens.count) tokens; need at least \(decodeSteps)"
            )
        }
        let timingPlan = try DecodeTimingPlan(
            seedTokenCount: seedTokens.count,
            decodeSteps: decodeSteps
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
        let validationDelayMS = try submissionValidationDelayMilliseconds()
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
            if validationDelayMS > 0 {
                progress?("decode validation delay enabled milliseconds_per_token=\(validationDelayMS)")
            }
            for decodedStep in 0..<timingPlan.decodeSteps {
                let inputToken = decodedStep == 0 ? expectedSeedToken : expectedTokens[decodedStep - 1]
                logits = try DeepSeekModel.logits(
                    inputIDs: inputIDsArray([inputToken]),
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
                if validationDelayMS > 0 {
                    Thread.sleep(forTimeInterval: Double(validationDelayMS) / 1_000.0)
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

    private static func measureWorkerDecode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        decodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps,
        worker: RuntimeWorkerClient,
        progress: ((String) -> Void)? = nil,
        peakRamGB: inout Double,
        expertStats: inout ExpertStreamingStats
    ) throws -> DecodeMeasurement {
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard expectedTokens.count >= decodeSteps else {
            throw MLXFastError.invalidInput(
                "benchmark decode oracle has \(expectedTokens.count) tokens; need at least \(decodeSteps)"
            )
        }
        progress?("decode measured start tokens=\(decodeSteps)")
        let response = try worker.decode(
            seedTokens: seedTokens,
            expectedSeedToken: expectedSeedToken,
            expectedTokens: expectedTokens,
            decodeSteps: decodeSteps
        )
        expertStats = response.expertStats ?? expertStats
        peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
        guard let seedToken = response.seedToken else {
            throw MLXFastError.invalidInput("runtime worker decode response missing seed token")
        }
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: expectedSeedToken,
                actualToken: seedToken
            )
        )
        let actualTokens = response.tokens ?? []
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeTokens(
                expectedTokens: Array(expectedTokens.prefix(decodeSteps)),
                actualTokens: actualTokens
            )
        )
        guard let secondsPerToken = response.secondsPerToken,
              let bandwidthGBPerToken = response.bandwidthGBPerToken,
              let bandwidthSource = response.bandwidthSource else {
            throw MLXFastError.invalidInput("runtime worker decode response missing timing or bandwidth")
        }
        progress?(
            "decode measured complete seconds=\(formatSeconds(response.seconds ?? 0)) "
                + "seconds_per_token=\(formatDouble(secondsPerToken)) "
                + "bandwidth_gb_per_token=\(formatDouble(bandwidthGBPerToken)) "
                + "bandwidth_source=\(bandwidthSource)"
        )
        return DecodeMeasurement(
            secondsPerToken: secondsPerToken,
            bandwidthGBPerToken: bandwidthGBPerToken,
            bandwidthSource: bandwidthSource
        )
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

    static func submissionValidationDelayMilliseconds() throws -> Int {
        let milliseconds = DeepSeekSubmissionControls.measuredDecodeDelayMilliseconds
        guard milliseconds >= 0 else {
            throw MLXFastError.invalidInput(
                "DeepSeekSubmissionControls.measuredDecodeDelayMilliseconds must be non-negative"
            )
        }
        return milliseconds
    }

    private static func expertStats(from weightCache: DeepSeekRuntimeWeightCache) -> ExpertStreamingStats {
        expertStats(from: weightCache.loader)
    }

    private static func expertStats(from loader: DeepSeekWeightLoader?) -> ExpertStreamingStats {
        loader?.expertStreamingMetrics?.snapshot().stats ?? .zero
    }

    private static func passedScore(
        score: Double,
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        benchmarkWallSeconds: Double,
        preflightSeconds: Double,
        correctnessSeconds: Double,
        timedBenchmarkSeconds: Double,
        numLayers: Int,
        correctness: CorrectnessReport,
        expertStats: ExpertStreamingStats,
        bandwidthSource: String,
        weightsDigest: DirectoryDigest?
    ) -> ScorePayload {
        ScorePayload(
            score: score,
            passed: true,
            metrics: ScoreMetrics(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                benchmarkWallSeconds: benchmarkWallSeconds,
                preflightSeconds: preflightSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedBenchmarkSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                passedCorrectness: true,
                numLayers: numLayers,
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
                bandwidthSource: bandwidthSource,
                error: "",
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

    private static func checkWorkerBenchmarkInputs(
        weightsPath: String,
        goldenPath: String
    ) throws {
        try requireDirectory(weightsPath, description: "transformed weights")
        let requiredFiles = [
            ("\(weightsPath)/config.json", "transformed config"),
            ("\(weightsPath)/model.safetensors.index.json", "dense safetensors index"),
            ("\(weightsPath)/experts/manifest.json", "expert manifest"),
            (goldenPath, "correctness golden file"),
        ]
        for (path, description) in requiredFiles {
            try requireRegularFile(path, description: description)
        }
    }

    private static func requireDirectory(_ path: String, description: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw MLXFastError.invalidInput("\(description) must not be a symlink: \(path)")
        }
        guard values.isDirectory == true else {
            throw MLXFastError.missingFile("\(description) directory missing at \(path)")
        }
    }

    private static func requireRegularFile(_ path: String, description: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw MLXFastError.invalidInput("\(description) must not be a symlink: \(path)")
        }
        guard values.isRegularFile == true else {
            throw MLXFastError.missingFile("\(description) missing at \(path)")
        }
    }

    private static func enforceTransformedWeightsByteLimit(_ byteCount: Int) throws {
        guard let maxByteCount = try transformedWeightsByteLimit() else {
            return
        }
        guard byteCount <= maxByteCount else {
            throw MLXFastError.invalidInput(
                "transformed weights are \(byteCount) bytes, above MLXFAST_MAX_WEIGHTS_BYTES=\(maxByteCount)"
            )
        }
    }

    private static func transformedWeightsByteLimit(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int? {
        let raw = environment["MLXFAST_MAX_WEIGHTS_BYTES"] ?? ""
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return MLXFastConstants.defaultMaxTransformedWeightsBytes
        }

        let lowercased = trimmed.lowercased()
        if lowercased == "0" || lowercased == "none" || lowercased == "unlimited" {
            return nil
        }
        guard let value = Int(trimmed), value > 0 else {
            throw MLXFastError.invalidInput(
                "MLXFAST_MAX_WEIGHTS_BYTES must be a positive byte count, 0, none, or unlimited"
            )
        }
        return value
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

    private struct WorkerCorrectnessResult {
        let comparison: CorrectnessTokenComparison
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
    }

    private static func compareTeacherForcedWithWorker(
        testCase: GoldenCase,
        worker: RuntimeWorkerClient,
        steps: Int = MLXFastConstants.correctnessSteps,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> WorkerCorrectnessResult {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("teacher-forced correctness prompt must not be empty")
        }
        guard testCase.expectedTokens.count >= steps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need at least \(steps)"
            )
        }

        var lastExpertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        var response = try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)
        lastExpertStats = response.expertStats ?? lastExpertStats
        peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)

        for step in 0..<steps {
            guard let actualToken = response.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness response missing token")
            }
            let expectedToken = testCase.expectedTokens[step]
            if !correctnessTokenAccepted(
                expectedToken: expectedToken,
                actualToken: actualToken,
                topLogits: response.topLogits
            ) {
                return WorkerCorrectnessResult(
                    comparison: CorrectnessTokenComparison(
                        passed: false,
                        checkedSteps: step + 1,
                        firstFailingStep: step,
                        expectedToken: expectedToken,
                        actualToken: actualToken
                    ),
                    expertStats: lastExpertStats,
                    peakRamGB: peakRamGB
                )
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

            response = try worker.teacherForcedCorrectnessStep(previousToken: expectedToken)
            lastExpertStats = response.expertStats ?? lastExpertStats
            peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
        }

        return WorkerCorrectnessResult(
            comparison: CorrectnessTokenComparison(
                passed: true,
                checkedSteps: steps,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            ),
            expertStats: lastExpertStats,
            peakRamGB: peakRamGB
        )
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

    private static func compareTeacherForcedCached(
        testCase: GoldenCase,
        weightCache: DeepSeekRuntimeWeightCache,
        steps: Int = MLXFastConstants.correctnessSteps,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> CorrectnessTokenComparison {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("teacher-forced correctness prompt must not be empty")
        }
        guard testCase.expectedTokens.count >= steps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need at least \(steps)"
            )
        }

        let cache = DeepSeekModelCache(config: weightCache.config)
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var actualToken = try DeepSeekCorrectness.greedyToken(from: logits)

        for step in 0..<steps {
            let expectedToken = testCase.expectedTokens[step]
            if !correctnessTokenAccepted(
                expectedToken: expectedToken,
                actualToken: actualToken,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
            ) {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
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

            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([expectedToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: testCase.promptTokens.count + step
            )
            actualToken = try DeepSeekCorrectness.greedyToken(from: logits)
        }

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: steps,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    private static func topLogits(from logits: MLXArray, topK: Int) throws -> [CorrectnessTraceLogit] {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("correctness logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        let last = rows[-1]
        eval(last)
        let values = last.asArray(Float.self).map(Double.init)
        guard values.count == vocabSize else {
            throw MLXFastError.invalidInput(
                "correctness logits materialized \(values.count) values, expected \(vocabSize)"
            )
        }

        let sortedIndices = values.indices.sorted {
            let lhs = values[$0]
            let rhs = values[$1]
            return lhs == rhs ? $0 < $1 : lhs > rhs
        }
        return sortedIndices.prefix(min(topK, sortedIndices.count)).map {
            CorrectnessTraceLogit(token: $0, logit: values[$0])
        }
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

private struct RuntimeWorkerRequest: Codable {
    let id: Int
    let kind: String
    let promptTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let expectedSeedToken: Int?
    let expectedTokens: [Int]?
    let steps: Int?
    let decodeSteps: Int?
    let validationDelayMilliseconds: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case token
        case seedTokens = "seed_tokens"
        case expectedSeedToken = "expected_seed_token"
        case expectedTokens = "expected_tokens"
        case steps
        case decodeSteps = "decode_steps"
        case validationDelayMilliseconds = "validation_delay_ms"
    }
}

private struct RuntimeWorkerState {
    var correctnessCache: DeepSeekModelCache?
    var correctnessPromptTokenCount = 0
    var correctnessStep = 0
}

private struct RuntimeWorkerResponse: Codable {
    let id: Int
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let seedToken: Int?
    let tokens: [Int]?
    let seconds: Double?
    let secondsPerToken: Double?
    let bandwidthGBPerToken: Double?
    let bandwidthSource: String?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?

    init(
        id: Int,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        seconds: Double? = nil,
        secondsPerToken: Double? = nil,
        bandwidthGBPerToken: Double? = nil,
        bandwidthSource: String? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil
    ) {
        self.id = id
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.seedToken = seedToken
        self.tokens = tokens
        self.seconds = seconds
        self.secondsPerToken = secondsPerToken
        self.bandwidthGBPerToken = bandwidthGBPerToken
        self.bandwidthSource = bandwidthSource
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
    }

    enum CodingKeys: String, CodingKey {
        case id
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case seedToken = "seed_token"
        case tokens
        case seconds
        case secondsPerToken = "seconds_per_token"
        case bandwidthGBPerToken = "bandwidth_gb_per_token"
        case bandwidthSource = "bandwidth_source"
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
    }
}

private final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let errorOutput: FileHandle
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var nextID = 1
    private var closed = false

    init(options: RuntimeWorkerOptions, weightsPath: String) throws {
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = [
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        }
        var environment = ProcessInfo.processInfo.environment
        environment["MLXFAST_USE_RUNTIME_WORKER"] = "0"
        for key in [
            "MLXFAST_CORRECTNESS_GOLDEN_PATH",
            "MLXFAST_CORRECTNESS_GOLDEN_URL",
            "MLXFAST_CORRECTNESS_GOLDEN_AUTH_HEADER",
            "MLXFAST_PRIVATE_DIR",
            "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE",
            "R2_ACCESS_KEY_ID",
            "R2_BUCKET_ENDPOINT",
            "R2_SECRET_ACCESS_KEY",
        ] {
            environment.removeValue(forKey: key)
        }
        process.environment = environment
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        self.process = process
        self.input = stdin.fileHandleForWriting
        self.output = stdout.fileHandleForReading
        self.errorOutput = stderr.fileHandleForReading
    }

    deinit {
        close()
    }

    func close() {
        guard !closed else {
            return
        }
        closed = true
        try? input.close()
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness",
            promptTokens: promptTokens,
            steps: steps
        )
    }

    func beginTeacherForcedCorrectness(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_begin",
            promptTokens: promptTokens
        )
    }

    func teacherForcedCorrectnessStep(previousToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_step",
            token: previousToken
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "prefill",
            promptTokens: promptTokens
        )
    }

    func decode(
        seedTokens: [Int],
        expectedSeedToken: Int,
        expectedTokens: [Int],
        decodeSteps: Int
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode",
            seedTokens: seedTokens,
            expectedSeedToken: expectedSeedToken,
            expectedTokens: expectedTokens,
            decodeSteps: decodeSteps
        )
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        expectedSeedToken: Int? = nil,
        expectedTokens: [Int]? = nil,
        steps: Int? = nil,
        decodeSteps: Int? = nil,
        validationDelayMilliseconds: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        guard process.isRunning else {
            throw MLXFastError.invalidInput("runtime worker exited before request \(kind): \(workerExitDiagnostic())")
        }
        let id = nextID
        nextID += 1
        let request = RuntimeWorkerRequest(
            id: id,
            kind: kind,
            promptTokens: promptTokens,
            token: token,
            seedTokens: seedTokens,
            expectedSeedToken: expectedSeedToken,
            expectedTokens: expectedTokens,
            steps: steps,
            decodeSteps: decodeSteps,
            validationDelayMilliseconds: validationDelayMilliseconds
        )
        var data = try encoder.encode(request)
        data.append(0x0a)
        try input.write(contentsOf: data)

        let response = try readResponseLine()
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine() throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            return try decoder.decode(RuntimeWorkerResponse.self, from: data)
        }
    }

    private func readWorkerOutputLine() throws -> Data {
        var data = Data()
        while true {
            let byte = output.readData(ofLength: 1)
            if byte.isEmpty {
                throw MLXFastError.invalidInput(
                    "runtime worker closed stdout before returning a response: \(workerExitDiagnostic())"
                )
            }
            if byte[byte.startIndex] == 0x0a {
                return data
            }
            data.append(byte)
        }
    }

    private func workerExitDiagnostic() -> String {
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        let stderr = String(data: errorOutput.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = sanitizeWorkerDiagnostic(trimmed)
        if redacted.isEmpty {
            return "exit_status=\(process.terminationStatus)"
        }
        return "exit_status=\(process.terminationStatus) stderr=\(redacted)"
    }

    private func sanitizeWorkerDiagnostic(_ value: String) -> String {
        let singleLine = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        if singleLine.range(of: "expected", options: .caseInsensitive) != nil
            || singleLine.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return singleLine
    }
}

func runtimeWorkerLineLooksLikeJSONResponse(_ data: Data) -> Bool {
    for byte in data where byte != 0x20 && byte != 0x09 && byte != 0x0d {
        return byte == 0x7b
    }
    return false
}

func correctnessTokenAccepted(
    expectedToken: Int,
    actualToken: Int,
    topLogits: [CorrectnessTraceLogit]?
) -> Bool {
    if actualToken == expectedToken {
        return true
    }
    guard let topLogits,
          let topLogit = topLogits.first?.logit,
          let expectedLogit = topLogits.first(where: { $0.token == expectedToken })?.logit
    else {
        return false
    }
    // Some Apple GPU/Metal combinations break exact argmax ties differently.
    // Accept only a true top-logit tie, and keep feeding the golden token.
    return topLogit - expectedLogit <= MLXFastConstants.correctnessLogitTieTolerance
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
