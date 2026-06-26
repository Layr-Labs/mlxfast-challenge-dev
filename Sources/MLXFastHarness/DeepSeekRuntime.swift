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

public struct GreedyGenerationOptions: Equatable {
    public let weightsPath: String
    public let promptTokens: [Int]
    public let steps: Int

    public init(weightsPath: String, promptTokens: [Int], steps: Int) {
        self.weightsPath = weightsPath
        self.promptTokens = promptTokens
        self.steps = steps
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
    public static func generateGreedyTokens(
        _ options: GreedyGenerationOptions,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        try generateGreedyTokens(options, worker: nil, progress: progress)
    }

    public static func generateGreedyTokens(
        _ options: GreedyGenerationOptions,
        worker workerOptions: RuntimeWorkerOptions?,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        if let workerOptions {
            let worker = try RuntimeWorkerClient(options: workerOptions, weightsPath: options.weightsPath)
            defer {
                worker.close()
            }
            let response = try worker.generateCorrectness(
                promptTokens: options.promptTokens,
                steps: options.steps
            )
            guard let tokens = response.tokens else {
                throw MLXFastError.invalidInput("runtime worker greedy generation response missing tokens")
            }
            try requireGeneratedTokenCount(tokens.count, expected: options.steps, label: "greedy generation")
            progress?(tokens.count, options.steps)
            return tokens
        }

        let config = try DeepSeekConfig.load(from: options.weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: options.weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        return try generateGreedyCached(
            promptTokens: options.promptTokens,
            steps: options.steps,
            weightCache: weightCache,
            progressIntervalSteps: 1,
            progress: progress
        )
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
            return runLayeredCorrectness(
                golden: golden,
                weightCache: weightCache,
                steps: MLXFastConstants.correctnessSteps
            )
        } catch {
            return failedCorrectnessReport(
                checkedSteps: 0,
                caseCount: loadedGolden?.totalCorrectnessCaseCount ?? 0,
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
            let result = runLayeredCorrectnessWithWorker(
                golden: golden,
                worker: worker,
                steps: MLXFastConstants.correctnessSteps
            )
            checkedSteps = result.report.checkedSteps
            lastExpertStats = result.expertStats
            return result.report
        } catch {
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: loadedGolden?.totalCorrectnessCaseCount ?? 0,
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
        // Keep protocol I/O off fd 0/1 so submitted model code cannot read
        // future request nonces or spoof JSON responses with normal stdio.
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))
        var state = RuntimeWorkerState()

        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                do {
                    response = try handleWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        weightCache: weightCache,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(id: -1, nonce: sessionNonce, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            try protocolIO.writeLine(data)
        }
    }

    private static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
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
                nonce: sessionNonce,
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
            let start = DispatchTime.now().uptimeNanoseconds
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
            let elapsed = secondsSince(start)
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
                seconds: elapsed,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "correctness_teacher_forced_batch":
            guard let promptTokens = request.promptTokens,
                  let expectedTokens = request.expectedTokens,
                  let steps = request.steps
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker batched teacher-forced request missing prompt_tokens, expected_tokens, or steps"
                )
            }
            guard !promptTokens.isEmpty else {
                throw MLXFastError.invalidInput("runtime worker batched teacher-forced prompt_tokens must not be empty")
            }
            guard steps > 0 else {
                throw MLXFastError.invalidInput("runtime worker batched teacher-forced steps must be positive")
            }
            guard expectedTokens.count >= steps else {
                throw MLXFastError.invalidInput(
                    "runtime worker batched teacher-forced expected_tokens has \(expectedTokens.count) tokens; expected at least \(steps)"
                )
            }
            let teacherForcedInput = promptTokens + Array(expectedTokens.prefix(max(steps - 1, 0)))
            let cache = DeepSeekModelCache(config: weightCache.config)
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray(teacherForcedInput),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            var tokens: [Int] = []
            var topLogitRows: [[CorrectnessTraceLogit]] = []
            tokens.reserveCapacity(steps)
            topLogitRows.reserveCapacity(steps)
            let firstLogitRow = promptTokens.count - 1
            for step in 0..<steps {
                let topLogits = try topLogits(
                    from: logits,
                    row: firstLogitRow + step,
                    topK: MLXFastConstants.correctnessTopLogits
                )
                guard let token = topLogits.first?.token else {
                    throw MLXFastError.invalidInput("runtime worker batched teacher-forced top logits missing token")
                }
                tokens.append(token)
                topLogitRows.append(topLogits)
            }
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                topLogitRows: topLogitRows,
                tokens: tokens,
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
                nonce: sessionNonce,
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
                nonce: sessionNonce,
                ok: true,
                token: token,
                seconds: elapsed,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "decode_begin":
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode_begin request missing seed_tokens")
            }
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
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray(seedTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            let token = try DeepSeekCorrectness.greedyToken(from: logits)
            let seedToken = token
            cache.materializeCachedState()
            state.decodeCache = cache
            state.decodeSeedTokenCount = seedTokens.count
            state.decodeStep = 0
            let elapsed = secondsSince(start)

            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken,
                seconds: elapsed,
                expertStats: expertStats(from: weightCache),
                peakRamGB: currentResidentMemoryGB()
            )

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            guard let cache = state.decodeCache else {
                throw MLXFastError.invalidInput("runtime worker decode_step before decode_begin")
            }
            let validationDelayMS = try submissionValidationDelayMilliseconds()
            guard validationDelayMS >= 0 else {
                throw MLXFastError.invalidInput("runtime worker validation delay must be non-negative")
            }
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([inputToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: state.decodeSeedTokenCount + state.decodeStep
            )
            let token = try DeepSeekCorrectness.greedyToken(from: logits)
            if validationDelayMS > 0 {
                Thread.sleep(forTimeInterval: Double(validationDelayMS) / 1_000.0)
            }
            let elapsed = secondsSince(start)
            state.decodeStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                seconds: elapsed,
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
                "golden load complete cases=\(golden.totalCorrectnessCaseCount) "
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
            progress("correctness start cases=\(golden.totalCorrectnessCaseCount)")
            let correctness = runLayeredCorrectness(
                golden: golden,
                weightCache: correctnessCache,
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
            let idleGBPerSecond = try measureMactopIdleGBPerSecond(progress: progress)

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
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken
            )
            let decodeSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken,
                candidateSecondsPerToken: decode.secondsPerToken
            )
            let prefillSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken,
                candidateSecondsPerToken: prefillSecondsPerToken
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
                    + "decode_speedup=\(formatDouble(decodeSpeedup)) "
                    + "prefill_speedup=\(formatDouble(prefillSpeedup)) "
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
                weightsDigest: transformedWeightsDigest,
                gpqaTTFT: .zero
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
                "golden load complete cases=\(golden.totalCorrectnessCaseCount) "
                    + "benchmark_oracle=\(golden.benchmark == nil ? "missing" : "present")"
            )

            let correctnessStart = DispatchTime.now().uptimeNanoseconds
            progress("correctness start cases=\(golden.totalCorrectnessCaseCount)")
            progress("correctness worker start")
            let correctnessResult: WorkerLayeredCorrectnessResult
            do {
                let correctnessWorker = try RuntimeWorkerClient(
                    options: workerOptions,
                    weightsPath: options.weightsPath
                )
                defer {
                    correctnessWorker.close()
                }
                correctnessResult = runLayeredCorrectnessWithWorker(
                    golden: golden,
                    worker: correctnessWorker,
                    steps: options.correctnessSteps,
                    progress: progress
                )
            }
            correctnessSeconds = secondsSince(correctnessStart)
            lastExpertStats = correctnessResult.expertStats
            peakRamGB = max(peakRamGB, correctnessResult.peakRamGB)
            let correctness = correctnessResult.report
            correctnessReport = correctness
            progress(
                "correctness complete passed=\(correctness.passed) "
                    + "checked_steps=\(correctness.checkedSteps) "
                    + "seconds=\(formatSeconds(correctnessSeconds))"
            )
            if correctnessResult.gpqaTTFT.caseCount > 0 {
                let gpqaTTFT = correctnessResult.gpqaTTFT
                progress(
                    "gpqa ttft complete cases=\(gpqaTTFT.caseCount) "
                        + "pass_count=\(gpqaTTFT.passCount) "
                        + "mean_seconds=\(formatSeconds(gpqaTTFT.meanSeconds)) "
                        + "p50_seconds=\(formatSeconds(gpqaTTFT.p50Seconds)) "
                        + "max_seconds=\(formatSeconds(gpqaTTFT.maxSeconds))"
                )
            }
            guard correctness.passed else {
                return makeFailedScore(
                    error: correctness.error.isEmpty ? "correctness gate failed" : correctness.error,
                    correctness: correctness,
                    passedCorrectness: false
                )
            }
            guard correctnessResult.gpqaTTFT.caseCount == 0 || correctnessResult.gpqaTTFT.passed else {
                return makeFailedScore(
                    error: "hidden GPQA TTFT gate failed",
                    correctness: correctness,
                    passedCorrectness: true
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
            let idleGBPerSecond = try measureMactopIdleGBPerSecond(progress: progress)
            progress("benchmark worker start")
            let benchmarkWorker = try RuntimeWorkerClient(
                options: workerOptions,
                weightsPath: options.weightsPath
            )
            defer {
                benchmarkWorker.close()
            }
            peakRamGB = 0
            lastExpertStats = .zero

            let timedBenchmarkStart = DispatchTime.now().uptimeNanoseconds
            progress("timed benchmark start")
            let prefillSecondsPerToken = try measureWorkerPrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                expectedToken: promptPlan.expectedPrefillToken,
                worker: benchmarkWorker,
                progress: progress,
                peakRamGB: &peakRamGB,
                expertStats: &lastExpertStats
            )
            let decode = try measureWorkerDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                expectedSeedToken: promptPlan.expectedDecodeSeedToken,
                expectedTokens: promptPlan.expectedDecodeTokens,
                decodeSteps: options.benchmarkDecodeSteps,
                worker: benchmarkWorker,
                idleGBPerSecond: idleGBPerSecond,
                progress: progress,
                peakRamGB: &peakRamGB,
                expertStats: &lastExpertStats
            )
            timedBenchmarkSeconds = secondsSince(timedBenchmarkStart)
            let score = BenchmarkScore.score(
                decodeSecondsPerToken: decode.secondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken
            )
            let decodeSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken,
                candidateSecondsPerToken: decode.secondsPerToken
            )
            let prefillSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken,
                candidateSecondsPerToken: prefillSecondsPerToken
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
                    + "decode_speedup=\(formatDouble(decodeSpeedup)) "
                    + "prefill_speedup=\(formatDouble(prefillSpeedup)) "
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
                weightsDigest: transformedWeightsDigest,
                gpqaTTFT: correctnessResult.gpqaTTFT
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

    private struct WorkerLayeredCorrectnessResult {
        let report: CorrectnessReport
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
        let gpqaTTFT: GPQATTFTSummary
    }

    private struct GPQATTFTSummary {
        static let zero = GPQATTFTSummary(passCount: 0, caseCount: 0, seconds: [])

        let passCount: Int
        let caseCount: Int
        let seconds: [Double]

        var passed: Bool {
            caseCount > 0 && passCount == caseCount && seconds.count == caseCount
        }

        var source: String {
            caseCount > 0 ? "hidden_gpqa_first_token" : ""
        }

        var meanSeconds: Double {
            guard !seconds.isEmpty else {
                return 0
            }
            return seconds.reduce(0, +) / Double(seconds.count)
        }

        var p50Seconds: Double {
            guard !seconds.isEmpty else {
                return 0
            }
            let sortedSeconds = seconds.sorted()
            return sortedSeconds[sortedSeconds.count / 2]
        }

        var maxSeconds: Double {
            seconds.max() ?? 0
        }
    }

    private static func runLayeredCorrectness(
        golden: GoldenFixture,
        weightCache: DeepSeekRuntimeWeightCache,
        steps: Int = MLXFastConstants.correctnessSteps,
        progress: ((String) -> Void)? = nil
    ) -> CorrectnessReport {
        let caseCount = golden.totalCorrectnessCaseCount
        var checkedSteps = 0
        var currentCase: String?

        func failure(
            caseName: String,
            comparison: CorrectnessTokenComparison,
            error: String
        ) -> CorrectnessReport {
            let stats = expertStats(from: weightCache)
            return CorrectnessReport(
                passed: false,
                checkedSteps: checkedSteps + comparison.checkedSteps,
                caseCount: caseCount,
                expertCacheHits: stats.cacheHits,
                expertCacheMisses: stats.cacheMisses,
                expertCacheEvictions: stats.cacheEvictions,
                expertBytesRead: stats.bytesRead,
                expertReadSeconds: stats.readSeconds,
                expertPeakCachedTensors: stats.peakCachedTensors,
                expertHitRate: stats.hitRate,
                firstFailingCase: caseName,
                firstFailingStep: comparison.firstFailingStep,
                expectedToken: comparison.expectedToken,
                actualToken: comparison.actualToken,
                goldenHash: golden.sha256,
                error: error
            )
        }

        do {
            for (caseIndex, testCase) in golden.cases.enumerated() {
                currentCase = testCase.name
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
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
                    return failure(
                        caseName: testCase.name,
                        comparison: comparison,
                        error: "teacher-forced token mismatch"
                    )
                }
                progress?("correctness case \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
                checkedSteps += comparison.checkedSteps
            }

            let gates = golden.correctnessGates
            for (caseIndex, anchor) in (gates?.anchorCases ?? []).enumerated() {
                currentCase = anchor.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.anchorCases.count ?? 0)"
                progress?("correctness anchor \(caseLabel) start context_tokens=\(anchor.contextTokens.count)")
                let comparison = try compareAnchorCached(anchor: anchor, weightCache: weightCache)
                if !comparison.passed {
                    progress?("correctness anchor \(caseLabel) failed")
                    return failure(
                        caseName: anchor.name,
                        comparison: comparison,
                        error: "anchor token mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness anchor \(caseLabel) complete")
            }

            for (caseIndex, freeRun) in (gates?.freeRunCases ?? []).enumerated() {
                currentCase = freeRun.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.freeRunCases.count ?? 0)"
                progress?("correctness free-run \(caseLabel) start tokens=\(freeRun.expectedTokens.count)")
                let comparison = try compareFreeRunCached(
                    testCase: freeRun,
                    weightCache: weightCache,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness free-run \(caseLabel) generated \(step)/\(total) tokens")
                    }
                )
                if !comparison.passed {
                    progress?("correctness free-run \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: freeRun.name,
                        comparison: comparison,
                        error: "free-run token mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness free-run \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
            }

            for (caseIndex, behavior) in (gates?.behaviorCases ?? []).enumerated() {
                currentCase = behavior.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.behaviorCases.count ?? 0)"
                progress?("correctness behavior \(caseLabel) start max_new_tokens=\(behavior.maxNewTokens)")
                let comparison = try compareBehaviorCached(
                    testCase: behavior,
                    weightCache: weightCache
                )
                if !comparison.passed {
                    progress?("correctness behavior \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: behavior.name,
                        comparison: comparison,
                        error: "behavior answer mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness behavior \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
            }
        } catch {
            progress?("correctness error=\(redactedProgressError("\(error)"))")
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: caseCount,
                firstFailingCase: currentCase,
                goldenHash: golden.sha256,
                expertStats: expertStats(from: weightCache),
                error: "\(error)"
            )
        }

        let stats = expertStats(from: weightCache)
        return CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: stats.cacheHits,
            expertCacheMisses: stats.cacheMisses,
            expertCacheEvictions: stats.cacheEvictions,
            expertBytesRead: stats.bytesRead,
            expertReadSeconds: stats.readSeconds,
            expertPeakCachedTensors: stats.peakCachedTensors,
            expertHitRate: stats.hitRate,
            firstFailingCase: nil,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: golden.sha256,
            error: ""
        )
    }

    private static func runLayeredCorrectnessWithWorker(
        golden: GoldenFixture,
        worker: RuntimeWorkerClient,
        steps: Int = MLXFastConstants.correctnessSteps,
        progress: ((String) -> Void)? = nil
    ) -> WorkerLayeredCorrectnessResult {
        let caseCount = golden.totalCorrectnessCaseCount
        var checkedSteps = 0
        var currentCase: String?
        var lastExpertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        var gpqaTTFTPassCount = 0
        var gpqaTTFTCaseCount = 0
        var gpqaTTFTSeconds: [Double] = []

        func result(report: CorrectnessReport) -> WorkerLayeredCorrectnessResult {
            WorkerLayeredCorrectnessResult(
                report: report,
                expertStats: lastExpertStats,
                peakRamGB: peakRamGB,
                gpqaTTFT: GPQATTFTSummary(
                    passCount: gpqaTTFTPassCount,
                    caseCount: gpqaTTFTCaseCount,
                    seconds: gpqaTTFTSeconds
                )
            )
        }

        func failure(
            caseName: String,
            comparison: CorrectnessTokenComparison,
            error: String
        ) -> WorkerLayeredCorrectnessResult {
            result(report: CorrectnessReport(
                passed: false,
                checkedSteps: checkedSteps + comparison.checkedSteps,
                caseCount: caseCount,
                expertCacheHits: lastExpertStats.cacheHits,
                expertCacheMisses: lastExpertStats.cacheMisses,
                expertCacheEvictions: lastExpertStats.cacheEvictions,
                expertBytesRead: lastExpertStats.bytesRead,
                expertReadSeconds: lastExpertStats.readSeconds,
                expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
                expertHitRate: lastExpertStats.hitRate,
                firstFailingCase: caseName,
                firstFailingStep: comparison.firstFailingStep,
                expectedToken: comparison.expectedToken,
                actualToken: comparison.actualToken,
                goldenHash: golden.sha256,
                error: error
            ))
        }

        do {
            for (caseIndex, testCase) in golden.cases.enumerated() {
                currentCase = testCase.name
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let check = try compareTeacherForcedWithWorker(
                    testCase: testCase,
                    worker: worker,
                    steps: steps,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness case \(caseLabel) checked \(step)/\(total) tokens")
                    }
                )
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness case \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: testCase.name,
                        comparison: check.comparison,
                        error: "teacher-forced token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness case \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }

            let gates = golden.correctnessGates
            for (caseIndex, anchor) in (gates?.anchorCases ?? []).enumerated() {
                currentCase = anchor.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.anchorCases.count ?? 0)"
                progress?("correctness anchor \(caseLabel) start context_tokens=\(anchor.contextTokens.count)")
                let check = try compareAnchorWithWorker(anchor: anchor, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness anchor \(caseLabel) failed")
                    return failure(
                        caseName: anchor.name,
                        comparison: check.comparison,
                        error: "anchor token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness anchor \(caseLabel) complete")
            }

            for (caseIndex, freeRun) in (gates?.freeRunCases ?? []).enumerated() {
                currentCase = freeRun.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.freeRunCases.count ?? 0)"
                progress?("correctness free-run \(caseLabel) start tokens=\(freeRun.expectedTokens.count)")
                let check = try compareFreeRunWithWorker(testCase: freeRun, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness free-run \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: freeRun.name,
                        comparison: check.comparison,
                        error: "free-run token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness free-run \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }

            for (caseIndex, behavior) in (gates?.behaviorCases ?? []).enumerated() {
                currentCase = behavior.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.behaviorCases.count ?? 0)"
                progress?("correctness behavior \(caseLabel) start max_new_tokens=\(behavior.maxNewTokens)")
                let check = try compareBehaviorWithWorker(testCase: behavior, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if behavior.maxNewTokens == 1 {
                    gpqaTTFTCaseCount += 1
                    if check.comparison.passed, let ttftSeconds = check.ttftSeconds, ttftSeconds > 0 {
                        gpqaTTFTPassCount += 1
                        gpqaTTFTSeconds.append(ttftSeconds)
                    }
                }
                if !check.comparison.passed {
                    progress?("correctness behavior \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: behavior.name,
                        comparison: check.comparison,
                        error: "behavior answer mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness behavior \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }
        } catch {
            progress?("correctness error=\(redactedProgressError("\(error)"))")
            return result(report: failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: caseCount,
                firstFailingCase: currentCase,
                goldenHash: golden.sha256,
                expertStats: lastExpertStats,
                error: "\(error)"
            ))
        }

        return result(report: CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
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
        ))
    }

    private struct DecodeMeasurement {
        let secondsPerToken: Double
        let bandwidthGBPerToken: Double
        let bandwidthSource: String
    }

    private static func measureMactopIdleGBPerSecond(progress: ((String) -> Void)? = nil) throws -> Double? {
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
            if requiresMactopHardwareBandwidth() {
                throw MLXFastError.invalidInput(
                    "mactop hardware bandwidth required but unavailable: \(redactedProgressError("\(error)"))"
                )
            }
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
                if requiresMactopHardwareBandwidth() {
                    throw MLXFastError.invalidInput(
                        "mactop hardware bandwidth required but decode measurement could not start: "
                            + "\(redactedProgressError("\(error)"))"
                    )
                }
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
                    if requiresMactopHardwareBandwidth() {
                        throw MLXFastError.invalidInput(
                            "mactop hardware bandwidth required but decode samples were unusable: "
                                + "\(redactedProgressError("\(error)"))"
                        )
                    }
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
        idleGBPerSecond: Double?,
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
        let beginResponse = try worker.beginDecode(seedTokens: seedTokens)
        let statsBeforeDecode = beginResponse.expertStats
        expertStats = beginResponse.expertStats ?? expertStats
        peakRamGB = max(peakRamGB, beginResponse.peakRamGB ?? 0)
        guard let seedToken = beginResponse.seedToken else {
            throw MLXFastError.invalidInput("runtime worker decode_begin response missing seed token")
        }
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: expectedSeedToken,
                actualToken: seedToken
            )
        )

        var actualTokens: [Int] = []
        actualTokens.reserveCapacity(decodeSteps)
        let validationDelayMS = try submissionValidationDelayMilliseconds()
        let session: MactopSession?
        if idleGBPerSecond != nil {
            do {
                session = try MactopSession.start()
                progress?("decode mactop measurement start")
            } catch {
                if requiresMactopHardwareBandwidth() {
                    throw MLXFastError.invalidInput(
                        "mactop hardware bandwidth required but decode measurement could not start: "
                            + "\(redactedProgressError("\(error)"))"
                    )
                }
                progress?(
                    "decode mactop measurement unavailable; using expert streaming byte fallback "
                        + "error=\(redactedProgressError("\(error)"))"
                )
                session = nil
            }
        } else {
            session = nil
        }
        var measuredSeconds = 0.0
        do {
            if validationDelayMS > 0 {
                progress?("decode validation delay enabled milliseconds_per_token=\(validationDelayMS)")
            }
            for decodedStep in 0..<decodeSteps {
                let inputToken = decodedStep == 0 ? expectedSeedToken : expectedTokens[decodedStep - 1]
                let response = try worker.decodeStep(inputToken: inputToken)
                expertStats = response.expertStats ?? expertStats
                peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
                guard let token = response.token, let elapsed = response.seconds else {
                    throw MLXFastError.invalidInput("runtime worker decode_step response missing token or seconds")
                }
                measuredSeconds += elapsed
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
                    total: decodeSteps,
                    intervalSteps: 64,
                    progress: { step, total in
                        progress?("decode measured generated \(step)/\(total) tokens")
                    }
                )
            }
        } catch {
            _ = try? session?.stop()
            throw error
        }

        let bandwidth: (gbPerToken: Double, source: String)
        if let session, let idleGBPerSecond {
            do {
                let samples = try session.stop()
                bandwidth = (
                    try MactopBandwidth.gigabytesPerToken(
                        samples: samples,
                        idleGBPerSecond: idleGBPerSecond,
                        decodeElapsedSeconds: measuredSeconds,
                        decodedTokens: decodeSteps
                    ),
                    "mactop_hardware"
                )
            } catch {
                if requiresMactopHardwareBandwidth() {
                    throw MLXFastError.invalidInput(
                        "mactop hardware bandwidth required but decode samples were unusable: "
                            + "\(redactedProgressError("\(error)"))"
                    )
                }
                progress?(
                    "decode mactop samples unavailable; using expert streaming byte fallback "
                        + "error=\(redactedProgressError("\(error)"))"
                )
                bandwidth = try expertStreamingBandwidthGBPerToken(
                    before: statsBeforeDecode,
                    after: expertStats,
                    decodedTokens: decodeSteps
                )
            }
        } else {
            bandwidth = try expertStreamingBandwidthGBPerToken(
                before: statsBeforeDecode,
                after: expertStats,
                decodedTokens: decodeSteps
            )
        }
        let secondsPerToken = measuredSeconds / Double(decodeSteps)
        let actualTokensComparison = BenchmarkOutputValidator.compareDecodeTokens(
            expectedTokens: Array(expectedTokens.prefix(decodeSteps)),
            actualTokens: actualTokens
        )
        try requireBenchmarkMatch(actualTokensComparison)
        progress?(
            "decode measured complete seconds=\(formatSeconds(measuredSeconds)) "
                + "seconds_per_token=\(formatDouble(secondsPerToken)) "
                + "bandwidth_gb_per_token=\(formatDouble(bandwidth.gbPerToken)) "
                + "bandwidth_source=\(bandwidth.source)"
        )
        return DecodeMeasurement(
            secondsPerToken: secondsPerToken,
            bandwidthGBPerToken: bandwidth.gbPerToken,
            bandwidthSource: bandwidth.source
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

    private static func expertStreamingBandwidthGBPerToken(
        before: ExpertStreamingStats?,
        after: ExpertStreamingStats?,
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

    static func requiresMactopHardwareBandwidth(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let raw = environment["MLXFAST_REQUIRE_MACTOP_BANDWIDTH"] ?? ""
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "1" || normalized == "true" || normalized == "yes"
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
        weightsDigest: DirectoryDigest?,
        gpqaTTFT: GPQATTFTSummary
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
                gpqaTTFTPassed: gpqaTTFT.passed,
                gpqaTTFTPassCount: gpqaTTFT.passCount,
                gpqaTTFTCaseCount: gpqaTTFT.caseCount,
                gpqaTTFTSeconds: gpqaTTFT.meanSeconds,
                gpqaTTFTP50Seconds: gpqaTTFT.p50Seconds,
                gpqaTTFTMaxSeconds: gpqaTTFT.maxSeconds,
                gpqaTTFTSource: gpqaTTFT.source,
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
                expectedToken: nil,
                actualToken: nil,
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
        let ttftSeconds: Double?

        init(
            comparison: CorrectnessTokenComparison,
            expertStats: ExpertStreamingStats,
            peakRamGB: Double,
            ttftSeconds: Double? = nil
        ) {
            self.comparison = comparison
            self.expertStats = expertStats
            self.peakRamGB = peakRamGB
            self.ttftSeconds = ttftSeconds
        }
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

        let response = try worker.teacherForcedCorrectnessBatch(
            promptTokens: testCase.promptTokens,
            expectedTokens: Array(testCase.expectedTokens.prefix(steps)),
            steps: steps
        )
        guard let actualTokens = response.tokens else {
            throw MLXFastError.invalidInput("runtime worker batched teacher-forced response missing tokens")
        }
        guard actualTokens.count == steps else {
            throw MLXFastError.invalidInput(
                "runtime worker batched teacher-forced response returned \(actualTokens.count) tokens; expected \(steps)"
            )
        }
        guard let topLogitRows = response.topLogitRows else {
            throw MLXFastError.invalidInput("runtime worker batched teacher-forced response missing top_logit_rows")
        }
        guard topLogitRows.count == steps else {
            throw MLXFastError.invalidInput(
                "runtime worker batched teacher-forced response returned \(topLogitRows.count) top_logit_rows; expected \(steps)"
            )
        }
        for step in 0..<steps {
            let actualToken = actualTokens[step]
            let topLogits = try validatedWorkerTopLogits(topLogitRows[step], actualToken: actualToken)
            let expectedToken = testCase.expectedTokens[step]
            if !correctnessTokenAccepted(
                expectedToken: expectedToken,
                actualToken: actualToken,
                topLogits: topLogits
            ) {
                return WorkerCorrectnessResult(
                    comparison: CorrectnessTokenComparison(
                        passed: false,
                        checkedSteps: step + 1,
                        firstFailingStep: step,
                        expectedToken: expectedToken,
                        actualToken: actualToken
                    ),
                    expertStats: response.expertStats ?? .zero,
                    peakRamGB: response.peakRamGB ?? 0
                )
            }
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )
        }

        return WorkerCorrectnessResult(
            comparison: CorrectnessTokenComparison(
                passed: true,
                checkedSteps: steps,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            ),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    private static func compareAnchorWithWorker(
        anchor: GoldenAnchorCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        let response = try worker.beginTeacherForcedCorrectness(promptTokens: anchor.contextTokens)
        guard let actualToken = response.token else {
            throw MLXFastError.invalidInput("runtime worker anchor response missing token")
        }
        let topLogits = try validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)
        return WorkerCorrectnessResult(
            comparison: compareAnchorToken(
                anchor: anchor,
                actualToken: actualToken,
                topLogits: topLogits
            ),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    private static func compareFreeRunWithWorker(
        testCase: GoldenFreeRunCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        let response = try worker.generateCorrectness(
            promptTokens: testCase.promptTokens,
            steps: testCase.expectedTokens.count
        )
        guard let generated = response.tokens else {
            throw MLXFastError.invalidInput("runtime worker free-run response missing tokens")
        }
        try requireGeneratedTokenCount(
            generated.count,
            expected: testCase.expectedTokens.count,
            label: "free-run"
        )
        return WorkerCorrectnessResult(
            comparison: compareFreeRunTokens(testCase: testCase, generated: generated),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    private static func compareBehaviorWithWorker(
        testCase: GoldenBehaviorCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        if testCase.maxNewTokens == 1 {
            let response = try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)
            guard let actualToken = response.token else {
                throw MLXFastError.invalidInput("runtime worker behavior response missing token")
            }
            let topLogits = try validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)
            return WorkerCorrectnessResult(
                comparison: compareBehaviorFirstToken(
                    testCase: testCase,
                    actualToken: actualToken,
                    topLogits: topLogits
                ),
                expertStats: response.expertStats ?? .zero,
                peakRamGB: response.peakRamGB ?? 0,
                ttftSeconds: response.seconds
            )
        }

        let response = try worker.generateCorrectness(
            promptTokens: testCase.promptTokens,
            steps: testCase.maxNewTokens
        )
        guard let generated = response.tokens else {
            throw MLXFastError.invalidInput("runtime worker behavior response missing tokens")
        }
        try requireGeneratedTokenCount(
            generated.count,
            expected: testCase.maxNewTokens,
            label: "behavior"
        )
        return WorkerCorrectnessResult(
            comparison: compareBehaviorTokens(testCase: testCase, generated: generated),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    private static func validatedWorkerTopLogits(
        _ topLogits: [CorrectnessTraceLogit]?,
        actualToken: Int
    ) throws -> [CorrectnessTraceLogit] {
        guard let topLogits, !topLogits.isEmpty else {
            throw MLXFastError.invalidInput("runtime worker response missing top_logits")
        }
        guard topLogits.count <= MLXFastConstants.correctnessTopLogits else {
            throw MLXFastError.invalidInput(
                "runtime worker returned \(topLogits.count) top_logits; maximum is \(MLXFastConstants.correctnessTopLogits)"
            )
        }
        guard topLogits[0].token == actualToken else {
            throw MLXFastError.invalidInput("runtime worker top_logits[0] does not match returned token")
        }

        var seen = Set<Int>()
        for (index, item) in topLogits.enumerated() {
            guard item.token >= 0, item.token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput("runtime worker top_logits[\(index)].token is outside vocab")
            }
            guard item.logit.isFinite else {
                throw MLXFastError.invalidInput("runtime worker top_logits[\(index)].logit must be finite")
            }
            guard seen.insert(item.token).inserted else {
                throw MLXFastError.invalidInput("runtime worker top_logits contains duplicate token \(item.token)")
            }
            if index > 0 {
                let previous = topLogits[index - 1]
                guard previous.logit > item.logit
                    || (previous.logit == item.logit && previous.token < item.token)
                else {
                    throw MLXFastError.invalidInput("runtime worker top_logits must be sorted by logit descending")
                }
            }
        }
        return topLogits
    }

    private static func requireGeneratedTokenCount(
        _ actual: Int,
        expected: Int,
        label: String
    ) throws {
        guard actual == expected else {
            throw MLXFastError.invalidInput(
                "runtime worker \(label) returned \(actual) tokens; expected \(expected)"
            )
        }
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

    private static func compareAnchorCached(
        anchor: GoldenAnchorCase,
        weightCache: DeepSeekRuntimeWeightCache
    ) throws -> CorrectnessTokenComparison {
        let cache = DeepSeekModelCache(config: weightCache.config)
        let logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(anchor.contextTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        let actualToken = try DeepSeekCorrectness.greedyToken(from: logits)
        return compareAnchorToken(
            anchor: anchor,
            actualToken: actualToken,
            topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
        )
    }

    private static func compareFreeRunCached(
        testCase: GoldenFreeRunCase,
        weightCache: DeepSeekRuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> CorrectnessTokenComparison {
        let generated = try generateGreedyCached(
            promptTokens: testCase.promptTokens,
            steps: testCase.expectedTokens.count,
            weightCache: weightCache,
            progressIntervalSteps: progressIntervalSteps,
            progress: progress
        )
        return compareFreeRunTokens(testCase: testCase, generated: generated)
    }

    private static func compareBehaviorCached(
        testCase: GoldenBehaviorCase,
        weightCache: DeepSeekRuntimeWeightCache
    ) throws -> CorrectnessTokenComparison {
        if testCase.maxNewTokens == 1 {
            let cache = DeepSeekModelCache(config: weightCache.config)
            let logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray(testCase.promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            let actualToken = try DeepSeekCorrectness.greedyToken(from: logits)
            return compareBehaviorFirstToken(
                testCase: testCase,
                actualToken: actualToken,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
            )
        }

        let generated = try generateGreedyCached(
            promptTokens: testCase.promptTokens,
            steps: testCase.maxNewTokens,
            weightCache: weightCache
        )
        return compareBehaviorTokens(testCase: testCase, generated: generated)
    }

    private static func compareAnchorToken(
        anchor: GoldenAnchorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> CorrectnessTokenComparison {
        if anchorTokenAccepted(anchor: anchor, actualToken: actualToken, topLogits: topLogits) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return CorrectnessTokenComparison(
            passed: false,
            checkedSteps: 1,
            firstFailingStep: 0,
            expectedToken: anchor.expectedToken,
            actualToken: actualToken
        )
    }

    private static func compareFreeRunTokens(
        testCase: GoldenFreeRunCase,
        generated: [Int]
    ) -> CorrectnessTokenComparison {
        let prefixTokens = testCase.exactPrefixTokens ?? testCase.expectedTokens.count
        let comparison = GoldenSequenceMatcher.firstPrefixMismatch(
            expected: testCase.expectedTokens,
            actual: generated,
            prefixTokens: prefixTokens
        )
        return CorrectnessTokenComparison(
            passed: comparison.passed,
            checkedSteps: comparison.passed ? prefixTokens : (comparison.step ?? 0) + 1,
            firstFailingStep: comparison.step,
            expectedToken: comparison.expectedToken,
            actualToken: comparison.actualToken
        )
    }

    private static func compareBehaviorTokens(
        testCase: GoldenBehaviorCase,
        generated: [Int]
    ) -> CorrectnessTokenComparison {
        let comparison = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
            acceptedSequences: testCase.acceptedTokenSequences,
            actual: generated
        )
        return CorrectnessTokenComparison(
            passed: comparison.passed,
            checkedSteps: comparison.passed ? testCase.maxNewTokens : (comparison.step ?? 0) + 1,
            firstFailingStep: comparison.step,
            expectedToken: comparison.expectedToken,
            actualToken: comparison.actualToken
        )
    }

    private static func compareBehaviorFirstToken(
        testCase: GoldenBehaviorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> CorrectnessTokenComparison {
        let acceptedTokens = Set(testCase.acceptedTokenSequences.compactMap(\.first))
        if acceptedTokens.contains(actualToken) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        for acceptedToken in acceptedTokens where correctnessTokenAccepted(
            expectedToken: acceptedToken,
            actualToken: actualToken,
            topLogits: topLogits
        ) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return CorrectnessTokenComparison(
            passed: false,
            checkedSteps: 1,
            firstFailingStep: 0,
            expectedToken: acceptedTokens.sorted().first,
            actualToken: actualToken
        )
    }

    private static func anchorTokenAccepted(
        anchor: GoldenAnchorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> Bool {
        var acceptedTokens = Set(anchor.acceptedTokens ?? [])
        acceptedTokens.insert(anchor.expectedToken)
        if acceptedTokens.contains(actualToken) {
            return true
        }
        for acceptedToken in acceptedTokens where correctnessTokenAccepted(
            expectedToken: acceptedToken,
            actualToken: actualToken,
            topLogits: topLogits
        ) {
            return true
        }
        guard let maxExpectedRank = anchor.maxExpectedRank,
              let topLogits,
              let topLogit = topLogits.first?.logit,
              let expectedIndex = topLogits.firstIndex(where: { $0.token == anchor.expectedToken })
        else {
            return false
        }
        let expectedRank = expectedIndex + 1
        let expectedDelta = topLogit - topLogits[expectedIndex].logit
        let maxDelta = anchor.maxTopLogitDelta ?? MLXFastConstants.correctnessLogitTieTolerance
        return expectedRank <= maxExpectedRank && expectedDelta <= maxDelta
    }

    private static func topLogits(from logits: MLXArray, topK: Int) throws -> [CorrectnessTraceLogit] {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("correctness logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        return try topLogits(fromRows: rows, row: rows.shape[0] - 1, vocabSize: vocabSize, topK: topK)
    }

    private static func topLogits(
        from logits: MLXArray,
        row: Int,
        topK: Int
    ) throws -> [CorrectnessTraceLogit] {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("correctness logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        return try topLogits(fromRows: rows, row: row, vocabSize: vocabSize, topK: topK)
    }

    private static func topLogits(
        fromRows rows: MLXArray,
        row: Int,
        vocabSize: Int,
        topK: Int
    ) throws -> [CorrectnessTraceLogit] {
        guard row >= 0, row < rows.shape[0] else {
            throw MLXFastError.invalidInput("correctness logits row \(row) is outside available rows \(rows.shape[0])")
        }
        let selected = rows[row]
        eval(selected)
        let values = selected.asArray(Float.self).map(Double.init)
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
    let expectedTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let steps: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case expectedTokens = "expected_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
    }
}

private struct RuntimeWorkerState {
    var correctnessCache: DeepSeekModelCache?
    var correctnessPromptTokenCount = 0
    var correctnessStep = 0
    var decodeCache: DeepSeekModelCache?
    var decodeSeedTokenCount = 0
    var decodeStep = 0
}

private struct RuntimeWorkerResponse: Codable {
    let id: Int
    let nonce: String?
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let topLogitRows: [[CorrectnessTraceLogit]]?
    let seedToken: Int?
    let tokens: [Int]?
    let seconds: Double?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?

    init(
        id: Int,
        nonce: String? = nil,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        topLogitRows: [[CorrectnessTraceLogit]]? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        seconds: Double? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil
    ) {
        self.id = id
        self.nonce = nonce
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.topLogitRows = topLogitRows
        self.seedToken = seedToken
        self.tokens = tokens
        self.seconds = seconds
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
    }

    enum CodingKeys: String, CodingKey {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case topLogitRows = "top_logit_rows"
        case seedToken = "seed_token"
        case tokens
        case seconds
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
    }
}

private final class RuntimeWorkerProtocolIO {
    private let input: FileHandle
    private let output: FileHandle

    private init(inputDescriptor: Int32, outputDescriptor: Int32) {
        self.input = FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        self.output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
    }

    static func isolatingStandardIO() throws -> RuntimeWorkerProtocolIO {
        let inputFD = try duplicatePrivateDescriptor(STDIN_FILENO, label: "stdin")
        let outputFD = try duplicatePrivateDescriptor(STDOUT_FILENO, label: "stdout")
        do {
            try redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY, label: "stdin")
            try redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY, label: "stdout")
        } catch {
            close(inputFD)
            close(outputFD)
            throw error
        }
        return RuntimeWorkerProtocolIO(inputDescriptor: inputFD, outputDescriptor: outputFD)
    }

    func readLine() throws -> String? {
        var data = Data()
        while true {
            let byte = input.readData(ofLength: 1)
            if byte.isEmpty {
                if data.isEmpty {
                    return nil
                }
                break
            }
            if byte[byte.startIndex] == 0x0a {
                break
            }
            data.append(byte)
        }
        guard let line = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput("runtime worker received non-UTF8 protocol input")
        }
        return line
    }

    func writeLine(_ data: Data) throws {
        try output.write(contentsOf: data)
        try output.write(contentsOf: Data([0x0a]))
    }
}

private func duplicatePrivateDescriptor(_ descriptor: Int32, label: String) throws -> Int32 {
    let lowerBound = Int32(64 + Int(arc4random_uniform(449)))
    let duplicatedFD = fcntl(descriptor, F_DUPFD_CLOEXEC, lowerBound)
    guard duplicatedFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to duplicate \(label) for protocol I/O")
    }
    return duplicatedFD
}

private func redirectDescriptorToDevNull(_ descriptor: Int32, flags: Int32, label: String) throws {
    let devNullFD = open("/dev/null", flags)
    guard devNullFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to open /dev/null for \(label) redirection")
    }
    defer {
        close(devNullFD)
    }
    guard dup2(devNullFD, descriptor) >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to redirect \(label) away from protocol I/O")
    }
}

private final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let errorOutput: FileHandle
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var sessionNonce = ""
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
            "ANTHROPIC_API_KEY",
            "MLXFAST_CORRECTNESS_GOLDEN_PATH",
            "MLXFAST_CORRECTNESS_GOLDEN_URL",
            "MLXFAST_CORRECTNESS_GOLDEN_AUTH_HEADER",
            "MLXFAST_GPQA_REFERENCE_PATH",
            "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH",
            "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH",
            "MLXFAST_SEMANTIC_GPQA_MODEL",
            "MLXFAST_PRIVATE_DIR",
            "MLXFAST_PRIVATE_PROMPTS_R2_PRESENT",
            "MLXFAST_ANTHROPIC_PRESENT",
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
        let hello = try readResponseLine(validateNonce: false)
        guard hello.id == 0, hello.ok, let nonce = hello.nonce, !nonce.isEmpty else {
            throw MLXFastError.invalidInput("runtime worker did not return a valid protocol hello")
        }
        self.sessionNonce = nonce
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

    func teacherForcedCorrectnessBatch(
        promptTokens: [Int],
        expectedTokens: [Int],
        steps: Int
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_teacher_forced_batch",
            promptTokens: promptTokens,
            expectedTokens: expectedTokens,
            steps: steps
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

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_begin",
            seedTokens: seedTokens
        )
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_step",
            token: inputToken
        )
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        expectedTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil
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
            expectedTokens: expectedTokens,
            token: token,
            seedTokens: seedTokens,
            steps: steps
        )
        var data = try encoder.encode(request)
        data.append(0x0a)
        try input.write(contentsOf: data)

        let response = try readResponseLine(validateNonce: true)
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine(validateNonce: Bool) throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            let response = try decoder.decode(RuntimeWorkerResponse.self, from: data)
            if validateNonce, response.nonce != sessionNonce {
                throw MLXFastError.invalidInput("runtime worker returned a response with an invalid nonce")
            }
            return response
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

private func generateRuntimeWorkerNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeMutableBytes { buffer in
        if let baseAddress = buffer.baseAddress {
            arc4random_buf(baseAddress, buffer.count)
        }
    }
    return bytes
        .map { String(format: "%02x", $0) }
        .joined()
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
