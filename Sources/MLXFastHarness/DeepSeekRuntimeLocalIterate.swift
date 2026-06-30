import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel

extension DeepSeekRuntime {
    public static func localIterate(
        _ options: LocalIterateOptions,
        worker: RuntimeWorkerOptions? = nil
    ) -> ScorePayload {
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: startedAt)
        var correctnessReport: CorrectnessReport?
        var transformedWeightsDigest: DirectoryDigest?
        var validationSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedSeconds = 0.0
        var expertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        let modeName = options.modeName

        progress(
            "\(modeName) start checked_tokens=\(options.benchmarkDecodeSteps + 1) "
                + "decode_steps=\(options.benchmarkDecodeSteps)"
        )

        func failed(
            _ error: String,
            passedCorrectness: Bool = false,
            decodeSecondsPerToken: Double = 0,
            prefillSecondsPerToken: Double = 0,
            bandwidthGBPerToken: Double = 0,
            bandwidthSource: String = ""
        ) -> ScorePayload {
            progress("\(modeName) failed error=\(redactedProgressError(error))")
            return failedScore(
                error: error,
                correctness: correctnessReport,
                passedCorrectness: passedCorrectness,
                expertStats: expertStats,
                weightsDigest: transformedWeightsDigest,
                benchmarkWallSeconds: secondsSince(startedAt),
                preflightSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                bandwidthSource: bandwidthSource,
                runtime: options.runtime
            )
        }

        do {
            try validateLocalIterateOptions(options)

            let validationStart = DispatchTime.now().uptimeNanoseconds
            progress("\(modeName) validation start")
            try checkWorkerBenchmarkInputs(weightsPath: options.weightsPath, goldenPath: options.goldenPath)
            transformedWeightsDigest = try directoryDigest(
                rootPath: options.weightsPath,
                ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
            )
            if let transformedWeightsDigest {
                try enforceTransformedWeightsByteLimit(transformedWeightsDigest.byteCount)
                progress(
                    "\(modeName) weights digest complete files=\(transformedWeightsDigest.fileCount) "
                        + "bytes=\(transformedWeightsDigest.byteCount)"
                )
            }
            let golden = try loadGoldenFixture(
                from: options.goldenPath,
                requiredSteps: options.benchmarkDecodeSteps + 1,
                requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
            )
            guard let localCase = golden.cases.first else {
                throw MLXFastError.invalidInput("\(modeName) public golden must contain at least one case")
            }
            validationSeconds = secondsSince(validationStart)
            progress(
                "\(modeName) validation complete prompt_tokens=\(localCase.promptTokens.count) "
                    + "seconds=\(formatSeconds(validationSeconds))"
            )

            let timedStart = DispatchTime.now().uptimeNanoseconds
            progress("\(modeName) checked timing start")
            let timing: LocalIterateTimingResult
            if let worker {
                timing = try runLocalIterateCheckedTimingWithWorker(
                    weightsPath: options.weightsPath,
                    testCase: localCase,
                    goldenHash: golden.sha256,
                    decodeSteps: options.benchmarkDecodeSteps,
                    modeName: modeName,
                    workerOptions: worker,
                    progress: progress
                )
            } else {
                timing = try runLocalIterateCheckedTiming(
                    weightsPath: options.weightsPath,
                    testCase: localCase,
                    goldenHash: golden.sha256,
                    decodeSteps: options.benchmarkDecodeSteps,
                    modeName: modeName,
                    progress: progress
                )
            }
            timedSeconds = secondsSince(timedStart)
            correctnessSeconds = timedSeconds
            correctnessReport = timing.correctness
            expertStats = timing.expertStats
            peakRamGB = timing.peakRamGB
            progress(
                "\(modeName) checked timing complete passed=\(timing.correctness.passed) "
                    + "checked_steps=\(timing.correctness.checkedSteps) "
                    + "prefill_seconds_per_token=\(formatDouble(timing.prefillSecondsPerToken)) "
                    + "decode_seconds_per_token=\(formatDouble(timing.decode.secondsPerToken)) "
                    + "seconds=\(formatSeconds(timedSeconds))"
            )
            guard timing.correctness.passed else {
                return failed(
                    timing.correctness.error.isEmpty ? "\(modeName) correctness failed" : timing.correctness.error,
                    passedCorrectness: false,
                    decodeSecondsPerToken: timing.decode.secondsPerToken,
                    prefillSecondsPerToken: timing.prefillSecondsPerToken,
                    bandwidthGBPerToken: timing.decode.bandwidthGBPerToken,
                    bandwidthSource: timing.decode.bandwidthSource
                )
            }

            return localIterateScore(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: timing.decode.bandwidthGBPerToken,
                decodeSecondsPerToken: timing.decode.secondsPerToken,
                prefillSecondsPerToken: timing.prefillSecondsPerToken,
                wallSeconds: secondsSince(startedAt),
                validationSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedSeconds: timedSeconds,
                correctness: timing.correctness,
                expertStats: expertStats,
                bandwidthSource: timing.decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest,
                runtime: options.runtime
            )
        } catch {
            return failed("\(error)", passedCorrectness: correctnessReport?.passed == true)
        }
    }

    static func validateLocalIterateOptions(_ options: LocalIterateOptions) throws {
        guard options.benchmarkDecodeSteps > 0 else {
            throw MLXFastError.invalidInput("\(options.modeName) decode steps must be positive")
        }
    }

    struct LocalIterateTimingResult {
        let correctness: CorrectnessReport
        let prefillSecondsPerToken: Double
        let decode: DecodeMeasurement
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
    }

    static func runLocalIterateCheckedTiming(
        weightsPath: String,
        testCase: GoldenCase,
        goldenHash: String,
        decodeSteps: Int,
        modeName: String,
        progress: ((String) -> Void)?
    ) throws -> LocalIterateTimingResult {
        let config = try DeepSeekConfig.load(from: weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(modeName) prompt must not be empty")
        }
        guard testCase.expectedTokens.count > decodeSteps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "need at least \(decodeSteps + 1)"
            )
        }

        let cache = DeepSeekModelCache(config: weightCache.config)
        let prefillStart = DispatchTime.now().uptimeNanoseconds
        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var actualToken = try DeepSeekCorrectness.greedyToken(from: logits)
        let prefillElapsed = secondsSince(prefillStart)
        let expectedSeedToken = testCase.expectedTokens[0]
        var latestStats = DeepSeekRuntime.expertStats(from: weightCache)
        var failureStep: Int?
        var failureExpected: Int?
        var failureActual: Int?
        if !correctnessTokenAccepted(
            expectedToken: expectedSeedToken,
            actualToken: actualToken,
            topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
        ) {
            failureStep = 0
            failureExpected = expectedSeedToken
            failureActual = actualToken
        }
        cache.materializeCachedState()
        let metricsBeforeDecode = weightCache.loader.expertStreamingMetrics?.snapshot()

        let decodeStart = DispatchTime.now().uptimeNanoseconds
        for decodedStep in 0..<decodeSteps {
            let previousToken = testCase.expectedTokens[decodedStep]
            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([previousToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: testCase.promptTokens.count + decodedStep
            )
            actualToken = try DeepSeekCorrectness.greedyToken(from: logits)
            let expectedToken = testCase.expectedTokens[decodedStep + 1]
            if failureStep == nil,
               !correctnessTokenAccepted(
                   expectedToken: expectedToken,
                   actualToken: actualToken,
                   topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
               )
            {
                failureStep = decodedStep + 1
                failureExpected = expectedToken
                failureActual = actualToken
            }
            latestStats = DeepSeekRuntime.expertStats(from: weightCache)
            reportProgress(
                step: decodedStep + 1,
                total: decodeSteps,
                intervalSteps: 8,
                progress: { step, total in
                    progress?("\(modeName) checked decode \(step)/\(total) tokens")
                }
            )
        }

        let decodeElapsed = secondsSince(decodeStart)
        let bandwidth = localIterateBandwidthGBPerToken(
            before: metricsBeforeDecode?.stats,
            after: weightCache.loader.expertStreamingMetrics?.snapshot().stats,
            decodedTokens: decodeSteps
        )
        latestStats = DeepSeekRuntime.expertStats(from: weightCache)
        let correctness = localIterateCorrectnessReport(
            passed: failureStep == nil,
            checkedSteps: failureStep.map { $0 + 1 } ?? decodeSteps + 1,
            caseCount: 1,
            firstFailingStep: failureStep,
            expectedToken: failureExpected,
            actualToken: failureActual,
            goldenHash: goldenHash,
            expertStats: latestStats,
            error: failureStep == nil ? "" : "\(modeName) teacher-forced token mismatch",
            modeName: modeName
        )
        return LocalIterateTimingResult(
            correctness: correctness,
            prefillSecondsPerToken: prefillElapsed / Double(testCase.promptTokens.count),
            decode: DecodeMeasurement(
                secondsPerToken: decodeElapsed / Double(decodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source
            ),
            expertStats: latestStats,
            peakRamGB: Double(Memory.peakMemory) / Double(1 << 30)
        )
    }

    static func runLocalIterateCheckedTimingWithWorker(
        weightsPath: String,
        testCase: GoldenCase,
        goldenHash: String,
        decodeSteps: Int,
        modeName: String,
        workerOptions: RuntimeWorkerOptions,
        progress: ((String) -> Void)?
    ) throws -> LocalIterateTimingResult {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(modeName) prompt must not be empty")
        }
        guard testCase.expectedTokens.count > decodeSteps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "need at least \(decodeSteps + 1)"
            )
        }

        let worker = try RuntimeWorkerClient(options: workerOptions, weightsPath: weightsPath)
        defer {
            worker.close()
        }

        let prefillStart = DispatchTime.now().uptimeNanoseconds
        var response = try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)
        let prefillElapsed = secondsSince(prefillStart)
        var latestStats = response.expertStats ?? .zero
        var peakRamGB = response.peakRamGB ?? 0
        guard var actualToken = response.token else {
            throw MLXFastError.invalidInput("runtime worker \(modeName) prefill response missing token")
        }
        let expectedSeedToken = testCase.expectedTokens[0]
        var failureStep: Int?
        var failureExpected: Int?
        var failureActual: Int?
        if !correctnessTokenAccepted(
            expectedToken: expectedSeedToken,
            actualToken: actualToken,
            topLogits: try validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)
        ) {
            failureStep = 0
            failureExpected = expectedSeedToken
            failureActual = actualToken
        }
        let statsBeforeDecode = response.expertStats

        let decodeStart = DispatchTime.now().uptimeNanoseconds
        for decodedStep in 0..<decodeSteps {
            response = try worker.teacherForcedCorrectnessStep(previousToken: testCase.expectedTokens[decodedStep])
            latestStats = response.expertStats ?? latestStats
            peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
            guard let token = response.token else {
                throw MLXFastError.invalidInput("runtime worker \(modeName) decode response missing token")
            }
            actualToken = token
            let expectedToken = testCase.expectedTokens[decodedStep + 1]
            if failureStep == nil,
               !correctnessTokenAccepted(
                   expectedToken: expectedToken,
                   actualToken: actualToken,
                   topLogits: try validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)
               )
            {
                failureStep = decodedStep + 1
                failureExpected = expectedToken
                failureActual = actualToken
            }
            reportProgress(
                step: decodedStep + 1,
                total: decodeSteps,
                intervalSteps: 8,
                progress: { step, total in
                    progress?("\(modeName) checked decode \(step)/\(total) tokens")
                }
            )
        }
        let decodeElapsed = secondsSince(decodeStart)
        let bandwidth = localIterateBandwidthGBPerToken(
            before: statsBeforeDecode,
            after: latestStats,
            decodedTokens: decodeSteps
        )
        let correctness = localIterateCorrectnessReport(
            passed: failureStep == nil,
            checkedSteps: failureStep.map { $0 + 1 } ?? decodeSteps + 1,
            caseCount: 1,
            firstFailingStep: failureStep,
            expectedToken: failureExpected,
            actualToken: failureActual,
            goldenHash: goldenHash,
            expertStats: latestStats,
            error: failureStep == nil ? "" : "\(modeName) teacher-forced token mismatch",
            modeName: modeName
        )
        return LocalIterateTimingResult(
            correctness: correctness,
            prefillSecondsPerToken: prefillElapsed / Double(testCase.promptTokens.count),
            decode: DecodeMeasurement(
                secondsPerToken: decodeElapsed / Double(decodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source
            ),
            expertStats: latestStats,
            peakRamGB: peakRamGB
        )
    }

    static func localIterateCorrectnessReport(
        passed: Bool,
        checkedSteps: Int,
        caseCount: Int,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        goldenHash: String,
        expertStats: ExpertStreamingStats,
        error: String,
        modeName: String
    ) -> CorrectnessReport {
        CorrectnessReport(
            passed: passed,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: expertStats.cacheHits,
            expertCacheMisses: expertStats.cacheMisses,
            expertCacheEvictions: expertStats.cacheEvictions,
            expertBytesRead: expertStats.bytesRead,
            expertReadSeconds: expertStats.readSeconds,
            expertPeakCachedTensors: expertStats.peakCachedTensors,
            expertHitRate: expertStats.hitRate,
            firstFailingCase: firstFailingStep == nil ? nil : modeName,
            firstFailingStep: firstFailingStep,
            expectedToken: expectedToken,
            actualToken: actualToken,
            goldenHash: goldenHash,
            error: error
        )
    }

    static func localIterateBandwidthGBPerToken(
        before: ExpertStreamingStats?,
        after: ExpertStreamingStats?,
        decodedTokens: Int
    ) -> (gbPerToken: Double, source: String) {
        guard decodedTokens > 0, let after else {
            return (0, "")
        }
        let beforeBytes = before?.bytesRead ?? 0
        let bytesRead = after.bytesRead >= beforeBytes ? after.bytesRead - beforeBytes : after.bytesRead
        guard bytesRead > 0 else {
            return (0, ExpertStreamingMetrics.bandwidthSource)
        }
        return (
            Double(bytesRead) / Double(1 << 30) / Double(decodedTokens),
            ExpertStreamingMetrics.bandwidthSource
        )
    }

    static func localIterateScore(
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        wallSeconds: Double,
        validationSeconds: Double,
        correctnessSeconds: Double,
        timedSeconds: Double,
        correctness: CorrectnessReport,
        expertStats: ExpertStreamingStats,
        bandwidthSource: String,
        weightsDigest: DirectoryDigest?,
        runtime: String
    ) -> ScorePayload {
        ScorePayload(
            score: nil,
            passed: true,
            metrics: ScoreMetrics(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                benchmarkWallSeconds: wallSeconds,
                preflightSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                passedCorrectness: true,
                numLayers: MLXFastConstants.numHiddenLayers,
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
                runtime: runtime
            )
        )
    }
}
