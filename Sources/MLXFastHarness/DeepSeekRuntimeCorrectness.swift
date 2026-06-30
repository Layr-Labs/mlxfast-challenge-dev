import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import Tokenizers

// DeepSeekRuntime is split across DeepSeekRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension DeepSeekRuntime {
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
        worker workerOptions: RuntimeWorkerOptions? = nil
    ) throws -> CorrectnessReport {
        var loadedGolden: GoldenFixture?
        do {
            if workerOptions != nil {
                try requireRegularFile(options.weightsPath + "/config.json", description: "transformed config")
                try requireRegularFile(options.goldenPath, description: "correctness golden file")
            }
            let golden = try loadGoldenFixture(from: options.goldenPath)
            loadedGolden = golden
            if let workerOptions {
                let worker = try RuntimeWorkerClient(options: workerOptions, weightsPath: options.weightsPath)
                defer { worker.close() }
                return runLayeredCorrectness(
                    golden: golden,
                    session: worker,
                    steps: MLXFastConstants.correctnessSteps
                ).report
            }
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            let loader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
            )
            let session = InProcessInferenceSession(
                weightCache: DeepSeekRuntimeWeightCache(loader: loader, config: config)
            )
            return runLayeredCorrectness(
                golden: golden,
                session: session,
                steps: MLXFastConstants.correctnessSteps
            ).report
        } catch {
            return failedCorrectnessReport(
                checkedSteps: 0,
                caseCount: loadedGolden?.totalCorrectnessCaseCount ?? 0,
                goldenHash: loadedGolden?.sha256 ?? "",
                expertStats: .zero,
                error: "\(error)"
            )
        }
    }


    struct WorkerLayeredCorrectnessResult {
        let report: CorrectnessReport
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
        let gpqaTTFT: GPQATTFTSummary
    }

    struct GPQATTFTSummary {
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

    static func runLayeredCorrectness(
        golden: GoldenFixture,
        session: RuntimeInferenceSession,
        steps: Int = MLXFastConstants.correctnessSteps,
        semanticCapture: SemanticGPQACaptureOptions? = nil,
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
        var semanticAnswers: [SemanticGPQAAnswerCase] = []

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
            let semanticTokenizer: (any Tokenizer)? = try semanticCapture.map {
                try loadLocalTokenizer(at: $0.tokenizerPath)
            }
            for (caseIndex, testCase) in golden.cases.enumerated() {
                currentCase = testCase.name
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let check = try compareTeacherForced(
                    testCase: testCase,
                    session: session,
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
                let check = try compareAnchor(anchor: anchor, session: session)
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
                let check = try compareFreeRun(testCase: freeRun, session: session)
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
                let check = try compareBehavior(testCase: behavior, session: session)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if check.ttftSeconds != nil {
                    gpqaTTFTCaseCount += 1
                    if check.comparison.passed, let ttftSeconds = check.ttftSeconds, ttftSeconds > 0 {
                        gpqaTTFTPassCount += 1
                        gpqaTTFTSeconds.append(ttftSeconds)
                    }
                }
                if let semanticCapture,
                   let semanticTokenizer,
                   semanticAnswers.count < semanticCapture.caseCount,
                   let generatedTokens = check.generatedTokens,
                   let answer = try semanticAnswerCase(
                       behavior: behavior,
                       generatedTokens: Array(generatedTokens.prefix(semanticCapture.maxNewTokens)),
                       tokenizer: semanticTokenizer,
                       maxNewTokens: semanticCapture.maxNewTokens
                   )
                {
                    semanticAnswers.append(answer)
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
            if let semanticCapture {
                guard semanticAnswers.count == semanticCapture.caseCount else {
                    throw MLXFastError.invalidInput(
                        "captured \(semanticAnswers.count) semantic GPQA answers; expected \(semanticCapture.caseCount)"
                    )
                }
                try writeSemanticGPQAAnswers(semanticAnswers, to: semanticCapture.outputPath)
                progress?("correctness semantic GPQA answers captured cases=\(semanticAnswers.count)")
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

    static func failedCorrectnessReport(
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
