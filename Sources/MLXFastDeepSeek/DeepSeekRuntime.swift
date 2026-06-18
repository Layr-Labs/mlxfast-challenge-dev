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
        case firstFailingCase = "first_failing_case"
        case firstFailingStep = "first_failing_step"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case goldenHash = "golden_hash"
        case error
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(passed, forKey: .passed)
        try container.encode(checkedSteps, forKey: .checkedSteps)
        try container.encode(caseCount, forKey: .caseCount)
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
}

public struct BenchmarkOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String

    public init(weightsPath: String, goldenPath: String) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
    }
}

public enum DeepSeekRuntime {
    public static func runCorrectness(_ options: CorrectnessOptions) throws -> CorrectnessReport {
        do {
            let golden = try loadGoldenFixture(from: options.goldenPath)
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            let loader = try DeepSeekWeightLoader(weightsPath: options.weightsPath)
            let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
            return runCorrectness(
                cases: golden.cases,
                weightCache: weightCache,
                goldenHash: golden.sha256
            )
        } catch {
            return failedCorrectnessReport(checkedSteps: 0, error: "\(error)")
        }
    }

    public static func benchmark(_ options: BenchmarkOptions) -> ScorePayload {
        var correctnessReport: CorrectnessReport?
        do {
            _ = try BenchmarkPreflight.check(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            let golden = try loadGoldenFixture(from: options.goldenPath)
            let config = try DeepSeekConfig.load(from: options.weightsPath)
            let correctnessLoader = try DeepSeekWeightLoader(weightsPath: options.weightsPath)
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

            let benchmarkLoader = try DeepSeekWeightLoader(
                weightsPath: options.weightsPath,
                expertStreamingConfig: ExpertStreamingConfig(recordsMetrics: true)
            )
            let benchmarkCache = DeepSeekRuntimeWeightCache(loader: benchmarkLoader, config: config)
            let promptPlan = try BenchmarkPrompt.plan(from: golden.cases)
            let idleSamples = try MactopSession.measureIdleSamples()
            guard !idleSamples.isEmpty else {
                throw MLXFastError.invalidInput("mactop idle measurement produced no samples")
            }
            let idleGBPerSecond = idleSamples.reduce(0, +) / Double(idleSamples.count)

            Memory.peakMemory = 0
            let prefillSecondsPerToken = try measurePrefillSecondsPerToken(
                promptTokens: promptPlan.prefillTokens,
                weightCache: benchmarkCache
            )
            let decode = try measureDecode(
                seedTokens: promptPlan.decodeSeedTokens,
                weightCache: benchmarkCache,
                idleGBPerSecond: idleGBPerSecond
            )
            let peakRamGB = Double(Memory.peakMemory) / Double(1 << 30)
            let score = peakRamGB
                * decode.bandwidthGBPerToken
                * decode.secondsPerToken
                * prefillSecondsPerToken

            guard score.isFinite, score >= 0 else {
                return failedScore(
                    error: "computed score was not finite",
                    correctness: correctnessReport,
                    passedCorrectness: true
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
        } catch {
            return failedScore(
                error: "\(error)",
                correctness: correctnessReport,
                passedCorrectness: correctnessReport?.passed == true
            )
        }
    }

    private static func runCorrectness(
        cases: [GoldenCase],
        weightCache: DeepSeekRuntimeWeightCache,
        goldenHash: String
    ) -> CorrectnessReport {
        var checkedSteps = 0
        do {
            for testCase in cases {
                let actualTokens = try generateGreedyCached(
                    promptTokens: testCase.promptTokens,
                    steps: MLXFastConstants.correctnessSteps,
                    weightCache: weightCache
                )
                let comparison = DeepSeekCorrectness.compareTokens(
                    expected: testCase.expectedTokens,
                    actual: actualTokens
                )
                if !comparison.passed {
                    return CorrectnessReport(
                        passed: false,
                        checkedSteps: checkedSteps + comparison.checkedSteps,
                        caseCount: cases.count,
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
                goldenHash: goldenHash,
                error: "\(error)"
            )
        }

        return CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: cases.count,
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
            _ = try DeepSeekCorrectness.greedyToken(from: logits)
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
        cache.materializeCachedState()

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
            }

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

    private static func failedScore(
        error: String,
        correctness: CorrectnessReport?,
        passedCorrectness: Bool
    ) -> ScorePayload {
        ScorePayload(
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
                firstFailingLayer: nil,
                firstFailingCase: correctness?.firstFailingCase,
                firstFailingStep: correctness?.firstFailingStep,
                expectedToken: correctness?.expectedToken,
                actualToken: correctness?.actualToken,
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
        guard steps > 0 else {
            return []
        }

        let config = weightCache.config
        let cache = DeepSeekModelCache(config: config)
        var generated: [Int] = []
        generated.reserveCapacity(steps)

        var logits = try DeepSeekModel.logits(
            inputIDs: inputIDsArray(promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var token = try DeepSeekCorrectness.greedyToken(from: logits)
        generated.append(token)

        while generated.count < steps {
            let positionOffset = promptTokens.count + generated.count - 1
            logits = try DeepSeekModel.logits(
                inputIDs: inputIDsArray([token]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: positionOffset
            )
            token = try DeepSeekCorrectness.greedyToken(from: logits)
            generated.append(token)
        }

        return generated
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
        goldenHash: String = "",
        error: String
    ) -> CorrectnessReport {
        CorrectnessReport(
            passed: false,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            firstFailingCase: nil,
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
