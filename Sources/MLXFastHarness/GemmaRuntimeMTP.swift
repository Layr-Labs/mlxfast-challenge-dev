import Foundation
import MLXFastCore
import MLXFastModel

/// Explicitly experimental target-block probe options. This type is not part
/// of `BenchmarkOptions`, so it cannot silently change the ranked benchmark.
public struct ExperimentalMTPProbeOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let maxBlockSize: Int

    public init(
        weightsPath: String,
        goldenPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalMTPMaxBlockSize
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.maxBlockSize = maxBlockSize
    }
}

/// Diagnostics only. In particular, this schema intentionally has no score or
/// speedup field.
public struct ExperimentalMTPProbeReport: Codable, Equatable {
    public let experimental: Bool
    public let protocolName: String
    public let generator: String
    public let usesTrainedDrafter: Bool
    public let assistantUnavailableReason: String
    public let oracleSource: String
    public let seedTokenCount: Int
    public let decodeTokenCount: Int
    public let maxBlockSize: Int
    public let blockRequestCount: Int
    public let elapsedSeconds: Double
    public let parentMeasuredSecondsPerToken: Double
    public let allTokensMatched: Bool
    public let officialScoreProduced: Bool

    enum CodingKeys: String, CodingKey {
        case experimental
        case protocolName = "protocol"
        case generator
        case usesTrainedDrafter = "uses_trained_drafter"
        case assistantUnavailableReason = "assistant_unavailable_reason"
        case oracleSource = "oracle_source"
        case seedTokenCount = "seed_token_count"
        case decodeTokenCount = "decode_token_count"
        case maxBlockSize = "max_block_size"
        case blockRequestCount = "block_request_count"
        case elapsedSeconds = "elapsed_seconds"
        case parentMeasuredSecondsPerToken = "parent_measured_seconds_per_token"
        case allTokensMatched = "all_tokens_matched"
        case officialScoreProduced = "official_score_produced"
    }
}

struct ExperimentalMTPPromptPlan: Equatable {
    let seedTokens: [Int]
    let expectedSeedToken: Int
    let expectedTokens: [Int]
    let oracleSource: String

    static func make(from golden: GoldenFixture) throws -> ExperimentalMTPPromptPlan {
        let requiredDecodeTokens = MLXFastConstants.experimentalMTPMaxTotalTokens
        if let benchmark = golden.benchmark {
            let benchmarkPlan = try BenchmarkPrompt.plan(from: benchmark)
            guard benchmarkPlan.expectedDecodeTokens.count >= requiredDecodeTokens else {
                throw MLXFastError.invalidInput(
                    "experimental MTP benchmark oracle needs at least "
                        + "\(requiredDecodeTokens) decode tokens"
                )
            }
            return ExperimentalMTPPromptPlan(
                seedTokens: benchmarkPlan.decodeSeedTokens,
                expectedSeedToken: benchmarkPlan.expectedDecodeSeedToken,
                expectedTokens: Array(
                    benchmarkPlan.expectedDecodeTokens.prefix(requiredDecodeTokens)
                ),
                oracleSource: "benchmark_oracle"
            )
        }

        guard let testCase = golden.cases.first else {
            throw MLXFastError.invalidInput(
                "experimental MTP public golden must contain at least one case"
            )
        }
        guard testCase.expectedTokens.count > requiredDecodeTokens else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "experimental MTP needs at least \(requiredDecodeTokens + 1)"
            )
        }
        return ExperimentalMTPPromptPlan(
            seedTokens: testCase.promptTokens,
            expectedSeedToken: testCase.expectedTokens[0],
            expectedTokens: Array(
                testCase.expectedTokens.dropFirst().prefix(requiredDecodeTokens)
            ),
            oracleSource: "first_golden_case"
        )
    }
}

/// Trusted-parent block validator. It commits a block only after its complete
/// shape and ordered token prefix have been checked.
struct ExperimentalMTPBlockValidator {
    let expectedTokens: [Int]
    let totalTokenCount: Int
    let protocolMaxBlockSize: Int
    private(set) var committedTokenCount = 0

    init(
        expectedTokens: [Int],
        totalTokenCount: Int = MLXFastConstants.experimentalMTPMaxTotalTokens,
        protocolMaxBlockSize: Int = MLXFastConstants.experimentalMTPMaxBlockSize
    ) throws {
        guard totalTokenCount > 0,
              totalTokenCount <= MLXFastConstants.experimentalMTPMaxTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP total token count is outside the trusted limit"
            )
        }
        guard protocolMaxBlockSize > 0,
              protocolMaxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP protocol block size is outside the trusted limit"
            )
        }
        guard expectedTokens.count >= totalTokenCount else {
            throw MLXFastError.invalidInput(
                "experimental MTP oracle does not contain enough decode tokens"
            )
        }
        self.expectedTokens = expectedTokens
        self.totalTokenCount = totalTokenCount
        self.protocolMaxBlockSize = protocolMaxBlockSize
    }

    var remainingTokenCount: Int {
        totalTokenCount - committedTokenCount
    }

    mutating func accept(
        _ tokens: [Int],
        requestedMaxBlockSize: Int
    ) throws {
        guard requestedMaxBlockSize > 0,
              requestedMaxBlockSize <= protocolMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP parent issued an invalid block request size"
            )
        }
        guard requestedMaxBlockSize <= remainingTokenCount else {
            throw MLXFastError.invalidInput(
                "experimental MTP parent request would overrun the configured decode length"
            )
        }
        guard !tokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "runtime worker decode_block response returned an empty block"
            )
        }
        guard tokens.count <= requestedMaxBlockSize else {
            throw MLXFastError.invalidInput(
                "runtime worker decode_block response exceeded the requested block size"
            )
        }
        guard tokens.count <= remainingTokenCount else {
            throw MLXFastError.invalidInput(
                "runtime worker decode_block response overran the configured decode length"
            )
        }

        for (blockIndex, token) in tokens.enumerated() {
            guard token >= 0, token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput(
                    "runtime worker decode_block response contained an invalid token"
                )
            }
            let step = committedTokenCount + blockIndex
            let expectedToken = expectedTokens[step]
            guard token == expectedToken else {
                throw BenchmarkTokenMismatchError(
                    comparison: BenchmarkTokenComparison(
                        passed: false,
                        label: "experimental MTP decode token",
                        step: step,
                        expectedToken: expectedToken,
                        actualToken: token
                    )
                )
            }
        }
        committedTokenCount += tokens.count
    }

    func requireComplete() throws {
        guard committedTokenCount == totalTokenCount else {
            throw MLXFastError.invalidInput(
                "experimental MTP decode ended before the configured token count"
            )
        }
    }
}

struct ExperimentalMTPDecodeMeasurement: Equatable {
    let elapsedSeconds: Double
    let secondsPerToken: Double
    let blockRequestCount: Int
}

extension GemmaRuntime {
    public static func experimentalMTPProbe(
        _ options: ExperimentalMTPProbeOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) throws -> ExperimentalMTPProbeReport {
        try validateExperimentalMTPProbeOptions(options)
        try checkWorkerBenchmarkInputs(
            weightsPath: options.weightsPath,
            goldenPath: options.goldenPath
        )
        let golden = try loadGoldenFixture(from: options.goldenPath)
        let plan = try ExperimentalMTPPromptPlan.make(from: golden)

        // Worker startup loads only prompt-independent model state. The first
        // prompt-dependent operation is decode_begin, inside the timer below.
        let worker = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.weightsPath
        )
        defer {
            worker.close()
        }
        let measurement = try measureExperimentalMTPWorkerDecode(
            plan: plan,
            maxBlockSize: options.maxBlockSize,
            worker: worker
        )
        let availability = Gemma4MTPAssistantAvailability.shippedCheckpoint
        return ExperimentalMTPProbeReport(
            experimental: true,
            protocolName: "decode_block_v1",
            generator: "serial_target_fallback",
            usesTrainedDrafter: false,
            assistantUnavailableReason: availability.reason,
            oracleSource: plan.oracleSource,
            seedTokenCount: plan.seedTokens.count,
            decodeTokenCount: MLXFastConstants.experimentalMTPMaxTotalTokens,
            maxBlockSize: options.maxBlockSize,
            blockRequestCount: measurement.blockRequestCount,
            elapsedSeconds: measurement.elapsedSeconds,
            parentMeasuredSecondsPerToken: measurement.secondsPerToken,
            allTokensMatched: true,
            officialScoreProduced: false
        )
    }

    static func validateExperimentalMTPProbeOptions(
        _ options: ExperimentalMTPProbeOptions
    ) throws {
        guard !options.weightsPath.isEmpty, !options.goldenPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "experimental MTP probe requires weights and golden paths"
            )
        }
        guard options.maxBlockSize > 0,
              options.maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP --block-size must be in "
                    + "1...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard MLXFastConstants.experimentalMTPMaxTotalTokens
            == MLXFastConstants.benchmarkDecodeSteps
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP token window changed and must be explicitly rebaselined"
            )
        }
    }

    static func measureExperimentalMTPWorkerDecode(
        plan: ExperimentalMTPPromptPlan,
        maxBlockSize: Int,
        worker: RuntimeWorkerClient
    ) throws -> ExperimentalMTPDecodeMeasurement {
        var validator = try ExperimentalMTPBlockValidator(
            expectedTokens: plan.expectedTokens
        )

        // Charge seed setup, target forwards, future drafting/verification and
        // rollback work, and every protocol serialization round trip.
        let phaseStart = DispatchTime.now().uptimeNanoseconds
        let beginResponse = try worker.beginDecode(seedTokens: plan.seedTokens)
        guard let seedToken = beginResponse.seedToken else {
            throw MLXFastError.invalidInput(
                "runtime worker decode_begin response missing seed token"
            )
        }
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: plan.expectedSeedToken,
                actualToken: seedToken
            )
        )

        var previousCommittedToken = seedToken
        var blockRequestCount = 0
        while validator.remainingTokenCount > 0 {
            let requestedMaxBlockSize = min(
                maxBlockSize,
                validator.remainingTokenCount
            )
            let response = try worker.decodeBlock(
                previousToken: previousCommittedToken,
                maxBlockSize: requestedMaxBlockSize
            )
            guard let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "runtime worker decode_block response missing tokens"
                )
            }
            try validator.accept(
                tokens,
                requestedMaxBlockSize: requestedMaxBlockSize
            )
            previousCommittedToken = tokens[tokens.count - 1]
            blockRequestCount += 1
        }
        try validator.requireComplete()
        let elapsedSeconds = secondsSince(phaseStart)
        let fixedTokenCount = MLXFastConstants.experimentalMTPMaxTotalTokens
        return ExperimentalMTPDecodeMeasurement(
            elapsedSeconds: elapsedSeconds,
            secondsPerToken: elapsedSeconds / Double(fixedTokenCount),
            blockRequestCount: blockRequestCount
        )
    }
}
