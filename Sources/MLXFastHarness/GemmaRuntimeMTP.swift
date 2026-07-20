import Foundation
import MLXFastCore
import MLXFastModel

/// Explicitly experimental target-block probe options. This type is not part
/// of `BenchmarkOptions`, so it cannot silently change the ranked benchmark.
public struct ExperimentalMTPProbeOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let maxBlockSize: Int
    public let totalTokenCount: Int

    public init(
        weightsPath: String,
        goldenPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalMTPMaxBlockSize,
        totalTokenCount: Int = MLXFastConstants.experimentalMTPMaxTotalTokens
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.maxBlockSize = maxBlockSize
        self.totalTokenCount = totalTokenCount
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
    public let seedPrefillSeconds: Double
    public let blockDecodeSeconds: Double
    public let meanBlockRequestSeconds: Double
    public let p50BlockRequestSeconds: Double
    public let maxBlockRequestSeconds: Double
    public let parentMeasuredSecondsPerToken: Double
    public let peakRamGB: Double
    public let mlxActiveMemoryBytes: Int
    public let mlxCacheMemoryBytes: Int
    public let mlxPeakMemoryBytes: Int
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
        case seedPrefillSeconds = "seed_prefill_seconds"
        case blockDecodeSeconds = "block_decode_seconds"
        case meanBlockRequestSeconds = "mean_block_request_seconds"
        case p50BlockRequestSeconds = "p50_block_request_seconds"
        case maxBlockRequestSeconds = "max_block_request_seconds"
        case parentMeasuredSecondsPerToken = "parent_measured_seconds_per_token"
        case peakRamGB = "peak_ram_gb"
        case mlxActiveMemoryBytes = "mlx_active_memory_bytes"
        case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case allTokensMatched = "all_tokens_matched"
        case officialScoreProduced = "official_score_produced"
    }
}

/// Explicit options for the real trained-assistant prototype. This type is not
/// reachable from `BenchmarkOptions`; the serial ranked/local commands cannot
/// opt into MTP through environment state.
public struct ExperimentalTrainedMTPOptions: Equatable {
    public let sourceTargetPath: String
    public let targetWeightsPath: String
    public let assistantPath: String
    public let contractPath: String
    public let goldenPath: String
    public let maxBlockSize: Int
    public let totalTokenCount: Int
    public let verificationMode: Gemma4MTPVerificationMode
    public let requireTrainedAssistant: Bool

    public init(
        sourceTargetPath: String,
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String,
        goldenPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalMTPMaxBlockSize,
        totalTokenCount: Int = MLXFastConstants.experimentalMTPMaxTotalTokens,
        verificationMode: Gemma4MTPVerificationMode = .exactPair,
        requireTrainedAssistant: Bool
    ) {
        self.sourceTargetPath = sourceTargetPath
        self.targetWeightsPath = targetWeightsPath
        self.assistantPath = assistantPath
        self.contractPath = contractPath
        self.goldenPath = goldenPath
        self.maxBlockSize = maxBlockSize
        self.totalTokenCount = totalTokenCount
        self.verificationMode = verificationMode
        self.requireTrainedAssistant = requireTrainedAssistant
    }
}

/// Diagnostics-only Phase 1 report. A reference baseline has not been
/// established, so this schema deliberately cannot carry score or speedup.
public struct ExperimentalTrainedMTPReport: Codable, Equatable {
    public let experimental: Bool
    public let trackID: String
    public let protocolName: String
    public let generator: String
    public let usesTrainedDrafter: Bool
    public let targetModelID: String
    public let targetRevision: String
    public let assistantModelID: String
    public let assistantRevision: String
    public let targetArtifactBytes: Int
    public let targetArtifactSHA256: String
    public let assistantArtifactBytes: Int
    public let assistantArtifactSHA256: String
    public let seedTokenCount: Int
    public let decodeTokenCount: Int
    public let maxBlockSize: Int
    public let targetVerificationMode: String
    public let blockRequestCount: Int
    public let exactPairSegmentCount: Int
    public let exactPairRollbackRowCount: Int
    public let serialVerificationRowCount: Int
    public let proposedDraftTokenCount: Int
    public let acceptedDraftTokenCount: Int
    public let rejectedDraftTokenCount: Int
    public let acceptedDraftRate: Double
    public let rollbackRoundCount: Int
    public let zeroAcceptanceRoundCount: Int
    public let fullAcceptanceRoundCount: Int
    public let targetOnlyTailTokenCount: Int
    public let elapsedSeconds: Double
    public let seedPrefillSeconds: Double
    public let blockDecodeSeconds: Double
    public let meanBlockRequestSeconds: Double
    public let p50BlockRequestSeconds: Double
    public let maxBlockRequestSeconds: Double
    public let parentMeasuredSecondsPerToken: Double
    public let peakRamGB: Double
    public let mlxActiveMemoryBytes: Int
    public let mlxCacheMemoryBytes: Int
    public let mlxPeakMemoryBytes: Int
    public let allTokensMatched: Bool
    public let referenceBaselineStatus: String
    public let officialScoreProduced: Bool

    enum CodingKeys: String, CodingKey {
        case experimental
        case trackID = "track_id"
        case protocolName = "protocol"
        case generator
        case usesTrainedDrafter = "uses_trained_drafter"
        case targetModelID = "target_model_id"
        case targetRevision = "target_revision"
        case assistantModelID = "assistant_model_id"
        case assistantRevision = "assistant_revision"
        case targetArtifactBytes = "target_artifact_bytes"
        case targetArtifactSHA256 = "target_artifact_sha256"
        case assistantArtifactBytes = "assistant_artifact_bytes"
        case assistantArtifactSHA256 = "assistant_artifact_sha256"
        case seedTokenCount = "seed_token_count"
        case decodeTokenCount = "decode_token_count"
        case maxBlockSize = "max_block_size"
        case targetVerificationMode = "target_verification_mode"
        case blockRequestCount = "block_request_count"
        case exactPairSegmentCount = "exact_pair_segment_count"
        case exactPairRollbackRowCount = "exact_pair_rollback_row_count"
        case serialVerificationRowCount = "serial_verification_row_count"
        case proposedDraftTokenCount = "proposed_draft_token_count"
        case acceptedDraftTokenCount = "accepted_draft_token_count"
        case rejectedDraftTokenCount = "rejected_draft_token_count"
        case acceptedDraftRate = "accepted_draft_rate"
        case rollbackRoundCount = "rollback_round_count"
        case zeroAcceptanceRoundCount = "zero_acceptance_round_count"
        case fullAcceptanceRoundCount = "full_acceptance_round_count"
        case targetOnlyTailTokenCount = "target_only_tail_token_count"
        case elapsedSeconds = "elapsed_seconds"
        case seedPrefillSeconds = "seed_prefill_seconds"
        case blockDecodeSeconds = "block_decode_seconds"
        case meanBlockRequestSeconds = "mean_block_request_seconds"
        case p50BlockRequestSeconds = "p50_block_request_seconds"
        case maxBlockRequestSeconds = "max_block_request_seconds"
        case parentMeasuredSecondsPerToken = "parent_measured_seconds_per_token"
        case peakRamGB = "peak_ram_gb"
        case mlxActiveMemoryBytes = "mlx_active_memory_bytes"
        case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case allTokensMatched = "all_tokens_matched"
        case referenceBaselineStatus = "reference_baseline_status"
        case officialScoreProduced = "official_score_produced"
    }
}

struct ExperimentalMTPPromptPlan: Equatable {
    let seedTokens: [Int]
    let expectedSeedToken: Int
    let expectedTokens: [Int]
    let oracleSource: String

    static func make(
        from golden: GoldenFixture,
        totalTokenCount: Int = MLXFastConstants.experimentalMTPMaxTotalTokens
    ) throws -> ExperimentalMTPPromptPlan {
        guard totalTokenCount > 0,
              totalTokenCount
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP total token count is outside the trusted limit"
            )
        }
        let requiredDecodeTokens = totalTokenCount
        if let benchmark = golden.benchmark {
            guard benchmark.expectedDecodeTokens.count >= requiredDecodeTokens else {
                throw MLXFastError.invalidInput(
                    "experimental MTP benchmark oracle needs at least "
                        + "\(requiredDecodeTokens) decode tokens"
                )
            }
            return ExperimentalMTPPromptPlan(
                seedTokens: benchmark.decodeSeedTokens,
                expectedSeedToken: benchmark.expectedDecodeSeedToken,
                expectedTokens: Array(
                    benchmark.expectedDecodeTokens.prefix(requiredDecodeTokens)
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
              totalTokenCount
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
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
    let seedPrefillSeconds: Double
    let blockDecodeSeconds: Double
    let meanBlockRequestSeconds: Double
    let p50BlockRequestSeconds: Double
    let maxBlockRequestSeconds: Double
    let targetVerificationMode: String
    let exactPairSegmentCount: Int
    let exactPairRollbackRowCount: Int
    let serialVerificationRowCount: Int
    let proposedDraftTokenCount: Int
    let acceptedDraftTokenCount: Int
    let rejectedDraftTokenCount: Int
    let rollbackRoundCount: Int
    let zeroAcceptanceRoundCount: Int
    let fullAcceptanceRoundCount: Int
    let targetOnlyTailTokenCount: Int
    let peakRamGB: Double
    let mlxActiveMemoryBytes: Int
    let mlxCacheMemoryBytes: Int
    let mlxPeakMemoryBytes: Int

    init(
        elapsedSeconds: Double,
        secondsPerToken: Double,
        blockRequestCount: Int,
        seedPrefillSeconds: Double = 0,
        blockDecodeSeconds: Double = 0,
        meanBlockRequestSeconds: Double = 0,
        p50BlockRequestSeconds: Double = 0,
        maxBlockRequestSeconds: Double = 0,
        targetVerificationMode: String = "",
        exactPairSegmentCount: Int = 0,
        exactPairRollbackRowCount: Int = 0,
        serialVerificationRowCount: Int = 0,
        proposedDraftTokenCount: Int = 0,
        acceptedDraftTokenCount: Int = 0,
        rejectedDraftTokenCount: Int = 0,
        rollbackRoundCount: Int = 0,
        zeroAcceptanceRoundCount: Int = 0,
        fullAcceptanceRoundCount: Int = 0,
        targetOnlyTailTokenCount: Int = 0,
        peakRamGB: Double = 0,
        mlxActiveMemoryBytes: Int = 0,
        mlxCacheMemoryBytes: Int = 0,
        mlxPeakMemoryBytes: Int = 0
    ) {
        self.elapsedSeconds = elapsedSeconds
        self.secondsPerToken = secondsPerToken
        self.blockRequestCount = blockRequestCount
        self.seedPrefillSeconds = seedPrefillSeconds
        self.blockDecodeSeconds = blockDecodeSeconds
        self.meanBlockRequestSeconds = meanBlockRequestSeconds
        self.p50BlockRequestSeconds = p50BlockRequestSeconds
        self.maxBlockRequestSeconds = maxBlockRequestSeconds
        self.targetVerificationMode = targetVerificationMode
        self.exactPairSegmentCount = exactPairSegmentCount
        self.exactPairRollbackRowCount = exactPairRollbackRowCount
        self.serialVerificationRowCount = serialVerificationRowCount
        self.proposedDraftTokenCount = proposedDraftTokenCount
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.rejectedDraftTokenCount = rejectedDraftTokenCount
        self.rollbackRoundCount = rollbackRoundCount
        self.zeroAcceptanceRoundCount = zeroAcceptanceRoundCount
        self.fullAcceptanceRoundCount = fullAcceptanceRoundCount
        self.targetOnlyTailTokenCount = targetOnlyTailTokenCount
        self.peakRamGB = peakRamGB
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes
        self.mlxCacheMemoryBytes = mlxCacheMemoryBytes
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes
    }
}

struct ExperimentalMTPBlockObservation: Equatable, Sendable {
    let startingLogicalTargetOffset: Int
    let requestedMaxBlockSize: Int
    let returnedTokenCount: Int
}

struct ExperimentalMTPVerificationDelta: Equatable, Sendable {
    let exactPairSegmentCount: Int
    let exactPairRollbackRowCount: Int
    let serialVerificationRowCount: Int
}

struct ExperimentalMTPAcceptanceAccumulator: Equatable {
    private(set) var proposedDraftTokenCount = 0
    private(set) var acceptedDraftTokenCount = 0
    private(set) var rollbackRoundCount = 0
    private(set) var zeroAcceptanceRoundCount = 0
    private(set) var fullAcceptanceRoundCount = 0
    private(set) var targetOnlyTailTokenCount = 0

    mutating func record(
        requestedMaxBlockSize: Int,
        returnedTokenCount: Int
    ) throws {
        guard requestedMaxBlockSize > 0,
              requestedMaxBlockSize
                  <= MLXFastConstants.experimentalMTPMaxBlockSize,
              returnedTokenCount > 0,
              returnedTokenCount <= requestedMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP acceptance accounting received invalid block bounds"
            )
        }
        if requestedMaxBlockSize == 1 {
            targetOnlyTailTokenCount += returnedTokenCount
            return
        }

        let proposed = requestedMaxBlockSize - 1
        let accepted = returnedTokenCount - 1
        proposedDraftTokenCount += proposed
        acceptedDraftTokenCount += accepted
        if accepted == 0 {
            zeroAcceptanceRoundCount += 1
        }
        if accepted == proposed {
            fullAcceptanceRoundCount += 1
        } else {
            rollbackRoundCount += 1
        }
    }

    var rejectedDraftTokenCount: Int {
        proposedDraftTokenCount - acceptedDraftTokenCount
    }
}

extension GemmaRuntime {
    public static func experimentalTrainedMTPBenchmark(
        _ options: ExperimentalTrainedMTPOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) throws -> ExperimentalTrainedMTPReport {
        try validateExperimentalTrainedMTPOptions(options)
        try checkWorkerBenchmarkInputs(
            weightsPath: options.targetWeightsPath,
            goldenPath: options.goldenPath
        )
        _ = try validateExperimentalMTPSourceTarget(
            sourceTargetPath: options.sourceTargetPath,
            contractPath: options.contractPath
        )
        let artifacts = try validateExperimentalMTPArtifacts(
            targetWeightsPath: options.targetWeightsPath,
            assistantPath: options.assistantPath,
            contractPath: options.contractPath
        )
        let golden = try loadGoldenFixture(from: options.goldenPath)
        let plan = try ExperimentalMTPPromptPlan.make(
            from: golden,
            totalTokenCount: options.totalTokenCount
        )

        // Worker startup may load and hash only prompt-independent artifacts.
        // The trusted timer starts before the first prompt-bearing request.
        let worker = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            launch: .trainedMTP(
                assistantPath: options.assistantPath,
                contractPath: options.contractPath,
                verificationMode: options.verificationMode
            )
        )
        defer {
            worker.close()
        }
        let measurement = try measureExperimentalTrainedMTPWorkerDecode(
            plan: plan,
            maxBlockSize: options.maxBlockSize,
            totalTokenCount: options.totalTokenCount,
            verificationMode: options.verificationMode,
            worker: worker
        )
        return ExperimentalTrainedMTPReport(
            experimental: true,
            trackID: artifacts.trackID,
            protocolName: "trusted_mtp_block_v1",
            generator:
                "google_gemma4_trained_mtp_\(measurement.targetVerificationMode)",
            usesTrainedDrafter: true,
            targetModelID: artifacts.targetModelID,
            targetRevision: artifacts.targetRevision,
            assistantModelID: artifacts.assistantModelID,
            assistantRevision: artifacts.assistantRevision,
            targetArtifactBytes: artifacts.targetByteCount,
            targetArtifactSHA256: artifacts.targetTreeSHA256,
            assistantArtifactBytes: artifacts.assistantByteCount,
            assistantArtifactSHA256: artifacts.assistantTreeSHA256,
            seedTokenCount: plan.seedTokens.count,
            decodeTokenCount: options.totalTokenCount,
            maxBlockSize: options.maxBlockSize,
            targetVerificationMode: measurement.targetVerificationMode,
            blockRequestCount: measurement.blockRequestCount,
            exactPairSegmentCount: measurement.exactPairSegmentCount,
            exactPairRollbackRowCount: measurement.exactPairRollbackRowCount,
            serialVerificationRowCount: measurement.serialVerificationRowCount,
            proposedDraftTokenCount: measurement.proposedDraftTokenCount,
            acceptedDraftTokenCount: measurement.acceptedDraftTokenCount,
            rejectedDraftTokenCount: measurement.rejectedDraftTokenCount,
            acceptedDraftRate: measurement.proposedDraftTokenCount > 0
                ? Double(measurement.acceptedDraftTokenCount)
                    / Double(measurement.proposedDraftTokenCount)
                : 0,
            rollbackRoundCount: measurement.rollbackRoundCount,
            zeroAcceptanceRoundCount: measurement.zeroAcceptanceRoundCount,
            fullAcceptanceRoundCount: measurement.fullAcceptanceRoundCount,
            targetOnlyTailTokenCount: measurement.targetOnlyTailTokenCount,
            elapsedSeconds: measurement.elapsedSeconds,
            seedPrefillSeconds: measurement.seedPrefillSeconds,
            blockDecodeSeconds: measurement.blockDecodeSeconds,
            meanBlockRequestSeconds: measurement.meanBlockRequestSeconds,
            p50BlockRequestSeconds: measurement.p50BlockRequestSeconds,
            maxBlockRequestSeconds: measurement.maxBlockRequestSeconds,
            parentMeasuredSecondsPerToken: measurement.secondsPerToken,
            peakRamGB: measurement.peakRamGB,
            mlxActiveMemoryBytes: measurement.mlxActiveMemoryBytes,
            mlxCacheMemoryBytes: measurement.mlxCacheMemoryBytes,
            mlxPeakMemoryBytes: measurement.mlxPeakMemoryBytes,
            allTokensMatched: true,
            referenceBaselineStatus: artifacts.referenceBaselineStatus,
            // This CLI invocation never emits the official score even on the
            // enabled track: the trusted workflow computes the paired score
            // from measure-mtp-job's sealed results, never from this report.
            officialScoreProduced: false
        )
    }

    static func validateExperimentalTrainedMTPOptions(
        _ options: ExperimentalTrainedMTPOptions
    ) throws {
        guard !options.sourceTargetPath.isEmpty,
              !options.targetWeightsPath.isEmpty,
              !options.assistantPath.isEmpty,
              !options.contractPath.isEmpty,
              !options.goldenPath.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "trained MTP benchmark requires source target, transformed target, "
                    + "assistant, contract, and golden paths"
            )
        }
        guard options.requireTrainedAssistant else {
            throw MLXFastError.invalidInput(
                "trained MTP benchmark requires --require-trained-assistant; "
                    + "use mtp-probe for the serial control"
            )
        }
        // K=1 would never draft, so the trained-assistant report fields
        // (generator, uses_trained_drafter) would be dishonest. The serial
        // K=1 control is mtp-probe; block-size one remains valid only for
        // the session's internal final-token tails.
        guard options.maxBlockSize >= 2,
              options.maxBlockSize
                  <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "trained MTP --block-size must be in "
                    + "2...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard options.totalTokenCount > 0,
              options.totalTokenCount
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "trained MTP --tokens must be in 1..."
                    + "\(MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens)"
            )
        }
    }

    static func experimentalMTPVerificationDeltas(
        for observation: ExperimentalMTPBlockObservation,
        verificationMode: Gemma4MTPVerificationMode
    ) throws -> [ExperimentalMTPVerificationDelta] {
        let k = observation.requestedMaxBlockSize
        let c = observation.returnedTokenCount
        guard observation.startingLogicalTargetOffset >= 0,
              (1...MLXFastConstants.experimentalMTPMaxBlockSize).contains(k),
              (1...k).contains(c)
        else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics contain an invalid block observation"
            )
        }

        if verificationMode == .serial {
            return [
                ExperimentalMTPVerificationDelta(
                    exactPairSegmentCount: 0,
                    exactPairRollbackRowCount: 0,
                    serialVerificationRowCount: c
                )
            ]
        }

        func delta(_ segments: Int, _ rollbacks: Int, _ serial: Int)
            -> ExperimentalMTPVerificationDelta
        {
            ExperimentalMTPVerificationDelta(
                exactPairSegmentCount: segments,
                exactPairRollbackRowCount: rollbacks,
                serialVerificationRowCount: serial
            )
        }

        let serialFallback = [delta(0, 0, c)]
        if k == 1 {
            return serialFallback
        }

        // The ranked track's rotating cache has a 1,024-token window.
        // canAppendExactRows(2) requires offset + 2 <= 1,024 before the first
        // wrap; direct-four/two-pair composition requires offset + 4 <= 1,024.
        // Engine ineligibility is fail-closed, so geometry is the only legal
        // exact-pair-mode serial fallback.
        let offset = observation.startingLogicalTargetOffset
        let firstPairFits = offset <= 1_022
        let fullFourRowsFit = offset <= 1_020
        if !firstPairFits {
            return serialFallback
        }

        switch (k, c) {
        case (2, 1), (3, 1):
            return [delta(1, 1, 0)]
        case (2, 2), (3, 2):
            return [delta(1, 0, 0)]
        case (3, 3):
            return [delta(1, 0, 1)]
        case (4, 1):
            return fullFourRowsFit
                ? [delta(1, 1, 0), delta(2, 3, 0)]
                : [delta(1, 1, 0)]
        case (4, 2):
            return fullFourRowsFit
                ? [delta(1, 0, 0), delta(2, 2, 0)]
                : [delta(1, 0, 0)]
        case (4, 3):
            return fullFourRowsFit
                ? [delta(2, 1, 0)]
                : [delta(1, 0, 1)]
        case (4, 4):
            return fullFourRowsFit
                ? [delta(2, 0, 0)]
                : [delta(1, 0, 2)]
        default:
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics contain an unsupported block observation"
            )
        }
    }

    static func validateExperimentalMTPVerificationDiagnostics(
        totalTokenCount: Int,
        verificationMode: Gemma4MTPVerificationMode,
        observedBlocks: [ExperimentalMTPBlockObservation],
        exactPairSegmentCount: Int,
        exactPairRollbackRowCount: Int,
        serialVerificationRowCount: Int
    ) throws {
        guard totalTokenCount > 0,
              totalTokenCount
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics received an invalid token total"
            )
        }

        // Cheap overflow-safe bounds and the exact physical-row equation run
        // before allocating the bounded reachability workspace.
        let maximumPairSegments = 2 * totalTokenCount
        let maximumRollbackRows = 3 * totalTokenCount
        guard (0...maximumPairSegments).contains(exactPairSegmentCount),
              (0...maximumRollbackRows).contains(exactPairRollbackRowCount),
              (0...totalTokenCount).contains(serialVerificationRowCount)
        else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics returned invalid verification data"
            )
        }
        let maximumRollbackRowsForSegments =
            exactPairSegmentCount + exactPairSegmentCount / 2
        guard exactPairRollbackRowCount <= maximumRollbackRowsForSegments else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics returned impossible rollback accounting"
            )
        }
        let physicalRowCount = 2 * exactPairSegmentCount
            - exactPairRollbackRowCount
            + serialVerificationRowCount
        guard physicalRowCount == totalTokenCount else {
            throw MLXFastError.invalidInput(
                "trained MTP verification rows (\(physicalRowCount)) diverged "
                    + "from the configured decode total (\(totalTokenCount))"
            )
        }

        // The observations are trusted parent history. Enforce final-request
        // truncation and prepare the exact per-block alternatives.
        guard !observedBlocks.isEmpty,
              observedBlocks.count <= totalTokenCount
        else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics have invalid block history length"
            )
        }
        var remainingTokenCount = totalTokenCount
        var expectedStartingOffset =
            observedBlocks[0].startingLogicalTargetOffset
        var alternatives: [[ExperimentalMTPVerificationDelta]] = []
        alternatives.reserveCapacity(observedBlocks.count)
        for observation in observedBlocks {
            guard observation.startingLogicalTargetOffset == expectedStartingOffset,
                  observation.requestedMaxBlockSize <= remainingTokenCount
            else {
                throw MLXFastError.invalidInput(
                    "trained MTP block history violates parent offset or truncation"
                )
            }
            let deltas = try experimentalMTPVerificationDeltas(
                for: observation,
                verificationMode: verificationMode
            )
            alternatives.append(deltas)
            remainingTokenCount -= observation.returnedTokenCount
            let (nextOffset, offsetOverflow) =
                expectedStartingOffset.addingReportingOverflow(
                    observation.returnedTokenCount
                )
            guard !offsetOverflow else {
                throw MLXFastError.invalidInput(
                    "trained MTP block history offset overflowed"
                )
            }
            expectedStartingOffset = nextOffset
        }
        guard remainingTokenCount == 0 else {
            throw MLXFastError.invalidInput(
                "trained MTP block history does not reach the configured total"
            )
        }

        // Every legal delta satisfies c = 2*S - R + V. Since the observed c
        // values sum to totalTokenCount and the reported tuple passed the same
        // equation, matching (S,R) also proves the exact reported V.
        //
        // Bitset DP complexity:
        // O(blocks * alternatives * targetS * ceil((targetR + 1) / 64))
        // time and O(targetS * ceil((targetR + 1) / 64)) memory. At the
        // protocol caps this is about 40M UInt64 operations and two 205 KiB
        // buffers, allocated only after timing has stopped.
        let targetSegments = exactPairSegmentCount
        let targetRollbacks = exactPairRollbackRowCount
        let wordsPerRow = targetRollbacks / 64 + 1
        let rowCount = targetSegments + 1
        let wordCount = rowCount * wordsPerRow
        var reachable = [UInt64](repeating: 0, count: wordCount)
        var next = [UInt64](repeating: 0, count: wordCount)
        reachable[0] = 1
        let validBitsInLastWord = targetRollbacks % 64 + 1
        let lastWordMask =
            validBitsInLastWord == 64
            ? UInt64.max
            : (UInt64(1) << UInt64(validBitsInLastWord)) - 1

        for blockAlternatives in alternatives {
            for index in next.indices {
                next[index] = 0
            }
            for delta in blockAlternatives {
                let segmentDelta = delta.exactPairSegmentCount
                let rollbackDelta = delta.exactPairRollbackRowCount
                guard segmentDelta <= targetSegments,
                      rollbackDelta <= targetRollbacks
                else {
                    continue
                }
                let rollbackWordShift = rollbackDelta / 64
                let rollbackBitShift = rollbackDelta % 64
                for sourceSegments in 0...(targetSegments - segmentDelta) {
                    let sourceBase = sourceSegments * wordsPerRow
                    let destinationBase =
                        (sourceSegments + segmentDelta) * wordsPerRow
                    for sourceWord in 0..<wordsPerRow {
                        let bits = reachable[sourceBase + sourceWord]
                        if bits == 0 { continue }
                        let destinationWord = sourceWord + rollbackWordShift
                        if destinationWord >= wordsPerRow { continue }
                        next[destinationBase + destinationWord] |=
                            bits << UInt64(rollbackBitShift)
                        if rollbackBitShift > 0,
                           destinationWord + 1 < wordsPerRow
                        {
                            next[destinationBase + destinationWord + 1] |=
                                bits >> UInt64(64 - rollbackBitShift)
                        }
                    }
                }
            }
            if lastWordMask != UInt64.max {
                for segments in 0...targetSegments {
                    next[(segments + 1) * wordsPerRow - 1] &= lastWordMask
                }
            }
            guard next.contains(where: { $0 != 0 }) else {
                throw MLXFastError.invalidInput(
                    "trained MTP diagnostics are unreachable from block history"
                )
            }
            swap(&reachable, &next)
        }

        let targetWord = targetRollbacks / 64
        let targetBit = UInt64(1) << UInt64(targetRollbacks % 64)
        guard reachable[targetSegments * wordsPerRow + targetWord] & targetBit != 0 else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics are unreachable from block history"
            )
        }
    }

    static func measureExperimentalTrainedMTPWorkerDecode(
        plan: ExperimentalMTPPromptPlan,
        maxBlockSize: Int,
        totalTokenCount: Int,
        verificationMode: Gemma4MTPVerificationMode,
        worker: RuntimeWorkerClient
    ) throws -> ExperimentalMTPDecodeMeasurement {
        var validator = try ExperimentalMTPBlockValidator(
            expectedTokens: plan.expectedTokens,
            totalTokenCount: totalTokenCount
        )
        // Reserve trusted bookkeeping before timing. Appends remain inside the
        // measured parent loop, but cannot grow either backing allocation.
        var observedBlocks: [ExperimentalMTPBlockObservation] = []
        observedBlocks.reserveCapacity(totalTokenCount)
        var blockRequestSeconds: [Double] = []
        blockRequestSeconds.reserveCapacity(totalTokenCount)

        // This is the only timing authority. It charges prefill, drafting,
        // multi-position target verification, rejection/rollback, IPC and all
        // final-short-block work. The denominator is the configured,
        // oracle-bounded parent-owned total, never a worker-reported count or
        // duration.
        let phaseStart = DispatchTime.now().uptimeNanoseconds
        let beginResponse = try worker.beginTrainedMTPDecode(
            seedTokens: plan.seedTokens
        )
        guard let seedToken = beginResponse.seedToken else {
            throw MLXFastError.invalidInput(
                "trained MTP worker begin response is missing the target seed token"
            )
        }
        try requireBenchmarkMatch(
            BenchmarkOutputValidator.compareDecodeSeedToken(
                expectedToken: plan.expectedSeedToken,
                actualToken: seedToken
            )
        )
        let seedPrefillSeconds = secondsSince(phaseStart)

        var previousCommittedToken = seedToken
        var blockRequestCount = 0
        var acceptance = ExperimentalMTPAcceptanceAccumulator()
        while validator.remainingTokenCount > 0 {
            let requestedMaxBlockSize = min(
                maxBlockSize,
                validator.remainingTokenCount
            )
            // Seed prefill has already materialized seedTokens.count target
            // rows. Every previously returned token commits exactly one more.
            let startingLogicalTargetOffset =
                plan.seedTokens.count
                + (totalTokenCount - validator.remainingTokenCount)
            let blockStart = DispatchTime.now().uptimeNanoseconds
            let response = try worker.trainedMTPDecodeBlock(
                previousToken: previousCommittedToken,
                maxBlockSize: requestedMaxBlockSize
            )
            blockRequestSeconds.append(secondsSince(blockStart))
            guard let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "trained MTP worker block response is missing tokens"
                )
            }
            try validator.accept(
                tokens,
                requestedMaxBlockSize: requestedMaxBlockSize
            )
            observedBlocks.append(
                ExperimentalMTPBlockObservation(
                    startingLogicalTargetOffset: startingLogicalTargetOffset,
                    requestedMaxBlockSize: requestedMaxBlockSize,
                    returnedTokenCount: tokens.count
                )
            )
            try acceptance.record(
                requestedMaxBlockSize: requestedMaxBlockSize,
                returnedTokenCount: tokens.count
            )
            previousCommittedToken = tokens[tokens.count - 1]
            blockRequestCount += 1
        }
        try validator.requireComplete()
        let elapsedSeconds = secondsSince(phaseStart)
        let blockDecodeSeconds = elapsedSeconds - seedPrefillSeconds
        let sortedBlockSeconds = blockRequestSeconds.sorted()
        let meanBlockRequestSeconds = blockRequestSeconds.isEmpty
            ? 0
            : blockRequestSeconds.reduce(0, +)
                / Double(blockRequestSeconds.count)
        let p50BlockRequestSeconds = sortedBlockSeconds.isEmpty
            ? 0
            : sortedBlockSeconds[sortedBlockSeconds.count / 2]
        let maxBlockRequestSeconds = sortedBlockSeconds.last ?? 0
        let diagnostics = try worker.trainedMTPDiagnostics()
        let peakRamGB = diagnostics.peakRamGB ?? 0
        let mlxActiveMemoryBytes = diagnostics.mlxActiveMemoryBytes ?? 0
        let mlxCacheMemoryBytes = diagnostics.mlxCacheMemoryBytes ?? 0
        let mlxPeakMemoryBytes = diagnostics.mlxPeakMemoryBytes ?? 0
        let targetVerificationMode = diagnostics.targetVerificationMode ?? ""
        let exactPairSegmentCount = diagnostics.exactPairSegmentCount ?? -1
        let exactPairRollbackRowCount =
            diagnostics.exactPairRollbackRowCount ?? -1
        let serialVerificationRowCount =
            diagnostics.serialVerificationRowCount ?? -1
        guard peakRamGB >= 0,
              mlxActiveMemoryBytes >= 0,
              mlxCacheMemoryBytes >= 0,
              mlxPeakMemoryBytes >= 0,
              targetVerificationMode == verificationMode.rawValue
        else {
            throw MLXFastError.invalidInput(
                "trained MTP diagnostics returned invalid verification or memory data"
            )
        }
        try validateExperimentalMTPVerificationDiagnostics(
            totalTokenCount: totalTokenCount,
            verificationMode: verificationMode,
            observedBlocks: observedBlocks,
            exactPairSegmentCount: exactPairSegmentCount,
            exactPairRollbackRowCount: exactPairRollbackRowCount,
            serialVerificationRowCount: serialVerificationRowCount
        )
        return ExperimentalMTPDecodeMeasurement(
            elapsedSeconds: elapsedSeconds,
            secondsPerToken: elapsedSeconds
                / Double(totalTokenCount),
            blockRequestCount: blockRequestCount,
            seedPrefillSeconds: seedPrefillSeconds,
            blockDecodeSeconds: blockDecodeSeconds,
            meanBlockRequestSeconds: meanBlockRequestSeconds,
            p50BlockRequestSeconds: p50BlockRequestSeconds,
            maxBlockRequestSeconds: maxBlockRequestSeconds,
            targetVerificationMode: targetVerificationMode,
            exactPairSegmentCount: exactPairSegmentCount,
            exactPairRollbackRowCount: exactPairRollbackRowCount,
            serialVerificationRowCount: serialVerificationRowCount,
            proposedDraftTokenCount: acceptance.proposedDraftTokenCount,
            acceptedDraftTokenCount: acceptance.acceptedDraftTokenCount,
            rejectedDraftTokenCount: acceptance.rejectedDraftTokenCount,
            rollbackRoundCount: acceptance.rollbackRoundCount,
            zeroAcceptanceRoundCount: acceptance.zeroAcceptanceRoundCount,
            fullAcceptanceRoundCount: acceptance.fullAcceptanceRoundCount,
            targetOnlyTailTokenCount: acceptance.targetOnlyTailTokenCount,
            peakRamGB: peakRamGB,
            mlxActiveMemoryBytes: mlxActiveMemoryBytes,
            mlxCacheMemoryBytes: mlxCacheMemoryBytes,
            mlxPeakMemoryBytes: mlxPeakMemoryBytes
        )
    }

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
        let plan = try ExperimentalMTPPromptPlan.make(
            from: golden,
            totalTokenCount: options.totalTokenCount
        )

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
            totalTokenCount: options.totalTokenCount,
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
            decodeTokenCount: options.totalTokenCount,
            maxBlockSize: options.maxBlockSize,
            blockRequestCount: measurement.blockRequestCount,
            elapsedSeconds: measurement.elapsedSeconds,
            seedPrefillSeconds: measurement.seedPrefillSeconds,
            blockDecodeSeconds: measurement.blockDecodeSeconds,
            meanBlockRequestSeconds: measurement.meanBlockRequestSeconds,
            p50BlockRequestSeconds: measurement.p50BlockRequestSeconds,
            maxBlockRequestSeconds: measurement.maxBlockRequestSeconds,
            parentMeasuredSecondsPerToken: measurement.secondsPerToken,
            peakRamGB: measurement.peakRamGB,
            mlxActiveMemoryBytes: measurement.mlxActiveMemoryBytes,
            mlxCacheMemoryBytes: measurement.mlxCacheMemoryBytes,
            mlxPeakMemoryBytes: measurement.mlxPeakMemoryBytes,
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
        guard options.totalTokenCount > 0,
              options.totalTokenCount
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "experimental MTP --tokens must be in 1..."
                    + "\(MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens)"
            )
        }
    }

    static func measureExperimentalMTPWorkerDecode(
        plan: ExperimentalMTPPromptPlan,
        maxBlockSize: Int,
        totalTokenCount: Int,
        worker: RuntimeWorkerClient
    ) throws -> ExperimentalMTPDecodeMeasurement {
        var validator = try ExperimentalMTPBlockValidator(
            expectedTokens: plan.expectedTokens,
            totalTokenCount: totalTokenCount
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
        let seedPrefillSeconds = secondsSince(phaseStart)

        var previousCommittedToken = seedToken
        var blockRequestCount = 0
        var blockRequestSeconds: [Double] = []
        while validator.remainingTokenCount > 0 {
            let requestedMaxBlockSize = min(
                maxBlockSize,
                validator.remainingTokenCount
            )
            let blockStart = DispatchTime.now().uptimeNanoseconds
            let response = try worker.decodeBlock(
                previousToken: previousCommittedToken,
                maxBlockSize: requestedMaxBlockSize
            )
            blockRequestSeconds.append(secondsSince(blockStart))
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
        let blockDecodeSeconds = elapsedSeconds - seedPrefillSeconds
        let sortedBlockSeconds = blockRequestSeconds.sorted()
        let meanBlockRequestSeconds = blockRequestSeconds.isEmpty
            ? 0
            : blockRequestSeconds.reduce(0, +)
                / Double(blockRequestSeconds.count)
        let p50BlockRequestSeconds = sortedBlockSeconds.isEmpty
            ? 0
            : sortedBlockSeconds[sortedBlockSeconds.count / 2]
        let maxBlockRequestSeconds = sortedBlockSeconds.last ?? 0
        let diagnostics = try worker.phaseDiagnostics()
        return ExperimentalMTPDecodeMeasurement(
            elapsedSeconds: elapsedSeconds,
            secondsPerToken: elapsedSeconds / Double(totalTokenCount),
            blockRequestCount: blockRequestCount,
            seedPrefillSeconds: seedPrefillSeconds,
            blockDecodeSeconds: blockDecodeSeconds,
            meanBlockRequestSeconds: meanBlockRequestSeconds,
            p50BlockRequestSeconds: p50BlockRequestSeconds,
            maxBlockRequestSeconds: maxBlockRequestSeconds,
            peakRamGB: diagnostics.peakRamGB ?? 0,
            mlxActiveMemoryBytes: diagnostics.mlxActiveMemoryBytes ?? 0,
            mlxCacheMemoryBytes: diagnostics.mlxCacheMemoryBytes ?? 0,
            mlxPeakMemoryBytes: diagnostics.mlxPeakMemoryBytes ?? 0
        )
    }
}
