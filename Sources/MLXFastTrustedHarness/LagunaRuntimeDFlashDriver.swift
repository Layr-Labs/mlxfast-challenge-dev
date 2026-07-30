import Foundation
import MLXFastCore

// Trusted-parent driver for the DFlash block-decode track. Links no MLX: it
// spawns the sandboxed worker, owns the timer and the block schedule, and feeds
// every emitted row to the Criterion E validator.

/// On-disk reference material for one DFlash prompt.
///
/// This is the artifact the pinned-baseline reference verifier (contract layer
/// L1) produces: for each decode position, the reference's K=1 sequential
/// argmax plus -- where a declared-frame replay was performed -- the reference's
/// argmax at that block width, and the reference top-2 readouts used for work
/// binding. Keeping it a file keeps the trusted parent free of MLX.
public struct DFlashReferenceGolden: Codable {
    public struct Row: Codable {
        public let sequentialArgmax: Int
        public let declaredFrameArgmax: [String: Int]?
        public let top2Tokens: [Int]?
        public let top2Logits: [Double]?

        enum CodingKeys: String, CodingKey {
            case sequentialArgmax = "sequential_argmax"
            case declaredFrameArgmax = "declared_frame_argmax"
            case top2Tokens = "top2_tokens"
            case top2Logits = "top2_logits"
        }
    }

    public let seedTokens: [Int]
    public let expectedSeedToken: Int
    public let rows: [Row]
    /// Set by the reference generator once its self-consistency replay passed
    /// (contract layer L1 requirement R5). A run must refuse to score against a
    /// golden that never proved reference determinism.
    public let referenceSelfConsistent: Bool?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case expectedSeedToken = "expected_seed_token"
        case rows
        case referenceSelfConsistent = "reference_self_consistent"
    }
}

/// Reference oracle backed by a pinned-baseline-generated golden file.
public struct DFlashGoldenReferenceOracle: DFlashReferenceOracle {
    private let rows: [DFlashReferenceGolden.Row]

    public init(golden: DFlashReferenceGolden) {
        self.rows = golden.rows
    }

    public func referenceRows(
        emittedPrefix: [Int],
        startOffset: Int,
        count: Int,
        declaredBlockWidth: Int
    ) throws -> [DFlashReferenceRow] {
        guard startOffset >= 0, count >= 0 else {
            throw MLXFastError.invalidInput(
                "DFlash reference request has a negative range"
            )
        }
        guard startOffset + count <= rows.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference golden covers \(rows.count) rows; run reached "
                    + "\(startOffset + count)"
            )
        }
        return (0 ..< count).map { index in
            let row = rows[startOffset + index]
            return DFlashReferenceRow(
                sequentialArgmax: row.sequentialArgmax,
                declaredFrameArgmax: row.declaredFrameArgmax?[
                    String(declaredBlockWidth)
                ],
                top2Tokens: row.top2Tokens ?? [],
                top2Logits: row.top2Logits ?? []
            )
        }
    }
}

extension LagunaRuntime {
    /// Run a validated, parent-timed DFlash block-decode measurement.
    ///
    /// Timing is parent-owned end to end: the clock starts after the seed
    /// prefill response and stops when the configured token total is committed,
    /// and the denominator is that configured total -- never a worker-reported
    /// count.
    public static func experimentalDFlashBenchmark(
        options: ExperimentalDFlashOptions,
        workerOptions: RuntimeWorkerOptions,
        scheduleSeed: UInt64,
        tolerance: DFlashWorkBindingTolerance = DFlashWorkBindingTolerance()
    ) throws -> ExperimentalDFlashReport {
        let goldenData = try Data(
            contentsOf: URL(fileURLWithPath: options.goldenPath)
        )
        let golden = try JSONDecoder().decode(
            DFlashReferenceGolden.self,
            from: goldenData
        )
        guard golden.referenceSelfConsistent != false else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                detail: "reference golden reports a failed self-consistency "
                    + "replay; this is an operator fault, not a submission fault"
            )
        }
        guard options.totalTokenCount > 0,
              options.totalTokenCount <= golden.rows.count
        else {
            throw MLXFastError.invalidInput(
                "DFlash golden carries \(golden.rows.count) reference rows but "
                    + "\(options.totalTokenCount) tokens were requested"
            )
        }

        let oracle = DFlashGoldenReferenceOracle(golden: golden)
        let validator = LagunaDFlashBlockValidator(
            oracle: oracle,
            seedTokenCount: golden.seedTokens.count,
            totalTokenCount: options.totalTokenCount,
            tolerance: tolerance
        )
        var schedule = DFlashBlockSchedule(
            seed: scheduleSeed,
            maxBlockSize: options.maxBlockSize
        )

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            dflashDrafterPath: options.drafterPath
        )
        defer { client.close() }

        // Seed prefill is untimed here only in the sense that it is charged to
        // the decode measurement as a whole (the retired MTP contract charged it
        // the same way): the clock starts immediately before the request so the
        // seed cost cannot be hidden.
        let started = Date()
        let begin = try client.beginDFlashDecode(seedTokens: golden.seedTokens)
        guard begin.ok, let seedToken = begin.seedToken else {
            throw MLXFastError.invalidInput(
                "DFlash worker failed the seed prefill: "
                    + (begin.error ?? "no seed token returned")
            )
        }
        guard seedToken == golden.expectedSeedToken else {
            throw DFlashContractViolation(
                kind: .tokenNotAdmissible,
                step: 0,
                detail: "seed prefill token disagreed with the reference"
            )
        }

        var previousCommittedToken = seedToken
        var acceptedTotal = 0
        var rejectedTotal = 0

        while validator.committedTokens.count < options.totalTokenCount {
            let blockSize = schedule.nextBlockSize()
            let roundStart = Date()
            let response = try client.dflashDecodeBlock(
                previousCommittedToken: previousCommittedToken,
                maxBlockSize: blockSize
            )
            let latency = Date().timeIntervalSince(roundStart)
            guard response.ok, let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "DFlash block request failed: "
                        + (response.error ?? "no tokens returned")
                )
            }

            let round = DFlashObservedRound(
                requestedBlockSize: blockSize,
                tokens: tokens,
                declaredRows: response.declaredRows ?? 0,
                perRowTop2Tokens: response.perRowTop2Tokens ?? [],
                perRowTop2Logits: response.perRowTop2Logits ?? [],
                acceptedDraftCount: response.acceptedDraftCount ?? 0,
                rejectedDraftCount: response.rejectedDraftCount ?? 0,
                targetCacheOffset: response.targetCacheOffset ?? -1,
                latencySeconds: latency
            )
            // The parent asks for a full block every time; if the round would
            // overrun the scored window, validate only the prefix that fits.
            let remaining = options.totalTokenCount
                - validator.committedTokens.count
            if tokens.count > remaining {
                let trimmed = DFlashObservedRound(
                    requestedBlockSize: blockSize,
                    tokens: Array(tokens.prefix(remaining)),
                    declaredRows: round.declaredRows,
                    perRowTop2Tokens: round.perRowTop2Tokens,
                    perRowTop2Logits: round.perRowTop2Logits,
                    acceptedDraftCount: round.acceptedDraftCount,
                    rejectedDraftCount: round.rejectedDraftCount,
                    // The worker's offset counts every token it committed; the
                    // parent only scores the prefix, so re-base the expectation.
                    targetCacheOffset: golden.seedTokens.count
                        + validator.committedTokens.count + remaining,
                    latencySeconds: latency
                )
                try validator.accept(round: trimmed)
            } else {
                try validator.accept(round: round)
            }
            acceptedTotal += round.acceptedDraftCount
            rejectedTotal += round.rejectedDraftCount
            previousCommittedToken = tokens.last ?? previousCommittedToken
        }
        let decodeSeconds = Date().timeIntervalSince(started)
        try validator.requireComplete()

        return ExperimentalDFlashReport(
            totalTokenCount: options.totalTokenCount,
            decodeSeconds: decodeSeconds,
            roundCount: validator.roundLatencies.count,
            acceptedDraftTotal: acceptedTotal,
            rejectedDraftTotal: rejectedTotal,
            declaredRowTotal: validator.declaredRowTotal,
            residualDivergenceCount: validator.residualDivergenceCount,
            maxOverMedianRoundLatency: validator.maxOverMedianRoundLatency(),
            allTokensAdmissible: true
        )
    }
}
