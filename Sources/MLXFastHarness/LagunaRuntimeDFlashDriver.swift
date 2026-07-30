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
    public let referenceSeedToken: Int
    public let rows: [Row]
    /// Set by the reference generator once its self-consistency replay passed
    /// (contract layer L1 requirement R5). A run must refuse to score against a
    /// golden that never proved reference determinism.
    public let referenceSelfConsistent: Bool?
    /// The emitted chain these rows were replayed against, when the reference
    /// generated it itself. Carried so a plan can be reconstructed from the
    /// golden alone; it leaks nothing the rows do not already contain, since
    /// `rows[i].sequentialArgmax` is the same chain for a generated golden.
    public let emittedTokens: [Int]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case referenceSeedToken = "reference_seed_token"
        case rows
        case referenceSelfConsistent = "reference_self_consistent"
        case emittedTokens = "emitted_tokens"
    }

    public init(
        seedTokens: [Int],
        referenceSeedToken: Int,
        rows: [Row],
        referenceSelfConsistent: Bool?,
        emittedTokens: [Int]? = nil
    ) {
        self.seedTokens = seedTokens
        self.referenceSeedToken = referenceSeedToken
        self.rows = rows
        self.referenceSelfConsistent = referenceSelfConsistent
        self.emittedTokens = emittedTokens
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
        // A maxBlockSize of 1 is the SERIAL CONTROL: pin the schedule to width 1
        // so every round is a one-row target step. Same worker, same protocol,
        // same forward -- only the width differs, which is what makes the paired
        // ratio a like-for-like comparison.
        //
        // `minBlockSize` 2 for every non-control run is LOAD-BEARING, not a
        // stylistic floor. A width-1 round advances the target without feeding the
        // drafter, whose cross-attention context must be exactly as wide as the
        // positions the target advanced since it last wrote. One such row still
        // leaves a gap of 1 and stays legal; two in a row does not, and the
        // session refuses the following block rather than draft from a prefix the
        // drafter never saw. If L6's randomized schedule is ever widened to
        // include width 1 alongside wider blocks, `LagunaDFlashBlockSession` has
        // to accumulate the skipped hidden rows first.
        var schedule = DFlashBlockSchedule(
            seed: scheduleSeed,
            maxBlockSize: options.maxBlockSize,
            minBlockSize: options.maxBlockSize == 1 ? 1 : 2
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
        guard seedToken == golden.referenceSeedToken else {
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
            admissibleExactCount: validator.admissibleExactCount,
            admissibleDeclaredFrameCount: validator.admissibleDeclaredFrameCount,
            maxOverMedianRoundLatency: validator.maxOverMedianRoundLatency(),
            allTokensAdmissible: true,
            seedTokenCount: golden.seedTokens.count,
            // Parent-derived, never worker-reported: the ledger the wrapper
            // checks must not be something the measured party can assert.
            targetCacheOffsetFinal: golden.seedTokens.count
                + validator.committedTokens.count,
            referenceCheckedRowTotal: validator.referenceCheckedRowTotal,
            // One target-produced tail token per round, by construction.
            targetTailTotal: validator.roundLatencies.count,
            maxBlockRequestSeconds: validator.maxBlockRequestSeconds,
            p50BlockRequestSeconds: validator.p50BlockRequestSeconds,
            blockSize: options.maxBlockSize,
            usesTrainedDrafter: options.maxBlockSize > 1,
            workBindingComparisonCount: validator.workBindingComparisonCount,
            maxTop2LogitDelta: validator.maxWorkBindingLogitDelta,
            meanTop2LogitDelta: validator.meanWorkBindingLogitDelta,
            p50Top2LogitDelta: validator.p50WorkBindingLogitDelta,
            p99Top2LogitDelta: validator.p99WorkBindingLogitDelta,
            maxTop2LogitRelativeDelta: validator.maxWorkBindingLogitRelativeDelta,
            p99Top2LogitRelativeDelta: validator.p99WorkBindingLogitRelativeDelta,
            workBindingLogitDeltas: validator.workBindingLogitDeltas,
            workBindingToleranceAbsolute: tolerance.absolute,
            workBindingToleranceRelative: tolerance.relative
        )
    }
}

// MARK: - L1 reference golden generation

/// The emitted plan a reference pass replays.
///
/// Produced by whoever ran the candidate (locally: the bench script; on the box:
/// the measurement wrapper). It carries only what the reference needs to rebuild
/// the same positions: the seed prompt, the tokens the candidate emitted, and
/// the block width the candidate declared per round.
public struct DFlashEmittedPlan: Codable {
    public struct Round: Codable {
        public let blockSize: Int
        public let count: Int

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case count
        }
    }

    public let seedTokens: [Int]
    public let emitted: [Int]
    public let rounds: [Round]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case emitted
        case rounds
    }
}

/// Reference-driven chain generation for `dflash-reference --generate`.
///
/// Without this, a golden can only ever be as long as an `emitted` array someone
/// typed by hand, which makes the long-context and wrap-seam cases untestable.
/// Here the REFERENCE produces the chain itself: sequential width-1 argmax over
/// its own growing context, one token at a time, in a single process with the
/// model resident once.
///
/// `seedExtensionSteps` is the seam lever. The trusted binary links no
/// tokenizer, so a long seed cannot be typed as text; instead the seed becomes
/// the supplied prompt plus that many reference-generated tokens. That is real
/// model-shaped context, and it lets the seed length be dialled to any position
/// relative to Laguna's 512-slot sliding-window ring -- including the ~505..511
/// band where a block round STARTS trimmable and ENDS wrapped.
public struct DFlashReferenceChainOptions {
    public let seedExtensionSteps: Int
    public let generateTokenCount: Int
    public let roundBlockSize: Int
    public let scheduleSeed: UInt64
    public let planOutputPath: String?

    public init(
        seedExtensionSteps: Int = 0,
        generateTokenCount: Int,
        roundBlockSize: Int = 1,
        scheduleSeed: UInt64 = 0,
        planOutputPath: String? = nil
    ) {
        self.seedExtensionSteps = seedExtensionSteps
        self.generateTokenCount = generateTokenCount
        self.roundBlockSize = roundBlockSize
        self.scheduleSeed = scheduleSeed
        self.planOutputPath = planOutputPath
    }
}

/// Outcome of a reference pass, including the self-consistency verdict.
public struct DFlashReferenceGoldenResult {
    public let rowCount: Int
    public let referenceSeedToken: Int
    public let selfConsistent: Bool
    public let selfConsistencyRowCount: Int
    public let selfConsistencyDetail: String
    /// Seed length actually used, i.e. after any `--seed-generate` extension.
    public let seedTokenCount: Int
    /// Declared block widths recorded per row, ascending.
    public let recordedFrameWidths: [Int]
    /// Where the reconstructed emitted plan was written, if it was.
    public let planOutputPath: String?

    public init(
        rowCount: Int,
        referenceSeedToken: Int,
        selfConsistent: Bool,
        selfConsistencyRowCount: Int,
        selfConsistencyDetail: String,
        seedTokenCount: Int = 0,
        recordedFrameWidths: [Int] = [],
        planOutputPath: String? = nil
    ) {
        self.rowCount = rowCount
        self.referenceSeedToken = referenceSeedToken
        self.selfConsistent = selfConsistent
        self.selfConsistencyRowCount = selfConsistencyRowCount
        self.selfConsistencyDetail = selfConsistencyDetail
        self.seedTokenCount = seedTokenCount
        self.recordedFrameWidths = recordedFrameWidths
        self.planOutputPath = planOutputPath
    }
}

extension LagunaRuntime {
    /// Generate the DFlash reference golden (contract layer L1).
    ///
    /// The worker spawned here MUST be the one built from the pinned baseline
    /// tree, loading organizer-transformed weights -- never the candidate's. The
    /// caller enforces that by pointing `workerOptions.executablePath` and
    /// `targetWeightsPath` at the pinned tree; this function additionally runs
    /// strictly on its own, after the timed phase, with the candidate gone.
    ///
    /// R5 self-consistency: one round is replayed twice in the SAME reference
    /// build and required to be bit-identical. Without that, the admissible sets
    /// this golden defines are not well defined -- the retired track measured
    /// reference-vs-reference instability, so the check is mandatory, and a
    /// failure is an OPERATOR fault rather than a submission fault.
    public static func experimentalDFlashReferenceGolden(
        plan: DFlashEmittedPlan,
        chain: DFlashReferenceChainOptions? = nil,
        targetWeightsPath: String,
        drafterPath: String,
        outputPath: String,
        workerOptions: RuntimeWorkerOptions
    ) throws -> DFlashReferenceGoldenResult {
        guard !plan.seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "DFlash reference plan has an empty seed"
            )
        }
        if let chain {
            guard chain.generateTokenCount > 0 else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation needs a positive token "
                        + "count"
                )
            }
            guard chain.seedExtensionSteps >= 0, chain.roundBlockSize >= 1,
                  chain.roundBlockSize
                      <= MLXFastConstants.experimentalDFlashMaxBlockSize
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation has an out-of-range "
                        + "seed extension or block width"
                )
            }
        } else {
            guard !plan.emitted.isEmpty else {
                throw MLXFastError.invalidInput(
                    "DFlash reference plan has no emitted tokens to verify"
                )
            }
        }

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: targetWeightsPath,
            dflashDrafterPath: drafterPath
        )
        defer { client.close() }

        // --- optional seed extension (one process, model resident once) ------
        var seedTokens = plan.seedTokens
        if let chain, chain.seedExtensionSteps > 0 {
            seedTokens += try generateReferenceChain(
                client: client,
                context: seedTokens,
                steps: chain.seedExtensionSteps,
                label: "seed-generate"
            )
        }

        // The seed token is the width-1 argmax with the seed as the whole
        // context: row 0 fed the seed's last token. Asking the reference for it
        // avoids trusting any candidate-supplied value.
        let seedProbe = try client.dflashReferenceRows(
            prefixTokens: seedTokens,
            startOffset: seedTokens.count - 1,
            rowCount: 1,
            declaredBlockWidth: 1
        )
        guard seedProbe.ok,
              let seedArgmax = seedProbe.referenceK1Argmax?.first
        else {
            throw MLXFastError.invalidInput(
                "DFlash reference could not establish the seed token: "
                    + (seedProbe.error ?? "no reference row returned")
            )
        }

        // --- the emitted chain: supplied, or generated by the reference ------
        let emitted: [Int]
        if let chain {
            emitted = try generateReferenceChain(
                client: client,
                context: seedTokens + [seedArgmax],
                steps: chain.generateTokenCount,
                label: "generate"
            )
        } else {
            emitted = plan.emitted
        }

        // Full context the emitted rows were produced against.
        let context = seedTokens + [seedArgmax] + emitted
        // Row i is fed context[seedLen + i] and predicts context[seedLen+i+1],
        // i.e. emitted[i].
        let rowInputBase = seedTokens.count

        // Rounds give the declared block widths. Absent, treat every row as its
        // own width-1 round so the golden is still usable for a serial control.
        let rounds: [DFlashEmittedPlan.Round]
        if let chain {
            // Lay the generated rounds out on the SAME parent schedule the run
            // will use. The parent owns block widths, so with full acceptance
            // the golden's frame boundaries land exactly where the run's do; a
            // partially-accepting round shifts them, which is what the capped
            // residual bucket is for.
            rounds = scheduledRounds(
                tokenCount: emitted.count,
                maxBlockSize: chain.roundBlockSize,
                scheduleSeed: chain.scheduleSeed
            )
        } else {
            rounds = plan.rounds
                ?? emitted.map { _ in
                    DFlashEmittedPlan.Round(blockSize: 1, count: 1)
                }
        }
        let plannedRowTotal = rounds.reduce(0) { $0 + $1.count }
        guard plannedRowTotal == emitted.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference plan rounds cover \(plannedRowTotal) rows but "
                    + "\(emitted.count) tokens were emitted"
            )
        }

        if let path = chain?.planOutputPath {
            let reconstructed = DFlashEmittedPlan(
                seedTokens: seedTokens,
                emitted: emitted,
                rounds: rounds
            )
            let planEncoder = JSONEncoder()
            planEncoder.outputFormatting = [
                .prettyPrinted, .sortedKeys, .withoutEscapingSlashes,
            ]
            try planEncoder.encode(reconstructed)
                .write(to: URL(fileURLWithPath: path))
        }

        var goldenRows = [DFlashReferenceGolden.Row]()
        goldenRows.reserveCapacity(emitted.count)
        var replayRequest: (offset: Int, count: Int, width: Int)?
        var replayExpected: RuntimeWorkerResponse?
        var recordedWidths = Set<Int>([1])

        var emittedOffset = 0
        for round in rounds {
            guard round.count > 0, round.blockSize >= round.count else {
                throw MLXFastError.invalidInput(
                    "DFlash reference plan has a round emitting \(round.count) "
                        + "tokens at declared width \(round.blockSize)"
                )
            }
            let startOffset = rowInputBase + emittedOffset

            // Record a GENUINE block frame for every width the parent could
            // have declared here, not just one.
            //
            // The parent picks each round's width from a randomized schedule, and
            // a round that rejects drafts emits fewer tokens than its declared
            // width, so the width a row is scored at is not known when the
            // golden is built. Replaying widths `count ... blockSize` covers
            // every width the schedule can request. Widening the frame is sound
            // for the scored rows: attention is causally masked, so rows
            // `0 ..< count` cannot see the tail rows this widening filled in
            // with later emitted tokens rather than the candidate's (unknown)
            // rejected drafts. Kernel tiling can still perturb the last bits of
            // a wider frame, which is what the top-2 tolerance absorbs.
            let availableRows = context.count - startOffset
            let widestFrame = Swift.min(round.blockSize, availableRows)
            var response: RuntimeWorkerResponse?
            var frames = [Int: [Int]]()
            for width in round.count ... Swift.max(round.count, widestFrame) {
                let framed = try client.dflashReferenceRows(
                    prefixTokens: context,
                    startOffset: startOffset,
                    rowCount: width,
                    declaredBlockWidth: width
                )
                guard framed.ok,
                      let block = framed.referenceBlockArgmax,
                      block.count == width
                else {
                    throw MLXFastError.invalidInput(
                        "DFlash reference returned an incomplete width-\(width) "
                            + "frame: " + (framed.error ?? "shape mismatch")
                    )
                }
                frames[width] = Array(block.prefix(round.count))
                recordedWidths.insert(width)
                if width == round.count { response = framed }
            }
            guard let response,
                  let k1 = response.referenceK1Argmax,
                  let top2Tokens = response.referenceTop2Tokens,
                  let top2Logits = response.referenceTop2Logits,
                  k1.count == round.count,
                  top2Tokens.count == round.count,
                  top2Logits.count == round.count
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference returned an incomplete row batch"
                )
            }

            for index in 0 ..< round.count {
                var declaredFrames = [String: Int]()
                for (width, argmax) in frames where index < argmax.count {
                    declaredFrames[String(width)] = argmax[index]
                }
                // Width 1 needs no request: a one-row frame IS the sequential
                // frame, so the serial control (max block size 1) always has a
                // declared-frame entry to be scored against.
                declaredFrames["1"] = k1[index]
                goldenRows.append(
                    DFlashReferenceGolden.Row(
                        sequentialArgmax: k1[index],
                        declaredFrameArgmax: declaredFrames,
                        top2Tokens: top2Tokens[index],
                        top2Logits: top2Logits[index]
                    )
                )
            }

            // Replay the FIRST round: it is the one whose context is shortest,
            // so a determinism failure there is unambiguous.
            if replayRequest == nil {
                replayRequest = (startOffset, round.count, round.count)
                replayExpected = response
            }
            emittedOffset += round.count
        }

        // --- R5: self-consistency replay in the same reference build ---------
        var selfConsistent = false
        var selfConsistencyDetail = "no round available to replay"
        var selfConsistencyRowCount = 0
        if let request = replayRequest, let expected = replayExpected {
            let again = try client.dflashReferenceRows(
                prefixTokens: context,
                startOffset: request.offset,
                rowCount: request.count,
                declaredBlockWidth: request.width
            )
            selfConsistencyRowCount = request.count
            if !again.ok {
                selfConsistencyDetail =
                    "replay failed: " + (again.error ?? "unknown error")
            } else if again.referenceK1Argmax != expected.referenceK1Argmax {
                selfConsistencyDetail = "width-1 argmax differed between replays"
            } else if again.referenceBlockArgmax != expected.referenceBlockArgmax {
                selfConsistencyDetail = "block-frame argmax differed between replays"
            } else if again.referenceTop2Tokens != expected.referenceTop2Tokens {
                selfConsistencyDetail = "top-2 token ids differed between replays"
            } else if again.referenceTop2Logits != expected.referenceTop2Logits {
                selfConsistencyDetail = "top-2 logit values differed between replays"
            } else {
                selfConsistent = true
                selfConsistencyDetail =
                    "replayed \(request.count) row(s) bit-identically"
            }
        }

        let golden = DFlashReferenceGolden(
            seedTokens: seedTokens,
            referenceSeedToken: seedArgmax,
            rows: goldenRows,
            referenceSelfConsistent: selfConsistent,
            emittedTokens: chain == nil ? nil : emitted
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(golden).write(to: URL(fileURLWithPath: outputPath))

        return DFlashReferenceGoldenResult(
            rowCount: goldenRows.count,
            referenceSeedToken: seedArgmax,
            selfConsistent: selfConsistent,
            selfConsistencyRowCount: selfConsistencyRowCount,
            selfConsistencyDetail: selfConsistencyDetail,
            seedTokenCount: seedTokens.count,
            recordedFrameWidths: recordedWidths.sorted(),
            planOutputPath: chain?.planOutputPath
        )
    }

    /// Generate `steps` tokens by sequential width-1 reference argmax.
    ///
    /// Every step is its own stateless reference request, so each generated token
    /// is produced by EXACTLY the computation the golden's own replay uses -- a
    /// stateful incremental generator would be faster but would generate the
    /// chain in a different frame than the one it is later checked in, which is
    /// the class of mismatch this whole contract exists to detect. The cost is
    /// quadratic in `steps`; that is an operator-side cost outside any timed
    /// window, paid once per golden.
    private static func generateReferenceChain(
        client: RuntimeWorkerClient,
        context: [Int],
        steps: Int,
        label: String
    ) throws -> [Int] {
        guard steps > 0 else { return [] }
        var context = context
        var generated = [Int]()
        generated.reserveCapacity(steps)
        for index in 0 ..< steps {
            let response = try client.dflashReferenceRows(
                prefixTokens: context,
                startOffset: context.count - 1,
                rowCount: 1,
                declaredBlockWidth: 1
            )
            guard response.ok,
                  let token = response.referenceK1Argmax?.first
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation failed at \(label) step "
                        + "\(index): " + (response.error ?? "no row returned")
                )
            }
            generated.append(token)
            context.append(token)
            if (index + 1) % 32 == 0 || index + 1 == steps {
                fputs(
                    "dflash-reference: \(label) \(index + 1)/\(steps) "
                        + "(context \(context.count))\n",
                    stderr
                )
            }
        }
        return generated
    }

    /// Round layout for a generated chain, replaying the parent block schedule.
    private static func scheduledRounds(
        tokenCount: Int,
        maxBlockSize: Int,
        scheduleSeed: UInt64
    ) -> [DFlashEmittedPlan.Round] {
        var schedule = DFlashBlockSchedule(
            seed: scheduleSeed,
            maxBlockSize: maxBlockSize,
            minBlockSize: maxBlockSize == 1 ? 1 : 2
        )
        var rounds = [DFlashEmittedPlan.Round]()
        var remaining = tokenCount
        while remaining > 0 {
            let width = schedule.nextBlockSize()
            let count = Swift.min(width, remaining)
            rounds.append(
                DFlashEmittedPlan.Round(blockSize: width, count: count)
            )
            remaining -= count
        }
        return rounds
    }
}
