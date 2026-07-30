import Foundation
import MLXFastCore

// Trusted parent for the DFlash block-decode track (laguna-xs-2.1-dflash-v1).
//
// This file links NO MLX and must never compute a logit: MLXFastCLI (the trusted
// binary) depends only on MLXFastCore / MLXFastTransform / MLXFastHarness /
// Tokenizers. Every reference verdict therefore arrives through
// `DFlashReferenceOracle`, which the caller backs with a SECOND worker process
// built from the pinned baseline tree and loading organizer-transformed weights
// (contract layer L1). The parent owns the timer, the token budget, the block
// schedule, and the arithmetic -- nothing else.

/// Reference verdict for one emitted row, supplied by the pinned-baseline
/// reference worker (never by the candidate).
public struct DFlashReferenceRow: Equatable, Sendable {
    /// Reference argmax in the K=1 sequential frame at this position.
    public let sequentialArgmax: Int
    /// Reference argmax in the block frame the candidate declared for this row,
    /// when the reference replayed that width. `nil` when no declared-frame
    /// replay exists for the row.
    public let declaredFrameArgmax: Int?
    /// Reference top-2 token ids at this position, highest logit first.
    public let top2Tokens: [Int]
    /// Reference top-2 logit values, aligned with `top2Tokens`.
    public let top2Logits: [Double]
    /// The reference's own top-1 logit at this row: the value every near-tie
    /// distance is measured from.
    public let top1Logit: Double
    /// The token this reference row was teacher-forced to PREDICT. On the live
    /// replay path the reference walks the candidate's own emitted chain, so
    /// this is the token the candidate emitted here; a pre-generated golden's
    /// row was forced on the GOLDEN's chain instead, which is why the validator
    /// re-checks the id before trusting the logit below.
    public let emittedToken: Int?
    /// The REFERENCE's logit for `emittedToken`, read out of the same width-1
    /// row the top-2 readout comes from. This is what makes a near tie a
    /// question about the emitted token itself rather than about membership in a
    /// two-slot list, which cannot express a three-way tie (Blocker 4c).
    public let emittedTokenLogit: Double?

    public init(
        sequentialArgmax: Int,
        declaredFrameArgmax: Int? = nil,
        top2Tokens: [Int] = [],
        top2Logits: [Double] = [],
        top1Logit: Double? = nil,
        emittedToken: Int? = nil,
        emittedTokenLogit: Double? = nil
    ) {
        self.sequentialArgmax = sequentialArgmax
        self.declaredFrameArgmax = declaredFrameArgmax
        self.top2Tokens = top2Tokens
        self.top2Logits = top2Logits
        // A reference that reports top-2 values but no explicit top-1 is
        // reporting the top-1 in `top2Logits[0]`; keep the two consistent
        // rather than defaulting to a zero that would read as a flat row.
        self.top1Logit = top1Logit ?? top2Logits.first ?? 0
        self.emittedToken = emittedToken
        self.emittedTokenLogit = emittedTokenLogit
    }

    /// Criterion E admissible set: the reference argmax in either exactly-defined
    /// frame. Systematic honest divergence lands here and is absorbed WITHOUT a
    /// budget, because the set has at most two specific members -- there is no
    /// epsilon margin for a cheating submission to spend.
    public var admissibleTokens: Set<Int> {
        var tokens: Set<Int> = [sequentialArgmax]
        if let declaredFrameArgmax {
            tokens.insert(declaredFrameArgmax)
        }
        return tokens
    }
}

/// Source of reference verdicts for the trusted parent.
public protocol DFlashReferenceOracle {
    /// Reference rows for `[startOffset, startOffset + count)` of the emitted
    /// sequence, teacher-forced on the candidate's own emitted prefix.
    func referenceRows(
        emittedPrefix: [Int],
        startOffset: Int,
        count: Int,
        declaredBlockWidth: Int
    ) throws -> [DFlashReferenceRow]
}

public enum DFlashValidationOutcome: String, Sendable {
    case admissibleExact
    case admissibleDeclaredFrame
    /// The reference itself had no defensible answer at this row: its own top-2
    /// logits are close enough that build-to-build drift can reorder them, and the
    /// candidate emitted the other member of that pair. Distinguished from
    /// `residualWithinBudget` because a coin-flip row is not evidence of anything,
    /// so it must not spend the budget reserved for real drift.
    case admissibleNearTie
    case residualWithinBudget
    case rejected
}

/// Why a run failed the contract. Carries no golden or reference token ids:
/// surfacing those against hidden material would turn the validator into a
/// query oracle for the hidden prompt (contract layer L6).
public struct DFlashContractViolation: Error, CustomStringConvertible {
    public enum Kind: String, Sendable {
        case emptyBlock
        case oversizedBlock
        case outOfVocabularyToken
        case tokenNotAdmissible
        case residualBudgetExhausted
        case rowAccountingMismatch
        case declaredRowsMissing
        case workBindingMissing
        case workBindingLogitMismatch
        case cacheOffsetDiverged
        case incompleteRun
    }

    public let kind: Kind
    public let step: Int?
    public let detail: String

    public init(kind: Kind, step: Int? = nil, detail: String = "") {
        self.kind = kind
        self.step = step
        self.detail = detail
    }

    public var description: String {
        var text = "DFlash contract violation: \(kind.rawValue)"
        if let step {
            text += " at step \(step)"
        }
        if !detail.isEmpty {
            text += " (\(detail))"
        }
        return text
    }
}

/// Tolerance for comparing candidate and reference per-row top-2 logit VALUES.
///
/// These are numeric readouts taken in two different frames, so they are compared
/// with a tolerance -- never for equality.
///
/// MEASURED BASIS, RE-TAKEN. The first calibration (Amendment 6) compared the
/// candidate against a PRE-GENERATED golden, so every gap it recorded was the sum
/// of candidate-vs-reference drift and the golden's own pre-generation drift. Only
/// the live post-run replay of the candidate's own chain (Amendment 15) isolates
/// the term this tolerance is supposed to price. Both regimes are kept below,
/// because the difference between them IS the finding.
///
/// LIVE REPLAY (M5-C, 2026-07-30, frozen 128-token window, 256 comparisons per
/// run, Laguna XS 2.1 NVFP4 target + the pinned 924 MB BF16 drafter, candidate and
/// reference from the SAME build). Gap = `|candidate - reference|` per top-2 slot;
/// logits are BF16, so one ULP at these magnitudes is 0.125:
///
///     width  fixture       seeds    max            p99      p50    mean
///     K=1    both          -        0              0        0      0
///     K=3    varied        7        1.500          1.375    0.188  0.280
///     K=3    repetitive    0        2.750          1.500    0.250  0.309
///     K=4    varied        0, 7     1.750, 2.4375  1.75     0.250  0.320
///     K=4    repetitive    0, 7     1.750, 2.000   1.688    0.250  0.319
///     K=8    varied        7        2.625          1.750    0.250  0.386
///     K=8    repetitive    0        3.250          2.125    0.188  0.365
///
/// Read that table before touching the constant. The K=1 control is EXACTLY zero:
/// at width 1 the candidate reproduces the reference's own width-1 walk bit for
/// bit, so there is no same-build drift at all. Every nonzero row is a BLOCK-FRAME
/// effect that grows with the width -- which also means the 3.375 the old
/// calibration attributed to build-to-build drift was mostly the golden's, and
/// what survives is a frame effect the reference could not have avoided.
///
/// So: absolute **4.875** is 2x the maximum observed at the widths this
/// recalibration covers (2.4375 at K<=4, both fixtures, several seeds), and 1.5x
/// the maximum observed at ANY width including the off-ranked K=8 diagnostic
/// (3.250) -- the same 1.5x discipline Amendment 6 applied, now against an
/// uncontaminated maximum. It is also strictly below the outgoing 5.0, which is a
/// hard constraint: a recalibration whose measured basis shrank must not end up
/// loosening the binder. Relative 0.25 is 1.5x the largest relative gap observed
/// live (0.166) -- the old 0.02 was dead code, below even the short-context gaps.
///
/// STORED-GOLDEN REGIME, retained for provenance because it is what the 5.0 was
/// built on: max 3.375 over 352 long-context comparisons, concentrated once the
/// context passed the 512-slot sliding window, spread across the run rather than
/// clustered at the wrap seam.
///
/// WHAT THIS DOES NOT ESTABLISH, stated because it bounds the claim:
///   * Candidate and reference were the same build, so the CROSS-BUILD term (a
///     submission's own kernels) is unmeasured and additive on top of these.
///   * The false-negative side is unmeasured: no deliberately degraded verifier
///     was run, so "a cheaper verifier cannot hide inside 5.0" is not a measured
///     claim. At 5.0 against top-1 logits near 21 the binder prices only GROSS
///     degradation. It remains a real constraint on a verifier that skips the
///     per-row lm_head entirely (it then has no values to report at all), but it
///     does not price a subtly reduced trunk.
/// Both gaps must be closed before `official_scoring_enabled` is turned on.
public struct DFlashWorkBindingTolerance: Sendable {
    public let absolute: Double
    public let relative: Double

    public init(absolute: Double = 4.875, relative: Double = 0.25) {
        self.absolute = absolute
        self.relative = relative
    }

    public func matches(candidate: Double, reference: Double) -> Bool {
        let delta = abs(candidate - reference)
        if delta <= absolute {
            return true
        }
        let scale = Swift.max(abs(candidate), abs(reference))
        return scale > 0 && delta / scale <= relative
    }
}

/// One round as reported by the candidate worker, in parent terms.
public struct DFlashObservedRound: Sendable {
    public let requestedBlockSize: Int
    public let tokens: [Int]
    public let declaredRows: Int
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    public let targetCacheOffset: Int
    public let latencySeconds: Double

    public init(
        requestedBlockSize: Int,
        tokens: [Int],
        declaredRows: Int,
        perRowTop2Tokens: [[Int]],
        perRowTop2Logits: [[Double]],
        acceptedDraftCount: Int,
        rejectedDraftCount: Int,
        targetCacheOffset: Int,
        latencySeconds: Double
    ) {
        self.requestedBlockSize = requestedBlockSize
        self.tokens = tokens
        self.declaredRows = declaredRows
        self.perRowTop2Tokens = perRowTop2Tokens
        self.perRowTop2Logits = perRowTop2Logits
        self.acceptedDraftCount = acceptedDraftCount
        self.rejectedDraftCount = rejectedDraftCount
        self.targetCacheOffset = targetCacheOffset
        self.latencySeconds = latencySeconds
    }
}

/// Criterion E validator: primary token predicate + L2 work binding + L3 row
/// accounting -- in TWO halves, deliberately.
///
/// `acceptStructural(round:)` is arithmetic on what the worker just reported:
/// vocabulary range, block bounds, emitted-vs-declared row accounting, the
/// KV-offset ledger, and the shape of the work-binding readout. None of it needs
/// a reference, so it runs inline in the timed loop and fails the round that
/// produced it.
///
/// `validateJournalAgainstReference(oracle:)` is everything that needs a
/// reference verdict: sequential / declared-frame / near-tie / residual
/// admissibility and the L2 top-2 logit comparison. It CANNOT run inline, and
/// that is a property of the contract rather than a performance choice. A
/// reference verdict is only meaningful when it was teacher-forced on the
/// candidate's OWN emitted prefix; the moment the candidate legitimately
/// diverges (a near-tie or residual row the contract admits), a pre-generated
/// file's later rows are anchored to a prefix the candidate no longer has, so
/// every one of them is scored against the wrong context. The parent therefore
/// journals the rounds, and the pinned reference replays the candidate's own
/// chain afterwards. Keeping the oracle out of the timed loop also keeps harness
/// work off the clock, which is a side benefit, not the reason.
public final class LagunaDFlashBlockValidator {
    /// Fallback oracle supplied at init (the golden file). The scored path passes
    /// a live reference oracle to `validateJournalAgainstReference` instead.
    private let oracle: any DFlashReferenceOracle
    private let totalTokenCount: Int
    private let seedTokenCount: Int
    private let tolerance: DFlashWorkBindingTolerance
    private let residualBudget: Int
    private let nearTieBudget: Int

    public private(set) var committedTokens = [Int]()
    /// Every structurally-accepted round, in emission order. This is the journal
    /// the post-run reference pass replays; it is what lets the reference be
    /// teacher-forced on the candidate's own chain instead of a stored one.
    public private(set) var journal = [DFlashObservedRound]()
    /// Set once the oracle half has run. A report must not be published without
    /// it: the structural half alone scores no token.
    public private(set) var referenceValidated = false
    public private(set) var outcomes = [DFlashValidationOutcome]()
    public private(set) var residualDivergenceCount = 0
    public private(set) var nearTieAdmissionCount = 0
    public private(set) var declaredRowTotal = 0
    public private(set) var roundLatencies = [Double]()
    /// Rows the parent actually obtained a reference verdict for. The box
    /// wrapper cross-checks this against the emitted total; it cannot yet equal
    /// `declaredRowTotal` because the worker does not report the token ids of
    /// REJECTED draft rows, so the reference has nothing to score them against
    /// (an L2 follow-up).
    public private(set) var referenceCheckedRowTotal = 0
    /// Every observed `|candidate - reference|` top-2 logit gap, including the
    /// ones comfortably inside tolerance.
    ///
    /// This is the calibration instrument for contract Amendment 1: the
    /// tolerance constant is only defensible if it was set from a MEASURED
    /// distribution of honest gaps, so the parent records the distribution on
    /// every run rather than asserting a number. It is also a liveness check on
    /// the binding itself -- a run that recorded no gaps at all compared no
    /// readouts.
    public private(set) var workBindingLogitDeltas = [Double]()
    /// The same gaps divided by the larger of the two magnitudes, i.e. the
    /// quantity the relative arm of the tolerance actually tests.
    public private(set) var workBindingLogitRelativeDeltas = [Double]()

    public init(
        oracle: any DFlashReferenceOracle,
        seedTokenCount: Int,
        totalTokenCount: Int,
        tolerance: DFlashWorkBindingTolerance = DFlashWorkBindingTolerance()
    ) {
        self.oracle = oracle
        self.seedTokenCount = seedTokenCount
        self.totalTokenCount = totalTokenCount
        self.tolerance = tolerance
        // Scale the small residual bucket with the window, rounding up so short
        // diagnostic runs still get at least one slot.
        let perThousand =
            MLXFastConstants.experimentalDFlashResidualDivergenceBudgetPerThousand
        self.residualBudget = Swift.max(
            1,
            (totalTokenCount * perThousand + 999) / 1_000
        )
        let nearTiePerThousand =
            MLXFastConstants.experimentalDFlashNearTieAdmissionBudgetPerThousand
        self.nearTieBudget = Swift.max(
            1,
            (totalTokenCount * nearTiePerThousand + 999) / 1_000
        )
    }

    /// A row the reference cannot break: the reference's own logit for the
    /// token the candidate emitted sits within the measured drift envelope of
    /// the reference's own top-1, so the ordering between them is not a property
    /// of the model but of accumulation order.
    ///
    /// This tests the EMITTED TOKEN'S OWN reference logit rather than membership
    /// in a fixed-size shortlist (Amendment 16). It subsumes Amendment 14's
    /// top-2 test -- the #2 token's distance from #1 IS the top-2 gap -- and is
    /// better shaped in both directions: at a confident row nothing but the
    /// top-1 qualifies, and at a flat row exactly those tokens the reference
    /// genuinely cannot separate qualify, however many of them there are. A
    /// two-member admissible set cannot express a three-way tie (Blocker 4c),
    /// and at such a row the reference cannot rank those three either.
    ///
    /// Still unfarmable for the same reason as before: the logits are the
    /// REFERENCE's, so a submission can neither manufacture a flat row nor learn
    /// which positions are flat without doing the reference's own work. And the
    /// admitted distance is bounded by the envelope, so a token the reference
    /// prices well below its top-1 is rejected here exactly as anywhere else.
    private func referenceRowAdmitsNearTie(
        _ row: DFlashReferenceRow,
        token: Int
    ) -> Bool {
        let envelope = MLXFastConstants.experimentalDFlashNearTieLogitEnvelope
        // The id is re-checked rather than assumed: the emitted-token logit is
        // only a statement about the candidate's token when the row was
        // teacher-forced on the candidate's token. The live replay walks the
        // candidate's own chain and so it always is; a stored golden's row was
        // forced on the golden's chain and so it is not, which would otherwise
        // make a golden row admit any divergence at zero distance.
        if let emittedTokenLogit = row.emittedTokenLogit,
           row.emittedToken == token
        {
            return row.top1Logit - emittedTokenLogit <= envelope
        }
        // Fallback for a reference row that predates the field: Amendment 14's
        // top-2 membership form, so a pre-generated golden still validates.
        guard row.top2Tokens.contains(token), row.top2Logits.count >= 2 else {
            return false
        }
        return (row.top2Logits[0] - row.top2Logits[1]) <= envelope
    }

    public var remainingTokenCount: Int {
        Swift.max(0, totalTokenCount - committedTokens.count)
    }

    /// STRUCTURAL half. Runs inline in the timed loop, needs no reference, and
    /// journals the round for the post-run oracle pass. Throws on the first
    /// structural contract violation.
    public func acceptStructural(round: DFlashObservedRound) throws {
        let step = committedTokens.count

        guard !round.tokens.isEmpty else {
            throw DFlashContractViolation(kind: .emptyBlock, step: step)
        }
        guard round.tokens.count <= round.requestedBlockSize else {
            throw DFlashContractViolation(
                kind: .oversizedBlock,
                step: step,
                detail: "emitted \(round.tokens.count) for a block of "
                    + "\(round.requestedBlockSize)"
            )
        }
        guard round.tokens.allSatisfy({
            $0 >= 0 && $0 < MLXFastConstants.vocabSize
        }) else {
            throw DFlashContractViolation(
                kind: .outOfVocabularyToken,
                step: step
            )
        }

        // L3 row accounting: the worker must have declared at least as many
        // target rows as it emitted tokens. Over-emitting -- returning more
        // tokens than rows it actually pushed through the target -- fails
        // arithmetically here, before any token is scored.
        guard round.declaredRows > 0 else {
            throw DFlashContractViolation(kind: .declaredRowsMissing, step: step)
        }
        guard round.declaredRows >= round.tokens.count,
              round.declaredRows <= round.requestedBlockSize
        else {
            throw DFlashContractViolation(
                kind: .rowAccountingMismatch,
                step: step,
                detail: "declared \(round.declaredRows) rows for "
                    + "\(round.tokens.count) emitted tokens at block "
                    + "\(round.requestedBlockSize)"
            )
        }
        guard round.acceptedDraftCount >= 0,
              round.rejectedDraftCount >= 0,
              round.acceptedDraftCount + 1 >= round.tokens.count
        else {
            throw DFlashContractViolation(
                kind: .rowAccountingMismatch,
                step: step,
                detail: "accepted/rejected counts inconsistent with the block"
            )
        }

        // The ledger the worker also checks locally; re-checked here because the
        // worker is the party with an incentive to skip rollback.
        let expectedOffset = seedTokenCount + step + round.tokens.count
        guard round.targetCacheOffset == expectedOffset else {
            throw DFlashContractViolation(
                kind: .cacheOffsetDiverged,
                step: step,
                detail: "reported offset \(round.targetCacheOffset)"
            )
        }

        // L2 work binding must cover EVERY declared row, including the rejected
        // tail rollback is about to discard.
        guard round.perRowTop2Tokens.count == round.declaredRows,
              round.perRowTop2Logits.count == round.declaredRows
        else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                step: step,
                detail: "per-row readouts cover "
                    + "\(round.perRowTop2Tokens.count) of \(round.declaredRows) "
                    + "declared rows"
            )
        }

        journal.append(round)
        committedTokens.append(contentsOf: round.tokens)
        declaredRowTotal += round.declaredRows
        roundLatencies.append(round.latencySeconds)
    }

    /// ORACLE half. Replays the journal in order against a reference, scoring
    /// every emitted token and closing the L2 work binding.
    ///
    /// `oracle` defaults to the one supplied at init -- the golden file, kept as
    /// a fallback for paths that have no live reference. The scored path passes a
    /// LIVE reference oracle, because reference rows have to be teacher-forced on
    /// the candidate's own emitted prefix and only a live reference can be.
    ///
    /// Replaying strictly in emission order matters twice: it is what makes
    /// `emittedPrefix` the candidate's actual prefix at that point, and the live
    /// reference session is a continuously-advanced cache that only stays a pure
    /// continuation for in-order requests.
    public func validateJournalAgainstReference(
        oracle overrideOracle: (any DFlashReferenceOracle)? = nil
    ) throws {
        guard !referenceValidated else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                detail: "the reference admissibility pass ran twice"
            )
        }
        let oracle = overrideOracle ?? self.oracle
        var emittedPrefix = [Int]()
        emittedPrefix.reserveCapacity(committedTokens.count)
        for round in journal {
            let step = emittedPrefix.count
            let reference = try oracle.referenceRows(
                emittedPrefix: emittedPrefix,
                startOffset: step,
                count: round.tokens.count,
                declaredBlockWidth: round.requestedBlockSize
            )
            guard reference.count == round.tokens.count else {
                throw DFlashContractViolation(
                    kind: .workBindingMissing,
                    step: step,
                    detail: "reference returned \(reference.count) rows for "
                        + "\(round.tokens.count) emitted tokens"
                )
            }
            try score(round: round, against: reference, step: step)
            emittedPrefix.append(contentsOf: round.tokens)
            referenceCheckedRowTotal += reference.count
        }
        referenceValidated = true
    }

    /// The oracle half must have run before any report is published: on its own,
    /// the structural half scores no token at all.
    public func requireReferenceValidated() throws {
        guard referenceValidated else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                detail: "no reference admissibility pass ran for this run"
            )
        }
    }

    private func score(
        round: DFlashObservedRound,
        against reference: [DFlashReferenceRow],
        step: Int
    ) throws {
        for (index, token) in round.tokens.enumerated() {
            let referenceRow = reference[index]
            let outcome: DFlashValidationOutcome
            if token == referenceRow.sequentialArgmax {
                outcome = .admissibleExact
            } else if referenceRow.admissibleTokens.contains(token) {
                // Honest frame divergence: the reference itself produces this
                // token at the width the candidate declared.
                outcome = .admissibleDeclaredFrame
            } else if referenceRowAdmitsNearTie(referenceRow, token: token) {
                // Numerical plausibility, not frame equality. At a row where the
                // reference prices the emitted token within the measured
                // build-to-build drift envelope of its own top-1, "the reference
                // says X" is not a fact about correctness -- either token is what
                // the reference would produce under a differently-ordered
                // accumulation. Rejecting the candidate here fails honest code for
                // a tie the reference cannot break. Still capped, so it cannot
                // become a free channel: near-tie rows are rare (measured 3 per
                // 128 positions on varied prose, 0 per 128 on repetitive text) and
                // a submission cannot manufacture them, because the logits are the
                // REFERENCE's own.
                nearTieAdmissionCount += 1
                guard nearTieAdmissionCount <= nearTieBudget else {
                    throw DFlashContractViolation(
                        kind: .residualBudgetExhausted,
                        step: step + index,
                        detail: "near-tie admissions exceeded \(nearTieBudget)"
                    )
                }
                outcome = .admissibleNearTie
            } else if referenceRow.top2Tokens.contains(token) {
                // Top-2 member, but at a row the reference answers confidently.
                // That is genuine candidate-vs-reference drift and keeps spending
                // the deliberately small residual budget.
                residualDivergenceCount += 1
                guard residualDivergenceCount <= residualBudget else {
                    throw DFlashContractViolation(
                        kind: .residualBudgetExhausted,
                        step: step + index,
                        detail: "residual divergences exceeded "
                            + "\(residualBudget)"
                    )
                }
                outcome = .residualWithinBudget
            } else {
                throw DFlashContractViolation(
                    kind: .tokenNotAdmissible,
                    step: step + index
                )
            }
            outcomes.append(outcome)

            // Work binding on the emitted rows: the candidate's top-2 logit
            // VALUES must track the reference's within tolerance. A verifier
            // that skipped or cheapened the per-row lm_head cannot produce
            // these.
            let candidateLogits = round.perRowTop2Logits[index]
            let referenceLogits = referenceRow.top2Logits
            if !referenceLogits.isEmpty, !candidateLogits.isEmpty {
                let pairCount = Swift.min(
                    candidateLogits.count,
                    referenceLogits.count
                )
                for pair in 0 ..< pairCount {
                    let candidateValue = candidateLogits[pair]
                    let referenceValue = referenceLogits[pair]
                    // Record first, judge second: the distribution is wanted even
                    // for the run that is about to fail.
                    let delta = abs(candidateValue - referenceValue)
                    workBindingLogitDeltas.append(delta)
                    let scale = Swift.max(
                        abs(candidateValue),
                        abs(referenceValue)
                    )
                    workBindingLogitRelativeDeltas.append(
                        scale > 0 ? delta / scale : 0
                    )
                    guard tolerance.matches(
                        candidate: candidateValue,
                        reference: referenceValue
                    ) else {
                        throw DFlashContractViolation(
                            kind: .workBindingLogitMismatch,
                            step: step + index,
                            detail: "row readout \(pair) outside tolerance"
                        )
                    }
                }
            }
        }
    }

    /// Require the run to have produced exactly the configured token total.
    public func requireComplete() throws {
        guard committedTokens.count == totalTokenCount else {
            throw DFlashContractViolation(
                kind: .incompleteRun,
                detail: "committed \(committedTokens.count) of "
                    + "\(totalTokenCount) tokens"
            )
        }
    }

    /// Stall guardrail input: a round whose latency exceeds `factor` times the
    /// median round is a measurement-invalid sample, not a participant fault.
    public func maxOverMedianRoundLatency() -> Double? {
        guard !roundLatencies.isEmpty else { return nil }
        let sorted = roundLatencies.sorted()
        let median = sorted[sorted.count / 2]
        guard median > 0, let maximum = sorted.last else { return nil }
        return maximum / median
    }

    // --- work-binding gap distribution (Amendment 1 calibration input) ------

    /// How many emitted tokens were the reference's own width-1 argmax.
    public var admissibleExactCount: Int {
        outcomes.filter { $0 == .admissibleExact }.count
    }
    /// How many were admitted only because the reference produces them in the
    /// declared block frame. A nonzero count on a run scored against a
    /// PRE-GENERATED golden also means the candidate's chain has diverged from
    /// the golden's, so every later row is being compared against a differently
    /// teacher-forced context.
    public var admissibleNearTieCount: Int {
        outcomes.filter { $0 == .admissibleNearTie }.count
    }

    public var admissibleDeclaredFrameCount: Int {
        outcomes.filter { $0 == .admissibleDeclaredFrame }.count
    }

    public var workBindingComparisonCount: Int { workBindingLogitDeltas.count }
    public var maxWorkBindingLogitDelta: Double {
        workBindingLogitDeltas.max() ?? 0
    }
    public var meanWorkBindingLogitDelta: Double {
        guard !workBindingLogitDeltas.isEmpty else { return 0 }
        return workBindingLogitDeltas.reduce(0, +)
            / Double(workBindingLogitDeltas.count)
    }
    public var p50WorkBindingLogitDelta: Double {
        workBindingLogitDeltaQuantile(0.50)
    }
    public var p99WorkBindingLogitDelta: Double {
        workBindingLogitDeltaQuantile(0.99)
    }

    public var maxWorkBindingLogitRelativeDelta: Double {
        workBindingLogitRelativeDeltas.max() ?? 0
    }
    public var p99WorkBindingLogitRelativeDelta: Double {
        Self.quantile(workBindingLogitRelativeDeltas, 0.99)
    }

    private func workBindingLogitDeltaQuantile(_ quantile: Double) -> Double {
        Self.quantile(workBindingLogitDeltas, quantile)
    }

    private static func quantile(_ values: [Double], _ quantile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = Int((Double(sorted.count - 1) * quantile).rounded())
        return sorted[Swift.min(Swift.max(index, 0), sorted.count - 1)]
    }

    /// Slowest single block request, in seconds.
    public var maxBlockRequestSeconds: Double { roundLatencies.max() ?? 0 }

    /// Median block request, in seconds. The box wrapper rejects a phase whose
    /// slowest round exceeds a fixed factor of this.
    public var p50BlockRequestSeconds: Double {
        guard !roundLatencies.isEmpty else { return 0 }
        let sorted = roundLatencies.sorted()
        return sorted[sorted.count / 2]
    }
}

/// Trusted-parent options for a DFlash block-decode measurement.
public struct ExperimentalDFlashOptions: Equatable {
    public let targetWeightsPath: String
    public let drafterPath: String
    public let goldenPath: String
    public let maxBlockSize: Int
    public let totalTokenCount: Int
    /// Weights the POST-RUN reference replay loads. `nil` means the same tree the
    /// candidate loaded, which is what a local diagnostic run wants; a ranked run
    /// points this (and the reference worker executable) at the pinned baseline.
    public let referenceWeightsPath: String?
    public let referenceDrafterPath: String?

    public init(
        targetWeightsPath: String,
        drafterPath: String,
        goldenPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalDFlashMaxBlockSize,
        totalTokenCount: Int = MLXFastConstants.experimentalDFlashMaxTotalTokens,
        referenceWeightsPath: String? = nil,
        referenceDrafterPath: String? = nil
    ) {
        self.targetWeightsPath = targetWeightsPath
        self.drafterPath = drafterPath
        self.goldenPath = goldenPath
        self.maxBlockSize = maxBlockSize
        self.totalTokenCount = totalTokenCount
        self.referenceWeightsPath = referenceWeightsPath
        self.referenceDrafterPath = referenceDrafterPath
    }
}

/// Result of a validated DFlash measurement. Timing is parent-owned: the
/// denominator is the parent-configured token total, never a worker-reported
/// count.
public struct ExperimentalDFlashReport: Equatable {
    public let totalTokenCount: Int
    public let decodeSeconds: Double
    public let decodeSecondsPerToken: Double
    public let roundCount: Int
    public let acceptedDraftTotal: Int
    public let rejectedDraftTotal: Int
    public let declaredRowTotal: Int
    public let residualDivergenceCount: Int
    public let admissibleExactCount: Int
    public let admissibleDeclaredFrameCount: Int
    /// Rows admitted because the REFERENCE could not break its own tie.
    public let admissibleNearTieCount: Int
    public let maxOverMedianRoundLatency: Double?
    public let allTokensAdmissible: Bool
    // Fields the box measurement wrapper consumes for the L3 ledger and the
    // stall guardrail.
    public let seedTokenCount: Int
    public let targetCacheOffsetFinal: Int
    public let referenceCheckedRowTotal: Int
    public let targetTailTotal: Int
    public let maxBlockRequestSeconds: Double
    public let p50BlockRequestSeconds: Double
    public let blockSize: Int
    public let usesTrainedDrafter: Bool
    // Measured work-binding gap distribution. Recorded on every run so the
    // tolerance constant stays traceable to observation (contract Amendment 1),
    // and so an audit can see whether the binding compared anything at all.
    public let workBindingComparisonCount: Int
    public let maxTop2LogitDelta: Double
    public let meanTop2LogitDelta: Double
    public let p50Top2LogitDelta: Double
    public let p99Top2LogitDelta: Double
    public let maxTop2LogitRelativeDelta: Double
    public let p99Top2LogitRelativeDelta: Double
    /// Every recorded gap, for calibration only. The CLI publishes this ONLY on a
    /// widened-tolerance calibration run (refused on the official path): on a
    /// ranked run a per-row proximity trace against a hidden prompt is an oracle
    /// signal, which is exactly what contract layer L6 keeps out of the output.
    public let workBindingLogitDeltas: [Double]
    public let workBindingToleranceAbsolute: Double
    public let workBindingToleranceRelative: Double

    public init(
        totalTokenCount: Int,
        decodeSeconds: Double,
        roundCount: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        declaredRowTotal: Int,
        residualDivergenceCount: Int,
        admissibleExactCount: Int = 0,
        admissibleDeclaredFrameCount: Int = 0,
        admissibleNearTieCount: Int = 0,
        maxOverMedianRoundLatency: Double?,
        allTokensAdmissible: Bool,
        seedTokenCount: Int = 0,
        targetCacheOffsetFinal: Int = 0,
        referenceCheckedRowTotal: Int = 0,
        targetTailTotal: Int = 0,
        maxBlockRequestSeconds: Double = 0,
        p50BlockRequestSeconds: Double = 0,
        blockSize: Int = 0,
        usesTrainedDrafter: Bool = true,
        workBindingComparisonCount: Int = 0,
        maxTop2LogitDelta: Double = 0,
        meanTop2LogitDelta: Double = 0,
        p50Top2LogitDelta: Double = 0,
        p99Top2LogitDelta: Double = 0,
        maxTop2LogitRelativeDelta: Double = 0,
        p99Top2LogitRelativeDelta: Double = 0,
        workBindingLogitDeltas: [Double] = [],
        workBindingToleranceAbsolute: Double = 0,
        workBindingToleranceRelative: Double = 0
    ) {
        self.totalTokenCount = totalTokenCount
        self.decodeSeconds = decodeSeconds
        self.decodeSecondsPerToken = totalTokenCount > 0
            ? decodeSeconds / Double(totalTokenCount)
            : 0
        self.roundCount = roundCount
        self.acceptedDraftTotal = acceptedDraftTotal
        self.rejectedDraftTotal = rejectedDraftTotal
        self.declaredRowTotal = declaredRowTotal
        self.residualDivergenceCount = residualDivergenceCount
        self.admissibleExactCount = admissibleExactCount
        self.admissibleDeclaredFrameCount = admissibleDeclaredFrameCount
        self.admissibleNearTieCount = admissibleNearTieCount
        self.maxOverMedianRoundLatency = maxOverMedianRoundLatency
        self.allTokensAdmissible = allTokensAdmissible
        self.seedTokenCount = seedTokenCount
        self.targetCacheOffsetFinal = targetCacheOffsetFinal
        self.referenceCheckedRowTotal = referenceCheckedRowTotal
        self.targetTailTotal = targetTailTotal
        self.maxBlockRequestSeconds = maxBlockRequestSeconds
        self.p50BlockRequestSeconds = p50BlockRequestSeconds
        self.blockSize = blockSize
        self.usesTrainedDrafter = usesTrainedDrafter
        self.workBindingComparisonCount = workBindingComparisonCount
        self.maxTop2LogitDelta = maxTop2LogitDelta
        self.meanTop2LogitDelta = meanTop2LogitDelta
        self.p50Top2LogitDelta = p50Top2LogitDelta
        self.p99Top2LogitDelta = p99Top2LogitDelta
        self.maxTop2LogitRelativeDelta = maxTop2LogitRelativeDelta
        self.p99Top2LogitRelativeDelta = p99Top2LogitRelativeDelta
        self.workBindingLogitDeltas = workBindingLogitDeltas
        self.workBindingToleranceAbsolute = workBindingToleranceAbsolute
        self.workBindingToleranceRelative = workBindingToleranceRelative
    }

    public var acceptedDraftRate: Double {
        let proposed = acceptedDraftTotal + rejectedDraftTotal
        return proposed > 0 ? Double(acceptedDraftTotal) / Double(proposed) : 0
    }
}

/// Parent-side block schedule.
///
/// The parent -- never the worker -- picks each round's width. A randomized
/// schedule (contract layer L6) stops a submission from tuning a
/// confidence threshold to one fixed cadence, and keeps the total decode length
/// undisclosed: the worker only ever sees the next width.
public struct DFlashBlockSchedule {
    private var generator: SplitMix64
    private let maxBlockSize: Int
    private let minBlockSize: Int

    public init(seed: UInt64, maxBlockSize: Int, minBlockSize: Int = 2) {
        self.generator = SplitMix64(seed: seed)
        self.maxBlockSize = Swift.max(minBlockSize, maxBlockSize)
        self.minBlockSize = minBlockSize
    }

    /// Next block width. Always a full legal width -- deliberately NOT clamped
    /// to the remaining token count, which would leak the budget near the tail.
    public mutating func nextBlockSize() -> Int {
        let span = UInt64(maxBlockSize - minBlockSize + 1)
        return minBlockSize + Int(generator.next() % span)
    }
}

/// Small deterministic PRNG so a schedule is reproducible from its seed for
/// audit without pulling in a dependency.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
