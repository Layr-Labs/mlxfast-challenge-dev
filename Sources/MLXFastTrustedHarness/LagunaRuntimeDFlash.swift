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

    public init(
        sequentialArgmax: Int,
        declaredFrameArgmax: Int? = nil,
        top2Tokens: [Int] = [],
        top2Logits: [Double] = []
    ) {
        self.sequentialArgmax = sequentialArgmax
        self.declaredFrameArgmax = declaredFrameArgmax
        self.top2Tokens = top2Tokens
        self.top2Logits = top2Logits
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
/// These are numeric readouts from two different builds, so they are compared
/// with a tolerance -- never for equality. The tolerance is deliberately much
/// tighter than the degradation a cheaper verifier would introduce (a
/// reduced-depth trunk or coarser dequantization moves these at every row,
/// including the confident ones where argmax hides it) but loose enough to
/// absorb benign kernel accumulation-order differences. Calibrate on M5-C
/// before enabling official scoring.
public struct DFlashWorkBindingTolerance: Sendable {
    public let absolute: Double
    public let relative: Double

    public init(absolute: Double = 0.75, relative: Double = 0.02) {
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
/// accounting, evaluated round by round against a pinned reference.
public final class LagunaDFlashBlockValidator {
    private let oracle: any DFlashReferenceOracle
    private let totalTokenCount: Int
    private let seedTokenCount: Int
    private let tolerance: DFlashWorkBindingTolerance
    private let residualBudget: Int

    public private(set) var committedTokens = [Int]()
    public private(set) var outcomes = [DFlashValidationOutcome]()
    public private(set) var residualDivergenceCount = 0
    public private(set) var declaredRowTotal = 0
    public private(set) var roundLatencies = [Double]()
    /// Rows the parent actually obtained a reference verdict for. The box
    /// wrapper cross-checks this against the emitted total; it cannot yet equal
    /// `declaredRowTotal` because the worker does not report the token ids of
    /// REJECTED draft rows, so the reference has nothing to score them against
    /// (an L2 follow-up).
    public private(set) var referenceCheckedRowTotal = 0

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
    }

    public var remainingTokenCount: Int {
        Swift.max(0, totalTokenCount - committedTokens.count)
    }

    /// Validate one observed round. Throws on the first contract violation.
    public func accept(round: DFlashObservedRound) throws {
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

        let reference = try oracle.referenceRows(
            emittedPrefix: committedTokens,
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

        for (index, token) in round.tokens.enumerated() {
            let referenceRow = reference[index]
            let outcome: DFlashValidationOutcome
            if token == referenceRow.sequentialArgmax {
                outcome = .admissibleExact
            } else if referenceRow.admissibleTokens.contains(token) {
                // Honest frame divergence: the reference itself produces this
                // token at the width the candidate declared.
                outcome = .admissibleDeclaredFrame
            } else if referenceRow.top2Tokens.contains(token) {
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
                for pair in 0 ..< pairCount
                where !tolerance.matches(
                    candidate: candidateLogits[pair],
                    reference: referenceLogits[pair]
                ) {
                    throw DFlashContractViolation(
                        kind: .workBindingLogitMismatch,
                        step: step + index,
                        detail: "row readout \(pair) outside tolerance"
                    )
                }
            }
        }

        committedTokens.append(contentsOf: round.tokens)
        declaredRowTotal += round.declaredRows
        referenceCheckedRowTotal += reference.count
        roundLatencies.append(round.latencySeconds)
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

    public init(
        targetWeightsPath: String,
        drafterPath: String,
        goldenPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalDFlashMaxBlockSize,
        totalTokenCount: Int = MLXFastConstants.experimentalDFlashMaxTotalTokens
    ) {
        self.targetWeightsPath = targetWeightsPath
        self.drafterPath = drafterPath
        self.goldenPath = goldenPath
        self.maxBlockSize = maxBlockSize
        self.totalTokenCount = totalTokenCount
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

    public init(
        totalTokenCount: Int,
        decodeSeconds: Double,
        roundCount: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        declaredRowTotal: Int,
        residualDivergenceCount: Int,
        maxOverMedianRoundLatency: Double?,
        allTokensAdmissible: Bool,
        seedTokenCount: Int = 0,
        targetCacheOffsetFinal: Int = 0,
        referenceCheckedRowTotal: Int = 0,
        targetTailTotal: Int = 0,
        maxBlockRequestSeconds: Double = 0,
        p50BlockRequestSeconds: Double = 0,
        blockSize: Int = 0,
        usesTrainedDrafter: Bool = true
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
