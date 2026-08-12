import Foundation
import MLXFastCore

// Trusted-parent contract types for the Qwen 3.6 native-MTP track
// (`qwen3.6-27b-mtp-v1`). Links no MLX: every type here is arithmetic over what
// the sandboxed worker reported plus what the pinned reference replayed.
//
// THE FIDELITY CONTRACT, and how it differs from DFlash's. Native MTP on this
// checkpoint is measured exact-greedy against the serial trajectory (12/12 runs
// to 512 tokens, all EOS branches, MTPLX-corroborated), which is a STRONGER
// property than the DFlash track can assert: DFlash needed a bounded near-tie
// budget because its target's own block-shaped forward diverges from its
// sequential forward with no drafter involved. Here the emitted stream is
// required to equal the serial trajectory OUTRIGHT -- `allTokensMatched` is not
// a budget, it is an equality.
//
// The one place a tolerance survives is the REJECTED TAIL, and it is not about
// emitted tokens at all. Those rows are replayed by the pinned reference in the
// candidate's own verify frame; the argmax should agree, but the two frames were
// built differently (the candidate's cache carries a verify + rollback + repair
// history, the reference's a width-1 walk), so a near-tie row can legitimately
// flip. Such a row is admitted and COUNTED, never hidden.
//
// A claim about this track should be worded "decision-level exact up to bf16
// shape-dependent reduction geometry", never "bitwise": the logit VALUES do move
// between a batched and a sequential frame; what does not move is which token
// wins.

/// Wire shape of the track's reference rows.
///
/// Deliberately its own type rather than a reuse of `DFlashReferenceGolden`,
/// despite carrying the same keys: the Laguna/DFlash surface is scheduled for
/// excision when the dedicated Qwen repository is created, and this track's
/// golden format must not go with it. The keys are identical on purpose -- the
/// local runner and the provisioning workflow already validate goldens in that
/// shape.
public struct QwenMTPReferenceGolden: Codable {
    public struct Row: Codable {
        public let sequentialArgmax: Int
        public let top2Tokens: [Int]?
        public let top2Logits: [Double]?
        public let top1Logit: Double?

        enum CodingKeys: String, CodingKey {
            case sequentialArgmax = "sequential_argmax"
            case top2Tokens = "top2_tokens"
            case top2Logits = "top2_logits"
            case top1Logit = "top1_logit"
        }

        public init(
            sequentialArgmax: Int,
            top2Tokens: [Int]?,
            top2Logits: [Double]?,
            top1Logit: Double?
        ) {
            self.sequentialArgmax = sequentialArgmax
            self.top2Tokens = top2Tokens
            self.top2Logits = top2Logits
            self.top1Logit = top1Logit
        }
    }

    public let seedTokens: [Int]
    /// The serial argmax of the seed's last row -- the run's FIRST emitted token.
    /// It has no `rows` entry because no row precedes it; `rows[i]` describes the
    /// token emitted at index `i + 1`.
    public let referenceSeedToken: Int
    public let rows: [Row]
    public let referenceSelfConsistent: Bool?
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

public struct QwenMTPContractViolation: Error, CustomStringConvertible {
    public enum Kind: String, Sendable, CaseIterable {
        case seedTokenMismatch = "seed_token_mismatch"
        case tokenNotExact = "token_not_exact"
        case rowLedgerNotClosed = "row_ledger_not_closed"
        case rowNotReferenceChecked = "row_not_reference_checked"
        case rejectedTailDiverged = "rejected_tail_diverged"
        case referenceNotSelfConsistent = "reference_not_self_consistent"
        case stopTokenInsideWindow = "stop_token_inside_window"
        case headNotUsed = "head_not_used"
    }

    public let kind: Kind
    public let step: Int?
    public let detail: String

    public init(kind: Kind, step: Int? = nil, detail: String) {
        self.kind = kind
        self.step = step
        self.detail = detail
    }

    public var description: String {
        "qwen-mtp contract violation [\(kind.rawValue)]"
            + (step.map { " at step \($0)" } ?? "") + ": " + detail
    }
}

/// One journalled round, as the parent recorded it.
public struct QwenMTPObservedRound: Sendable {
    /// Index into the emitted stream of this round's primary.
    public let emittedBaseIndex: Int
    public let requestedDepth: Int
    public let tokens: [Int]
    public let declaredRows: Int
    public let draftTokens: [Int]
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
    public let targetCacheOffset: Int
    public let latencySeconds: Double

    public init(
        emittedBaseIndex: Int,
        requestedDepth: Int,
        tokens: [Int],
        declaredRows: Int,
        draftTokens: [Int],
        acceptedDraftCount: Int,
        rejectedDraftCount: Int,
        perRowTop2Tokens: [[Int]],
        perRowTop2Logits: [[Double]],
        targetCacheOffset: Int,
        latencySeconds: Double
    ) {
        self.emittedBaseIndex = emittedBaseIndex
        self.requestedDepth = requestedDepth
        self.tokens = tokens
        self.declaredRows = declaredRows
        self.draftTokens = draftTokens
        self.acceptedDraftCount = acceptedDraftCount
        self.rejectedDraftCount = rejectedDraftCount
        self.perRowTop2Tokens = perRowTop2Tokens
        self.perRowTop2Logits = perRowTop2Logits
        self.targetCacheOffset = targetCacheOffset
        self.latencySeconds = latencySeconds
    }

    /// The candidate's own verify input for this round, reconstructed by the
    /// PARENT from its committed primary plus the journalled drafts, so the
    /// worker never chooses the row-0 token of its own audit.
    public var verifyBlockTokens: [Int] {
        guard let primary = tokens.first, !draftTokens.isEmpty else { return [] }
        return [primary] + draftTokens
    }
}

/// One audited row of the ledger, and how it was reference-checked.
public struct QwenMTPLedgerRow: Sendable {
    public enum Kind: String, Sendable {
        /// A row of the batched verify forward that scored a head proposal.
        case draft
        /// The single per-round row whose argmax becomes the next primary.
        case targetTail
    }

    public enum ReferenceSource: String, Sendable {
        /// Checked against the serial golden's row at this emitted index.
        case serialGolden = "serial_golden"
        /// Checked against the pinned reference's replay of the candidate's own
        /// verify block -- the only way to reach a row conditioned on a rejected
        /// prefix, which no serial golden describes.
        case verifyBlockReplay = "verify_block_replay"
    }

    public let rowIndex: Int
    public let round: Int
    public let kind: Kind
    /// Position within the round's draft window, or nil for the tail row.
    public let draftIndex: Int?
    public let accepted: Bool
    /// The head's proposal for a draft row; the committed token for a tail row.
    public let token: Int
    public let top2Tokens: [Int]
    public let top2Logits: [Double]
    public let referenceToken: Int
    public let referenceCheckedBy: ReferenceSource
    /// Reference top1-top2 gap. A near-zero gap is where a batched-versus-
    /// sequential argmax may legitimately flip; a large gap means the two paths
    /// genuinely disagree, which is a logic bug and not numerics.
    public let referenceMargin: Double

    public var top2Margin: Double {
        guard top2Logits.count >= 2 else { return .infinity }
        return top2Logits[0] - top2Logits[1]
    }
}

/// How close two logit readouts of the same row must be before a token
/// disagreement is read as numerics rather than as a logic bug.
///
/// PROVISIONAL AND UNCALIBRATED FOR THIS CHECKPOINT. It applies ONLY to
/// rejected-tail rows replayed in a different cache history; no emitted token is
/// ever admitted by it. `1e-2` is the margin the validated driver used to
/// classify a mismatch as a tie-flip rather than a logic bug, carried over
/// unchanged. Box-3 calibration required before the first ranked session.
public struct QwenMTPNearTieTolerance: Sendable {
    public let referenceMargin: Double

    public init(referenceMargin: Double = 1e-2) {
        self.referenceMargin = referenceMargin
    }
}

/// Options shared by both verbs. `depth == 1` is the serial control.
public struct QwenMTPOptions: Equatable {
    public let targetWeightsPath: String
    public let mtpHeadPath: String
    public let goldenPath: String
    public let depth: Int
    public let totalTokenCount: Int
    public let referenceWeightsPath: String?
    public let referenceHeadPath: String?

    public init(
        targetWeightsPath: String,
        mtpHeadPath: String,
        goldenPath: String,
        depth: Int,
        totalTokenCount: Int,
        referenceWeightsPath: String? = nil,
        referenceHeadPath: String? = nil
    ) {
        self.targetWeightsPath = targetWeightsPath
        self.mtpHeadPath = mtpHeadPath
        self.goldenPath = goldenPath
        self.depth = depth
        self.totalTokenCount = totalTokenCount
        self.referenceWeightsPath = referenceWeightsPath
        self.referenceHeadPath = referenceHeadPath
    }
}

/// The evidence payload both verbs emit. Field names are the consumer contract:
/// `deploy/qwen36-mtp/measure-qwen-mtp-job.sh`, the ranked workflow's gate jq,
/// and `benchmark-qwen-mtp.sh`.
public struct QwenMTPReport: Equatable {
    public let verb: String
    public let depth: Int
    public let seedTokenCount: Int
    public let decodeTokenCount: Int
    public let emittedTokenTotal: Int
    public let allTokensMatched: Bool
    public let firstDivergenceIndex: Int?
    public let firstDivergenceReferenceMargin: Double?
    public let roundCount: Int
    public let acceptedDraftTotal: Int
    public let rejectedDraftTotal: Int
    public let targetTailTotal: Int
    public let declaredRowTotal: Int
    public let referenceCheckedRowTotal: Int
    public let rejectedRowsReferenceChecked: Int
    public let verifyBlockReplayedRoundCount: Int
    public let residualDivergenceCount: Int
    public let maxRejectedTailLogitDelta: Double
    public let targetCacheOffsetFinal: Int
    public let decodeSeconds: Double
    public let maxRoundRequestSeconds: Double
    public let p50RoundRequestSeconds: Double
    /// Retained ledger rows. `mtp-verify` carries the whole ledger as the
    /// fidelity evidence; `mtp-timed` carries none (the timed report is a timing
    /// artifact, and a 512-token ledger of top-2 readouts would dominate it).
    public let ledger: [QwenMTPLedgerRow]

    public var usesNativeMTPHead: Bool { depth > 1 }
    public var decodeSecondsPerToken: Double {
        decodeSeconds / Double(Swift.max(decodeTokenCount, 1))
    }
    public var acceptedDraftRate: Double {
        let drafted = acceptedDraftTotal + rejectedDraftTotal
        return drafted > 0 ? Double(acceptedDraftTotal) / Double(drafted) : 0
    }
    /// `parity_all_ok`: the bit-exact token gate as a single top-level boolean.
    /// It is an AND of the exactness verdict and ledger closure, not a copy of
    /// either -- a run whose ledger did not close has not proven its tokens were
    /// produced by the work it declared, whatever the tokens happened to be.
    public var parityAllOK: Bool {
        allTokensMatched
            && referenceCheckedRowTotal == declaredRowTotal
            && acceptedDraftTotal + rejectedDraftTotal + targetTailTotal
                == declaredRowTotal
    }

    public init(
        verb: String,
        depth: Int,
        seedTokenCount: Int,
        decodeTokenCount: Int,
        emittedTokenTotal: Int,
        allTokensMatched: Bool,
        firstDivergenceIndex: Int?,
        firstDivergenceReferenceMargin: Double?,
        roundCount: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        targetTailTotal: Int,
        declaredRowTotal: Int,
        referenceCheckedRowTotal: Int,
        rejectedRowsReferenceChecked: Int,
        verifyBlockReplayedRoundCount: Int,
        residualDivergenceCount: Int,
        maxRejectedTailLogitDelta: Double,
        targetCacheOffsetFinal: Int,
        decodeSeconds: Double,
        maxRoundRequestSeconds: Double,
        p50RoundRequestSeconds: Double,
        ledger: [QwenMTPLedgerRow] = []
    ) {
        self.verb = verb
        self.depth = depth
        self.seedTokenCount = seedTokenCount
        self.decodeTokenCount = decodeTokenCount
        self.emittedTokenTotal = emittedTokenTotal
        self.allTokensMatched = allTokensMatched
        self.firstDivergenceIndex = firstDivergenceIndex
        self.firstDivergenceReferenceMargin = firstDivergenceReferenceMargin
        self.roundCount = roundCount
        self.acceptedDraftTotal = acceptedDraftTotal
        self.rejectedDraftTotal = rejectedDraftTotal
        self.targetTailTotal = targetTailTotal
        self.declaredRowTotal = declaredRowTotal
        self.referenceCheckedRowTotal = referenceCheckedRowTotal
        self.rejectedRowsReferenceChecked = rejectedRowsReferenceChecked
        self.verifyBlockReplayedRoundCount = verifyBlockReplayedRoundCount
        self.residualDivergenceCount = residualDivergenceCount
        self.maxRejectedTailLogitDelta = maxRejectedTailLogitDelta
        self.targetCacheOffsetFinal = targetCacheOffsetFinal
        self.decodeSeconds = decodeSeconds
        self.maxRoundRequestSeconds = maxRoundRequestSeconds
        self.p50RoundRequestSeconds = p50RoundRequestSeconds
        self.ledger = ledger
    }
}

extension QwenMTPLedgerRow: Equatable {}

/// Parent-side ledger arithmetic, shared by both verbs.
///
/// Every equation the box wrapper's `check_row_accounting` enforces is closed
/// HERE first, from rows the parent reference-checked itself, so the wrapper is
/// auditing a ledger rather than a set of worker-reported counters.
public enum QwenMTPRowAccounting {
    /// Rows a round of `depth` declares: `depth` draft rows plus one target tail
    /// row. This is the constant the wrapper's `rows_per_round` mirrors; if one
    /// moves, both move.
    public static func rowsPerRound(depth: Int) -> Int { depth + 1 }

    /// The wrapper's L3 equations, as a single predicate.
    public static func closes(
        emittedTokenTotal: Int,
        configuredTokenTotal: Int,
        declaredRowTotal: Int,
        referenceCheckedRowTotal: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        targetTailTotal: Int,
        roundCount: Int,
        depth: Int,
        seedTokenCount: Int,
        targetCacheOffsetFinal: Int
    ) -> Bool {
        emittedTokenTotal == configuredTokenTotal
            && declaredRowTotal >= emittedTokenTotal
            && declaredRowTotal <= rowsPerRound(depth: depth) * roundCount
            && acceptedDraftTotal + rejectedDraftTotal + targetTailTotal
                == declaredRowTotal
            && referenceCheckedRowTotal == declaredRowTotal
            && targetCacheOffsetFinal == seedTokenCount + emittedTokenTotal
    }
}
