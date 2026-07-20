import Foundation

// COMMITTED-KV FAITHFULNESS AUDIT (experimental MTP track).
//
// Purpose: catch kernels that pass the greedy returned-token gate but drift
// the COMMITTED KV state, compounding through commit/rollback until a token
// flips beyond the gated horizon (the documented M5 step-48 incident class).
//
// Data flow (all untimed, strictly after the trusted parent captured
// elapsedSeconds):
//   1. The CANDIDATE worker reads its ACTUAL committed block KV cache and
//      emits bounded per-layer digests, a strided set of raw committed rows
//      (bf16, base64), and a next-decision probe (the pre-argmax logit
//      digest for the last committed token).
//   2. The PINNED-REFERENCE worker (rotated baseline tree) serially replays
//      the SAME committed token sequence (seed prefill + K=1 teacher-forced
//      forwards into a fresh cache) and emits the SAME digest schema plus
//      the serial argmax after every replayed token.
//   3. The TRUSTED PARENT compares candidate digests against reference
//      digests within the envelopes below, requires the serial replay
//      argmaxes to equal the committed tokens exactly, requires the probe
//      argmax to match, and recomputes the strided raw-row statistics in
//      plain Swift as an anti-tamper spot check on the candidate's own
//      (editable-kernel-computed) reductions.
//
// Everything in this file is pure scalar/host math: no MLX, no model code.
// It compiles into both the trusted parent and the worker so the schema can
// never skew between them.

public struct MTPCommittedTensorStats: Codable, Equatable, Sendable {
    public let l2: Double
    public let maxAbs: Double
    public let mean: Double

    public init(l2: Double, maxAbs: Double, mean: Double) {
        self.l2 = l2
        self.maxAbs = maxAbs
        self.mean = mean
    }

    enum CodingKeys: String, CodingKey {
        case l2
        case maxAbs = "max_abs"
        case mean
    }

    public var isFinite: Bool {
        l2.isFinite && maxAbs.isFinite && mean.isFinite
    }
}

public struct MTPCommittedLayerDigest: Codable, Equatable, Sendable {
    public let layerIndex: Int
    public let retainedPositionCount: Int
    public let keyStats: MTPCommittedTensorStats
    public let valueStats: MTPCommittedTensorStats

    public init(
        layerIndex: Int,
        retainedPositionCount: Int,
        keyStats: MTPCommittedTensorStats,
        valueStats: MTPCommittedTensorStats
    ) {
        self.layerIndex = layerIndex
        self.retainedPositionCount = retainedPositionCount
        self.keyStats = keyStats
        self.valueStats = valueStats
    }

    enum CodingKeys: String, CodingKey {
        case layerIndex = "layer_index"
        case retainedPositionCount = "retained_position_count"
        case keyStats = "key_stats"
        case valueStats = "value_stats"
    }
}

public struct MTPCommittedRowSample: Codable, Equatable, Sendable {
    public let layerIndex: Int
    /// Physical position within the retained window (post-wrap positions are
    /// physical slots, not logical offsets; both sides sample identically).
    public let position: Int
    public let keyStats: MTPCommittedTensorStats
    public let valueStats: MTPCommittedTensorStats
    public let keyRowBF16Base64: String
    public let valueRowBF16Base64: String

    public init(
        layerIndex: Int,
        position: Int,
        keyStats: MTPCommittedTensorStats,
        valueStats: MTPCommittedTensorStats,
        keyRowBF16Base64: String,
        valueRowBF16Base64: String
    ) {
        self.layerIndex = layerIndex
        self.position = position
        self.keyStats = keyStats
        self.valueStats = valueStats
        self.keyRowBF16Base64 = keyRowBF16Base64
        self.valueRowBF16Base64 = valueRowBF16Base64
    }

    enum CodingKeys: String, CodingKey {
        case layerIndex = "layer_index"
        case position
        case keyStats = "key_stats"
        case valueStats = "value_stats"
        case keyRowBF16Base64 = "key_row_bf16_base64"
        case valueRowBF16Base64 = "value_row_bf16_base64"
    }
}

public struct MTPNextDecisionProbe: Codable, Equatable, Sendable {
    /// The last committed token that was forwarded to produce these logits.
    public let inputToken: Int
    public let argmaxToken: Int
    public let topLogit: Double
    public let secondLogit: Double
    public let logitStats: MTPCommittedTensorStats

    public init(
        inputToken: Int,
        argmaxToken: Int,
        topLogit: Double,
        secondLogit: Double,
        logitStats: MTPCommittedTensorStats
    ) {
        self.inputToken = inputToken
        self.argmaxToken = argmaxToken
        self.topLogit = topLogit
        self.secondLogit = secondLogit
        self.logitStats = logitStats
    }

    enum CodingKeys: String, CodingKey {
        case inputToken = "input_token"
        case argmaxToken = "argmax_token"
        case topLogit = "top_logit"
        case secondLogit = "second_logit"
        case logitStats = "logit_stats"
    }

    public var topTwoMargin: Double {
        topLogit - secondLogit
    }
}

public struct MTPCommittedStateDigest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let committedTokenCount: Int
    public let cacheOffset: Int
    public let layerDigests: [MTPCommittedLayerDigest]
    public let rowSamples: [MTPCommittedRowSample]
    public let probe: MTPNextDecisionProbe
    /// Reference-replay only: the argmax after the seed prefill.
    public let replaySeedArgmaxToken: Int?
    /// Reference-replay only: the serial argmax after each replayed token
    /// except the final probe forward.
    public let replayArgmaxTokens: [Int]?

    public init(
        schemaVersion: Int,
        committedTokenCount: Int,
        cacheOffset: Int,
        layerDigests: [MTPCommittedLayerDigest],
        rowSamples: [MTPCommittedRowSample],
        probe: MTPNextDecisionProbe,
        replaySeedArgmaxToken: Int? = nil,
        replayArgmaxTokens: [Int]? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.committedTokenCount = committedTokenCount
        self.cacheOffset = cacheOffset
        self.layerDigests = layerDigests
        self.rowSamples = rowSamples
        self.probe = probe
        self.replaySeedArgmaxToken = replaySeedArgmaxToken
        self.replayArgmaxTokens = replayArgmaxTokens
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case committedTokenCount = "committed_token_count"
        case cacheOffset = "cache_offset"
        case layerDigests = "layer_digests"
        case rowSamples = "row_samples"
        case probe
        case replaySeedArgmaxToken = "replay_seed_argmax_token"
        case replayArgmaxTokens = "replay_argmax_tokens"
    }
}

/// Bounded, publication-safe audit summary carried in the untimed gate
/// report. `would_fail_reasons` strings are built only from the fixed
/// vocabulary in `MTPCommittedStateAuditMath` plus integer counts/indices:
/// never token IDs, logit values, or any other prompt-derived analog value,
/// so surfacing them in a public workflow warning cannot leak hidden-oracle
/// content.
public struct MTPCommittedStateAuditReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let status: String
    public let referenceStatus: String
    public let wouldFail: Bool
    public let wouldFailReasons: [String]
    public let comparedLayerCount: Int
    public let comparedRowSampleCount: Int
    public let replayTokenMismatchCount: Int
    public let probeArgmaxMatched: Bool?
    public let probeTopTwoMarginCandidate: Double?
    public let probeTopTwoMarginReference: Double?
    public let maxLayerStatRelativeError: Double?
    public let maxRowSampleElementAbsoluteError: Double?
    public let selfConsistencyMaxAbsoluteError: Double?
    public let envelopeRelativeTolerance: Double
    public let envelopeAbsoluteTolerance: Double

    public init(
        schemaVersion: Int,
        status: String,
        referenceStatus: String,
        wouldFail: Bool,
        wouldFailReasons: [String],
        comparedLayerCount: Int,
        comparedRowSampleCount: Int,
        replayTokenMismatchCount: Int,
        probeArgmaxMatched: Bool?,
        probeTopTwoMarginCandidate: Double?,
        probeTopTwoMarginReference: Double?,
        maxLayerStatRelativeError: Double?,
        maxRowSampleElementAbsoluteError: Double?,
        selfConsistencyMaxAbsoluteError: Double?,
        envelopeRelativeTolerance: Double,
        envelopeAbsoluteTolerance: Double
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.referenceStatus = referenceStatus
        self.wouldFail = wouldFail
        self.wouldFailReasons = wouldFailReasons
        self.comparedLayerCount = comparedLayerCount
        self.comparedRowSampleCount = comparedRowSampleCount
        self.replayTokenMismatchCount = replayTokenMismatchCount
        self.probeArgmaxMatched = probeArgmaxMatched
        self.probeTopTwoMarginCandidate = probeTopTwoMarginCandidate
        self.probeTopTwoMarginReference = probeTopTwoMarginReference
        self.maxLayerStatRelativeError = maxLayerStatRelativeError
        self.maxRowSampleElementAbsoluteError = maxRowSampleElementAbsoluteError
        self.selfConsistencyMaxAbsoluteError = selfConsistencyMaxAbsoluteError
        self.envelopeRelativeTolerance = envelopeRelativeTolerance
        self.envelopeAbsoluteTolerance = envelopeAbsoluteTolerance
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case referenceStatus = "reference_status"
        case wouldFail = "would_fail"
        case wouldFailReasons = "would_fail_reasons"
        case comparedLayerCount = "compared_layer_count"
        case comparedRowSampleCount = "compared_row_sample_count"
        case replayTokenMismatchCount = "replay_token_mismatch_count"
        case probeArgmaxMatched = "probe_argmax_matched"
        case probeTopTwoMarginCandidate = "probe_top_two_margin_candidate"
        case probeTopTwoMarginReference = "probe_top_two_margin_reference"
        case maxLayerStatRelativeError = "max_layer_stat_relative_error"
        case maxRowSampleElementAbsoluteError = "max_row_sample_element_absolute_error"
        case selfConsistencyMaxAbsoluteError = "self_consistency_max_absolute_error"
        case envelopeRelativeTolerance = "envelope_relative_tolerance"
        case envelopeAbsoluteTolerance = "envelope_absolute_tolerance"
    }
}

public struct MTPCommittedStateEnvelope: Equatable, Sendable {
    public let relativeTolerance: Double
    public let absoluteTolerance: Double

    public init(relativeTolerance: Double, absoluteTolerance: Double) {
        self.relativeTolerance = relativeTolerance
        self.absoluteTolerance = absoluteTolerance
    }

    public func admits(_ candidate: Double, _ reference: Double) -> Bool {
        guard candidate.isFinite, reference.isFinite else { return false }
        let scale = max(abs(candidate), abs(reference))
        return abs(candidate - reference)
            <= absoluteTolerance + relativeTolerance * scale
    }
}

public struct MTPCommittedStateAuditFindings: Equatable, Sendable {
    public var comparedLayerCount = 0
    public var comparedRowSampleCount = 0
    public var replayTokenMismatchCount = 0
    public var probeArgmaxMatched: Bool?
    public var maxLayerStatRelativeError = 0.0
    public var maxRowSampleElementAbsoluteError = 0.0
    public var selfConsistencyMaxAbsoluteError = 0.0
    public var wouldFailReasons: [String] = []

    public init() {}

    public var wouldFail: Bool {
        !wouldFailReasons.isEmpty
    }
}

public enum MTPCommittedStateAuditMath {
    /// Bounded reason vocabulary: category strings plus integer counts and
    /// indices only. Never embed token IDs, logits, or raw tensor values.
    static let reasonNonFiniteCandidate = "candidate_digest_non_finite"
    static let reasonNonFiniteReference = "reference_digest_non_finite"
    static let reasonSelfConsistency = "candidate_row_stats_disagree_with_raw_rows"
    static let reasonReplaySeedMismatch = "reference_replay_seed_argmax_mismatch"
    static let reasonReplayTokenMismatch = "reference_replay_argmax_mismatch"
    static let reasonReplayShape = "reference_replay_shape_invalid"
    static let reasonGeometryMismatch = "digest_geometry_mismatch"
    static let reasonLayerEnvelope = "layer_digest_outside_envelope"
    static let reasonRowSampleEnvelope = "row_sample_outside_envelope"
    static let reasonRowSampleUndecodable = "row_sample_payload_undecodable"
    static let reasonProbeArgmaxMismatch = "probe_argmax_mismatch"
    static let reasonProbeEnvelope = "probe_logit_digest_outside_envelope"

    /// Decode a base64 payload of little-endian bf16 values into Float32.
    /// bf16 -> f32 is exact (bf16 is the upper half of the f32 bit pattern).
    public static func bf16Base64ToFloats(_ base64: String) -> [Float]? {
        guard let data = Data(base64Encoded: base64),
              data.count % 2 == 0
        else {
            return nil
        }
        var result = [Float]()
        result.reserveCapacity(data.count / 2)
        var index = data.startIndex
        while index < data.endIndex {
            let low = UInt16(data[index])
            let high = UInt16(data[data.index(after: index)])
            let bits = UInt32(low | (high << 8)) << 16
            result.append(Float(bitPattern: bits))
            index = data.index(index, offsetBy: 2)
        }
        return result
    }

    /// Encode Float32 values that are exactly representable in bf16 (i.e.
    /// values that originated as bf16) into a little-endian bf16 base64
    /// payload. Truncation is exact for such values.
    public static func floatsToBF16Base64(_ values: [Float]) -> String {
        var data = Data(capacity: values.count * 2)
        for value in values {
            let upper = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            data.append(UInt8(truncatingIfNeeded: upper))
            data.append(UInt8(truncatingIfNeeded: upper >> 8))
        }
        return data.base64EncodedString()
    }

    /// Plain-Swift float64-accumulated statistics. This is the parent's
    /// anti-tamper recompute; the worker uses the identical function on the
    /// identical decoded floats so an honest worker matches bit-for-bit.
    public static func stats(of values: [Float]) -> MTPCommittedTensorStats {
        var sumSquares = 0.0
        var sum = 0.0
        var maxAbs = 0.0
        for value in values {
            let v = Double(value)
            sumSquares += v * v
            sum += v
            maxAbs = max(maxAbs, abs(v))
        }
        let count = Double(max(values.count, 1))
        return MTPCommittedTensorStats(
            l2: sumSquares.squareRoot(),
            maxAbs: maxAbs,
            mean: sum / count
        )
    }

    static func statsWithinEnvelope(
        _ candidate: MTPCommittedTensorStats,
        _ reference: MTPCommittedTensorStats,
        envelope: MTPCommittedStateEnvelope
    ) -> Bool {
        envelope.admits(candidate.l2, reference.l2)
            && envelope.admits(candidate.maxAbs, reference.maxAbs)
            && envelope.admits(candidate.mean, reference.mean)
    }

    static func maxRelativeError(
        _ candidate: MTPCommittedTensorStats,
        _ reference: MTPCommittedTensorStats
    ) -> Double {
        func relative(_ c: Double, _ r: Double) -> Double {
            guard c.isFinite, r.isFinite else { return .infinity }
            let scale = max(abs(c), abs(r), 1e-12)
            return abs(c - r) / scale
        }
        return max(
            relative(candidate.l2, reference.l2),
            max(
                relative(candidate.maxAbs, reference.maxAbs),
                relative(candidate.mean, reference.mean)
            )
        )
    }

    static func digestIsFinite(_ digest: MTPCommittedStateDigest) -> Bool {
        digest.layerDigests.allSatisfy {
            $0.keyStats.isFinite && $0.valueStats.isFinite
        }
            && digest.rowSamples.allSatisfy {
                $0.keyStats.isFinite && $0.valueStats.isFinite
            }
            && digest.probe.logitStats.isFinite
            && digest.probe.topLogit.isFinite
            && digest.probe.secondLogit.isFinite
    }

    /// Candidate-only checks: finiteness plus the raw-row anti-tamper
    /// recompute (parent recomputes the strided raw rows' statistics in
    /// plain Swift and requires the candidate's declared row stats to match
    /// exactly up to accumulation determinism -- both sides use `stats(of:)`
    /// on identical floats, so the comparison is exact equality with a tiny
    /// slack for future implementations that batch differently).
    public static func auditCandidateSelfConsistency(
        _ digest: MTPCommittedStateDigest,
        findings: inout MTPCommittedStateAuditFindings
    ) {
        if !digestIsFinite(digest) {
            findings.wouldFailReasons.append(reasonNonFiniteCandidate)
        }
        let selfEnvelope = MTPCommittedStateEnvelope(
            relativeTolerance: 1e-6,
            absoluteTolerance: 1e-9
        )
        for sample in digest.rowSamples {
            guard let keyRow = bf16Base64ToFloats(sample.keyRowBF16Base64),
                  let valueRow = bf16Base64ToFloats(sample.valueRowBF16Base64),
                  !keyRow.isEmpty,
                  !valueRow.isEmpty
            else {
                findings.wouldFailReasons.append(
                    "\(reasonRowSampleUndecodable)[layer=\(sample.layerIndex)]"
                )
                continue
            }
            let recomputedKey = stats(of: keyRow)
            let recomputedValue = stats(of: valueRow)
            let keyError = maxAbsoluteStatError(recomputedKey, sample.keyStats)
            let valueError = maxAbsoluteStatError(
                recomputedValue,
                sample.valueStats
            )
            findings.selfConsistencyMaxAbsoluteError = max(
                findings.selfConsistencyMaxAbsoluteError,
                max(keyError, valueError)
            )
            if !statsWithinEnvelope(
                recomputedKey,
                sample.keyStats,
                envelope: selfEnvelope
            )
                || !statsWithinEnvelope(
                    recomputedValue,
                    sample.valueStats,
                    envelope: selfEnvelope
                )
            {
                findings.wouldFailReasons.append(
                    "\(reasonSelfConsistency)[layer=\(sample.layerIndex),position=\(sample.position)]"
                )
            }
        }
    }

    static func maxAbsoluteStatError(
        _ candidate: MTPCommittedTensorStats,
        _ reference: MTPCommittedTensorStats
    ) -> Double {
        guard candidate.isFinite, reference.isFinite else { return .infinity }
        return max(
            abs(candidate.l2 - reference.l2),
            max(
                abs(candidate.maxAbs - reference.maxAbs),
                abs(candidate.mean - reference.mean)
            )
        )
    }

    /// Full candidate-vs-reference audit. `expectedSeedToken` is the gated
    /// seed argmax and `expectedCommittedTokens` the gated decode tokens;
    /// both already passed the exact-token gate, so mismatches here indict
    /// the REPLAY (reference/infrastructure) or the candidate's committed
    /// state, never re-litigate the gate itself.
    public static func audit(
        candidate: MTPCommittedStateDigest,
        reference: MTPCommittedStateDigest?,
        expectedSeedToken: Int,
        expectedCommittedTokens: [Int],
        envelope: MTPCommittedStateEnvelope
    ) -> MTPCommittedStateAuditFindings {
        var findings = MTPCommittedStateAuditFindings()
        auditCandidateSelfConsistency(candidate, findings: &findings)

        guard let reference else {
            return findings
        }
        if !digestIsFinite(reference) {
            findings.wouldFailReasons.append(reasonNonFiniteReference)
            return findings
        }

        // Serial replay argmaxes must reproduce the committed sequence with
        // zero mismatches: replayArgmaxTokens[i] is the reference argmax
        // after forwarding [seedToken] + committed[0..<(count-1)], so it
        // must equal committed[i].
        if let seedArgmax = reference.replaySeedArgmaxToken,
           seedArgmax != expectedSeedToken
        {
            findings.wouldFailReasons.append(reasonReplaySeedMismatch)
        }
        if let replayArgmaxTokens = reference.replayArgmaxTokens {
            if replayArgmaxTokens.count != expectedCommittedTokens.count {
                findings.wouldFailReasons.append(
                    "\(reasonReplayShape)[count=\(replayArgmaxTokens.count)]"
                )
            } else {
                for (index, argmax) in replayArgmaxTokens.enumerated()
                where argmax != expectedCommittedTokens[index] {
                    findings.replayTokenMismatchCount += 1
                }
                if findings.replayTokenMismatchCount > 0 {
                    findings.wouldFailReasons.append(
                        "\(reasonReplayTokenMismatch)[count=\(findings.replayTokenMismatchCount)]"
                    )
                }
            }
        } else {
            findings.wouldFailReasons.append("\(reasonReplayShape)[missing]")
        }

        guard candidate.committedTokenCount == reference.committedTokenCount,
              candidate.cacheOffset == reference.cacheOffset,
              candidate.layerDigests.count == reference.layerDigests.count
        else {
            findings.wouldFailReasons.append(reasonGeometryMismatch)
            return findings
        }

        for (candidateLayer, referenceLayer) in zip(
            candidate.layerDigests,
            reference.layerDigests
        ) {
            guard candidateLayer.layerIndex == referenceLayer.layerIndex,
                  candidateLayer.retainedPositionCount
                      == referenceLayer.retainedPositionCount
            else {
                findings.wouldFailReasons.append(
                    "\(reasonGeometryMismatch)[layer=\(candidateLayer.layerIndex)]"
                )
                continue
            }
            findings.comparedLayerCount += 1
            findings.maxLayerStatRelativeError = max(
                findings.maxLayerStatRelativeError,
                max(
                    maxRelativeError(
                        candidateLayer.keyStats,
                        referenceLayer.keyStats
                    ),
                    maxRelativeError(
                        candidateLayer.valueStats,
                        referenceLayer.valueStats
                    )
                )
            )
            if !statsWithinEnvelope(
                candidateLayer.keyStats,
                referenceLayer.keyStats,
                envelope: envelope
            )
                || !statsWithinEnvelope(
                    candidateLayer.valueStats,
                    referenceLayer.valueStats,
                    envelope: envelope
                )
            {
                findings.wouldFailReasons.append(
                    "\(reasonLayerEnvelope)[layer=\(candidateLayer.layerIndex)]"
                )
            }
        }

        var referenceSamples = [String: MTPCommittedRowSample]()
        for sample in reference.rowSamples {
            referenceSamples["\(sample.layerIndex):\(sample.position)"] = sample
        }
        for candidateSample in candidate.rowSamples {
            let key = "\(candidateSample.layerIndex):\(candidateSample.position)"
            guard let referenceSample = referenceSamples[key] else {
                findings.wouldFailReasons.append(
                    "\(reasonGeometryMismatch)[row=\(key)]"
                )
                continue
            }
            guard
                let candidateKeyRow = bf16Base64ToFloats(
                    candidateSample.keyRowBF16Base64
                ),
                let candidateValueRow = bf16Base64ToFloats(
                    candidateSample.valueRowBF16Base64
                ),
                let referenceKeyRow = bf16Base64ToFloats(
                    referenceSample.keyRowBF16Base64
                ),
                let referenceValueRow = bf16Base64ToFloats(
                    referenceSample.valueRowBF16Base64
                ),
                candidateKeyRow.count == referenceKeyRow.count,
                candidateValueRow.count == referenceValueRow.count
            else {
                findings.wouldFailReasons.append(
                    "\(reasonRowSampleUndecodable)[row=\(key)]"
                )
                continue
            }
            findings.comparedRowSampleCount += 1
            var rowWithinEnvelope = true
            for (candidateElement, referenceElement) in zip(
                candidateKeyRow + candidateValueRow,
                referenceKeyRow + referenceValueRow
            ) {
                let candidateValue = Double(candidateElement)
                let referenceValue = Double(referenceElement)
                findings.maxRowSampleElementAbsoluteError = max(
                    findings.maxRowSampleElementAbsoluteError,
                    abs(candidateValue - referenceValue)
                )
                if !envelope.admits(candidateValue, referenceValue) {
                    rowWithinEnvelope = false
                }
            }
            if !rowWithinEnvelope {
                findings.wouldFailReasons.append(
                    "\(reasonRowSampleEnvelope)[row=\(key)]"
                )
            }
        }

        findings.probeArgmaxMatched =
            candidate.probe.argmaxToken == reference.probe.argmaxToken
            && candidate.probe.inputToken == reference.probe.inputToken
        if findings.probeArgmaxMatched != true {
            findings.wouldFailReasons.append(reasonProbeArgmaxMismatch)
        }
        if !statsWithinEnvelope(
            candidate.probe.logitStats,
            reference.probe.logitStats,
            envelope: envelope
        )
            || !envelope.admits(
                candidate.probe.topLogit,
                reference.probe.topLogit
            )
            || !envelope.admits(
                candidate.probe.secondLogit,
                reference.probe.secondLogit
            )
        {
            findings.wouldFailReasons.append(reasonProbeEnvelope)
        }

        return findings
    }
}
