import Foundation
@testable import MLXFastCore
@testable import MLXFastRuntimeWorkerSupport
import Testing

// Correctness-gate hardening for the MTP track: committed-KV audit math,
// the new worker-protocol surface, per-run gate rotation, and the untimed
// workflow wiring (extended legs, reference replay, paired semantic floor).

// MARK: - Committed-state audit math (pure MLXFastCore)

private func hardeningStats(_ values: [Float]) -> MTPCommittedTensorStats {
    MTPCommittedStateAuditMath.stats(of: values)
}

private func hardeningRowSample(
    layerIndex: Int,
    position: Int,
    keyRow: [Float],
    valueRow: [Float]
) -> MTPCommittedRowSample {
    MTPCommittedRowSample(
        layerIndex: layerIndex,
        position: position,
        keyStats: hardeningStats(keyRow),
        valueStats: hardeningStats(valueRow),
        keyRowBF16Base64: MTPCommittedStateAuditMath.floatsToBF16Base64(keyRow),
        valueRowBF16Base64: MTPCommittedStateAuditMath.floatsToBF16Base64(
            valueRow
        )
    )
}

private func hardeningDigest(
    committedTokenCount: Int = 4,
    cacheOffset: Int = 16,
    layerValues: [Float] = [0.5, -1.25, 3.0, 0.0078125],
    rowKey: [Float] = [0.5, -1.25, 3.0, 0.0078125],
    rowValue: [Float] = [2.0, -0.5, 0.25, 1.0],
    probeArgmax: Int = 42,
    probeInput: Int = 7,
    replaySeedArgmaxToken: Int? = nil,
    replayArgmaxTokens: [Int]? = nil
) -> MTPCommittedStateDigest {
    let layerStats = hardeningStats(layerValues)
    return MTPCommittedStateDigest(
        schemaVersion: MLXFastConstants.mtpCommittedStateAuditSchemaVersion,
        committedTokenCount: committedTokenCount,
        cacheOffset: cacheOffset,
        layerDigests: [
            MTPCommittedLayerDigest(
                layerIndex: 0,
                retainedPositionCount: 16,
                keyStats: layerStats,
                valueStats: layerStats
            ),
            MTPCommittedLayerDigest(
                layerIndex: 1,
                retainedPositionCount: 16,
                keyStats: layerStats,
                valueStats: layerStats
            ),
        ],
        rowSamples: [
            hardeningRowSample(
                layerIndex: 0,
                position: 3,
                keyRow: rowKey,
                valueRow: rowValue
            )
        ],
        probe: MTPNextDecisionProbe(
            inputToken: probeInput,
            argmaxToken: probeArgmax,
            topLogit: 12.5,
            secondLogit: 9.25,
            logitStats: hardeningStats([12.5, 9.25, -3.0])
        ),
        replaySeedArgmaxToken: replaySeedArgmaxToken,
        replayArgmaxTokens: replayArgmaxTokens
    )
}

private let hardeningEnvelope = MTPCommittedStateEnvelope(
    relativeTolerance:
        MLXFastConstants.mtpCommittedStateAuditRelativeTolerance,
    absoluteTolerance:
        MLXFastConstants.mtpCommittedStateAuditAbsoluteTolerance
)

@Test
func committedAuditBF16PayloadRoundTripsExactly() throws {
    // Every value is exactly representable in bf16 (8-bit exponent, 7-bit
    // mantissa), so the base64 round trip must be bit-exact.
    let values: [Float] = [0.0, 0.5, -1.25, 3.0, 0.0078125, -65_536.0]
    let base64 = MTPCommittedStateAuditMath.floatsToBF16Base64(values)
    let decoded = try #require(
        MTPCommittedStateAuditMath.bf16Base64ToFloats(base64)
    )
    #expect(decoded == values)
    #expect(MTPCommittedStateAuditMath.bf16Base64ToFloats("!!!") == nil)
    // Odd byte counts cannot be bf16 payloads.
    #expect(
        MTPCommittedStateAuditMath.bf16Base64ToFloats(
            Data([0x01]).base64EncodedString()
        ) == nil
    )
}

@Test
func committedAuditStatsAreDeterministicPlainAccumulation() {
    let stats = hardeningStats([3.0, -4.0])
    #expect(abs(stats.l2 - 5.0) < 1e-12)
    #expect(stats.maxAbs == 4.0)
    #expect(abs(stats.mean - (-0.5)) < 1e-12)
    let empty = hardeningStats([])
    #expect(empty.l2 == 0 && empty.maxAbs == 0 && empty.mean == 0)
}

@Test
func committedAuditEnvelopeAdmitsAndRejects() {
    let envelope = MTPCommittedStateEnvelope(
        relativeTolerance: 1e-2,
        absoluteTolerance: 1e-3
    )
    #expect(envelope.admits(1.0, 1.0))
    #expect(envelope.admits(1.0, 1.005))
    #expect(!envelope.admits(1.0, 1.5))
    #expect(!envelope.admits(Double.nan, 1.0))
    #expect(!envelope.admits(1.0, Double.infinity))
}

@Test
func committedAuditCleanComparisonHasNoWouldFailReasons() {
    let committed = [11, 12, 13, 14]
    let candidate = hardeningDigest()
    let reference = hardeningDigest(
        replaySeedArgmaxToken: 7,
        replayArgmaxTokens: committed
    )
    let findings = MTPCommittedStateAuditMath.audit(
        candidate: candidate,
        reference: reference,
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(findings.wouldFailReasons.isEmpty)
    #expect(!findings.wouldFail)
    #expect(findings.comparedLayerCount == 2)
    #expect(findings.comparedRowSampleCount == 1)
    #expect(findings.replayTokenMismatchCount == 0)
    #expect(findings.probeArgmaxMatched == true)
}

@Test
func committedAuditCandidateOnlyModeStillChecksSelfConsistency() {
    let clean = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(),
        reference: nil,
        expectedSeedToken: 7,
        expectedCommittedTokens: [11, 12, 13, 14],
        envelope: hardeningEnvelope
    )
    #expect(clean.wouldFailReasons.isEmpty)

    // Tamper the declared row stats away from the raw payload: the plain
    // Swift recompute must catch it even without a reference.
    let row = hardeningRowSample(
        layerIndex: 0,
        position: 3,
        keyRow: [0.5, -1.25, 3.0, 0.0078125],
        valueRow: [2.0, -0.5, 0.25, 1.0]
    )
    let tamperedRow = MTPCommittedRowSample(
        layerIndex: row.layerIndex,
        position: row.position,
        keyStats: MTPCommittedTensorStats(l2: 999.0, maxAbs: 999.0, mean: 0),
        valueStats: row.valueStats,
        keyRowBF16Base64: row.keyRowBF16Base64,
        valueRowBF16Base64: row.valueRowBF16Base64
    )
    let base = hardeningDigest()
    let tampered = MTPCommittedStateDigest(
        schemaVersion: base.schemaVersion,
        committedTokenCount: base.committedTokenCount,
        cacheOffset: base.cacheOffset,
        layerDigests: base.layerDigests,
        rowSamples: [tamperedRow],
        probe: base.probe
    )
    let findings = MTPCommittedStateAuditMath.audit(
        candidate: tampered,
        reference: nil,
        expectedSeedToken: 7,
        expectedCommittedTokens: [11, 12, 13, 14],
        envelope: hardeningEnvelope
    )
    #expect(
        findings.wouldFailReasons.contains {
            $0.hasPrefix("candidate_row_stats_disagree_with_raw_rows")
        }
    )
}

@Test
func committedAuditFlagsReplayMismatchLayerDriftAndProbeFlip() {
    let committed = [11, 12, 13, 14]

    // Serial replay argmax diverging from the committed sequence.
    let replayMismatch = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(),
        reference: hardeningDigest(
            replaySeedArgmaxToken: 7,
            replayArgmaxTokens: [11, 99, 13, 14]
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(replayMismatch.replayTokenMismatchCount == 1)
    #expect(
        replayMismatch.wouldFailReasons.contains(
            "reference_replay_argmax_mismatch[count=1]"
        )
    )

    // Layer digests drifting outside the envelope.
    let layerDrift = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(layerValues: [0.5, -1.25, 3.0, 0.0078125]),
        reference: hardeningDigest(
            layerValues: [5.0, -12.5, 30.0, 0.078125],
            replaySeedArgmaxToken: 7,
            replayArgmaxTokens: committed
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(
        layerDrift.wouldFailReasons.contains {
            $0.hasPrefix("layer_digest_outside_envelope")
        }
    )
    #expect(layerDrift.maxLayerStatRelativeError > 0.5)

    // Raw committed rows drifting outside the envelope element-wise.
    let rowDrift = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(rowKey: [0.5, -1.25, 3.0, 0.0078125]),
        reference: hardeningDigest(
            rowKey: [0.5, -1.25, 4.0, 0.0078125],
            replaySeedArgmaxToken: 7,
            replayArgmaxTokens: committed
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(
        rowDrift.wouldFailReasons.contains {
            $0.hasPrefix("row_sample_outside_envelope")
        }
    )
    #expect(rowDrift.maxRowSampleElementAbsoluteError >= 1.0)

    // Next-decision probe argmax flip.
    let probeFlip = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(probeArgmax: 42),
        reference: hardeningDigest(
            probeArgmax: 43,
            replaySeedArgmaxToken: 7,
            replayArgmaxTokens: committed
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(probeFlip.probeArgmaxMatched == false)
    #expect(probeFlip.wouldFailReasons.contains("probe_argmax_mismatch"))

    // Geometry mismatch fails before tensor comparison.
    let geometry = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(cacheOffset: 16),
        reference: hardeningDigest(
            cacheOffset: 17,
            replaySeedArgmaxToken: 7,
            replayArgmaxTokens: committed
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(
        geometry.wouldFailReasons.contains("digest_geometry_mismatch")
    )

    // Seed replay mismatch.
    let seedMismatch = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(),
        reference: hardeningDigest(
            replaySeedArgmaxToken: 8,
            replayArgmaxTokens: committed
        ),
        expectedSeedToken: 7,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(
        seedMismatch.wouldFailReasons.contains(
            "reference_replay_seed_argmax_mismatch"
        )
    )
}

@Test
func committedAuditReasonsStayCategoricalAndTokenFree() {
    // The reasons surface in a public workflow warning, so they must never
    // embed token IDs, logits, or other prompt-derived analog values --
    // only fixed categories plus layer/position/count integers.
    let committed = [123_456, 234_567, 111, 222]
    let findings = MTPCommittedStateAuditMath.audit(
        candidate: hardeningDigest(probeArgmax: 123_456),
        reference: hardeningDigest(
            probeArgmax: 654_321,
            replaySeedArgmaxToken: 99_999,
            replayArgmaxTokens: [123_456, 9_999, 111, 222]
        ),
        expectedSeedToken: 88_888,
        expectedCommittedTokens: committed,
        envelope: hardeningEnvelope
    )
    #expect(!findings.wouldFailReasons.isEmpty)
    for reason in findings.wouldFailReasons {
        #expect(!reason.contains("123456"), Comment(rawValue: reason))
        #expect(!reason.contains("654321"), Comment(rawValue: reason))
        #expect(!reason.contains("99999"), Comment(rawValue: reason))
        #expect(!reason.contains("88888"), Comment(rawValue: reason))
        #expect(!reason.contains("9999"), Comment(rawValue: reason))
    }
}

// MARK: - Worker protocol surface

@Test
func committedStateAuditRequestCarriesOnlyIDAndKind() throws {
    let request = RuntimeWorkerRequest(
        id: 5,
        kind: "mtp_committed_state_audit"
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(Set(object.keys) == Set(["id", "kind"]))
}

@Test
func auditSerialReplayRequestRoundTripsAndStaysStrict() throws {
    let request = RuntimeWorkerRequest(
        id: 6,
        kind: "mtp_audit_serial_replay",
        seedTokens: [1, 2, 3],
        replayTokens: [4, 5, 6]
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(
        Set(object.keys)
            == Set(["id", "kind", "seed_tokens", "replay_tokens"])
    )
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerRequest.self,
        from: data
    )
    #expect(decoded.replayTokens == [4, 5, 6])
    #expect(decoded.seedTokens == [1, 2, 3])

    // Unknown fields remain rejected on the shared strict decoder.
    let forged = Data(
        """
        {"id":6,"kind":"mtp_audit_serial_replay","seed_tokens":[1],"replay_tokens":[2],"future_oracle":[3]}
        """.utf8
    )
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(RuntimeWorkerRequest.self, from: forged)
    }
}

@Test
func replayTokensAreRejectedOnEveryOtherRequestKind() {
    // Trained-MTP block requests must not smuggle a replay payload.
    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalTrainedMTPBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "mtp_decode_block",
                token: 7,
                maxBlockSize: 4,
                replayTokens: [1]
            ),
            decodedTokenCount: 0
        )
    }
    // Serial decode_block requests reject it as a cross-kind field.
    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalDecodeBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "decode_block",
                token: 12,
                maxBlockSize: 2,
                replayTokens: [1]
            ),
            decodedTokenCount: 0
        )
    }
}

@Test
func committedStateAuditResponseRoundTripsWithBoundedPayload() throws {
    let digest = hardeningDigest(
        replaySeedArgmaxToken: 7,
        replayArgmaxTokens: [11, 12, 13, 14]
    )
    let response = RuntimeWorkerResponse(
        id: 9,
        nonce: "nonce",
        ok: true,
        committedStateAudit: digest
    )
    let data = try JSONEncoder().encode(response)
    #expect(data.count < BufferedFileLineReader.defaultMaximumLineByteCount)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(
        Set(object.keys)
            == Set(["id", "nonce", "ok", "committed_state_audit"])
    )
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self,
        from: data
    )
    let decodedDigest = try #require(decoded.committedStateAudit)
    #expect(decodedDigest == digest)

    // Responses without the audit block keep their historical shape.
    let plain = RuntimeWorkerResponse(
        id: 10,
        nonce: "nonce",
        ok: true,
        tokens: [1]
    )
    let plainObject = try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(plain))
            as? [String: Any]
    )
    #expect(plainObject["committed_state_audit"] == nil)
}

@Test
func trainedMTPReportCarriesAuditBlockOnlyWhenRequested() throws {
    // The timed measurement never passes --audit-committed-state, so the
    // timed report must stay byte-shape-identical to the pre-audit schema.
    let source = try String(
        contentsOfFile: "Sources/MLXFastTrustedHarness/GemmaRuntimeMTP.swift",
        encoding: .utf8
    )
    #expect(source.contains("case committedStateAudit = \"committed_state_audit\""))
    #expect(source.contains("if options.auditCommittedState {"))
    let workerCopy = try String(
        contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeMTP.swift",
        encoding: .utf8
    )
    #expect(workerCopy.contains("case committedStateAudit = \"committed_state_audit\""))
    #expect(workerCopy.contains("if options.auditCommittedState {"))
}

// MARK: - Ordering: the audit runs strictly after elapsed capture

@Test
func committedStateAuditRunsStrictlyAfterElapsedCapture() throws {
    for path in [
        "Sources/MLXFastHarness/GemmaRuntimeMTP.swift",
        "Sources/MLXFastTrustedHarness/GemmaRuntimeMTP.swift",
    ] {
        let source = try String(contentsOfFile: path, encoding: .utf8)

        // The measurement function (which owns phaseStart/elapsedSeconds)
        // must never invoke the audit.
        let measureStart = try #require(
            source.range(
                of: "    static func measureExperimentalTrainedMTPWorkerDecode("
            ),
            Comment(rawValue: path)
        )
        let measureEnd = try #require(
            source.range(
                of: "    public static func experimentalMTPProbe(",
                range: measureStart.upperBound..<source.endIndex
            ),
            Comment(rawValue: path)
        )
        let measureBody = source[measureStart.lowerBound..<measureEnd.lowerBound]
        #expect(
            !measureBody.contains("CommittedStateAudit"),
            Comment(rawValue: path)
        )

        // Inside the benchmark entry point the audit call is sequenced after
        // the measurement returns (elapsedSeconds is captured inside it).
        let benchmarkStart = try #require(
            source.range(
                of: "public static func experimentalTrainedMTPBenchmark("
            ),
            Comment(rawValue: path)
        )
        let measurementCall = try #require(
            source.range(
                of: "let measurement = try measureExperimentalTrainedMTPWorkerDecode(",
                range: benchmarkStart.upperBound..<source.endIndex
            ),
            Comment(rawValue: path)
        )
        let auditCall = try #require(
            source.range(
                of: "runExperimentalTrainedMTPCommittedStateAudit(",
                range: benchmarkStart.upperBound..<source.endIndex
            ),
            Comment(rawValue: path)
        )
        let reportReturn = try #require(
            source.range(
                of: "return ExperimentalTrainedMTPReport(",
                range: benchmarkStart.upperBound..<source.endIndex
            ),
            Comment(rawValue: path)
        )
        #expect(
            measurementCall.upperBound < auditCall.lowerBound,
            Comment(rawValue: path)
        )
        #expect(
            auditCall.upperBound < reportReturn.lowerBound,
            Comment(rawValue: path)
        )
    }

    // The audit driver itself is warn-only: it never throws.
    for path in [
        "Sources/MLXFastHarness/GemmaRuntimeMTPCommittedAudit.swift",
        "Sources/MLXFastTrustedHarness/GemmaRuntimeMTPCommittedAudit.swift",
    ] {
        let source = try String(contentsOfFile: path, encoding: .utf8)
        let driverStart = try #require(
            source.range(
                of: "static func runExperimentalTrainedMTPCommittedStateAudit("
            )
        )
        let driverEnd = try #require(
            source.range(
                of: "private static func committedStateAuditReport(",
                range: driverStart.upperBound..<source.endIndex
            )
        )
        let driver = source[driverStart.lowerBound..<driverEnd.lowerBound]
        #expect(!driver.contains(") throws -> MTPCommittedStateAuditReport"))
        #expect(driver.contains(") -> MTPCommittedStateAuditReport"))
    }

    // The two shared audit files stay byte-identical.
    let trusted = try String(
        contentsOfFile:
            "Sources/MLXFastTrustedHarness/GemmaRuntimeMTPCommittedAudit.swift",
        encoding: .utf8
    )
    let worker = try String(
        contentsOfFile:
            "Sources/MLXFastHarness/GemmaRuntimeMTPCommittedAudit.swift",
        encoding: .utf8
    )
    #expect(trusted == worker)
}

// MARK: - Workflow wiring

private func hardeningWorkflow() throws -> String {
    try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
}

private func hardeningStepBody(
    _ workflow: String,
    from stepName: String,
    to nextStepName: String
) throws -> String {
    let start = try #require(
        workflow.range(of: stepName),
        "missing step \(stepName)"
    )
    let end = try #require(
        workflow.range(
            of: nextStepName,
            range: start.upperBound..<workflow.endIndex
        ),
        "missing step \(nextStepName)"
    )
    return String(workflow[start.lowerBound..<end.lowerBound])
}

@Test
func committedStateAuditFlagIsOnUntimedGateOnlyAndWarnOnly() throws {
    let workflow = try hardeningWorkflow()

    // Exactly ONE invocation passes the audit flag: the untimed base gate.
    #expect(
        workflow.components(separatedBy: "--audit-committed-state").count - 1
            == 1
    )
    let gateBody = try hardeningStepBody(
        workflow,
        from: "- name: MTP correctness and parity gate (untimed)",
        to: "- name: MTP extended correctness legs (untimed)"
    )
    #expect(gateBody.contains("--audit-committed-state"))
    #expect(gateBody.contains(".committed_state_audit"))
    #expect(gateBody.contains("committed-KV audit WOULD FAIL"))
    #expect(gateBody.contains("WARN-ONLY"))

    // The timed measure step never carries the audit or the reference
    // workspace argument.
    let timedBody = try hardeningStepBody(
        workflow,
        from: "- name: Timed paired MTP benchmark (measure-mtp-job)",
        to: "- name: Compute MTP score and enforce floor"
    )
    #expect(!timedBody.contains("--audit-committed-state"))
    #expect(!timedBody.contains("--audit-reference-workspace"))

    // The local runner does not opt into the audit implicitly.
    let localRunner = try String(
        contentsOfFile: "benchmark-mtp.sh",
        encoding: .utf8
    )
    #expect(!localRunner.contains("--audit-committed-state"))
}

@Test
func extendedCorrectnessLegsAreUntimedExactTokenAndScrubbed() throws {
    let workflow = try hardeningWorkflow()

    // Ordering: base gate -> extended legs -> reap -> scrub -> semantic.
    let baseGate = try #require(
        workflow.range(of: "- name: MTP correctness and parity gate (untimed)")
    )
    let extended = try #require(
        workflow.range(of: "- name: MTP extended correctness legs (untimed)")
    )
    let reap = try #require(
        workflow.range(
            of: "- name: Reap exact MTP worker before semantic material"
        )
    )
    let timed = try #require(
        workflow.range(
            of: "- name: Timed paired MTP benchmark (measure-mtp-job)"
        )
    )
    #expect(baseGate.lowerBound < extended.lowerBound)
    #expect(extended.lowerBound < reap.lowerBound)
    #expect(reap.lowerBound < timed.lowerBound)

    let legs = try hardeningStepBody(
        workflow,
        from: "- name: MTP extended correctness legs (untimed)",
        to: "- name: Reap exact MTP worker before semantic material"
    )
    // Fail-closed exact-token validation identical in spirit to the base
    // gate: matched tokens, requested block size, no score fields.
    #expect(legs.contains(".all_tokens_matched == true"))
    #expect(legs.contains(".max_block_size == $block_size"))
    #expect(legs.contains("(has(\"score\") | not)"))
    #expect(legs.contains("--deny-worker-file-writes"))
    // Block-size sweep plus the artifact-dependent legs with graceful skip.
    #expect(legs.contains("MLXFAST_MTP_GATE_BLOCK_SIZE_SWEEP"))
    #expect(legs.contains("mtp_correctness_golden_wrap.json"))
    #expect(legs.contains("mtp_correctness_golden_rotation.json"))
    #expect(legs.contains("window-wrap correctness leg skipped"))
    #expect(legs.contains("rotation correctness leg skipped"))

    // Job-level knobs exist with the sweep enabled by default.
    #expect(workflow.contains("MLXFAST_MTP_GATE_BLOCK_SIZE_SWEEP: \"2 3\""))
    #expect(workflow.contains("MLXFAST_MTP_WRAP_GOLDEN_TOKENS: \"1536\""))
    // Wrap pins ship empty (leg skipped) until the operator provisions them.
    #expect(workflow.contains("MLXFAST_MTP_WRAP_GOLDEN_SHA256: \"\""))

    // The staged extended goldens are scrubbed everywhere the base golden
    // is: before semantic capture, in the semantic scrub, and pre-timing.
    for scrubStep in [
        "- name: Scrub exact MTP oracle before semantic capture",
        "- name: Scrub MTP semantic GPQA material",
        "- name: Scrub hidden material from bench workspace",
    ] {
        let start = try #require(workflow.range(of: scrubStep))
        let body = String(
            workflow[start.lowerBound...].prefix(6_000)
        )
        #expect(
            body.contains("mtp_correctness_golden_wrap.json"),
            Comment(rawValue: scrubStep)
        )
        #expect(
            body.contains("mtp_correctness_golden_rotation.json"),
            Comment(rawValue: scrubStep)
        )
    }
}

@Test
func referenceOracleReplayIsUntimedAndClassifiesInfrastructureDrift() throws {
    let workflow = try hardeningWorkflow()
    let replay = try #require(
        workflow.range(of: "- name: MTP reference oracle replay (untimed)")
    )
    let baseGate = try #require(
        workflow.range(of: "- name: MTP correctness and parity gate (untimed)")
    )
    #expect(replay.lowerBound < baseGate.lowerBound)

    let body = try hardeningStepBody(
        workflow,
        from: "- name: MTP reference oracle replay (untimed)",
        to: "- name: MTP correctness and parity gate (untimed)"
    )
    // Runs the pinned baseline tree's own serial probe, never bench-exec.
    #expect(body.contains("mtp-probe"))
    #expect(body.contains("MLXFAST_MTP_BASELINE_WS"))
    #expect(!body.contains("MLXFAST_BENCH_EXEC"))
    // Divergence is INFRASTRUCTURE, launch problems skip with a warning.
    #expect(body.contains("INFRASTRUCTURE"))
    #expect(body.contains("NOT a candidate failure"))
    #expect(body.contains("SKIPPING"))
    #expect(body.contains("mismatch at step"))
}

@Test
func pairedSemanticFloorShipsWarnOnlyWithBestEffortReferenceCapture() throws {
    let workflow = try hardeningWorkflow()
    #expect(workflow.contains("MLXFAST_MTP_SEMANTIC_GPQA_PAIRED_DELTA: \"1\""))
    #expect(
        workflow.contains(
            "MLXFAST_MTP_SEMANTIC_GPQA_REFERENCE_MIN_PASS: \"1\""
        )
    )
    // WARN-ONLY until operator calibration.
    #expect(
        workflow.contains("MLXFAST_MTP_SEMANTIC_GPQA_PAIRED_ENFORCE: \"0\"")
    )

    let capture = try hardeningStepBody(
        workflow,
        from: "- name: Capture reference semantic GPQA answers (best-effort, untimed)",
        to: "- name: MTP semantic GPQA gate (untimed)"
    )
    #expect(capture.contains("mtp-generate-gpqa-answers"))
    #expect(capture.contains("predates mtp-generate-gpqa-answers"))
    #expect(capture.contains("falls back to the absolute floor"))
    #expect(!capture.contains("ANTHROPIC_API_KEY"))

    let judge = try hardeningStepBody(
        workflow,
        from: "- name: MTP semantic GPQA gate (untimed)",
        to: "- name: Scrub MTP semantic GPQA material"
    )
    #expect(judge.contains("MLXFAST_SEMANTIC_GPQA_REFERENCE_OUTPUT_PATH"))
    #expect(judge.contains("MLXFAST_SEMANTIC_GPQA_PAIRED_DELTA"))
    #expect(judge.contains("MLXFAST_SEMANTIC_GPQA_PAIRED_ENFORCE"))
    #expect(!judge.contains("score.json"))

    // The judge script preserves the verdict-only fence in paired mode and
    // keeps counts out of its paired warn lines.
    let script = try String(
        contentsOfFile: ".github/scripts/run-semantic-gpqa-gate.sh",
        encoding: .utf8
    )
    #expect(
        script.contains(
            "paired semantic GPQA requires verdict-only mode"
        )
    )
    #expect(script.contains("WOULD FAIL the paired semantic floor"))
    #expect(script.contains("details runner-private"))
    #expect(script.contains("INFRASTRUCTURE"))
    #expect(script.contains("reference_session_valid"))
}

@Test
func rotationSelectionIsDeterministicAndSkipsUnprovisionedPools() throws {
    let fileManager = FileManager.default
    let repositoryRoot = fileManager.currentDirectoryPath
    let script = URL(fileURLWithPath: repositoryRoot)
        .appendingPathComponent(".github/scripts/select-mtp-gate-rotation.sh")
        .path
    guard fileManager.isExecutableFile(atPath: script) else {
        Issue.record("rotation selection script missing or not executable")
        return
    }

    let root = fileManager.temporaryDirectory.appendingPathComponent(
        "mtp-rotation-\(UUID().uuidString)"
    )
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    func run(_ manifest: String, seed: String) throws -> (Int32, String, String) {
        let manifestURL = root.appendingPathComponent(
            "pool-\(UUID().uuidString).json"
        )
        try Data(manifest.utf8).write(to: manifestURL)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script, manifestURL.path, seed]
        process.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(decoding: out, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines),
            String(decoding: err, as: UTF8.self)
        )
    }

    let sha = String(repeating: "1", count: 64)
    let sha2 = String(repeating: "2", count: 64)
    let mixedPool = """
    {
      "schema_version": 1,
      "track_id": "gemma4-31b-it-mtp-v1",
      "members": [
        {"id": "a", "r2_key": "k/a.json", "sha256": "\(sha)", "bytes": "10", "decode_tokens": 256, "block_size": 4},
        {"id": "unprovisioned", "r2_key": "k/b.json", "sha256": "", "bytes": "", "decode_tokens": 256, "block_size": 3},
        {"id": "c", "r2_key": "k/c.json", "sha256": "\(sha2)", "bytes": "20", "decode_tokens": 384, "block_size": 2}
      ]
    }
    """
    // Deterministic: the same seed always draws the same member; the modulo
    // over ELIGIBLE members skips the unprovisioned one.
    let first = try run(mixedPool, seed: "10")
    let second = try run(mixedPool, seed: "10")
    #expect(first.0 == 0)
    #expect(first.1 == second.1)
    let selected = try #require(
        try JSONSerialization.jsonObject(with: Data(first.1.utf8))
            as? [String: Any]
    )
    #expect(selected["id"] as? String == "a")
    #expect(first.2.contains("seed=10"))
    #expect(first.2.contains("eligible=2"))
    let odd = try run(mixedPool, seed: "11")
    let oddSelected = try #require(
        try JSONSerialization.jsonObject(with: Data(odd.1.utf8))
            as? [String: Any]
    )
    #expect(oddSelected["id"] as? String == "c")

    // Fully unprovisioned pools skip gracefully with {}.
    let emptyPool = """
    {
      "schema_version": 1,
      "track_id": "gemma4-31b-it-mtp-v1",
      "members": [
        {"id": "x", "r2_key": "k/x.json", "sha256": "", "bytes": "", "decode_tokens": 256, "block_size": 4}
      ]
    }
    """
    let skipped = try run(emptyPool, seed: "3")
    #expect(skipped.0 == 0)
    #expect(skipped.1 == "{}")

    // Out-of-bounds members are ineligible even when pinned.
    let outOfBounds = """
    {
      "schema_version": 1,
      "track_id": "gemma4-31b-it-mtp-v1",
      "members": [
        {"id": "too-long", "r2_key": "k/t.json", "sha256": "\(sha)", "bytes": "10", "decode_tokens": 100000, "block_size": 4},
        {"id": "bad-block", "r2_key": "k/u.json", "sha256": "\(sha)", "bytes": "10", "decode_tokens": 256, "block_size": 9}
      ]
    }
    """
    let bounded = try run(outOfBounds, seed: "0")
    #expect(bounded.0 == 0)
    #expect(bounded.1 == "{}")

    // Malformed manifests fail closed.
    let malformed = try run("{\"schema_version\": 2, \"members\": []}", seed: "0")
    #expect(malformed.0 != 0)

    // The checked-in pool manifest is valid and (until the operator
    // provisions R2 objects) selects nothing.
    let checkedIn = Process()
    checkedIn.executableURL = URL(fileURLWithPath: "/bin/bash")
    checkedIn.arguments = [
        script,
        "fixtures/mtp_correctness_gate_rotation_pool.json",
        "12345",
    ]
    checkedIn.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
    let checkedInStdout = Pipe()
    checkedIn.standardOutput = checkedInStdout
    checkedIn.standardError = Pipe()
    try checkedIn.run()
    let checkedInOut = checkedInStdout.fileHandleForReading
        .readDataToEndOfFile()
    checkedIn.waitUntilExit()
    #expect(checkedIn.terminationStatus == 0)
    #expect(
        String(decoding: checkedInOut, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines) == "{}"
    )
}

@Test
func pairedSemanticFloorMathWarnsWithoutFailingAndEnforcesWhenFlipped() throws {
    let fileManager = FileManager.default
    let repositoryRoot = fileManager.currentDirectoryPath
    let script = URL(fileURLWithPath: repositoryRoot)
        .appendingPathComponent(".github/scripts/run-semantic-gpqa-gate.sh")
        .path
    let temporaryRoot = fileManager.temporaryDirectory.appendingPathComponent(
        "mtp-paired-floor-\(UUID().uuidString)"
    )
    try fileManager.createDirectory(
        at: temporaryRoot,
        withIntermediateDirectories: true
    )
    let root = URL(
        fileURLWithPath: temporaryRoot.path.hasPrefix("/var/")
            ? "/private" + temporaryRoot.path
            : temporaryRoot.path
    )
    defer { try? fileManager.removeItem(at: root) }
    let privateRoot = root.appendingPathComponent("private")
    let shimRoot = root.appendingPathComponent("bin")
    try fileManager.createDirectory(
        at: privateRoot,
        withIntermediateDirectories: true
    )
    try fileManager.createDirectory(
        at: shimRoot,
        withIntermediateDirectories: true
    )

    func writeAnswers(name: String, marker: String) throws -> URL {
        let url = root.appendingPathComponent("\(name).json")
        let object: [String: Any] = [
            "version": 1,
            "cases": [
                [
                    "id": "case-one",
                    "prompt": "q1",
                    "answer_key": "A",
                    "reference_answer": "A. reference",
                    "candidate_answer": "\(marker)-ONE",
                    "candidate_tokens": [1, 2],
                    "max_new_tokens": 2,
                ],
                [
                    "id": "case-two",
                    "prompt": "q2",
                    "answer_key": "B",
                    "reference_answer": "B. reference",
                    "candidate_answer": "\(marker)-TWO",
                    "candidate_tokens": [3, 4],
                    "max_new_tokens": 2,
                ],
            ],
        ]
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        try fileManager.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
        return url
    }
    let candidateAnswers = try writeAnswers(name: "candidate", marker: "CAND")
    let referenceAnswers = try writeAnswers(name: "reference", marker: "REF")

    // Judge shim: the reference passes both cases; the candidate passes only
    // case one. Absolute floor (min_pass=1) passes; the paired floor with
    // delta 0 requires 2 and therefore would fail.
    let curl = shimRoot.appendingPathComponent("curl")
    try """
    #!/usr/bin/env bash
    set -euo pipefail
    output=""
    data=""
    previous=""
    for argument in "$@"; do
      if [[ "${previous}" == "--output" ]]; then
        output="${argument}"
      elif [[ "${previous}" == "--data" ]]; then
        data="${argument}"
      fi
      previous="${argument}"
    done
    request="${data#@}"
    if grep -q "CAND-TWO" "${request}"; then
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":false}"}]}' > "${output}"
    else
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' > "${output}"
    fi
    """.write(to: curl, atomically: true, encoding: .utf8)
    try fileManager.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: curl.path
    )

    func run(
        enforce: String,
        resultsName: String,
        delta: String
    ) throws -> (status: Int32, output: String, results: URL) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script]
        process.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
        let results = privateRoot.appendingPathComponent(resultsName)
        var environment = ProcessInfo.processInfo.environment
        for key in environment.keys
        where key.hasPrefix("MLXFAST_") || key.hasPrefix("ANTHROPIC_") {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] = shimRoot.path + ":"
            + (ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin")
        environment["ANTHROPIC_API_KEY"] = "test-key"
        environment["MLXFAST_PRIVATE_DIR"] = privateRoot.path
        environment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] = candidateAnswers.path
        environment["MLXFAST_SEMANTIC_GPQA_REFERENCE_OUTPUT_PATH"] =
            referenceAnswers.path
        environment["MLXFAST_SEMANTIC_GPQA_RESULTS_PATH"] = results.path
        environment["MLXFAST_SEMANTIC_GPQA_MIN_PASS"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_REQUIRED"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_VERDICT_ONLY"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_EXPECTED_CASE_COUNT"] = "2"
        environment["MLXFAST_SEMANTIC_GPQA_EXPECTED_MAX_NEW_TOKENS"] = "2"
        environment["MLXFAST_SEMANTIC_GPQA_PAIRED_DELTA"] = delta
        environment["MLXFAST_SEMANTIC_GPQA_REFERENCE_MIN_PASS"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_PAIRED_ENFORCE"] = enforce
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(decoding: data, as: UTF8.self),
            results
        )
    }

    // WARN-ONLY: the paired shortfall logs a would-fail warning but the gate
    // exits 0 on the absolute floor.
    let warnOnly = try run(enforce: "0", resultsName: "warn.json", delta: "0")
    #expect(warnOnly.status == 0)
    #expect(warnOnly.output.contains("WOULD FAIL the paired semantic floor"))
    #expect(warnOnly.output.contains("semantic-gpqa: verdict passed"))
    #expect(!warnOnly.output.contains("pass_count="))
    let results = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: warnOnly.results)
        ) as? [String: Any]
    )
    #expect(results["paired_mode"] as? Bool == true)
    #expect(results["pass_count"] as? Int == 1)
    #expect(results["reference_pass_count"] as? Int == 2)
    #expect(results["paired_required_pass_count"] as? Int == 2)
    #expect(results["paired_passed"] as? Bool == false)
    #expect(results["reference_session_valid"] as? Bool == true)
    #expect(results["case_count"] as? Int == 2)

    // ENFORCED: the same shortfall fails the gate once the operator flips
    // the knob after calibration.
    let enforced = try run(
        enforce: "1",
        resultsName: "enforced.json",
        delta: "0"
    )
    #expect(enforced.status != 0)
    #expect(enforced.output.contains("semantic GPQA paired floor failed"))

    // With the shipped delta of 1 the same session passes the paired floor.
    let withinDelta = try run(
        enforce: "1",
        resultsName: "delta.json",
        delta: "1"
    )
    #expect(withinDelta.status == 0)
    #expect(withinDelta.output.contains("semantic-gpqa: paired floor satisfied"))

    // A missing reference file degrades to the absolute floor with a notice.
    try fileManager.removeItem(at: referenceAnswers)
    let fallback = try run(
        enforce: "1",
        resultsName: "fallback.json",
        delta: "0"
    )
    #expect(fallback.status == 0)
    #expect(fallback.output.contains("falling back to the absolute floor"))
    #expect(fallback.output.contains("semantic-gpqa: verdict passed"))
}

@Test
func rotationAndWrapLegsRotateOnlyTheUntimedGateNeverTheTimedGolden() throws {
    let workflow = try hardeningWorkflow()
    // The timed benchmark golden pin and measure invocation are untouched by
    // rotation: measure-mtp-job still receives the single pinned benchmark
    // golden path.
    let timedBody = try hardeningStepBody(
        workflow,
        from: "- name: Timed paired MTP benchmark (measure-mtp-job)",
        to: "- name: Compute MTP score and enforce floor"
    )
    #expect(
        timedBody.contains(
            "--golden \"${MLXFAST_PRIVATE_DIR}/mtp_benchmark_golden.json\""
        )
    )
    #expect(!timedBody.contains("rotation"))
    #expect(!timedBody.contains("wrap"))
    // Rotation download is pin-verified and logged with seed + member id.
    let download = try hardeningStepBody(
        workflow,
        from: "- name: Prepare hidden MTP goldens",
        to: "- name: Verify trusted harness before MTP correctness gate"
    )
    #expect(download.contains("select-mtp-gate-rotation.sh"))
    #expect(download.contains("rotation_seed=$((GITHUB_RUN_ID + GITHUB_RUN_ATTEMPT))"))
    #expect(download.contains("rotation leg drew member"))
    #expect(download.contains("rotation-golden pin mismatch"))
    #expect(download.contains("wrap-golden pin mismatch"))
}
