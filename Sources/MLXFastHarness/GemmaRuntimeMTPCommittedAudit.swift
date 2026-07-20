import Foundation
import MLXFastCore
#if !MLXFAST_TRUSTED_HARNESS
import MLX
import MLXFastModel
import MLXLLM
import MLXLMCommon
#endif

// COMMITTED-KV FAITHFULNESS AUDIT (experimental MTP track).
//
// Parent-side driver (compiles into both the trusted CLI and the worker
// support target) plus, under `#if !MLXFAST_TRUSTED_HARNESS`, the
// model-linked digest computation the workers execute. The scalar audit
// math itself lives in MLXFastCore/MTPCommittedStateAudit.swift so the
// trusted parent handles only numbers and bounded strings.
//
// WARN-ONLY CONTRACT (first release): the driver NEVER throws and NEVER
// fails a run. Every failure mode -- candidate audit error, missing
// reference workspace, a pinned baseline tree that predates the replay
// protocol, or an out-of-envelope comparison -- is folded into the returned
// report's `status` / `reference_status` / `would_fail` fields for the
// workflow to surface as a warning. The operator flips enforcement in the
// workflow only after calibrating the envelopes in
// MLXFastConstants (see the calibration procedure there).

extension GemmaRuntime {
    /// Runs strictly AFTER `measureExperimentalTrainedMTPWorkerDecode` has
    /// captured elapsedSeconds: the committed tokens being audited are the
    /// gate-validated `plan.expectedTokens` prefix, the candidate worker is
    /// still alive inside the confirmed-termination guard, and the pinned
    /// reference workspace (when runnable) replays the same sequence
    /// serially. Nothing here can start before the timer stops because the
    /// caller sequences it after the measurement returns.
    static func runExperimentalTrainedMTPCommittedStateAudit(
        plan: ExperimentalMTPPromptPlan,
        totalTokenCount: Int,
        candidateWorker: RuntimeWorkerClient,
        referenceWorkspacePath: String,
        workerOptions: RuntimeWorkerOptions
    ) -> MTPCommittedStateAuditReport {
        let envelope = MTPCommittedStateEnvelope(
            relativeTolerance:
                MLXFastConstants.mtpCommittedStateAuditRelativeTolerance,
            absoluteTolerance:
                MLXFastConstants.mtpCommittedStateAuditAbsoluteTolerance
        )
        let expectedCommittedTokens = Array(
            plan.expectedTokens.prefix(totalTokenCount)
        )

        var status = "completed"
        var candidateDigest: MTPCommittedStateDigest?
        do {
            let response = try candidateWorker.trainedMTPCommittedStateAudit()
            if let digest = response.committedStateAudit {
                candidateDigest = digest
            } else {
                status = "candidate_audit_missing_payload"
            }
        } catch {
            // Never leak worker error text into the report: worker errors can
            // embed request/session details. The category alone is enough for
            // the warn-only phase; details stay in the private gate log.
            status = "candidate_audit_failed"
        }

        guard let candidateDigest else {
            return committedStateAuditReport(
                status: status,
                referenceStatus: "not_attempted",
                findings: {
                    var findings = MTPCommittedStateAuditFindings()
                    findings.wouldFailReasons.append("candidate_digest_unavailable")
                    return findings
                }(),
                candidateProbeMargin: nil,
                referenceProbeMargin: nil,
                envelope: envelope
            )
        }

        var findings = MTPCommittedStateAuditFindings()
        if candidateDigest.committedTokenCount != totalTokenCount
            || candidateDigest.cacheOffset
                != plan.seedTokens.count + totalTokenCount
        {
            findings.wouldFailReasons.append("digest_geometry_mismatch[candidate]")
        }

        var referenceStatus = "unavailable"
        var referenceDigest: MTPCommittedStateDigest?
        if !referenceWorkspacePath.isEmpty {
            let referenceExecutable = referenceWorkspacePath
                + "/.build-worker/release/mlxfast-runtime-worker"
            let referenceWeights = referenceWorkspacePath + "/mtp-weights"
            var isDirectory = ObjCBool(false)
            let weightsExist = FileManager.default.fileExists(
                atPath: referenceWeights,
                isDirectory: &isDirectory
            ) && isDirectory.boolValue
            if FileManager.default.isExecutableFile(atPath: referenceExecutable),
               weightsExist
            {
                do {
                    let referenceOptions = RuntimeWorkerOptions(
                        executablePath: referenceExecutable,
                        sandboxProfilePath: workerOptions.sandboxProfilePath,
                        forwardsWorkerStderr: false,
                        helloTimeoutSeconds: workerOptions.helloTimeoutSeconds,
                        requestTimeoutSeconds:
                            workerOptions.requestTimeoutSeconds,
                        shutdownTimeoutSeconds:
                            workerOptions.shutdownTimeoutSeconds,
                        terminationGraceSeconds:
                            workerOptions.terminationGraceSeconds
                    )
                    let referenceWorker = try RuntimeWorkerClient(
                        options: referenceOptions,
                        weightsPath: referenceWeights
                    )
                    let response = try withConfirmedRuntimeWorkerTermination(
                        worker: referenceWorker
                    ) {
                        try referenceWorker.auditSerialReplay(
                            seedTokens: plan.seedTokens,
                            replayTokens: [plan.expectedSeedToken]
                                + expectedCommittedTokens
                        )
                    }
                    if let digest = response.committedStateAudit {
                        referenceDigest = digest
                        referenceStatus = "compared"
                    } else {
                        referenceStatus = "reference_missing_payload"
                    }
                } catch {
                    // The pinned baseline tree predating the replay protocol
                    // lands here (its worker rejects the request kind), as do
                    // sandbox/permission gaps on the box. All are
                    // reference-side conditions, never candidate failures.
                    referenceStatus = "unsupported_or_failed"
                }
            }
        }

        let comparisonFindings = MTPCommittedStateAuditMath.audit(
            candidate: candidateDigest,
            reference: referenceDigest,
            expectedSeedToken: plan.expectedSeedToken,
            expectedCommittedTokens: expectedCommittedTokens,
            envelope: envelope
        )
        findings.comparedLayerCount = comparisonFindings.comparedLayerCount
        findings.comparedRowSampleCount =
            comparisonFindings.comparedRowSampleCount
        findings.replayTokenMismatchCount =
            comparisonFindings.replayTokenMismatchCount
        findings.probeArgmaxMatched = comparisonFindings.probeArgmaxMatched
        findings.maxLayerStatRelativeError =
            comparisonFindings.maxLayerStatRelativeError
        findings.maxRowSampleElementAbsoluteError =
            comparisonFindings.maxRowSampleElementAbsoluteError
        findings.selfConsistencyMaxAbsoluteError =
            comparisonFindings.selfConsistencyMaxAbsoluteError
        findings.wouldFailReasons.append(
            contentsOf: comparisonFindings.wouldFailReasons
        )

        return committedStateAuditReport(
            status: status,
            referenceStatus: referenceStatus,
            findings: findings,
            candidateProbeMargin: candidateDigest.probe.topTwoMargin,
            referenceProbeMargin: referenceDigest?.probe.topTwoMargin,
            envelope: envelope
        )
    }

    private static func committedStateAuditReport(
        status: String,
        referenceStatus: String,
        findings: MTPCommittedStateAuditFindings,
        candidateProbeMargin: Double?,
        referenceProbeMargin: Double?,
        envelope: MTPCommittedStateEnvelope
    ) -> MTPCommittedStateAuditReport {
        MTPCommittedStateAuditReport(
            schemaVersion:
                MLXFastConstants.mtpCommittedStateAuditSchemaVersion,
            status: status,
            referenceStatus: referenceStatus,
            wouldFail: findings.wouldFail,
            wouldFailReasons: findings.wouldFailReasons,
            comparedLayerCount: findings.comparedLayerCount,
            comparedRowSampleCount: findings.comparedRowSampleCount,
            replayTokenMismatchCount: findings.replayTokenMismatchCount,
            probeArgmaxMatched: findings.probeArgmaxMatched,
            probeTopTwoMarginCandidate: candidateProbeMargin,
            probeTopTwoMarginReference: referenceProbeMargin,
            maxLayerStatRelativeError: findings.maxLayerStatRelativeError,
            maxRowSampleElementAbsoluteError:
                findings.maxRowSampleElementAbsoluteError,
            selfConsistencyMaxAbsoluteError:
                findings.selfConsistencyMaxAbsoluteError,
            envelopeRelativeTolerance: envelope.relativeTolerance,
            envelopeAbsoluteTolerance: envelope.absoluteTolerance
        )
    }
}

#if !MLXFAST_TRUSTED_HARNESS
extension GemmaRuntime {
    /// Sampled (layer, position) grid shared by the candidate and reference
    /// digests: every Nth layer plus the final layer; up to
    /// `positionCount` evenly spaced positions per sampled layer plus the
    /// final retained position. Pure index math so both sides sample
    /// identically given identical geometry.
    static func committedAuditSampledPositions(
        retainedPositionCount: Int,
        positionCount: Int =
            MLXFastConstants.mtpCommittedStateAuditRowSamplePositionCount
    ) -> [Int] {
        guard retainedPositionCount > 0, positionCount > 0 else { return [] }
        var positions = Set<Int>()
        let strideLength = max(1, retainedPositionCount / positionCount)
        var position = 0
        while position < retainedPositionCount, positions.count < positionCount {
            positions.insert(position)
            position += strideLength
        }
        positions.insert(retainedPositionCount - 1)
        return positions.sorted()
    }

    static func committedAuditSampledLayers(
        layerCount: Int,
        layerStride: Int =
            MLXFastConstants.mtpCommittedStateAuditRowSampleLayerStride
    ) -> [Int] {
        guard layerCount > 0, layerStride > 0 else { return [] }
        var layers = Set(Swift.stride(from: 0, to: layerCount, by: layerStride))
        layers.insert(layerCount - 1)
        return layers.sorted()
    }

    /// Candidate-side digest of the session's ACTUAL committed cache. Reads
    /// the per-layer K/V state, emits window aggregates (MLX float32
    /// reductions), the strided raw-row samples (host floats, Swift stats),
    /// and finally the next-decision probe -- which forwards the last
    /// committed token and therefore advances the cache PAST the audited
    /// state; callers only invoke this after decode is complete and never
    /// reuse the session afterwards.
    static func computeCommittedStateDigest(
        target: Gemma4RuntimeModel,
        caches: [any KVCache],
        committedTokenCount: Int,
        cacheOffset: Int,
        lastCommittedToken: Int?
    ) throws -> MTPCommittedStateDigest {
        guard let lastCommittedToken else {
            throw MLXFastError.invalidInput(
                "committed-state audit requires a completed decode session"
            )
        }
        let digestBody = try committedStateDigestBody(caches: caches)
        let probe = try committedAuditProbe(
            target: target,
            caches: caches,
            inputToken: lastCommittedToken
        )
        return MTPCommittedStateDigest(
            schemaVersion:
                MLXFastConstants.mtpCommittedStateAuditSchemaVersion,
            committedTokenCount: committedTokenCount,
            cacheOffset: cacheOffset,
            layerDigests: digestBody.layerDigests,
            rowSamples: digestBody.rowSamples,
            probe: probe
        )
    }

    /// Reference-side serial replay for the plain (target-only) worker: seed
    /// prefill plus one teacher-forced K=1 forward per replay token into a
    /// FRESH cache, recording the argmax after every forward except the
    /// last, whose logits become the next-decision probe. The final replay
    /// token is the last committed token, so the pre-probe cache state has
    /// offset seed + (replayTokens.count - 1), exactly matching the
    /// candidate's committed state.
    static func runAuditSerialReplay(
        weightCache: Gemma4RuntimeWeightCache,
        seedTokens: [Int],
        replayTokens: [Int]
    ) throws -> MTPCommittedStateDigest {
        guard !seedTokens.isEmpty,
              seedTokens.count <= 4_096,
              seedTokens.allSatisfy({
                  $0 >= 0 && $0 < MLXFastConstants.vocabSize
              }),
              replayTokens.count >= 1,
              replayTokens.count
                  <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens + 1,
              replayTokens.allSatisfy({
                  $0 >= 0 && $0 < MLXFastConstants.vocabSize
              })
        else {
            throw MLXFastError.invalidInput(
                "audit serial replay request is outside the trusted bounds"
            )
        }
        let target = try weightCache.requireLibraryModel()
        let caches = target.mtpNewCache(parameters: nil)

        func greedyArgmax(from forward: Gemma4MTPForward) throws -> Int {
            let logitsRow = forward.logits[0..., -1, 0...]
                .flattened()
                .asType(.float32)
            eval(logitsRow)
            let logits = logitsRow.asArray(Float.self)
            guard let first = logits.first, first.isFinite || !logits.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "audit serial replay produced empty logits"
                )
            }
            var bestIndex = 0
            var bestValue = first
            for (index, value) in logits.enumerated() where value > bestValue {
                bestIndex = index
                bestValue = value
            }
            return bestIndex
        }

        let seedInput = MLXArray(
            seedTokens.map(Int32.init),
            [1, seedTokens.count]
        )
        let seedForward = target.forwardForMTP(seedInput, cache: caches)
        let seedArgmax = try greedyArgmax(from: seedForward)

        var replayArgmaxTokens: [Int] = []
        replayArgmaxTokens.reserveCapacity(max(replayTokens.count - 1, 0))
        for token in replayTokens.dropLast() {
            let forward = target.forwardForMTP(
                MLXArray([Int32(token)], [1, 1]),
                cache: caches
            )
            replayArgmaxTokens.append(try greedyArgmax(from: forward))
        }

        // Pre-probe committed state: digest BEFORE the final probe forward.
        let digestBody = try committedStateDigestBody(caches: caches)
        let cacheOffset = seedTokens.count + replayTokens.count - 1
        guard let probeInput = replayTokens.last else {
            throw MLXFastError.invalidInput(
                "audit serial replay requires at least one replay token"
            )
        }
        let probe = try committedAuditProbe(
            target: target,
            caches: caches,
            inputToken: probeInput
        )
        return MTPCommittedStateDigest(
            schemaVersion:
                MLXFastConstants.mtpCommittedStateAuditSchemaVersion,
            committedTokenCount: replayTokens.count - 1,
            cacheOffset: cacheOffset,
            layerDigests: digestBody.layerDigests,
            rowSamples: digestBody.rowSamples,
            probe: probe,
            replaySeedArgmaxToken: seedArgmax,
            replayArgmaxTokens: replayArgmaxTokens
        )
    }

    private static func committedStateDigestBody(
        caches: [any KVCache]
    ) throws -> (
        layerDigests: [MTPCommittedLayerDigest],
        rowSamples: [MTPCommittedRowSample]
    ) {
        guard !caches.isEmpty else {
            throw MLXFastError.invalidInput(
                "committed-state audit found no cache layers"
            )
        }
        var layerDigests: [MTPCommittedLayerDigest] = []
        layerDigests.reserveCapacity(caches.count)
        var rowSamples: [MTPCommittedRowSample] = []
        let sampledLayers = Set(
            committedAuditSampledLayers(layerCount: caches.count)
        )

        for (layerIndex, cache) in caches.enumerated() {
            let state = cache.state
            guard state.count == 2,
                  state[0].ndim == 4,
                  state[1].ndim == 4,
                  state[0].shape == state[1].shape
            else {
                throw MLXFastError.invalidInput(
                    "committed-state audit found an unexpected cache state shape"
                )
            }
            let keys = state[0]
            let values = state[1]
            let retained = keys.dim(2)

            // Window aggregates via MLX float32 reductions (evaluated in one
            // batch per layer; strictly untimed).
            let keysF32 = keys.asType(.float32)
            let valuesF32 = values.asType(.float32)
            let keySumSquares = (keysF32 * keysF32).sum()
            let keyMaxAbs = MLX.abs(keysF32).max()
            let keyMean = keysF32.mean()
            let valueSumSquares = (valuesF32 * valuesF32).sum()
            let valueMaxAbs = MLX.abs(valuesF32).max()
            let valueMean = valuesF32.mean()
            eval(
                keySumSquares, keyMaxAbs, keyMean,
                valueSumSquares, valueMaxAbs, valueMean
            )
            layerDigests.append(
                MTPCommittedLayerDigest(
                    layerIndex: layerIndex,
                    retainedPositionCount: retained,
                    keyStats: MTPCommittedTensorStats(
                        l2: Double(keySumSquares.item(Float.self)).squareRoot(),
                        maxAbs: Double(keyMaxAbs.item(Float.self)),
                        mean: Double(keyMean.item(Float.self))
                    ),
                    valueStats: MTPCommittedTensorStats(
                        l2: Double(valueSumSquares.item(Float.self)).squareRoot(),
                        maxAbs: Double(valueMaxAbs.item(Float.self)),
                        mean: Double(valueMean.item(Float.self))
                    )
                )
            )

            guard sampledLayers.contains(layerIndex), retained > 0 else {
                continue
            }
            for position in committedAuditSampledPositions(
                retainedPositionCount: retained
            ) {
                let keyRowArray = keys[
                    0..., 0..., position..<(position + 1), 0...
                ].flattened().asType(.float32)
                let valueRowArray = values[
                    0..., 0..., position..<(position + 1), 0...
                ].flattened().asType(.float32)
                eval(keyRowArray, valueRowArray)
                let keyRow = keyRowArray.asArray(Float.self)
                let valueRow = valueRowArray.asArray(Float.self)
                rowSamples.append(
                    MTPCommittedRowSample(
                        layerIndex: layerIndex,
                        position: position,
                        keyStats: MTPCommittedStateAuditMath.stats(of: keyRow),
                        valueStats: MTPCommittedStateAuditMath.stats(
                            of: valueRow
                        ),
                        keyRowBF16Base64:
                            MTPCommittedStateAuditMath.floatsToBF16Base64(
                                keyRow
                            ),
                        valueRowBF16Base64:
                            MTPCommittedStateAuditMath.floatsToBF16Base64(
                                valueRow
                            )
                    )
                )
            }
        }
        return (layerDigests, rowSamples)
    }

    private static func committedAuditProbe(
        target: Gemma4RuntimeModel,
        caches: [any KVCache],
        inputToken: Int
    ) throws -> MTPNextDecisionProbe {
        guard inputToken >= 0, inputToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "committed-state audit probe token is out of vocabulary"
            )
        }
        let forward = target.forwardForMTP(
            MLXArray([Int32(inputToken)], [1, 1]),
            cache: caches
        )
        let logitsRow = forward.logits[0..., -1, 0...]
            .flattened()
            .asType(.float32)
        eval(logitsRow)
        let logits = logitsRow.asArray(Float.self)
        guard !logits.isEmpty else {
            throw MLXFastError.invalidInput(
                "committed-state audit probe produced empty logits"
            )
        }
        var bestIndex = 0
        var bestValue = -Float.infinity
        var secondValue = -Float.infinity
        for (index, value) in logits.enumerated() {
            if value > bestValue {
                secondValue = bestValue
                bestValue = value
                bestIndex = index
            } else if value > secondValue {
                secondValue = value
            }
        }
        return MTPNextDecisionProbe(
            inputToken: inputToken,
            argmaxToken: bestIndex,
            topLogit: Double(bestValue),
            secondLogit: Double(secondValue),
            logitStats: MTPCommittedStateAuditMath.stats(of: logits)
        )
    }
}
#endif
