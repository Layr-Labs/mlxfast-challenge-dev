#if !MLXFAST_TRUSTED_HARNESS
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXSpeculative
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

/// Validated `dflash_decode_block` request.
struct ExperimentalDFlashBlockRequest: Equatable {
    let previousToken: Int
    let maxBlockSize: Int
}

struct ExperimentalDFlashWorkerState {
    var began = false
    var poisoned = false
    var seedTokenCount = 0
    var decodedTokenCount = 0
}

/// Strict validation for a DFlash block request.
///
/// Deliberately does NOT bound `maxBlockSize` by any remaining-token count: the
/// worker is never told how much of the decode window is left, so it cannot
/// special-case the tail. The trusted parent always asks for a full block and
/// truncates the scored prefix itself.
func validateExperimentalDFlashBlockRequest(
    _ request: RuntimeWorkerRequest,
    decodedTokenCount: Int
) throws -> ExperimentalDFlashBlockRequest {
    guard request.id > 0, request.kind == "dflash_decode_block" else {
        throw MLXFastError.invalidInput(
            "DFlash block request has an invalid id or kind"
        )
    }
    guard request.promptTokens == nil,
          request.seedTokens == nil,
          request.steps == nil,
          request.topK == nil,
          request.expectedToken == nil,
          request.prefixTokens == nil,
          request.startOffset == nil,
          request.rowCount == nil,
          request.declaredBlockWidth == nil,
          let previousToken = request.token,
          previousToken >= 0,
          previousToken < MLXFastConstants.vocabSize,
          let maxBlockSize = request.maxBlockSize,
          // 1 is legal: it is the serial control the paired score divides by,
          // served by the same worker and the same protocol.
          maxBlockSize >= 1,
          maxBlockSize <= MLXFastConstants.experimentalDFlashMaxBlockSize
    else {
        throw MLXFastError.invalidInput(
            "DFlash block request has invalid or cross-kind fields"
        )
    }
    guard decodedTokenCount >= 0 else {
        throw MLXFastError.invalidInput(
            "DFlash worker has a negative committed token count"
        )
    }
    let (requestedTotal, overflow) =
        decodedTokenCount.addingReportingOverflow(maxBlockSize)
    guard !overflow,
          requestedTotal
              <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
    else {
        throw MLXFastError.invalidInput(
            "DFlash block request exceeds the configured decode ceiling"
        )
    }
    return ExperimentalDFlashBlockRequest(
        previousToken: previousToken,
        maxBlockSize: maxBlockSize
    )
}

extension LagunaRuntime {
    /// Runtime worker for the DFlash block-decode track.
    ///
    /// Loads the organizer-pinned target and drafter, warms the block graph, and
    /// then serves exactly three request kinds over the isolated protocol pipe.
    /// Everything expensive happens before the protocol hello, i.e. outside
    /// every scored window.
    public static func runExperimentalDFlashWorker(
        targetWeightsPath: String,
        drafterPath: String
    ) throws {
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()

        // Load the target through the vendored factory: `LagunaModel` is the
        // type that conforms to `DFlashTargetModel` (the scored serial model,
        // LagunaRuntimeModel, deliberately does not), and this is the same load
        // path the validated `mlx-bench dflash` run used.
        let targetURL = URL(fileURLWithPath: targetWeightsPath)
        let context = try waitForExperimentalDFlashAsync {
            try await LLMModelFactory.shared.load(
                from: targetURL,
                using: #huggingFaceTokenizerLoader()
            )
        }
        guard let target = context.model as? any DFlashTargetModel else {
            throw MLXFastError.invalidInput(
                "DFlash target model does not conform to DFlashTargetModel: "
                    + "\(type(of: context.model))"
            )
        }
        let drafter = try waitForExperimentalDFlashAsync {
            try await DFlashDraftModel.load(
                from: URL(fileURLWithPath: drafterPath),
                bindTo: target
            )
        }
        eval(context.model, drafter)

        // Warm the block-decode shapes on throwaway cache state before the
        // hello. The real begin request performs the trusted allocator clear and
        // re-warms the working set it frees.
        // Same warm the post-allocator-reset re-warm runs, via the same helper:
        // a seed past the sliding-window ring and EVERY legal block width. These
        // two warm points used to warm different things.
        let warmup = try LagunaDFlashBlockSession(target: target, drafter: drafter)
        try warmup.warmAllBlockWidths()

        let session = try LagunaDFlashBlockSession(target: target, drafter: drafter)

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))

        var state = ExperimentalDFlashWorkerState()
        var expectedRequestID = 1
        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(
                    RuntimeWorkerRequest.self,
                    from: Data(line.utf8)
                )
                guard request.id == expectedRequestID else {
                    throw MLXFastError.invalidInput(
                        "DFlash request id must be monotonic; expected "
                            + "\(expectedRequestID), got \(request.id)"
                    )
                }
                expectedRequestID += 1
                do {
                    response = try handleExperimentalDFlashWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        session: session,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(
                    id: -1,
                    nonce: sessionNonce,
                    ok: false,
                    error: "\(error)"
                )
            }
            try protocolIO.writeLine(try encoder.encode(response))
        }
    }

    static func handleExperimentalDFlashWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        session: LagunaDFlashBlockSession,
        state: inout ExperimentalDFlashWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard !state.poisoned else {
            throw MLXFastError.invalidInput(
                "DFlash decode session is poisoned after an earlier failure"
            )
        }

        switch request.kind {
        case "dflash_decode_begin":
            guard !state.began,
                  request.id > 0,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "DFlash begin request is repeated or malformed"
                )
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            do {
                // Charged but input-independent (the seed is not applied yet):
                // re-touch the working set the trusted clear just freed so the
                // first timed block does not pay a first-touch spike.
                try session.warmWorkingSetAfterAllocatorReset()
                let seedToken = try session.begin(seedTokens: seedTokens)
                state.began = true
                state.seedTokenCount = seedTokens.count
                state.decodedTokenCount = 0
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    seedToken: seedToken
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "dflash_decode_block":
            guard state.began else {
                throw MLXFastError.invalidInput(
                    "DFlash block requested before begin"
                )
            }
            let block = try validateExperimentalDFlashBlockRequest(
                request,
                decodedTokenCount: state.decodedTokenCount
            )
            do {
                let result = try session.generateBlock(
                    previousCommittedToken: block.previousToken,
                    maxBlockSize: block.maxBlockSize
                )
                let (nextCount, overflow) =
                    state.decodedTokenCount.addingReportingOverflow(
                        result.tokens.count
                    )
                let (expectedCacheOffset, offsetOverflow) =
                    state.seedTokenCount.addingReportingOverflow(nextCount)
                guard !overflow,
                      !offsetOverflow,
                      nextCount
                          <= MLXFastConstants
                              .experimentalDFlashMaxConfiguredTotalTokens,
                      result.targetCacheOffset == expectedCacheOffset
                else {
                    throw MLXFastError.invalidInput(
                        "DFlash logical token count and target cache offset "
                            + "diverged"
                    )
                }
                state.decodedTokenCount = nextCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    tokens: result.tokens,
                    declaredRows: result.declaredRows,
                    perRowHiddenDigest: result.perRowHiddenDigest,
                    perRowTop2Tokens: result.perRowTop2Tokens,
                    perRowTop2Logits: result.perRowTop2Logits,
                    acceptedDraftCount: result.acceptedDraftCount,
                    rejectedDraftCount: result.rejectedDraftCount,
                    targetCacheOffset: result.targetCacheOffset
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "dflash_phase_diagnostics":
            guard state.began,
                  request.promptTokens == nil,
                  request.seedTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "DFlash diagnostics request is malformed or before begin"
                )
            }
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                peakRamGB: peakResidentMemoryGB(),
                mlxActiveMemoryBytes: Memory.activeMemory,
                mlxCacheMemoryBytes: Memory.cacheMemory,
                mlxPeakMemoryBytes: Memory.peakMemory,
                acceptedDraftCount: session.acceptedDraftTotal,
                rejectedDraftCount: session.rejectedDraftTotal,
                rollbackRoundCount: session.rollbackRoundCount,
                targetCacheOffset: session.seedTokenCount
                    + session.decodedTokenCount
            )

        case "dflash_reference_rows":
            // REFERENCE SIDE ONLY (contract layer L1). Served by a worker built
            // from the pinned baseline tree over organizer-transformed weights,
            // strictly after the timed window with the candidate torn down. It
            // is stateless on purpose: every request rebuilds its own KV cache
            // from the supplied context, which is what makes the L1
            // self-consistency replay meaningful -- two identical requests have
            // no carried state that could explain a difference away.
            guard request.token == nil,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  let prefixTokens = request.prefixTokens,
                  !prefixTokens.isEmpty,
                  let startOffset = request.startOffset,
                  startOffset >= 0,
                  let rowCount = request.rowCount,
                  rowCount > 0,
                  let declaredBlockWidth = request.declaredBlockWidth,
                  declaredBlockWidth >= rowCount,
                  declaredBlockWidth
                      <= MLXFastConstants.experimentalDFlashMaxBlockSize
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference-rows request is malformed or has "
                        + "cross-kind fields"
                )
            }
            let rows = try LagunaDFlashReference.rows(
                target: session.referenceTarget,
                targetLayerIds: session.referenceTargetLayerIds,
                tokens: prefixTokens,
                startOffset: startOffset,
                count: rowCount
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                referenceK1Argmax: rows.map(\.sequentialArgmax),
                referenceBlockArgmax: rows.map(\.blockArgmax),
                referenceTop2Tokens: rows.map(\.top2Tokens),
                referenceTop2Logits: rows.map(\.top2Logits)
            )

        default:
            throw MLXFastError.invalidInput(
                "DFlash worker rejects request kind \(request.kind)"
            )
        }
    }
}

private final class ExperimentalDFlashAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

/// Bridge the vendored async loaders into the synchronous worker startup path.
///
/// T is deliberately NOT constrained to `Sendable` and the operation is not a
/// `@Sendable` closure: the values crossing this boundary are the loaded
/// `ModelContext` and `any DFlashTargetModel`, neither of which is Sendable and
/// neither of which can be made so from here (both are vendored types). The
/// hop is safe because this bridge is only ever used during single-threaded
/// worker startup -- once, before the protocol loop begins -- so the loaded
/// model is never touched concurrently. `nonisolated(unsafe)` states that
/// reasoning explicitly instead of hiding it behind a false conformance.
private func waitForExperimentalDFlashAsync<T>(
    _ operation: @escaping () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ExperimentalDFlashAsyncResultBox<T>()
    nonisolated(unsafe) let unsafeOperation = operation
    nonisolated(unsafe) let unsafeBox = box
    Task {
        do {
            unsafeBox.result = .success(try await unsafeOperation())
        } catch {
            unsafeBox.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "DFlash async model load completed without a result"
        )
    }
    return try result.get()
}
#endif
