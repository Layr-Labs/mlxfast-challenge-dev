import Foundation
import MLXFastCore

// Trusted operator-only collection of model observations. The participant
// worker receives token arrays, never the private artifact bytes, and is
// closed before the caller can write a calibrated artifact.
extension LagunaRuntime {
    public static func collectPrivateArtifactCalibration(
        weightsPath: String,
        anchors: [PrivateAnchorCalibrationRequest],
        sequences: [PrivateSequenceCalibrationRequest],
        worker workerOptions: RuntimeWorkerOptions
    ) throws -> PrivateReferenceCalibrationObservations {
        try validatePrivateCalibrationRequests(
            anchors: anchors,
            sequences: sequences
        )
        let worker = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: weightsPath
        )
        defer {
            worker.close()
        }

        var anchorObservations: [PrivateAnchorCalibrationObservation] = []
        anchorObservations.reserveCapacity(anchors.count)
        for request in anchors {
            let response = try worker.beginTeacherForcedCorrectness(
                promptTokens: request.contextTokens,
                topK: MLXFastConstants.correctnessTopLogits,
                expectedToken: request.expectedToken
            )
            guard let generatedToken = response.token else {
                throw MLXFastError.invalidInput(
                    "runtime worker anchor calibration response missing token"
                )
            }
            let topLogits = try validatedWorkerTopLogits(
                response.topLogits,
                actualToken: generatedToken,
                maximumCount: MLXFastConstants.correctnessTopLogits
            )
            let expectedTokenLogit: Double
            let expectedTokenRank: Int
            if let workerLogit = response.expectedTokenLogit,
               let workerRank = response.expectedTokenRank
            {
                guard workerLogit.isFinite,
                      workerRank > 0,
                      workerRank <= MLXFastConstants.correctnessTopLogits
                else {
                    throw MLXFastError.invalidInput(
                        "runtime worker anchor calibration returned invalid expected-token diagnostics"
                    )
                }
                expectedTokenLogit = workerLogit
                expectedTokenRank = workerRank
                guard workerRank <= topLogits.count,
                      topLogits[workerRank - 1].token == request.expectedToken,
                      topLogits[workerRank - 1].logit == workerLogit
                else {
                    throw MLXFastError.invalidInput(
                        "runtime worker anchor calibration diagnostics disagree with top_logits"
                    )
                }
            } else {
                guard let expectedIndex = topLogits.firstIndex(
                    where: { $0.token == request.expectedToken }
                ) else {
                    throw MLXFastError.invalidInput(
                        "runtime worker anchor calibration omitted expected-token diagnostics"
                    )
                }
                expectedTokenLogit = topLogits[expectedIndex].logit
                expectedTokenRank = expectedIndex + 1
            }
            let delta = topLogits[0].logit - expectedTokenLogit
            guard delta.isFinite, delta >= 0 else {
                throw MLXFastError.invalidInput(
                    "runtime worker anchor calibration returned an invalid top-logit delta"
                )
            }
            anchorObservations.append(
                PrivateAnchorCalibrationObservation(
                    id: request.id,
                    generatedToken: generatedToken,
                    expectedTokenRank: expectedTokenRank,
                    expectedTokenTopLogitDelta: delta
                )
            )
        }

        var sequenceObservations: [PrivateSequenceCalibrationObservation] = []
        sequenceObservations.reserveCapacity(sequences.count)
        for request in sequences {
            let response = try worker.generateCorrectness(
                promptTokens: request.promptTokens,
                steps: request.steps
            )
            guard let generatedTokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "runtime worker sequence calibration response missing tokens"
                )
            }
            try requireGeneratedTokenCount(
                generatedTokens.count,
                expected: request.steps,
                label: "private sequence calibration"
            )
            guard generatedTokens.allSatisfy({
                $0 >= 0 && $0 < MLXFastConstants.vocabSize
            }) else {
                throw MLXFastError.invalidInput(
                    "runtime worker sequence calibration returned an invalid token"
                )
            }
            sequenceObservations.append(
                PrivateSequenceCalibrationObservation(
                    id: request.id,
                    generatedTokens: generatedTokens
                )
            )
        }
        return PrivateReferenceCalibrationObservations(
            anchors: anchorObservations,
            sequences: sequenceObservations
        )
    }

    private static func validatePrivateCalibrationRequests(
        anchors: [PrivateAnchorCalibrationRequest],
        sequences: [PrivateSequenceCalibrationRequest]
    ) throws {
        var ids = Set<String>()
        for request in anchors {
            guard isPrivateCalibrationID(request.id),
                  ids.insert(request.id).inserted,
                  !request.contextTokens.isEmpty,
                  request.contextTokens.count
                    <= MLXFastConstants.correctnessMaxAnchorContextTokens,
                  request.contextTokens.allSatisfy({
                    $0 >= 0 && $0 < MLXFastConstants.vocabSize
                  }),
                  request.expectedToken >= 0,
                  request.expectedToken < MLXFastConstants.vocabSize
            else {
                throw MLXFastError.invalidInput(
                    "private anchor calibration request is invalid"
                )
            }
        }
        for request in sequences {
            guard isPrivateCalibrationID(request.id),
                  ids.insert(request.id).inserted,
                  !request.promptTokens.isEmpty,
                  request.promptTokens.count
                    <= MLXFastConstants.correctnessMaxBehaviorPromptTokens,
                  request.promptTokens.allSatisfy({
                    $0 >= 0 && $0 < MLXFastConstants.vocabSize
                  }),
                  request.steps > 0,
                  request.steps <= MLXFastConstants.correctnessMaxBehaviorSteps
            else {
                throw MLXFastError.invalidInput(
                    "private sequence calibration request is invalid"
                )
            }
        }
    }

    private static func isPrivateCalibrationID(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty
            && trimmed == value
            && !value.unicodeScalars.contains {
                CharacterSet.controlCharacters.contains($0)
            }
    }
}
