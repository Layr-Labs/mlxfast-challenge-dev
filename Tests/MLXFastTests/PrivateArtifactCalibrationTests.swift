import CryptoKit
import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
import Testing

@Suite(.serialized)
struct PrivateArtifactCalibrationTests {
    @Test
    func goldenCalibrationIsDeterministicAndConsumable() throws {
        let templateData = try fixtureData(
            "synthetic-private-golden-template.json"
        )
        let observations = PrivateReferenceCalibrationObservations(
            anchors: [
                PrivateAnchorCalibrationObservation(
                    id: "synthetic-anchor",
                    generatedToken: 30,
                    expectedTokenRank: 2,
                    expectedTokenTopLogitDelta: 0.125
                )
            ],
            sequences: [
                PrivateSequenceCalibrationObservation(
                    id: "synthetic-behavior-a",
                    generatedTokens: [31, 32]
                ),
                PrivateSequenceCalibrationObservation(
                    id: "synthetic-behavior-b",
                    generatedTokens: [33]
                ),
            ]
        )
        let policy = PrivateAnchorCalibrationPolicy(
            maximumExpectedRank: 3,
            maximumTopLogitDelta: 0.25
        )

        let first = try PrivateArtifactCalibrator.calibrateGolden(
            templateData: templateData,
            observations: observations,
            anchorPolicy: policy,
            requiredSteps: 2,
            requiredPromptTokens: 3
        )
        let second = try PrivateArtifactCalibrator.calibrateGolden(
            templateData: templateData,
            observations: observations,
            anchorPolicy: policy,
            requiredSteps: 2,
            requiredPromptTokens: 3
        )
        #expect(first == second)

        let document = try decodeGoldenDocument(
            from: first.data,
            requiredSteps: 2,
            requiredPromptTokens: 3
        )
        let gates = try #require(document.correctnessGates)
        let anchor = try #require(gates.anchorCases.first)
        #expect(anchor.name == "synthetic-anchor")
        #expect(anchor.contextTokens == [6, 7])
        #expect(anchor.expectedToken == 8)
        #expect(anchor.acceptedTokens == [30])
        #expect(anchor.maxExpectedRank == 2)
        #expect(anchor.maxTopLogitDelta == 0.125)
        #expect(gates.freeRunCases.first?.name == "synthetic-free-run")
        #expect(
            gates.behaviorCases.map(\.name)
                == ["synthetic-behavior-a", "synthetic-behavior-b"]
        )
        #expect(
            gates.behaviorCases[0].acceptedTokenSequences == [[31, 32]]
        )
        #expect(gates.behaviorCases[0].maxNewTokens == 2)
        #expect(gates.behaviorCases[0].semanticDomain == "synthetic-domain")
        #expect(
            gates.behaviorCases[0].semanticSubdomain
                == "synthetic-subdomain"
        )
        #expect(gates.behaviorCases[1].acceptedTokenSequences == [[33]])

        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("calibrated.json")
        _ = try PrivateArtifactWriter.write(first.data, to: output.path)
        let fixture = try loadGoldenFixture(
            from: output.path,
            requiredSteps: 2,
            requiredPromptTokens: 3
        )
        #expect(fixture.correctnessGates == gates)
        let acceptedSequences =
            fixture.correctnessGates?.behaviorCases[0]
                .acceptedTokenSequences ?? []
        let behaviorMatch = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
            acceptedSequences: acceptedSequences,
            actual: [31, 32]
        )
        #expect(behaviorMatch.passed)
    }

    @Test
    func goldenCalibrationRejectsMismatchesAndUnapprovedTolerance() throws {
        let templateData = try fixtureData(
            "synthetic-private-golden-template.json"
        )
        let matchingAnchor = PrivateAnchorCalibrationObservation(
            id: "synthetic-anchor",
            generatedToken: 30,
            expectedTokenRank: 2,
            expectedTokenTopLogitDelta: 0.125
        )
        let reversedBehaviors = [
            PrivateSequenceCalibrationObservation(
                id: "synthetic-behavior-b",
                generatedTokens: [33]
            ),
            PrivateSequenceCalibrationObservation(
                id: "synthetic-behavior-a",
                generatedTokens: [31, 32]
            ),
        ]
        #expect(throws: MLXFastError.self) {
            _ = try PrivateArtifactCalibrator.calibrateGolden(
                templateData: templateData,
                observations: PrivateReferenceCalibrationObservations(
                    anchors: [matchingAnchor],
                    sequences: reversedBehaviors
                ),
                anchorPolicy: PrivateAnchorCalibrationPolicy(
                    maximumExpectedRank: 3,
                    maximumTopLogitDelta: 0.25
                ),
                requiredSteps: 2,
                requiredPromptTokens: 3
            )
        }

        #expect(throws: MLXFastError.self) {
            _ = try PrivateArtifactCalibrator.calibrateGolden(
                templateData: templateData,
                observations: PrivateReferenceCalibrationObservations(
                    anchors: [
                        PrivateAnchorCalibrationObservation(
                            id: "synthetic-anchor",
                            generatedToken: 30,
                            expectedTokenRank: 3,
                            expectedTokenTopLogitDelta: 0.5
                        )
                    ],
                    sequences: [
                        PrivateSequenceCalibrationObservation(
                            id: "synthetic-behavior-a",
                            generatedTokens: [31, 32]
                        ),
                        PrivateSequenceCalibrationObservation(
                            id: "synthetic-behavior-b",
                            generatedTokens: [33]
                        ),
                    ]
                ),
                anchorPolicy: PrivateAnchorCalibrationPolicy(
                    maximumExpectedRank: 2,
                    maximumTopLogitDelta: 0.25
                ),
                requiredSteps: 2,
                requiredPromptTokens: 3
            )
        }
    }

    @Test
    func gpqaCalibrationChangesOnlyAcceptedFirstTokens() throws {
        let templateData = try fixtureData(
            "synthetic-private-gpqa-template.json"
        )
        let observations = [
            PrivateSequenceCalibrationObservation(
                id: "synthetic-gpqa-a",
                generatedTokens: [41, 42, 43]
            ),
            PrivateSequenceCalibrationObservation(
                id: "synthetic-gpqa-b",
                generatedTokens: [51, 52]
            ),
        ]
        let first = try PrivateArtifactCalibrator.calibrateGPQA(
            templateData: templateData,
            observations: observations
        )
        let second = try PrivateArtifactCalibrator.calibrateGPQA(
            templateData: templateData,
            observations: observations
        )
        #expect(first == second)
        #expect(first.caseCount == 2)
        #expect(
            try PrivateArtifactCalibrator.gpqaCases(from: first.data)
                == [
                    PrivateGPQACalibrationCase(
                        id: "synthetic-gpqa-a",
                        prompt:
                            "Synthetic public question A?\nA. Alpha\nB. Beta"
                    ),
                    PrivateGPQACalibrationCase(
                        id: "synthetic-gpqa-b",
                        prompt:
                            "Synthetic public question B?\nA. Gamma\nB. Delta"
                    ),
                ]
        )

        let original = try jsonObject(templateData)
        let calibrated = try jsonObject(first.data)
        #expect(original["version"] as? Int == calibrated["version"] as? Int)
        #expect(original["status"] as? String == calibrated["status"] as? String)
        #expect(
            original["needs_reference_output"] as? Bool
                == calibrated["needs_reference_output"] as? Bool
        )
        let originalCases = try #require(
            original["cases"] as? [[String: Any]]
        )
        let calibratedCases = try #require(
            calibrated["cases"] as? [[String: Any]]
        )
        #expect(calibratedCases.count == originalCases.count)
        for index in originalCases.indices {
            var originalNonModel = originalCases[index]
            var calibratedNonModel = calibratedCases[index]
            originalNonModel.removeValue(
                forKey: "accepted_token_sequences"
            )
            calibratedNonModel.removeValue(
                forKey: "accepted_token_sequences"
            )
            #expect(
                try canonicalJSON(originalNonModel)
                    == canonicalJSON(calibratedNonModel)
            )
        }
        #expect(
            tokenSequences(calibratedCases[0]["accepted_token_sequences"])
                == [[41]]
        )
        #expect(
            tokenSequences(calibratedCases[1]["accepted_token_sequences"])
                == [[51]]
        )
    }

    @Test
    func gpqaCalibrationRejectsMalformedAndMismatchedCases() throws {
        let templateData = try fixtureData(
            "synthetic-private-gpqa-template.json"
        )
        let malformed = String(decoding: templateData, as: UTF8.self)
            .replacingOccurrences(
                of: #""status": "synthetic-template""#,
                with: #""unknown_status": "synthetic-template""#
            )
        #expect(throws: MLXFastError.self) {
            _ = try PrivateArtifactCalibrator.gpqaCases(
                from: Data(malformed.utf8)
            )
        }
        #expect(throws: MLXFastError.self) {
            _ = try PrivateArtifactCalibrator.calibrateGPQA(
                templateData: templateData,
                observations: [
                    PrivateSequenceCalibrationObservation(
                        id: "synthetic-gpqa-b",
                        generatedTokens: [51]
                    ),
                    PrivateSequenceCalibrationObservation(
                        id: "synthetic-gpqa-a",
                        generatedTokens: [41]
                    ),
                ]
            )
        }
    }

    @Test
    func privateWriterAtomicallyReplacesWithMode0600() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("private.json")
        try Data("old-private-value".utf8).write(to: output)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o644],
            ofItemAtPath: output.path
        )

        let newData = Data(#"{"synthetic":"private-value"}\n"#.utf8)
        let result = try PrivateArtifactWriter.write(
            newData,
            to: output.path
        )
        #expect(try Data(contentsOf: output) == newData)
        let attributes = try FileManager.default.attributesOfItem(
            atPath: output.path
        )
        #expect(
            (attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600
        )
        let expectedHash = SHA256.hash(data: newData)
            .map { String(format: "%02x", $0) }
            .joined()
        #expect(result.sha256 == expectedHash)
        #expect(result.byteCount == newData.count)
        #expect(
            try FileManager.default.contentsOfDirectory(
                atPath: root.path
            ).allSatisfy { !$0.contains(".private-") }
        )

        #expect(throws: MLXFastError.self) {
            _ = try PrivateArtifactWriter.write(Data(), to: output.path)
        }
        #expect(try Data(contentsOf: output) == newData)
    }

    @Test
    func trustedCollectorUsesOneWorkerWithoutForwardingSecrets() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let worker = root.appendingPathComponent("synthetic-worker")
        try writeExecutable(
            """
            #!/bin/bash
            printf '%s\n' 'synthetic-worker-secret-must-not-be-forwarded' >&2
            printf '%s\n' '{"id":0,"nonce":"private-calibration-nonce","ok":true}'
            IFS= read -r anchor_request
            printf '%s\n' '{"id":1,"nonce":"private-calibration-nonce","ok":true,"token":30,"top_logits":[{"token":30,"logit":10.0},{"token":8,"logit":9.75}],"expected_token_logit":9.75,"expected_token_rank":2,"top_logit_margin":0.25}'
            IFS= read -r sequence_request
            printf '%s\n' '{"id":2,"nonce":"private-calibration-nonce","ok":true,"tokens":[31,32]}'
            """,
            to: worker
        )

        let observations = try LagunaRuntime
            .collectPrivateArtifactCalibration(
                weightsPath: "/synthetic/weights",
                anchors: [
                    PrivateAnchorCalibrationRequest(
                        id: "synthetic-anchor",
                        contextTokens: [6, 7],
                        expectedToken: 8
                    )
                ],
                sequences: [
                    PrivateSequenceCalibrationRequest(
                        id: "synthetic-behavior",
                        promptTokens: [10, 11],
                        steps: 2
                    )
                ],
                worker: RuntimeWorkerOptions(
                    executablePath: worker.path,
                    forwardsWorkerStderr: false,
                    requestTimeoutSeconds: 5
                )
            )
        #expect(
            observations.anchors
                == [
                    PrivateAnchorCalibrationObservation(
                        id: "synthetic-anchor",
                        generatedToken: 30,
                        expectedTokenRank: 2,
                        expectedTokenTopLogitDelta: 0.25
                    )
                ]
        )
        #expect(
            observations.sequences
                == [
                    PrivateSequenceCalibrationObservation(
                        id: "synthetic-behavior",
                        generatedTokens: [31, 32]
                    )
                ]
        )
    }

    @Test
    func privateCommandLogsContainOnlyCountsHashesAndStatus() throws {
        let hashA = String(repeating: "a", count: 64)
        let hashB = String(repeating: "b", count: 64)
        let secretMarkers = [
            "synthetic-anchor",
            "synthetic-gpqa-a",
            "Synthetic public question",
            "A. Alpha",
            "[98765]",
            "/private/output.json",
        ]
        let lines = [
            PrivateArtifactLogSummary.goldenSuccess(
                anchorCount: 1,
                behaviorCount: 2,
                sha256: hashA
            ),
            PrivateArtifactLogSummary.gpqaSuccess(
                caseCount: 2,
                calibratedSHA256: hashA,
                answersSHA256: hashB
            ),
            PrivateArtifactLogSummary.goldenFailure,
            PrivateArtifactLogSummary.gpqaFailure,
        ]
        for line in lines {
            for marker in secretMarkers {
                #expect(!line.contains(marker))
            }
        }

        let cli = try String(
            contentsOfFile: "Sources/MLXFastCLI/main.swift",
            encoding: .utf8
        )
        #expect(cli.contains("case \"calibrate-private-golden\""))
        #expect(!cli.contains("case \"calibrate-gpqa-gates\""))
        #expect(cli.contains("_ = Darwin.umask(mode_t(0o077))"))
        #expect(cli.contains("requirePrivateIsolation: true"))
        #expect(cli.contains("PrivateArtifactLogSummary.goldenSuccess"))
        #expect(cli.contains("PrivateArtifactLogSummary.gpqaSuccess"))
        #expect(
            cli.contains(
                "generate-gpqa-answers requires --gpqa-output PATH"
            )
        )
    }

    private func fixtureData(_ name: String) throws -> Data {
        try Data(
            contentsOf: URL(
                fileURLWithPath:
                    "Tests/Fixtures/PrivateArtifactCalibration/\(name)"
            )
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true
        )
        return url
    }

    private func writeExecutable(_ contents: String, to url: URL) throws {
        try contents.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: url.path
        )
    }

    private func jsonObject(_ data: Data) throws -> [String: Any] {
        try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    private func canonicalJSON(_ value: Any) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: value,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
    }

    private func tokenSequences(_ value: Any?) -> [[Int]]? {
        value as? [[Int]]
    }
}
