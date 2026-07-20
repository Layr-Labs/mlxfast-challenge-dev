import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
import Testing

private final class ScriptedMTPSemanticWorker:
    TrainedMTPSemanticGPQAWorker
{
    let name: String
    let beginResponse: RuntimeWorkerResponse
    var blockResponses: [RuntimeWorkerResponse]
    var observedSeedTokens: [[Int]] = []
    var observedBlockRequests: [(previousToken: Int, maxBlockSize: Int)] = []
    var closed = false
    var closeSucceeds = true
    let recordEvent: (String) -> Void

    init(
        name: String,
        beginResponse: RuntimeWorkerResponse,
        blockResponses: [RuntimeWorkerResponse],
        recordEvent: @escaping (String) -> Void = { _ in }
    ) {
        self.name = name
        self.beginResponse = beginResponse
        self.blockResponses = blockResponses
        self.recordEvent = recordEvent
    }

    func beginTrainedMTPDecode(
        seedTokens: [Int]
    ) throws -> RuntimeWorkerResponse {
        observedSeedTokens.append(seedTokens)
        recordEvent("\(name)-begin")
        return beginResponse
    }

    func trainedMTPDecodeBlock(
        previousToken: Int,
        maxBlockSize: Int
    ) throws -> RuntimeWorkerResponse {
        observedBlockRequests.append((previousToken, maxBlockSize))
        recordEvent("\(name)-block")
        guard !blockResponses.isEmpty else {
            throw MLXFastError.invalidInput("test worker has no block response")
        }
        return blockResponses.removeFirst()
    }

    @discardableResult
    func close() -> Bool {
        guard !closed else {
            return closeSucceeds
        }
        closed = true
        recordEvent("\(name)-close")
        return closeSucceeds
    }
}

private func semanticCaptureCase(
    id: String,
    prompt: String,
    reference: String,
    promptTokens: [Int]
) -> MTPSemanticGPQACaptureCase {
    MTPSemanticGPQACaptureCase(
        id: id,
        domain: "science",
        subdomain: "test",
        prompt: prompt,
        answerKey: "A",
        referenceAnswer: reference,
        promptTokens: promptTokens
    )
}

private func semanticWorkerOptions() -> RuntimeWorkerOptions {
    RuntimeWorkerOptions(executablePath: "/test/worker")
}

@Test
func trainedMTPSemanticCaptureUsesExactPairFreshWorkersAndTruncatesTail()
    throws
{
    var events: [String] = []
    let first = ScriptedMTPSemanticWorker(
        name: "first",
        beginResponse: RuntimeWorkerResponse(
            id: 1,
            nonce: "first-nonce",
            ok: true,
            seedToken: 10
        ),
        blockResponses: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "first-nonce",
                ok: true,
                tokens: [11, 12, 13, 14]
            ),
            RuntimeWorkerResponse(
                id: 3,
                nonce: "first-nonce",
                ok: true,
                tokens: [15]
            ),
        ],
        recordEvent: { events.append($0) }
    )
    let second = ScriptedMTPSemanticWorker(
        name: "second",
        beginResponse: RuntimeWorkerResponse(
            id: 1,
            nonce: "second-nonce",
            ok: true,
            seedToken: 20
        ),
        blockResponses: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "second-nonce",
                ok: true,
                tokens: [21, 22]
            ),
            RuntimeWorkerResponse(
                id: 3,
                nonce: "second-nonce",
                ok: true,
                tokens: [23, 24, 25]
            ),
        ],
        recordEvent: { events.append($0) }
    )
    let workers = [first, second]
    var workerIndex = 0
    var launches: [RuntimeWorkerLaunch] = []
    var encodedDocument: Data?
    let secretPromptOne = "PRIVATE QUESTION ONE"
    let secretPromptTwo = "PRIVATE QUESTION TWO"
    let secretReferenceOne = "PRIVATE REFERENCE ONE"
    let secretReferenceTwo = "PRIVATE REFERENCE TWO"
    let secretReferencePath = "/private/gpqa-reference.json"
    let secretOutputPath = "/private/semantic-answers.json"

    let summary = try GemmaRuntime
        .captureExperimentalTrainedMTPSemanticGPQACases(
            [
                semanticCaptureCase(
                    id: "case-1",
                    prompt: secretPromptOne,
                    reference: secretReferenceOne,
                    promptTokens: [101, 102]
                ),
                semanticCaptureCase(
                    id: "case-2",
                    prompt: secretPromptTwo,
                    reference: secretReferenceTwo,
                    promptTokens: [201, 202]
                ),
            ],
            targetWeightsPath: "mtp-weights",
            assistantPath: "/organizer/assistant",
            contractPath: "fixtures/contract.json",
            maxBlockSize: 4,
            maxNewTokens: 6,
            workerOptions: semanticWorkerOptions(),
            makeWorker: { _, weightsPath, launch in
                #expect(weightsPath == "mtp-weights")
                launches.append(launch)
                let worker = workers[workerIndex]
                workerIndex += 1
                events.append("\(worker.name)-launch")
                return worker
            },
            decode: { tokens in
                let launchedWorkersAreClosed = workers
                    .prefix(workerIndex)
                    .reduce(true) { $0 && $1.closed }
                #expect(launchedWorkersAreClosed)
                events.append("decode")
                return tokens.map(String.init).joined(separator: " ")
            },
            write: { answers in
                let allWorkersAreClosed = workers.reduce(true) {
                    $0 && $1.closed
                }
                #expect(allWorkersAreClosed)
                events.append("write")
                encodedDocument = try GemmaRuntime
                    .semanticGPQAAnswerDocumentData(answers)
            }
        )

    #expect(summary.caseCount == 2)
    #expect(summary.generatedTokenCount == 12)
    #expect(workerIndex == 2)
    #expect(first.observedSeedTokens == [[101, 102]])
    #expect(second.observedSeedTokens == [[201, 202]])
    #expect(first.observedBlockRequests.map(\.previousToken) == [10, 14])
    #expect(first.observedBlockRequests.map(\.maxBlockSize) == [4, 1])
    #expect(second.observedBlockRequests.map(\.previousToken) == [20, 22])
    #expect(second.observedBlockRequests.map(\.maxBlockSize) == [4, 3])
    #expect(events.last == "write")
    #expect(events.firstIndex(of: "first-close")! < events.firstIndex(of: "write")!)
    #expect(events.firstIndex(of: "second-close")! < events.firstIndex(of: "write")!)

    for launch in launches {
        guard case .trainedMTP(
            let assistantPath,
            let contractPath,
            let verificationMode
        ) = launch else {
            Issue.record("semantic capture selected the serial worker")
            continue
        }
        #expect(assistantPath == "/organizer/assistant")
        #expect(contractPath == "fixtures/contract.json")
        #expect(verificationMode == .exactPair)
        let arguments = launch.arguments(weightsPath: "mtp-weights")
        #expect(arguments.first == "mtp-runtime-worker")
        #expect(arguments.contains("--require-trained-assistant"))
        #expect(!arguments.contains(secretPromptOne))
        #expect(!arguments.contains(secretPromptTwo))
        #expect(!arguments.contains(secretReferenceOne))
        #expect(!arguments.contains(secretReferenceTwo))
        #expect(!arguments.contains(secretReferencePath))
        #expect(!arguments.contains(secretOutputPath))
    }

    let data = try #require(encodedDocument)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )
    #expect(Set(object.keys) == Set(["version", "cases"]))
    #expect(object["version"] as? Int == 1)
    let jsonCases = try #require(object["cases"] as? [[String: Any]])
    #expect(jsonCases.count == 2)
    #expect(
        Set(jsonCases[0].keys) == Set([
            "answer_key",
            "candidate_answer",
            "candidate_tokens",
            "domain",
            "id",
            "max_new_tokens",
            "prompt",
            "reference_answer",
            "subdomain",
        ])
    )
    #expect(jsonCases[0]["prompt"] as? String == secretPromptOne)
    #expect(
        jsonCases[0]["reference_answer"] as? String == secretReferenceOne
    )
    #expect(jsonCases[0]["candidate_tokens"] as? [Int] == [10, 11, 12, 13, 14, 15])
    #expect(jsonCases[0]["max_new_tokens"] as? Int == 6)
}

@Test
func trainedMTPSemanticCollectorRejectsMalformedAndOversizedResponses() {
    func expectFailure(
        begin: RuntimeWorkerResponse,
        blocks: [RuntimeWorkerResponse],
        maxNewTokens: Int = 3
    ) {
        let worker = ScriptedMTPSemanticWorker(
            name: "invalid",
            beginResponse: begin,
            blockResponses: blocks
        )
        #expect(throws: MLXFastError.self) {
            _ = try GemmaRuntime
                .collectExperimentalTrainedMTPSemanticTokens(
                    promptTokens: [1, 2],
                    maxBlockSize: 2,
                    maxNewTokens: maxNewTokens,
                    worker: worker
                )
        }
    }

    expectFailure(
        begin: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true
        ),
        blocks: []
    )
    expectFailure(
        begin: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 1
        ),
        blocks: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "n",
                ok: true,
                tokens: []
            )
        ]
    )
    expectFailure(
        begin: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 1
        ),
        blocks: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "n",
                ok: true,
                tokens: [2, 3, 4]
            )
        ]
    )
    expectFailure(
        begin: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 1
        ),
        blocks: [
            RuntimeWorkerResponse(
                id: 3,
                nonce: "different",
                ok: true,
                tokens: [2]
            )
        ]
    )
    expectFailure(
        begin: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 1
        ),
        blocks: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "n",
                ok: true,
                tokens: [MLXFastConstants.vocabSize]
            )
        ],
        maxNewTokens: 2
    )
}

@Test
func trainedMTPSemanticCaptureClosesWorkerOnFailureAndWritesNothing() {
    var events: [String] = []
    let worker = ScriptedMTPSemanticWorker(
        name: "failing",
        beginResponse: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 10
        ),
        blockResponses: [
            RuntimeWorkerResponse(
                id: 2,
                nonce: "n",
                ok: true,
                tokens: []
            )
        ],
        recordEvent: { events.append($0) }
    )
    var didWrite = false

    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime
            .captureExperimentalTrainedMTPSemanticGPQACases(
                [
                    semanticCaptureCase(
                        id: "case",
                        prompt: "private",
                        reference: "reference",
                        promptTokens: [1, 2]
                    )
                ],
                targetWeightsPath: "weights",
                assistantPath: "assistant",
                contractPath: "contract",
                maxBlockSize: 2,
                maxNewTokens: 2,
                workerOptions: semanticWorkerOptions(),
                makeWorker: { _, _, _ in worker },
                decode: { _ in "answer" },
                write: { _ in didWrite = true }
            )
    }
    #expect(worker.closed)
    #expect(events.last == "failing-close")
    #expect(!didWrite)
}

@Test
func trainedMTPSemanticCaptureRequiresCleanShutdownBeforeDecodeOrWrite() {
    let worker = ScriptedMTPSemanticWorker(
        name: "unclean",
        beginResponse: RuntimeWorkerResponse(
            id: 1,
            nonce: "n",
            ok: true,
            seedToken: 10
        ),
        blockResponses: []
    )
    worker.closeSucceeds = false
    var didDecode = false
    var didWrite = false

    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime
            .captureExperimentalTrainedMTPSemanticGPQACases(
                [
                    semanticCaptureCase(
                        id: "case",
                        prompt: "private",
                        reference: "reference",
                        promptTokens: [1, 2]
                    )
                ],
                targetWeightsPath: "weights",
                assistantPath: "assistant",
                contractPath: "contract",
                maxBlockSize: 2,
                maxNewTokens: 1,
                workerOptions: semanticWorkerOptions(),
                makeWorker: { _, _, _ in worker },
                decode: { _ in
                    didDecode = true
                    return "answer"
                },
                write: { _ in didWrite = true }
            )
    }
    #expect(worker.closed)
    #expect(!didDecode)
    #expect(!didWrite)
}

@Test
func mtpSemanticPrivateOutputMustBeStrictDescendant() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-private-path-\(UUID().uuidString)"
    )
    let privateRoot = root.appendingPathComponent("private")
    let outsideRoot = root.appendingPathComponent("outside")
    try FileManager.default.createDirectory(
        at: privateRoot,
        withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
        at: outsideRoot,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    try GemmaRuntime.validateMTPSemanticPrivateOutputPath(
        privateRoot.appendingPathComponent("nested/answers.json").path,
        privateDirectoryPath: privateRoot.path
    )
    for (output, privateDirectory) in [
        (privateRoot.appendingPathComponent("answers.json").path, ""),
        (privateRoot.path, privateRoot.path),
        (outsideRoot.appendingPathComponent("answers.json").path, privateRoot.path),
        (
            root.appendingPathComponent("private-sibling/answers.json").path,
            privateRoot.path
        ),
    ] {
        #expect(throws: MLXFastError.self) {
            try GemmaRuntime.validateMTPSemanticPrivateOutputPath(
                output,
                privateDirectoryPath: privateDirectory
            )
        }
    }

    let escape = privateRoot.appendingPathComponent("escape")
    try FileManager.default.createSymbolicLink(
        at: escape,
        withDestinationURL: outsideRoot
    )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticPrivateOutputPath(
            escape.appendingPathComponent("answers.json").path,
            privateDirectoryPath: privateRoot.path
        )
    }
}

@Test
func mtpSemanticSandboxDeniesPrivateReadsAndCrossCaseWrites() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-sandbox-\(UUID().uuidString)"
    )
    let privateRoot = root.appendingPathComponent("private")
    let reference = root.appendingPathComponent("reference.json")
    try FileManager.default.createDirectory(
        at: privateRoot,
        withIntermediateDirectories: true
    )
    try "{}".write(to: reference, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: root) }

    func options(profile: URL) -> RuntimeWorkerOptions {
        RuntimeWorkerOptions(
            executablePath: "/test/worker",
            sandboxProfilePath: profile.path
        )
    }
    let valid = root.appendingPathComponent("valid.sb")
    try """
    (version 1)
    (allow default)
    (deny file-read* (literal "\(reference.path)"))
    (deny file-read* (subpath "\(privateRoot.path)"))
    (deny ipc-posix-shm*)
    (deny ipc-posix-sem*)
    (deny ipc-sysv*)
    (allow ipc-posix-shm-read*
      (ipc-posix-name "apple.shm.notification_center")
      (ipc-posix-name "apple.shm.cfprefsd.daemon")
      (ipc-posix-name-prefix "apple.cfprefs.")
      (ipc-posix-name-prefix "apple.shm.cfprefsd."))
    (deny user-preference-write)
    (deny mach-lookup (global-name-prefix "com.apple.pasteboard."))
    (deny mach-lookup (global-name-prefix "com.apple.logd"))
    (deny mach-lookup (global-name "com.apple.system.logger"))
    (deny file-write*)
    (allow file-write* (literal "/dev/null"))
    """.write(to: valid, atomically: true, encoding: .utf8)
    try GemmaRuntime.validateMTPSemanticWorkerSandbox(
        options(profile: valid),
        referenceCasesPath: reference.path,
        privateDirectoryPath: privateRoot.path
    )

    let leakedPrivate = root.appendingPathComponent("leaked-private.sb")
    try """
    (version 1)
    (allow default)
    (deny file-read* (literal "\(reference.path)"))
    (deny ipc-posix-shm*)
    (deny ipc-posix-sem*)
    (deny ipc-sysv*)
    (allow ipc-posix-shm-read*
      (ipc-posix-name "apple.shm.notification_center")
      (ipc-posix-name "apple.shm.cfprefsd.daemon")
      (ipc-posix-name-prefix "apple.cfprefs.")
      (ipc-posix-name-prefix "apple.shm.cfprefsd."))
    (deny user-preference-write)
    (deny mach-lookup (global-name-prefix "com.apple.logd"))
    (deny mach-lookup (global-name "com.apple.system.logger"))
    (deny file-write*)
    (allow file-write* (literal "/dev/null"))
    """.write(to: leakedPrivate, atomically: true, encoding: .utf8)
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticWorkerSandbox(
            options(profile: leakedPrivate),
            referenceCasesPath: reference.path,
            privateDirectoryPath: privateRoot.path
        )
    }

    let writable = root.appendingPathComponent("writable.sb")
    try """
    (version 1)
    (allow default)
    (deny file-read* (literal "\(reference.path)"))
    (deny file-read* (subpath "\(privateRoot.path)"))
    (deny ipc-posix-shm*)
    (deny ipc-posix-sem*)
    (deny ipc-sysv*)
    (deny file-write*)
    (allow file-write* (literal "/dev/null"))
    (allow file-write* (subpath "\(root.path)/shared"))
    """.write(to: writable, atomically: true, encoding: .utf8)
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticWorkerSandbox(
            options(profile: writable),
            referenceCasesPath: reference.path,
            privateDirectoryPath: privateRoot.path
        )
    }

    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticWorkerSandbox(
            RuntimeWorkerOptions(executablePath: "/test/worker"),
            referenceCasesPath: reference.path,
            privateDirectoryPath: privateRoot.path
        )
    }

    let namedIPC = root.appendingPathComponent("named-ipc.sb")
    try """
    (version 1)
    (allow default)
    (deny file-read* (literal "\(reference.path)"))
    (deny file-read* (subpath "\(privateRoot.path)"))
    (deny file-write*)
    (allow file-write* (literal "/dev/null"))
    """.write(to: namedIPC, atomically: true, encoding: .utf8)
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticWorkerSandbox(
            options(profile: namedIPC),
            referenceCasesPath: reference.path,
            privateDirectoryPath: privateRoot.path
        )
    }

    let pasteboard = root.appendingPathComponent("pasteboard.sb")
    try """
    (version 1)
    (allow default)
    (deny file-read* (literal "\(reference.path)"))
    (deny file-read* (subpath "\(privateRoot.path)"))
    (deny ipc-posix-shm*)
    (deny ipc-posix-sem*)
    (deny ipc-sysv*)
    (allow ipc-posix-shm-read*
      (ipc-posix-name "apple.shm.notification_center")
      (ipc-posix-name "apple.shm.cfprefsd.daemon")
      (ipc-posix-name-prefix "apple.cfprefs.")
      (ipc-posix-name-prefix "apple.shm.cfprefsd."))
    (deny user-preference-write)
    (deny mach-lookup (global-name-prefix "com.apple.logd"))
    (deny mach-lookup (global-name "com.apple.system.logger"))
    (deny file-write*)
    (allow file-write* (literal "/dev/null"))
    """.write(to: pasteboard, atomically: true, encoding: .utf8)
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateMTPSemanticWorkerSandbox(
            options(profile: pasteboard),
            referenceCasesPath: reference.path,
            privateDirectoryPath: privateRoot.path
        )
    }
}

@Test
func trainedMTPSemanticOptionsAndReferenceFilesFailClosed() throws {
    let valid = ExperimentalTrainedMTPSemanticGPQAOptions(
        sourceTargetPath: "source",
        targetWeightsPath: "weights",
        assistantPath: "assistant",
        contractPath: "contract",
        tokenizerPath: "source",
        referenceCasesPath: "reference",
        privateDirectoryPath: "private",
        outputPath: "private/output",
        maxBlockSize: 4,
        caseCount: 5,
        maxNewTokens: 64,
        requireTrainedAssistant: true
    )
    try GemmaRuntime
        .validateExperimentalTrainedMTPSemanticGPQAOptions(valid)

    let unpinnedTokenizer =
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "different-tokenizer",
            referenceCasesPath: "reference",
            privateDirectoryPath: "private",
            outputPath: "private/output",
            requireTrainedAssistant: true
        )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime
            .validateExperimentalTrainedMTPSemanticGPQAOptions(
                unpinnedTokenizer
            )
    }
    let missingPrivateDirectory =
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "source",
            referenceCasesPath: "reference",
            privateDirectoryPath: "",
            outputPath: "private/output",
            requireTrainedAssistant: true
        )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime
            .validateExperimentalTrainedMTPSemanticGPQAOptions(
                missingPrivateDirectory
            )
    }

    for invalid in [
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "source",
            referenceCasesPath: "reference",
            privateDirectoryPath: "private",
            outputPath: "private/output",
            maxBlockSize: 1,
            requireTrainedAssistant: true
        ),
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "source",
            referenceCasesPath: "reference",
            privateDirectoryPath: "private",
            outputPath: "private/output",
            caseCount: GemmaRuntime.mtpSemanticGPQAMaxCaseCount + 1,
            requireTrainedAssistant: true
        ),
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "source",
            referenceCasesPath: "reference",
            privateDirectoryPath: "private",
            outputPath: "private/output",
            maxNewTokens:
                MLXFastConstants.correctnessMaxBehaviorSteps + 1,
            requireTrainedAssistant: true
        ),
        ExperimentalTrainedMTPSemanticGPQAOptions(
            sourceTargetPath: "source",
            targetWeightsPath: "weights",
            assistantPath: "assistant",
            contractPath: "contract",
            tokenizerPath: "source",
            referenceCasesPath: "reference",
            privateDirectoryPath: "private",
            outputPath: "private/output",
            requireTrainedAssistant: false
        ),
    ] {
        #expect(throws: MLXFastError.self) {
            try GemmaRuntime
                .validateExperimentalTrainedMTPSemanticGPQAOptions(invalid)
        }
    }

    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-reference-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(
        at: root,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let malformed = root.appendingPathComponent("malformed.json")
    try Data(#"{"cases":["#.utf8).write(to: malformed)
    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime.loadMTPSemanticGPQAReferenceDocument(
            from: malformed.path
        )
    }

    let oversized = root.appendingPathComponent("oversized.json")
    try Data(
        repeating: 0x20,
        count: GemmaRuntime.mtpSemanticGPQAMaxReferenceFileByteCount + 1
    ).write(to: oversized)
    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime.loadMTPSemanticGPQAReferenceDocument(
            from: oversized.path
        )
    }

    let duplicateIDs = root.appendingPathComponent("duplicate.json")
    try Data(
        """
        {"cases":[
          {"id":"same","prompt":"Question\\nA. one","answer_key":"A"},
          {"id":"same","prompt":"Question\\nA. two","answer_key":"A"}
        ]}
        """.utf8
    ).write(to: duplicateIDs)
    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime.loadMTPSemanticGPQAReferenceDocument(
            from: duplicateIDs.path
        )
    }
}

@Test
func serialAndMTPSemanticPathsShareOneGPQAReferenceContract() throws {
    #expect(GPQAReferenceDocument.jsonKeys == Set(["cases"]))
    #expect(
        GPQAReferenceCase.jsonKeys == Set([
            "id",
            "prompt",
            "expected_response",
            "answer_key",
            "accepted_token_sequences",
            "accepted_responses",
            "domain",
            "subdomain",
        ])
    )
    let document = try JSONDecoder().decode(
        GPQAReferenceDocument.self,
        from: Data(
            """
            {"cases":[{
              "id":"shared",
              "prompt":"Question\\nA. first\\nB. second",
              "expected_response":null,
              "answer_key":"B",
              "accepted_token_sequences":[[11],[12]],
              "accepted_responses":["B"],
              "domain":"science",
              "subdomain":"physics"
            }]}
            """.utf8
        )
    )
    let item = try #require(document.cases.first)
    #expect(item.identifier == "shared")
    #expect(item.acceptedTokenSequences == [[11], [12]])
    #expect(item.semanticReferenceAnswer == "B")

    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let mtp = try String(
        contentsOfFile:
            "Sources/MLXFastTrustedHarness/GemmaRuntimeMTPSemanticGPQA.swift",
        encoding: .utf8
    )
    #expect(!cli.contains("private struct GPQAReferenceDocument"))
    #expect(!cli.contains("private struct GPQAReferenceCase"))
    #expect(!mtp.contains("struct MTPSemanticGPQAReferenceDocument"))
    #expect(!mtp.contains("struct MTPSemanticGPQAReferenceCase"))
    #expect(cli.contains("JSONDecoder().decode(GPQAReferenceDocument.self"))
    #expect(mtp.contains("JSONDecoder().decode("))
    #expect(mtp.contains("GPQAReferenceDocument.self"))
}

@Test
func gpqaReferenceDecoderRejectsUnknownMalformedEmptyOversizedAndDuplicateValues()
    throws
{
    func expectRejected(_ object: Any) throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(
                GPQAReferenceDocument.self,
                from: data
            )
        }
    }

    let validCase: [String: Any] = [
        "id": "case",
        "prompt": "Question\nA. answer",
        "answer_key": "A",
        "accepted_token_sequences": [[1]],
    ]
    try expectRejected([
        "cases": [validCase],
        "unexpected": true,
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": "Question",
            "answer_key": "A",
            "unexpected": true,
        ]],
    ])
    try expectRejected(["cases": []])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": 7,
            "answer_key": "A",
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": " \n ",
            "answer_key": "A",
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": String(
                repeating: "x",
                count: GPQAReferenceCase.maximumIdentifierByteCount + 1
            ),
            "prompt": "Question",
            "answer_key": "A",
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": String(
                repeating: "x",
                count: GPQAReferenceCase.maximumPromptByteCount + 1
            ),
            "answer_key": "A",
        ]],
    ])
    try expectRejected([
        "cases": Array(
            repeating: validCase,
            count: GPQAReferenceDocument.maximumCaseCount + 1
        ),
    ])
    try expectRejected([
        "cases": [validCase, validCase],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": "Question",
            "accepted_responses": ["A", "A"],
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": "Question",
            "accepted_token_sequences": [[1], [1]],
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": "Question",
            "accepted_token_sequences": [
                [MLXFastConstants.vocabSize],
            ],
        ]],
    ])
    try expectRejected([
        "cases": [[
            "id": "case",
            "prompt": "Question",
        ]],
    ])
}

@Test
func semanticAnswerWriterIsBoundedPrivateAndSymlinkSafe() throws {
    let answer = GemmaRuntime.SemanticGPQAAnswerCase(
        id: "case",
        domain: nil,
        subdomain: nil,
        prompt: "question",
        answerKey: "A",
        referenceAnswer: "A. answer",
        candidateAnswer: "answer",
        candidateTokens: [1, 2],
        maxNewTokens: 2
    )
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-answer-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(
        at: root,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let output = root.appendingPathComponent("answers.json")
    try GemmaRuntime.writeSemanticGPQAAnswers(
        [answer],
        to: output.path
    )
    let attributes = try FileManager.default.attributesOfItem(
        atPath: output.path
    )
    let permissions = try #require(
        attributes[.posixPermissions] as? NSNumber
    ).intValue
    #expect(permissions & 0o777 == 0o600)
    #expect(
        try JSONSerialization.jsonObject(with: Data(contentsOf: output))
            is [String: Any]
    )

    let symlink = root.appendingPathComponent("answers-link.json")
    try FileManager.default.createSymbolicLink(
        at: symlink,
        withDestinationURL: output
    )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.writeSemanticGPQAAnswers(
            [answer],
            to: symlink.path,
            permissions: 0o600
        )
    }
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.writeSemanticGPQAAnswers(
            [answer],
            to: root.appendingPathComponent("public.json").path,
            permissions: 0o644
        )
    }

    let oversized = GemmaRuntime.SemanticGPQAAnswerCase(
        id: "case",
        domain: nil,
        subdomain: nil,
        prompt: "question",
        answerKey: nil,
        referenceAnswer: "answer",
        candidateAnswer: String(
            repeating: "x",
            count: GemmaRuntime.semanticGPQAMaxAnswerDocumentByteCount
        ),
        candidateTokens: [1],
        maxNewTokens: 1
    )
    #expect(throws: MLXFastError.self) {
        _ = try GemmaRuntime.semanticGPQAAnswerDocumentData([oversized])
    }
}

@Test
func legacyGenerateGPQAAnswersUsesAtomicMode0600Writer() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let start = try #require(
        source.range(of: "private static func runGenerateGPQAAnswers(")
    )
    let end = try #require(
        source.range(
            of: "private static func loadLocalTokenizer(",
            range: start.upperBound..<source.endIndex
        )
    )
    let body = source[start.lowerBound..<end.lowerBound]
    #expect(body.contains("PrivateFileWriter.writeAtomically("))
    #expect(!body.contains(".write(to: outputURL"))

    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "legacy-gpqa-private-writer-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(
        at: root,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("answers.json")

    try PrivateFileWriter.writeAtomically(
        Data(#"{"version":1}"#.utf8),
        to: output.path,
        maximumByteCount: 1024
    )
    var attributes = try FileManager.default.attributesOfItem(
        atPath: output.path
    )
    var mode = try #require(
        attributes[.posixPermissions] as? NSNumber
    ).intValue
    #expect(mode & 0o777 == 0o600)

    try PrivateFileWriter.writeAtomically(
        Data(#"{"version":2}"#.utf8),
        to: output.path,
        maximumByteCount: 1024
    )
    #expect(
        try String(contentsOf: output, encoding: .utf8)
            == #"{"version":2}"#
    )
    attributes = try FileManager.default.attributesOfItem(
        atPath: output.path
    )
    mode = try #require(
        attributes[.posixPermissions] as? NSNumber
    ).intValue
    #expect(mode & 0o777 == 0o600)
    let leftovers = try FileManager.default.contentsOfDirectory(
        atPath: root.path
    )
    #expect(leftovers == ["answers.json"])
}

@Test
func mtpSemanticCLIIsExplicitAndCannotUseSerialGeneration() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    #expect(source.contains("case \"mtp-generate-gpqa-answers\":"))
    #expect(source.contains(
        "mlxfast-swift mtp-generate-gpqa-answers --target-source IT_SOURCE "
            + "--weights IT_PATH --assistant PATH --contract PATH "
            + "--tokenizer PATH --gpqa PATH --output PATH "
            + "--require-trained-assistant"
    ))
    let start = try #require(
        source.range(
            of: "private static func runExperimentalTrainedMTPSemanticGPQA("
        )
    )
    let end = try #require(
        source.range(
            of: "// Emits the in-memory payload",
            range: start.upperBound..<source.endIndex
        )
    )
    let body = source[start.lowerBound..<end.lowerBound]
    #expect(body.contains(
        "captureExperimentalTrainedMTPSemanticGPQAAnswers("
    ))
    #expect(body.contains("blockedGoldenPath: gpqaPath"))
    #expect(body.contains(
        "mtp-generate-gpqa-answers requires MLXFAST_PRIVATE_DIR"
    ))
    #expect(body.contains(
        "validateMTPSemanticPrivateOutputPath("
    ))
    #expect(body.contains("deniesWorkerFileWrites: true"))
    #expect(body.contains("requiresWorkerSandbox: true"))
    #expect(body.contains(
        #"print("mtp-generate-gpqa-answers: completed")"#
    ))
    #expect(!body.contains("generateGreedyTokens"))
    #expect(!body.contains("runGenerateGPQAAnswers"))
    #expect(source.contains(
        "sandboxProfile = try augmentRuntimeWorkerSandboxProfile("
    ))
    #expect(source.contains(
        "Trusted-parent private input/output binding"
    ))
    #expect(source.contains(
        "deny file-read* (subpath"
    ))
    #expect(source.contains("if deniesFileWrites"))
    #expect(source.contains("var workerIsolationRules = ["))
    #expect(source.contains(
        #"trimmed.hasPrefix("(allow file-write")"#
    ))
    #expect(source.contains(
        "multiline file-write allow"
    ))
    #expect(source.contains("(deny file-write*)"))
    #expect(source.contains("(deny ipc-posix-shm*)"))
    #expect(source.contains("(deny ipc-posix-sem*)"))
    #expect(source.contains("(deny ipc-sysv*)"))
    #expect(source.contains("(deny user-preference-write)"))
    #expect(source.contains(
        "(ipc-posix-name-prefix \"apple.cfprefs.\")"
    ))
    #expect(source.contains(
        "(deny mach-lookup (global-name-prefix "
            + "\\\"com.apple.pasteboard.\\\"))"
    ))
    #expect(source.contains(
        "(deny mach-lookup (global-name-prefix \\\"com.apple.logd\\\"))"
    ))
    #expect(source.contains(
        "(deny mach-lookup (global-name "
            + "\\\"com.apple.system.logger\\\"))"
    ))
    #expect(source.contains(
        #"(allow file-write* (literal "/dev/null"))"#
    ))

    let harness = try String(
        contentsOfFile:
            "Sources/MLXFastTrustedHarness/GemmaRuntimeMTPSemanticGPQA.swift",
        encoding: .utf8
    )
    #expect(harness.contains("RuntimeWorkerLaunch.trainedMTP("))
    #expect(harness.contains("verificationMode: .exactPair"))
    #expect(harness.contains("permissions: 0o600"))
    #expect(!harness.contains("ExperimentalMTPBlockValidator"))
    #expect(!harness.contains("expectedTokens"))
    #expect(!harness.contains("expectedToken:"))
}
