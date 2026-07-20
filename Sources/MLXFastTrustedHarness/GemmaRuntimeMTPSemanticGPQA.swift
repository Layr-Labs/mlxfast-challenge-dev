import Foundation
import MLXFastCore
import Tokenizers

/// Explicit options for the untimed semantic GPQA backstop on the trained-MTP
/// track. This path is intentionally separate from `ExperimentalTrainedMTPOptions`:
/// it has no serial token oracle and cannot produce timing or score fields.
public struct ExperimentalTrainedMTPSemanticGPQAOptions: Equatable {
    public let sourceTargetPath: String
    public let targetWeightsPath: String
    public let assistantPath: String
    public let contractPath: String
    public let tokenizerPath: String
    public let referenceCasesPath: String
    public let privateDirectoryPath: String
    public let outputPath: String
    public let maxBlockSize: Int
    public let caseCount: Int
    public let maxNewTokens: Int
    public let requireTrainedAssistant: Bool

    public init(
        sourceTargetPath: String,
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String,
        tokenizerPath: String,
        referenceCasesPath: String,
        privateDirectoryPath: String,
        outputPath: String,
        maxBlockSize: Int = MLXFastConstants.experimentalMTPMaxBlockSize,
        caseCount: Int = MLXFastConstants.semanticGPQACaseCount,
        maxNewTokens: Int = MLXFastConstants.semanticGPQAMaxNewTokens,
        requireTrainedAssistant: Bool
    ) {
        self.sourceTargetPath = sourceTargetPath
        self.targetWeightsPath = targetWeightsPath
        self.assistantPath = assistantPath
        self.contractPath = contractPath
        self.tokenizerPath = tokenizerPath
        self.referenceCasesPath = referenceCasesPath
        self.privateDirectoryPath = privateDirectoryPath
        self.outputPath = outputPath
        self.maxBlockSize = maxBlockSize
        self.caseCount = caseCount
        self.maxNewTokens = maxNewTokens
        self.requireTrainedAssistant = requireTrainedAssistant
    }
}

/// Non-sensitive completion summary. The CLI deliberately emits only a fixed
/// success line rather than serializing this value or any per-case data.
public struct ExperimentalTrainedMTPSemanticGPQACaptureSummary: Equatable, Sendable {
    public let caseCount: Int
    public let generatedTokenCount: Int
}

struct MTPSemanticGPQACaptureCase: Equatable {
    let id: String
    let domain: String?
    let subdomain: String?
    let prompt: String
    let answerKey: String?
    let referenceAnswer: String
    let promptTokens: [Int]
}

protocol TrainedMTPSemanticGPQAWorker: AnyObject {
    func beginTrainedMTPDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse
    func trainedMTPDecodeBlock(
        previousToken: Int,
        maxBlockSize: Int
    ) throws -> RuntimeWorkerResponse
    @discardableResult
    func close() -> Bool
}

extension RuntimeWorkerClient: TrainedMTPSemanticGPQAWorker {}

extension GemmaRuntime {
    static let mtpSemanticGPQAMaxReferenceFileByteCount = 1 * 1024 * 1024
    static let mtpSemanticGPQAMaxCaseCount = 32

    /// Capture a semantic-only answer document through the real trained-MTP
    /// worker launch. Each case gets a fresh worker/session because the strict
    /// protocol permits exactly one `mtp_decode_begin` per process. This costs
    /// one model load per case, but prevents prompt/KV/session state from
    /// crossing cases without adding a reset protocol to submitted code.
    public static func captureExperimentalTrainedMTPSemanticGPQAAnswers(
        _ options: ExperimentalTrainedMTPSemanticGPQAOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) throws -> ExperimentalTrainedMTPSemanticGPQACaptureSummary {
        try validateExperimentalTrainedMTPSemanticGPQAOptions(options)
        try validateMTPSemanticWorkerSandbox(
            workerOptions,
            referenceCasesPath: options.referenceCasesPath,
            privateDirectoryPath: options.privateDirectoryPath
        )
        try requireFile(
            options.referenceCasesPath,
            description: "MTP semantic GPQA reference cases"
        )
        try requireFile(
            URL(fileURLWithPath: options.tokenizerPath)
                .appendingPathComponent("tokenizer.json").path,
            description: "MTP semantic GPQA tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: options.tokenizerPath)
                .appendingPathComponent("tokenizer_config.json").path,
            description: "MTP semantic GPQA tokenizer_config.json"
        )

        // Parent-side provenance validation remains identical to mtp-benchmark.
        // Each fresh worker independently validates target/assistant/contract
        // again before and after model load.
        _ = try validateExperimentalMTPSourceTarget(
            sourceTargetPath: options.sourceTargetPath,
            contractPath: options.contractPath
        )
        _ = try validateExperimentalMTPArtifacts(
            targetWeightsPath: options.targetWeightsPath,
            assistantPath: options.assistantPath,
            contractPath: options.contractPath
        )

        let referenceDocument = try loadMTPSemanticGPQAReferenceDocument(
            from: options.referenceCasesPath
        )
        let tokenizer = try loadLocalTokenizer(at: options.tokenizerPath)
        let cases = try makeMTPSemanticGPQACaptureCases(
            referenceDocument,
            tokenizer: tokenizer,
            caseCount: options.caseCount
        )

        return try captureExperimentalTrainedMTPSemanticGPQACases(
            cases,
            targetWeightsPath: options.targetWeightsPath,
            assistantPath: options.assistantPath,
            contractPath: options.contractPath,
            maxBlockSize: options.maxBlockSize,
            maxNewTokens: options.maxNewTokens,
            workerOptions: workerOptions,
            makeWorker: { workerOptions, weightsPath, launch in
                try RuntimeWorkerClient(
                    options: workerOptions,
                    weightsPath: weightsPath,
                    launch: launch
                )
            },
            decode: { tokens in
                tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            },
            write: { answers in
                try writeSemanticGPQAAnswers(
                    answers,
                    to: options.outputPath,
                    permissions: 0o600
                )
            }
        )
    }

    static func validateExperimentalTrainedMTPSemanticGPQAOptions(
        _ options: ExperimentalTrainedMTPSemanticGPQAOptions
    ) throws {
        guard !options.sourceTargetPath.isEmpty,
              !options.targetWeightsPath.isEmpty,
              !options.assistantPath.isEmpty,
              !options.contractPath.isEmpty,
              !options.tokenizerPath.isEmpty,
              !options.referenceCasesPath.isEmpty,
              !options.privateDirectoryPath.isEmpty,
              !options.outputPath.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA capture requires target source, transformed "
                    + "target, assistant, contract, tokenizer, reference cases, "
                    + "private directory, and output paths"
            )
        }
        try validateMTPSemanticPrivateOutputPath(
            options.outputPath,
            privateDirectoryPath: options.privateDirectoryPath
        )
        let sourceTargetURL = URL(fileURLWithPath: options.sourceTargetPath)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let tokenizerURL = URL(fileURLWithPath: options.tokenizerPath)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        guard sourceTargetURL == tokenizerURL else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA tokenizer must be the pinned target source"
            )
        }
        guard options.requireTrainedAssistant else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA capture requires --require-trained-assistant"
            )
        }
        guard options.maxBlockSize >= 2,
              options.maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA --block-size must be in "
                    + "2...\(MLXFastConstants.experimentalMTPMaxBlockSize)"
            )
        }
        guard options.caseCount > 0,
              options.caseCount <= mtpSemanticGPQAMaxCaseCount
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA --case-count must be in "
                    + "1...\(mtpSemanticGPQAMaxCaseCount)"
            )
        }
        guard options.maxNewTokens > 0,
              options.maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA --max-new-tokens must be in "
                    + "1...\(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }
    }

    public static func validateMTPSemanticPrivateOutputPath(
        _ outputPath: String,
        privateDirectoryPath: String
    ) throws {
        guard !privateDirectoryPath.isEmpty, !outputPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA requires MLXFAST_PRIVATE_DIR and a private output"
            )
        }
        let currentDirectory = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
        func canonicalPath(_ path: String) throws -> String {
            let absoluteURL = path.hasPrefix("/")
                ? URL(fileURLWithPath: path)
                : URL(fileURLWithPath: path, relativeTo: currentDirectory)
            let standardizedURL = absoluteURL.standardizedFileURL
            var existingAncestor = standardizedURL
            var missingComponents: [String] = []
            while !FileManager.default.fileExists(
                atPath: existingAncestor.path
            ) {
                let values = try? existingAncestor.resourceValues(
                    forKeys: [.isSymbolicLinkKey]
                )
                if values?.isSymbolicLink == true {
                    throw MLXFastError.invalidInput(
                        "MTP semantic GPQA private path contains a dangling symlink"
                    )
                }
                guard existingAncestor.path != "/" else {
                    break
                }
                missingComponents.insert(
                    existingAncestor.lastPathComponent,
                    at: 0
                )
                existingAncestor.deleteLastPathComponent()
            }
            var resolved = existingAncestor.resolvingSymlinksInPath()
            for component in missingComponents {
                resolved.appendPathComponent(component)
            }
            return resolved.standardizedFileURL.path
        }
        let privatePath = try canonicalPath(privateDirectoryPath)
        let candidatePath = try canonicalPath(outputPath)
        guard privatePath != "/",
              candidatePath != privatePath,
              candidatePath.hasPrefix(privatePath + "/")
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA output must be a strict descendant of "
                    + "MLXFAST_PRIVATE_DIR"
            )
        }
    }

    static func validateMTPSemanticWorkerSandbox(
        _ workerOptions: RuntimeWorkerOptions,
        referenceCasesPath: String,
        privateDirectoryPath: String
    ) throws {
        guard let profilePath = workerOptions.sandboxProfilePath,
              !profilePath.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA requires a runtime worker sandbox profile"
            )
        }
        let source = try String(
            contentsOfFile: profilePath,
            encoding: .utf8
        )
        let currentDirectory = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
        func canonicalPath(_ path: String) -> String {
            let url = path.hasPrefix("/")
                ? URL(fileURLWithPath: path)
                : URL(fileURLWithPath: path, relativeTo: currentDirectory)
            return url.standardizedFileURL.resolvingSymlinksInPath().path
        }
        func seatbeltEscaped(_ path: String) -> String {
            path
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
        }
        let referenceRule =
            "(deny file-read* (literal \""
            + seatbeltEscaped(canonicalPath(referenceCasesPath))
            + "\"))"
        let privateRule =
            "(deny file-read* (subpath \""
            + seatbeltEscaped(canonicalPath(privateDirectoryPath))
            + "\"))"
        let devNullRule =
            "(allow file-write* (literal \"/dev/null\"))"
        let unexpectedWriteAllow = source.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ).map {
            $0.trimmingCharacters(in: .whitespaces)
        }.first {
            $0.hasPrefix("(allow file-write") && $0 != devNullRule
        }
        guard source.contains(referenceRule),
              source.contains(privateRule),
              source.contains("(deny ipc-posix-shm*)"),
              source.contains("(deny ipc-posix-sem*)"),
              source.contains("(deny ipc-sysv*)"),
              source.contains("(allow ipc-posix-shm-read*"),
              source.contains("(ipc-posix-name-prefix \"apple.cfprefs.\")"),
              source.contains("(deny user-preference-write)"),
              source.contains(
                  "(deny mach-lookup (global-name-prefix "
                      + "\"com.apple.pasteboard.\"))"
              ),
              source.contains(
                  "(deny mach-lookup (global-name-prefix "
                      + "\"com.apple.logd\"))"
              ),
              source.contains(
                  "(deny mach-lookup (global-name "
                      + "\"com.apple.system.logger\"))"
              ),
              source.contains("(deny file-write*)"),
              source.contains(devNullRule),
              unexpectedWriteAllow == nil
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA worker sandbox does not deny private "
                    + "fixture/output reads, named IPC, and cross-case "
                    + "filesystem writes"
            )
        }
    }

    static func loadMTPSemanticGPQAReferenceDocument(
        from path: String
    ) throws -> GPQAReferenceDocument {
        let url = URL(fileURLWithPath: path)
        let values = try url.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isRegularFile == true,
              values.isSymbolicLink != true,
              let byteCount = values.fileSize,
              byteCount > 0,
              byteCount <= mtpSemanticGPQAMaxReferenceFileByteCount
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA reference cases must be a nonempty regular "
                    + "file no larger than "
                    + "\(mtpSemanticGPQAMaxReferenceFileByteCount) bytes"
            )
        }
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count == byteCount else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA reference cases changed while being read"
            )
        }
        let document: GPQAReferenceDocument
        do {
            document = try JSONDecoder().decode(
                GPQAReferenceDocument.self,
                from: data
            )
        } catch {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA reference cases are malformed"
            )
        }
        guard !document.cases.isEmpty,
              document.cases.count <= mtpSemanticGPQAMaxCaseCount
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA reference case count is outside the trusted limit"
            )
        }

        var identifiers = Set<String>()
        for (index, testCase) in document.cases.enumerated() {
            guard let identifier = trimmedNonEmpty(testCase.id),
                  identifiers.insert(identifier).inserted,
                  trimmedNonEmpty(testCase.prompt) != nil,
                  trimmedNonEmpty(testCase.semanticReferenceAnswer) != nil
            else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA reference case \(index + 1) is invalid"
                )
            }
        }
        return document
    }

    static func makeMTPSemanticGPQACaptureCases(
        _ document: GPQAReferenceDocument,
        tokenizer: any Tokenizer,
        caseCount: Int
    ) throws -> [MTPSemanticGPQACaptureCase] {
        var cases: [MTPSemanticGPQACaptureCase] = []
        cases.reserveCapacity(caseCount)
        for (index, testCase) in document.cases.enumerated() {
            guard cases.count < caseCount else {
                break
            }
            let promptTokens = tokenizer.encode(
                text: testCase.prompt,
                addSpecialTokens: false
            )
            guard !promptTokens.isEmpty,
                  promptTokens.allSatisfy({
                      $0 >= 0 && $0 < MLXFastConstants.vocabSize
                  })
            else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA case \(index + 1) tokenized to invalid input"
                )
            }
            guard promptTokens.count
                    <= MLXFastConstants.correctnessMaxBehaviorPromptTokens
            else {
                continue
            }
            guard let identifier = trimmedNonEmpty(testCase.id),
                  let referenceAnswer = trimmedNonEmpty(
                      testCase.semanticReferenceAnswer
                  )
            else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA case \(index + 1) is missing trusted metadata"
                )
            }
            cases.append(MTPSemanticGPQACaptureCase(
                id: identifier,
                domain: trimmedNonEmpty(testCase.domain),
                subdomain: trimmedNonEmpty(testCase.subdomain),
                prompt: testCase.prompt,
                answerKey: trimmedNonEmpty(testCase.answerKey),
                referenceAnswer: referenceAnswer,
                promptTokens: promptTokens
            ))
        }
        guard cases.count == caseCount else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA reference does not contain enough "
                    + "token-budget-valid cases"
            )
        }
        return cases
    }

    static func captureExperimentalTrainedMTPSemanticGPQACases(
        _ cases: [MTPSemanticGPQACaptureCase],
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String,
        maxBlockSize: Int,
        maxNewTokens: Int,
        workerOptions: RuntimeWorkerOptions,
        makeWorker: (
            RuntimeWorkerOptions,
            String,
            RuntimeWorkerLaunch
        ) throws -> any TrainedMTPSemanticGPQAWorker,
        decode: ([Int]) throws -> String,
        write: ([SemanticGPQAAnswerCase]) throws -> Void
    ) throws -> ExperimentalTrainedMTPSemanticGPQACaptureSummary {
        guard !cases.isEmpty,
              cases.count <= mtpSemanticGPQAMaxCaseCount,
              maxBlockSize >= 2,
              maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize,
              maxNewTokens > 0,
              maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA capture bounds are invalid"
            )
        }

        let launch = RuntimeWorkerLaunch.trainedMTP(
            assistantPath: assistantPath,
            contractPath: contractPath,
            verificationMode: .exactPair
        )
        var answers: [SemanticGPQAAnswerCase] = []
        answers.reserveCapacity(cases.count)
        var generatedTokenCount = 0

        for (index, testCase) in cases.enumerated() {
            guard !testCase.promptTokens.isEmpty,
                  testCase.promptTokens.count
                    <= MLXFastConstants.correctnessMaxBehaviorPromptTokens,
                  testCase.promptTokens.allSatisfy({
                      $0 >= 0 && $0 < MLXFastConstants.vocabSize
                  })
            else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA case \(index + 1) has invalid prompt tokens"
                )
            }

            let worker = try makeWorker(
                workerOptions,
                targetWeightsPath,
                launch
            )
            let generatedTokens: [Int]
            do {
                generatedTokens = try collectExperimentalTrainedMTPSemanticTokens(
                    promptTokens: testCase.promptTokens,
                    maxBlockSize: maxBlockSize,
                    maxNewTokens: maxNewTokens,
                    worker: worker
                )
            } catch {
                _ = worker.close()
                throw error
            }
            // Decode/reference handling and every disk write happen only after
            // the process running editable model code has been torn down.
            guard worker.close() else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA worker did not shut down cleanly"
                )
            }

            let candidateAnswer = try decode(generatedTokens)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            answers.append(SemanticGPQAAnswerCase(
                id: testCase.id,
                domain: testCase.domain,
                subdomain: testCase.subdomain,
                prompt: testCase.prompt,
                answerKey: testCase.answerKey,
                referenceAnswer: testCase.referenceAnswer,
                candidateAnswer: candidateAnswer,
                candidateTokens: generatedTokens,
                maxNewTokens: maxNewTokens
            ))
            generatedTokenCount += generatedTokens.count
        }

        // No worker survives this point. A failure above writes no partial
        // hidden answer document.
        try write(answers)
        return ExperimentalTrainedMTPSemanticGPQACaptureSummary(
            caseCount: answers.count,
            generatedTokenCount: generatedTokenCount
        )
    }

    static func collectExperimentalTrainedMTPSemanticTokens(
        promptTokens: [Int],
        maxBlockSize: Int,
        maxNewTokens: Int,
        worker: any TrainedMTPSemanticGPQAWorker
    ) throws -> [Int] {
        let begin = try worker.beginTrainedMTPDecode(seedTokens: promptTokens)
        guard begin.id == 1,
              begin.ok,
              begin.error == nil,
              let nonce = begin.nonce,
              !nonce.isEmpty,
              let seedToken = begin.seedToken,
              seedToken >= 0,
              seedToken < MLXFastConstants.vocabSize,
              begin.tokens == nil,
              mtpSemanticResponseHasNoUnrelatedPayload(begin, allowing: .seed)
        else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA begin response is malformed"
            )
        }

        var generated = [seedToken]
        generated.reserveCapacity(maxNewTokens)
        var previousCommittedToken = seedToken
        var expectedResponseID = 2
        while generated.count < maxNewTokens {
            let remaining = maxNewTokens - generated.count
            let requestedMaxBlockSize = min(maxBlockSize, remaining)
            let response = try worker.trainedMTPDecodeBlock(
                previousToken: previousCommittedToken,
                maxBlockSize: requestedMaxBlockSize
            )
            guard response.id == expectedResponseID,
                  response.ok,
                  response.error == nil,
                  response.nonce == nonce,
                  response.seedToken == nil,
                  let tokens = response.tokens,
                  !tokens.isEmpty,
                  tokens.count <= requestedMaxBlockSize,
                  tokens.count <= remaining,
                  tokens.allSatisfy({
                      $0 >= 0 && $0 < MLXFastConstants.vocabSize
                  }),
                  mtpSemanticResponseHasNoUnrelatedPayload(
                      response,
                      allowing: .tokens
                  )
            else {
                throw MLXFastError.invalidInput(
                    "MTP semantic GPQA block response is malformed or oversized"
                )
            }
            generated.append(contentsOf: tokens)
            previousCommittedToken = tokens[tokens.count - 1]
            expectedResponseID += 1
        }
        guard generated.count == maxNewTokens else {
            throw MLXFastError.invalidInput(
                "MTP semantic GPQA capture did not reach its trusted token budget"
            )
        }
        return generated
    }

    private enum MTPSemanticAllowedResponsePayload {
        case seed
        case tokens
    }

    private static func mtpSemanticResponseHasNoUnrelatedPayload(
        _ response: RuntimeWorkerResponse,
        allowing payload: MTPSemanticAllowedResponsePayload
    ) -> Bool {
        let expectedPayloadIsPresent: Bool
        switch payload {
        case .seed:
            expectedPayloadIsPresent =
                response.seedToken != nil && response.tokens == nil
        case .tokens:
            expectedPayloadIsPresent =
                response.seedToken == nil && response.tokens != nil
        }
        return expectedPayloadIsPresent
            && response.token == nil
            && response.topLogits == nil
            && response.expectedTokenLogit == nil
            && response.expectedTokenRank == nil
            && response.topLogitMargin == nil
            && response.expertStats == nil
            && response.peakRamGB == nil
            && response.mlxActiveMemoryBytes == nil
            && response.mlxCacheMemoryBytes == nil
            && response.mlxPeakMemoryBytes == nil
            && response.targetVerificationMode == nil
            && response.exactPairSegmentCount == nil
            && response.exactPairRollbackRowCount == nil
            && response.serialVerificationRowCount == nil
    }

}
