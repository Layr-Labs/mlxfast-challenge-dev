import CoreFoundation
import CryptoKit
import Darwin
import Foundation

public struct PrivateAnchorCalibrationRequest: Equatable, Sendable {
    public let id: String
    public let contextTokens: [Int]
    public let expectedToken: Int

    public init(id: String, contextTokens: [Int], expectedToken: Int) {
        self.id = id
        self.contextTokens = contextTokens
        self.expectedToken = expectedToken
    }
}

public struct PrivateSequenceCalibrationRequest: Equatable, Sendable {
    public let id: String
    public let promptTokens: [Int]
    public let steps: Int

    public init(id: String, promptTokens: [Int], steps: Int) {
        self.id = id
        self.promptTokens = promptTokens
        self.steps = steps
    }
}

public struct PrivateAnchorCalibrationObservation: Equatable, Sendable {
    public let id: String
    public let generatedToken: Int
    public let expectedTokenRank: Int
    public let expectedTokenTopLogitDelta: Double

    public init(
        id: String,
        generatedToken: Int,
        expectedTokenRank: Int,
        expectedTokenTopLogitDelta: Double
    ) {
        self.id = id
        self.generatedToken = generatedToken
        self.expectedTokenRank = expectedTokenRank
        self.expectedTokenTopLogitDelta = expectedTokenTopLogitDelta
    }
}

public struct PrivateSequenceCalibrationObservation: Equatable, Sendable {
    public let id: String
    public let generatedTokens: [Int]

    public init(id: String, generatedTokens: [Int]) {
        self.id = id
        self.generatedTokens = generatedTokens
    }
}

public struct PrivateReferenceCalibrationObservations: Equatable, Sendable {
    public let anchors: [PrivateAnchorCalibrationObservation]
    public let sequences: [PrivateSequenceCalibrationObservation]

    public init(
        anchors: [PrivateAnchorCalibrationObservation],
        sequences: [PrivateSequenceCalibrationObservation]
    ) {
        self.anchors = anchors
        self.sequences = sequences
    }
}

/// Explicit operator ceiling for anchor tolerance derivation. The calibrator
/// records the observed rank/delta, but refuses to widen past these limits.
public struct PrivateAnchorCalibrationPolicy: Equatable, Sendable {
    public let maximumExpectedRank: Int
    public let maximumTopLogitDelta: Double

    public init(maximumExpectedRank: Int, maximumTopLogitDelta: Double) {
        self.maximumExpectedRank = maximumExpectedRank
        self.maximumTopLogitDelta = maximumTopLogitDelta
    }
}

public struct PrivateGoldenCalibrationResult: Equatable, Sendable {
    public let data: Data
    public let anchorCount: Int
    public let behaviorCount: Int

    public init(data: Data, anchorCount: Int, behaviorCount: Int) {
        self.data = data
        self.anchorCount = anchorCount
        self.behaviorCount = behaviorCount
    }
}

public struct PrivateGPQACalibrationCase: Equatable, Sendable {
    public let id: String
    public let prompt: String

    public init(id: String, prompt: String) {
        self.id = id
        self.prompt = prompt
    }
}

public struct PrivateGPQACalibrationResult: Equatable, Sendable {
    public let data: Data
    public let caseCount: Int

    public init(data: Data, caseCount: Int) {
        self.data = data
        self.caseCount = caseCount
    }
}

public enum PrivateArtifactCalibrator {
    public static func calibrateGolden(
        templateData: Data,
        observations: PrivateReferenceCalibrationObservations,
        anchorPolicy: PrivateAnchorCalibrationPolicy?,
        requiredSteps: Int = MLXFastConstants.correctnessSteps,
        requiredPromptTokens: Int = MLXFastConstants.correctnessPromptTokens
    ) throws -> PrivateGoldenCalibrationResult {
        let template = try decodeGoldenDocument(
            from: templateData,
            requiredSteps: requiredSteps,
            requiredPromptTokens: requiredPromptTokens
        )
        let anchors = template.correctnessGates?.anchorCases ?? []
        let behaviors = template.correctnessGates?.behaviorCases ?? []
        guard !anchors.isEmpty || !behaviors.isEmpty else {
            throw MLXFastError.invalidInput(
                "private golden contains no anchor or behavior cases to calibrate"
            )
        }
        try requireExactCaseOrder(
            expected: anchors.map(\.name),
            actual: observations.anchors.map(\.id),
            field: "anchor"
        )
        try requireExactCaseOrder(
            expected: behaviors.map(\.name),
            actual: observations.sequences.map(\.id),
            field: "behavior"
        )

        let calibratedAnchors: [GoldenAnchorCase]
        if anchors.isEmpty {
            guard anchorPolicy == nil else {
                throw MLXFastError.invalidInput(
                    "anchor tolerance policy was provided for a golden with no anchors"
                )
            }
            calibratedAnchors = []
        } else {
            guard let anchorPolicy else {
                throw MLXFastError.invalidInput(
                    "private golden anchor calibration requires an explicit tolerance policy"
                )
            }
            try validate(anchorPolicy)
            calibratedAnchors = try zip(anchors, observations.anchors).map {
                anchor, observation in
                guard observation.generatedToken >= 0,
                      observation.generatedToken < MLXFastConstants.vocabSize,
                      observation.expectedTokenRank > 0,
                      observation.expectedTokenRank <= anchorPolicy.maximumExpectedRank,
                      observation.expectedTokenTopLogitDelta.isFinite,
                      observation.expectedTokenTopLogitDelta >= 0,
                      observation.expectedTokenTopLogitDelta
                        <= anchorPolicy.maximumTopLogitDelta
                else {
                    throw MLXFastError.invalidInput(
                        "anchor observation exceeds the explicit tolerance policy"
                    )
                }
                return GoldenAnchorCase(
                    name: anchor.name,
                    contextTokens: anchor.contextTokens,
                    expectedToken: anchor.expectedToken,
                    acceptedTokens: [observation.generatedToken],
                    maxExpectedRank: observation.expectedTokenRank,
                    maxTopLogitDelta: observation.expectedTokenTopLogitDelta
                )
            }
        }

        let calibratedBehaviors = try zip(behaviors, observations.sequences).map {
            behavior, observation in
            guard observation.generatedTokens.count == behavior.maxNewTokens else {
                throw MLXFastError.invalidInput(
                    "behavior observation token count does not match max_new_tokens"
                )
            }
            return GoldenBehaviorCase(
                name: behavior.name,
                promptTokens: behavior.promptTokens,
                acceptedTokenSequences: [observation.generatedTokens],
                maxNewTokens: behavior.maxNewTokens,
                semanticPrompt: behavior.semanticPrompt,
                semanticAnswerKey: behavior.semanticAnswerKey,
                semanticReferenceAnswer: behavior.semanticReferenceAnswer,
                semanticDomain: behavior.semanticDomain,
                semanticSubdomain: behavior.semanticSubdomain
            )
        }

        let existingGates = template.correctnessGates
        let calibratedGates = GoldenCorrectnessGates(
            anchors: existingGates?.anchors == nil ? nil : calibratedAnchors,
            freeRun: existingGates?.freeRun,
            behavior: existingGates?.behavior == nil ? nil : calibratedBehaviors
        )
        let calibrated = GoldenDocument(
            version: template.version ?? 1,
            cases: template.cases,
            correctnessGates: calibratedGates,
            benchmark: template.benchmark
        )
        let data = try deterministicJSONData(calibrated)
        _ = try decodeGoldenDocument(
            from: data,
            requiredSteps: requiredSteps,
            requiredPromptTokens: requiredPromptTokens
        )
        return PrivateGoldenCalibrationResult(
            data: data,
            anchorCount: calibratedAnchors.count,
            behaviorCount: calibratedBehaviors.count
        )
    }

    public static func gpqaCases(from templateData: Data) throws
        -> [PrivateGPQACalibrationCase]
    {
        try parseGPQATemplate(templateData).descriptors
    }

    public static func calibrateGPQA(
        templateData: Data,
        observations: [PrivateSequenceCalibrationObservation]
    ) throws -> PrivateGPQACalibrationResult {
        let parsed = try parseGPQATemplate(templateData)
        try requireExactCaseOrder(
            expected: parsed.descriptors.map(\.id),
            actual: observations.map(\.id),
            field: "GPQA"
        )

        var cases = parsed.caseObjects
        for index in cases.indices {
            guard observations[index].generatedTokens.count >= 1 else {
                throw MLXFastError.invalidInput(
                    "GPQA calibration observation must contain a first token"
                )
            }
            let firstToken = observations[index].generatedTokens[0]
            guard firstToken >= 0, firstToken < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput(
                    "GPQA calibration observation contains an invalid token"
                )
            }
            // This is the only field the GPQA calibrator is permitted to
            // change. Reference answers and every other operator-authored
            // field remain untouched.
            cases[index]["accepted_token_sequences"] = [[firstToken]]
        }
        var root = parsed.root
        root["cases"] = cases
        var data = try JSONSerialization.data(
            withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        data.append(0x0a)

        let validated = try parseGPQATemplate(data)
        guard validated.descriptors == parsed.descriptors else {
            throw MLXFastError.invalidInput(
                "GPQA calibration changed case identity or ordering"
            )
        }
        return PrivateGPQACalibrationResult(
            data: data,
            caseCount: cases.count
        )
    }

    private struct ParsedGPQATemplate {
        let root: [String: Any]
        let caseObjects: [[String: Any]]
        let descriptors: [PrivateGPQACalibrationCase]
    }

    private static func parseGPQATemplate(_ data: Data) throws
        -> ParsedGPQATemplate
    {
        let raw = try JSONSerialization.jsonObject(with: data)
        guard let root = raw as? [String: Any] else {
            throw MLXFastError.invalidInput(
                "GPQA calibration template must be a JSON object"
            )
        }
        try rejectUnknownKeys(
            root,
            allowed: ["version", "cases", "status", "needs_reference_output"],
            field: "GPQA calibration template"
        )
        if let version = root["version"] {
            guard integerValue(version) == 1 else {
                throw MLXFastError.invalidInput(
                    "GPQA calibration template version must be 1"
                )
            }
        }
        try validateOptionalString(root["status"], field: "status")
        try validateOptionalBool(
            root["needs_reference_output"],
            field: "needs_reference_output"
        )
        guard let cases = root["cases"] as? [[String: Any]], !cases.isEmpty else {
            throw MLXFastError.invalidInput(
                "GPQA calibration template must contain a non-empty cases array"
            )
        }

        let allowedCaseKeys: Set<String> = [
            "id",
            "prompt",
            "expected_response",
            "answer_key",
            "accepted_token_sequences",
            "accepted_responses",
            "domain",
            "subdomain",
            "needs_reference_output",
        ]
        var ids = Set<String>()
        var descriptors: [PrivateGPQACalibrationCase] = []
        descriptors.reserveCapacity(cases.count)
        for testCase in cases {
            try rejectUnknownKeys(
                testCase,
                allowed: allowedCaseKeys,
                field: "GPQA calibration case"
            )
            guard let id = testCase["id"] as? String,
                  isCanonicalIdentifier(id),
                  ids.insert(id).inserted
            else {
                throw MLXFastError.invalidInput(
                    "GPQA calibration case IDs must be non-empty, canonical, and unique"
                )
            }
            guard let prompt = testCase["prompt"] as? String,
                  !prompt.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "GPQA calibration cases require a non-empty prompt"
                )
            }
            for field in [
                "expected_response",
                "answer_key",
                "domain",
                "subdomain",
            ] {
                try validateOptionalString(testCase[field], field: field)
            }
            try validateOptionalBool(
                testCase["needs_reference_output"],
                field: "needs_reference_output"
            )
            try validateAcceptedResponses(testCase["accepted_responses"])
            try validateTokenSequences(testCase["accepted_token_sequences"])
            descriptors.append(PrivateGPQACalibrationCase(id: id, prompt: prompt))
        }
        return ParsedGPQATemplate(
            root: root,
            caseObjects: cases,
            descriptors: descriptors
        )
    }

    private static func validate(_ policy: PrivateAnchorCalibrationPolicy) throws {
        guard policy.maximumExpectedRank > 0,
              policy.maximumExpectedRank <= MLXFastConstants.correctnessTopLogits,
              policy.maximumTopLogitDelta.isFinite,
              policy.maximumTopLogitDelta >= 0
        else {
            throw MLXFastError.invalidInput(
                "anchor tolerance policy is outside the correctness contract"
            )
        }
    }

    private static func requireExactCaseOrder(
        expected: [String],
        actual: [String],
        field: String
    ) throws {
        guard expected == actual else {
            throw MLXFastError.invalidInput(
                "\(field) calibration case count, IDs, or ordering do not match"
            )
        }
    }

    private static func deterministicJSONData<T: Encodable>(_ value: T) throws
        -> Data
    {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [
            .prettyPrinted,
            .sortedKeys,
            .withoutEscapingSlashes,
        ]
        var data = try encoder.encode(value)
        data.append(0x0a)
        return data
    }

    private static func rejectUnknownKeys(
        _ object: [String: Any],
        allowed: Set<String>,
        field: String
    ) throws {
        let unknown = Set(object.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(field) contains unknown keys"
            )
        }
    }

    private static func validateOptionalString(
        _ value: Any?,
        field: String
    ) throws {
        guard let value, !(value is NSNull) else {
            return
        }
        guard value is String else {
            throw MLXFastError.invalidInput("\(field) must be a string or null")
        }
    }

    private static func validateOptionalBool(
        _ value: Any?,
        field: String
    ) throws {
        guard let value, !(value is NSNull) else {
            return
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID()
        else {
            throw MLXFastError.invalidInput("\(field) must be a boolean or null")
        }
    }

    private static func validateAcceptedResponses(_ value: Any?) throws {
        guard let value, !(value is NSNull) else {
            return
        }
        guard let responses = value as? [Any],
              responses.allSatisfy({ $0 is String })
        else {
            throw MLXFastError.invalidInput(
                "accepted_responses must be an array of strings"
            )
        }
    }

    private static func validateTokenSequences(_ value: Any?) throws {
        guard let value, !(value is NSNull) else {
            return
        }
        guard let sequences = value as? [Any] else {
            throw MLXFastError.invalidInput(
                "accepted_token_sequences must be an array"
            )
        }
        for sequence in sequences {
            guard let rawTokens = sequence as? [Any], !rawTokens.isEmpty else {
                throw MLXFastError.invalidInput(
                    "accepted_token_sequences entries must be non-empty arrays"
                )
            }
            for rawToken in rawTokens {
                guard let token = integerValue(rawToken),
                      token >= 0,
                      token < MLXFastConstants.vocabSize
                else {
                    throw MLXFastError.invalidInput(
                        "accepted_token_sequences contains an invalid token"
                    )
                }
            }
        }
    }

    private static func integerValue(_ value: Any) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else {
            return nil
        }
        let double = number.doubleValue
        guard double.isFinite,
              double.rounded() == double,
              double >= Double(Int.min),
              double <= Double(Int.max)
        else {
            return nil
        }
        return Int(double)
    }

    private static func isCanonicalIdentifier(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty
            && trimmed == value
            && !value.unicodeScalars.contains {
                CharacterSet.controlCharacters.contains($0)
            }
    }
}

public struct PrivateArtifactWriteResult: Equatable, Sendable {
    public let sha256: String
    public let byteCount: Int

    public init(sha256: String, byteCount: Int) {
        self.sha256 = sha256
        self.byteCount = byteCount
    }
}

public enum PrivateArtifactWriter {
    /// Atomically replaces a private artifact with a freshly-created 0600
    /// regular file in the destination directory.
    public static func write(_ data: Data, to path: String) throws
        -> PrivateArtifactWriteResult
    {
        do {
            return try writeImpl(data, to: path)
        } catch {
            throw MLXFastError.invalidInput(
                "private artifact atomic write failed"
            )
        }
    }

    private static func writeImpl(_ data: Data, to path: String) throws
        -> PrivateArtifactWriteResult
    {
        guard !data.isEmpty, !path.isEmpty else {
            throw MLXFastError.invalidInput("private artifact data/path is empty")
        }
        let outputURL = URL(fileURLWithPath: path).standardizedFileURL
        let directoryURL = outputURL.deletingLastPathComponent()
        let fileName = outputURL.lastPathComponent
        guard !fileName.isEmpty, fileName != ".", fileName != ".." else {
            throw MLXFastError.invalidInput("private artifact path is invalid")
        }
        try FileManager.default.createDirectory(
            at: directoryURL,
            withIntermediateDirectories: true
        )

        let directoryFD = Darwin.open(
            directoryURL.path,
            O_RDONLY | O_DIRECTORY | O_CLOEXEC
        )
        guard directoryFD >= 0 else {
            throw MLXFastError.invalidInput(
                "private artifact directory could not be opened"
            )
        }
        defer {
            _ = Darwin.close(directoryFD)
        }

        let temporaryName = ".\(fileName).private-\(UUID().uuidString).tmp"
        var temporaryFD = temporaryName.withCString {
            Darwin.openat(
                directoryFD,
                $0,
                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                mode_t(0o600)
            )
        }
        guard temporaryFD >= 0 else {
            throw MLXFastError.invalidInput(
                "private artifact temporary file could not be created"
            )
        }
        var shouldRemoveTemporary = true
        defer {
            if temporaryFD >= 0 {
                _ = Darwin.close(temporaryFD)
            }
            if shouldRemoveTemporary {
                temporaryName.withCString {
                    _ = Darwin.unlinkat(directoryFD, $0, 0)
                }
            }
        }

        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw MLXFastError.invalidInput(
                    "private artifact data buffer is unavailable"
                )
            }
            var offset = 0
            while offset < rawBuffer.count {
                let count = Darwin.write(
                    temporaryFD,
                    baseAddress.advanced(by: offset),
                    rawBuffer.count - offset
                )
                if count < 0, errno == EINTR {
                    continue
                }
                guard count > 0 else {
                    throw MLXFastError.invalidInput(
                        "private artifact temporary write failed"
                    )
                }
                offset += count
            }
        }
        guard Darwin.fchmod(temporaryFD, mode_t(0o600)) == 0 else {
            throw MLXFastError.invalidInput(
                "private artifact permissions could not be set"
            )
        }
        while Darwin.fsync(temporaryFD) != 0 {
            guard errno == EINTR else {
                throw MLXFastError.invalidInput(
                    "private artifact temporary file could not be synced"
                )
            }
        }
        guard Darwin.close(temporaryFD) == 0 else {
            temporaryFD = -1
            throw MLXFastError.invalidInput(
                "private artifact temporary file could not be closed"
            )
        }
        temporaryFD = -1

        let renameResult = temporaryName.withCString { temporaryPath in
            fileName.withCString { outputPath in
                Darwin.renameat(
                    directoryFD,
                    temporaryPath,
                    directoryFD,
                    outputPath
                )
            }
        }
        guard renameResult == 0 else {
            throw MLXFastError.invalidInput(
                "private artifact atomic replacement failed"
            )
        }
        shouldRemoveTemporary = false
        _ = Darwin.fsync(directoryFD)

        var fileStatus = stat()
        guard Darwin.lstat(outputURL.path, &fileStatus) == 0,
              (fileStatus.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG),
              (fileStatus.st_mode & mode_t(0o777)) == mode_t(0o600)
        else {
            throw MLXFastError.invalidInput(
                "private artifact post-write verification failed"
            )
        }

        let digest = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
        return PrivateArtifactWriteResult(
            sha256: digest,
            byteCount: data.count
        )
    }
}

public enum PrivateArtifactLogSummary {
    public static func goldenSuccess(
        anchorCount: Int,
        behaviorCount: Int,
        sha256: String
    ) -> String {
        "private-golden-calibration success anchors=\(anchorCount) "
            + "behaviors=\(behaviorCount) sha256=\(sha256)"
    }

    public static func gpqaSuccess(
        caseCount: Int,
        calibratedSHA256: String,
        answersSHA256: String
    ) -> String {
        "private-gpqa-generation success cases=\(caseCount) "
            + "calibrated_sha256=\(calibratedSHA256) "
            + "answers_sha256=\(answersSHA256)"
    }

    public static let goldenFailure =
        "private-golden-calibration failure"
    public static let gpqaFailure =
        "private-gpqa-generation failure"
}
