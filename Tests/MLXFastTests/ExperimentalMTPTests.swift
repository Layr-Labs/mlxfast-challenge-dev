import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastModel
import Testing

@Test
func experimentalMTPSerialFallbackChainsTokensAndOffsetsExactly() throws {
    var inputTokens: [Int] = []
    var offsets: [Int] = []
    let tokens = try Gemma4TargetBlockGeneration.generateSerialBlock(
        previousToken: 10,
        maxBlockSize: 4,
        positionOffset: 512
    ) { inputToken, offset in
        inputTokens.append(inputToken)
        offsets.append(offset)
        return inputToken + 1
    }

    #expect(tokens == [11, 12, 13, 14])
    #expect(inputTokens == [10, 11, 12, 13])
    #expect(offsets == [512, 513, 514, 515])
}

@Test
func experimentalMTPSerialFallbackRejectsInvalidBounds() {
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: 1,
            maxBlockSize: 0,
            positionOffset: 0
        ) { token, _ in token }
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: 1,
            maxBlockSize: MLXFastConstants.experimentalMTPMaxBlockSize + 1,
            positionOffset: 0
        ) { token, _ in token }
    }
    #expect(throws: MLXFastError.self) {
        _ = try Gemma4TargetBlockGeneration.generateSerialBlock(
            previousToken: MLXFastConstants.vocabSize,
            maxBlockSize: 1,
            positionOffset: 0
        ) { token, _ in token }
    }
}

@Test
func shippedCheckpointReportsAssistantUnavailable() {
    let availability = Gemma4MTPAssistantAvailability.shippedCheckpoint
    #expect(!availability.isAvailable)
    #expect(availability.reason.contains("unavailable/incompatible"))
    #expect(availability.reason.contains("base checkpoint"))
    #expect(availability.reason.contains("IT models"))
    #expect(throws: MLXFastError.self) {
        try availability.requireAvailable()
    }
}

@Test
func decodeBlockRequestCodableShapeHasNoOracle() throws {
    let request = RuntimeWorkerRequest(
        id: 7,
        kind: "decode_block",
        token: 42,
        maxBlockSize: 4
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "kind", "token", "max_block_size"]))
    #expect(object["expected_tokens"] == nil)
    #expect(object["expected_seed_token"] == nil)
    #expect(object["prompt_tokens"] == nil)
    #expect(data.count < BufferedFileLineReader.defaultMaximumLineByteCount)

    let decoded = try JSONDecoder().decode(RuntimeWorkerRequest.self, from: data)
    #expect(decoded.id == 7)
    #expect(decoded.kind == "decode_block")
    #expect(decoded.token == 42)
    #expect(decoded.maxBlockSize == 4)

    let requestWithOracleField = Data(
        """
        {"id":7,"kind":"decode_block","token":42,"max_block_size":4,"expected_tokens":[1,2]}
        """.utf8
    )
    #expect(throws: DecodingError.self) {
        _ = try JSONDecoder().decode(
            RuntimeWorkerRequest.self,
            from: requestWithOracleField
        )
    }
}

@Test
func legacyDecodeStepRequestShapeRemainsUnchanged() throws {
    let request = RuntimeWorkerRequest(
        id: 8,
        kind: "decode_step",
        token: 17
    )
    let data = try JSONEncoder().encode(request)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "kind", "token"]))
    #expect(object["max_block_size"] == nil)
    let decoded = try JSONDecoder().decode(RuntimeWorkerRequest.self, from: data)
    #expect(decoded.maxBlockSize == nil)
}

@Test
func decodeBlockResponseCodableShapeIsBounded() throws {
    let response = RuntimeWorkerResponse(
        id: 9,
        nonce: "nonce",
        ok: true,
        tokens: [1, 2, 3, 4]
    )
    let data = try JSONEncoder().encode(response)
    let object = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any]
    )

    #expect(Set(object.keys) == Set(["id", "nonce", "ok", "tokens"]))
    #expect(data.count < BufferedFileLineReader.defaultMaximumLineByteCount)
}

@Test
func decodeBlockRequestValidationFailsClosed() throws {
    let valid = RuntimeWorkerRequest(
        id: 1,
        kind: "decode_block",
        token: 12,
        maxBlockSize: 4
    )
    #expect(
        try validateExperimentalDecodeBlockRequest(
            valid,
            decodedTokenCount: 124
        ) == ExperimentalDecodeBlockRequest(previousToken: 12, maxBlockSize: 4)
    )

    let invalidRequests = [
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 0
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 5
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            promptTokens: [1],
            token: 12,
            maxBlockSize: 1
        ),
        RuntimeWorkerRequest(
            id: 0,
            kind: "decode_block",
            token: 12,
            maxBlockSize: 1
        ),
        RuntimeWorkerRequest(
            id: 1,
            kind: "decode_block",
            token: MLXFastConstants.vocabSize,
            maxBlockSize: 1
        ),
    ]
    for request in invalidRequests {
        #expect(throws: MLXFastError.self) {
            _ = try validateExperimentalDecodeBlockRequest(
                request,
                decodedTokenCount: 0
            )
        }
    }
    #expect(throws: MLXFastError.self) {
        _ = try validateExperimentalDecodeBlockRequest(
            RuntimeWorkerRequest(
                id: 1,
                kind: "decode_block",
                token: 12,
                maxBlockSize: 2
            ),
            decodedTokenCount: 127
        )
    }
}

@Test
func trustedMTPBlockValidatorAcceptsOnlyExactOrderedPrefix() throws {
    var validator = try ExperimentalMTPBlockValidator(
        expectedTokens: [10, 11, 12, 13, 14],
        totalTokenCount: 5,
        protocolMaxBlockSize: 4
    )
    try validator.accept([10, 11], requestedMaxBlockSize: 4)
    try validator.accept([12, 13, 14], requestedMaxBlockSize: 3)
    try validator.requireComplete()

    #expect(validator.committedTokenCount == 5)
    #expect(validator.remainingTokenCount == 0)
}

@Test
func trustedMTPBlockValidatorRejectsEmptyAndOversizedBlocks() throws {
    var emptyValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    #expect(throws: MLXFastError.self) {
        try emptyValidator.accept([], requestedMaxBlockSize: 2)
    }

    var oversizedValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    #expect(throws: MLXFastError.self) {
        try oversizedValidator.accept([1, 2, 3], requestedMaxBlockSize: 2)
    }
}

@Test
func trustedMTPBlockValidatorRejectsOverrunAndMismatch() throws {
    var overrunValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    try overrunValidator.accept([1, 2], requestedMaxBlockSize: 2)
    #expect(throws: MLXFastError.self) {
        try overrunValidator.accept([3, 4], requestedMaxBlockSize: 2)
    }

    var mismatchValidator = try ExperimentalMTPBlockValidator(
        expectedTokens: [1, 2, 3],
        totalTokenCount: 3
    )
    do {
        try mismatchValidator.accept([1, 99], requestedMaxBlockSize: 2)
        Issue.record("expected token mismatch")
    } catch let mismatch as BenchmarkTokenMismatchError {
        #expect(mismatch.step == 1)
        #expect(mismatch.description == "experimental MTP decode token mismatch at step 1")
    }
    #expect(mismatchValidator.committedTokenCount == 0)
}

@Test
func experimentalMTPUsesFixedFrozenLengthAndBoundedBlocks() throws {
    #expect(MLXFastConstants.experimentalMTPMaxBlockSize == 4)
    #expect(MLXFastConstants.experimentalMTPMaxTotalTokens == 128)
    #expect(
        MLXFastConstants.experimentalMTPMaxTotalTokens
            == MLXFastConstants.benchmarkDecodeSteps
    )
    try GemmaRuntime.validateExperimentalMTPProbeOptions(
        ExperimentalMTPProbeOptions(
            weightsPath: "weights",
            goldenPath: "public.json",
            maxBlockSize: 4
        )
    )
    #expect(throws: MLXFastError.self) {
        try GemmaRuntime.validateExperimentalMTPProbeOptions(
            ExperimentalMTPProbeOptions(
                weightsPath: "weights",
                goldenPath: "public.json",
                maxBlockSize: 5
            )
        )
    }
}

@Test
func experimentalMTPCLIIsExplicitAndDoesNotAlterBenchmarkCommand() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    #expect(source.contains("case \"mtp-probe\":"))
    #expect(
        source.contains(
            "mlxfast-swift mtp-probe --weights PATH --golden PATH [--block-size N]"
        )
    )
    #expect(source.contains("mtp-probe requires --weights PATH"))
    #expect(source.contains("mtp-probe requires --golden PATH"))
    #expect(!source.contains("MLXFAST_MTP"))

    let benchmarkStart = try #require(
        source.range(of: "private static func runBenchmark(")
    )
    let probeStart = try #require(
        source.range(
            of: "private static func runExperimentalMTPProbe(",
            range: benchmarkStart.upperBound..<source.endIndex
        )
    )
    let benchmarkBody = source[benchmarkStart.lowerBound..<probeStart.lowerBound]
    #expect(!benchmarkBody.contains("experimentalMTPProbe"))
    #expect(!benchmarkBody.contains("decodeBlock"))
    #expect(!benchmarkBody.contains("mtp-probe"))
}

@Test
func experimentalMTPReportCannotRepresentOfficialScore() throws {
    let report = ExperimentalMTPProbeReport(
        experimental: true,
        protocolName: "decode_block_v1",
        generator: "serial_target_fallback",
        usesTrainedDrafter: false,
        assistantUnavailableReason: "unavailable",
        oracleSource: "first_golden_case",
        seedTokenCount: 512,
        decodeTokenCount: 128,
        maxBlockSize: 4,
        blockRequestCount: 32,
        elapsedSeconds: 1,
        parentMeasuredSecondsPerToken: 1.0 / 128.0,
        allTokensMatched: true,
        officialScoreProduced: false
    )
    let object = try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(report))
            as? [String: Any]
    )
    #expect(object["score"] == nil)
    #expect(object["speedup"] == nil)
    #expect(object["official_score_produced"] as? Bool == false)
}
