import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastModel
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

private struct ExactPairNumericTolerance: Sendable {
    let relative: Double
    let absolute: Double
}

// Provisional heuristic regression envelopes: MLX reports unit-scale
// epsilons of 9.77e-4 for Float16 and 7.8125e-3 for BF16. Existing tests in
// this repository use rtol=1e-5/atol=1e-5 for Gemma 4 logits and
// rtol=1e-2/atol=2e-3 for BF16 attention outputs. KV keeps the BF16-relative
// allowance with a smaller absolute heuristic to increase sensitivity near
// zero; it does not guarantee detection of every stale or shifted value. These
// values are local/trusted regression heuristics, not ranked submission gates
// or M5-calibrated optimization limits. Reassociated kernels must be calibrated
// on M5 before relying on or loosening any envelope.
private let exactPairLogitsTolerance = ExactPairNumericTolerance(
    relative: 1e-5,
    absolute: 1e-5
)
private let exactPairHiddenStateTolerance = ExactPairNumericTolerance(
    relative: 1e-2,
    absolute: 2e-3
)
private let exactPairKVTolerance = ExactPairNumericTolerance(
    relative: 1e-2,
    absolute: 1e-3
)
private let exactPairAttentionTolerance = ExactPairNumericTolerance(
    relative: 1e-2,
    absolute: 2e-3
)

@Test
func exactPairDecisionCoversFullPartialAndZeroAcceptance() throws {
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11, 12]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 2,
            committedTokens: [11, 12]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 1,
            committedTokens: [11, 99]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [98, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 0,
            committedTokens: [98]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11],
            targetTokens: [11, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 1,
            committedTokens: [11, 99]
        )
    )
    #expect(
        try gemma4ExactPairDecision(
            drafts: [11],
            targetTokens: [98, 99]
        ) == Gemma4ExactPairDecision(
            acceptedDrafts: 0,
            committedTokens: [98]
        )
    )
    #expect(throws: MLXFastError.self) {
        _ = try gemma4ExactPairDecision(
            drafts: [11, 12],
            targetTokens: [11]
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try gemma4ExactPairDecision(
            drafts: [],
            targetTokens: [11, 12]
        )
    }
}

@Test
func exactPairEligibilityRequiresEveryVerificationComponent() {
    struct Components {
        var hasQKV = true
        var hasAttentionPreparation = true
        var hasAttentionBoundary = true
        var hasNextBoundary = true
        var hasOutput = true
        var hasGateUp = true
        var hasDown = true
        var usesFusedActivation = true

        var isEligible: Bool {
            gemma4ExactTwoVectorLayerIsEligible(
                hasQKV: hasQKV,
                hasAttentionPreparation: hasAttentionPreparation,
                hasAttentionBoundary: hasAttentionBoundary,
                hasNextBoundary: hasNextBoundary,
                hasOutput: hasOutput,
                hasGateUp: hasGateUp,
                hasDown: hasDown,
                usesFusedActivation: usesFusedActivation
            )
        }
    }

    #expect(gemma4ExactTwoVectorShapeIsSupported([2, 5_376], width: 5_376))
    #expect(!gemma4ExactTwoVectorShapeIsSupported([1, 2, 5_376], width: 5_376))
    let complete = Components()
    #expect(complete.isEligible)

    let dependencies: [(String, WritableKeyPath<Components, Bool>)] = [
        ("QKV", \.hasQKV),
        ("attention preparation", \.hasAttentionPreparation),
        ("attention boundary", \.hasAttentionBoundary),
        ("next boundary", \.hasNextBoundary),
        ("output", \.hasOutput),
        ("gate/up", \.hasGateUp),
        ("down", \.hasDown),
        ("fused activation", \.usesFusedActivation),
    ]
    for (name, dependency) in dependencies {
        var missing = complete
        missing[keyPath: dependency] = false
        #expect(
            !missing.isEligible,
            Comment(rawValue: "eligibility unexpectedly ignored \(name)")
        )
    }
}

@Test
func exactPairAttentionMatchesSerialRowsAcrossSlidingDispatchBoundary() {
    guard ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_MLX_RUNTIME_TESTS"
    ] == "1" else {
        return
    }

    for (headDimension, keyCount) in [
        (256, 514),
        (256, 1_023),
        (256, 1_024),
        (512, 514),
    ] {
        let keyElements = keyCount * headDimension
        let keyPositions = MLXArray(0..<keyElements).asType(.float32)
        let keysWithDraft = sin(keyPositions * 0.000_976_562_5)
            .asType(.bfloat16)
            .reshaped(1, 1, keyCount, headDimension)
        let valuesWithDraft = cos(keyPositions * 0.001_953_125)
            .asType(.bfloat16)
            .reshaped(1, 1, keyCount, headDimension)
        let keysBeforeDraft = keysWithDraft[
            0..., 0..., 0..<(keyCount - 1), 0...
        ]
        let valuesBeforeDraft = valuesWithDraft[
            0..., 0..., 0..<(keyCount - 1), 0...
        ]
        let queryPositions = MLXArray(0..<(2 * headDimension)).asType(.float32)
        let queries = sin(queryPositions * 0.003_906_25)
            .asType(.bfloat16)
            .reshaped(1, 1, 2, headDimension)

        let candidate = gemma4ExactTwoTokenAttention(
            queries: queries,
            keysBeforeDraft: keysBeforeDraft,
            valuesBeforeDraft: valuesBeforeDraft,
            keysWithDraft: keysWithDraft,
            valuesWithDraft: valuesWithDraft,
            scale: 1.0
        )
        let serialFirst = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 0..<1, 0...],
            keys: keysBeforeDraft,
            values: valuesBeforeDraft,
            scale: 1.0,
            mask: .none
        )
        let serialSecond = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 1..<2, 0...],
            keys: keysWithDraft,
            values: valuesWithDraft,
            scale: 1.0,
            mask: .none
        )
        expectActivationNumericParity(
            candidate,
            concatenated([serialFirst, serialSecond], axis: 2),
            expectedShape: [1, 1, 2, headDimension],
            context:
                "attention D=\(headDimension) keys=\(keyCount)"
        )
    }
}

@Test
func exactPairAndFourRuntimePreserveTokensWithinNumericEnvelope() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let goldenPath = try #require(
        environment["MLXFAST_MTP_PAIR_GOLDEN_PATH"]
    )
    let golden = try loadGoldenFixture(
        from: goldenPath,
        requiredSteps: 5,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    let testCase = try #require(golden.cases.first)
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()
    let tensorConfiguration = model.configuration

    let pairCache = model.newCache(parameters: nil)
    let serialCache = model.newCache(parameters: nil)
    let rollbackCache = model.newCache(parameters: nil)
    let serialPrefixCache = model.newCache(parameters: nil)
    let fourCache = model.newCache(parameters: nil)
    let serialFourCache = model.newCache(parameters: nil)
    let prompt = MLXArray(
        testCase.promptTokens.map(Int32.init),
        [1, testCase.promptTokens.count]
    )
    let pairPrefill = try #require(model.fastMTPForward(prompt, cache: pairCache))
    let serialPrefill = try #require(model.fastMTPForward(prompt, cache: serialCache))
    _ = try #require(model.fastMTPForward(prompt, cache: rollbackCache))
    _ = try #require(model.fastMTPForward(prompt, cache: serialPrefixCache))
    _ = try #require(model.fastMTPForward(prompt, cache: fourCache))
    _ = try #require(model.fastMTPForward(prompt, cache: serialFourCache))
    let seedArray = pairPrefill.logits[
        0..., -1, 0...
    ].asType(.float32).argMax(axis: -1)
    let serialSeedArray = serialPrefill.logits[
        0..., -1, 0...
    ].asType(.float32).argMax(axis: -1)
    eval(seedArray, serialSeedArray)
    let seed = Int(seedArray.item(Int32.self))
    #expect(seed == testCase.expectedTokens[0])
    #expect(seedArray.item(Int32.self) == serialSeedArray.item(Int32.self))

    // exactMTPFour is a distinct four-row forward with dedicated QKV,
    // projection, MLP, and vocabulary-head kernels. It is not implemented as
    // two exactMTPPair calls, so cover its tensors directly.
    let fourInputs = Array(testCase.expectedTokens[0..<4])
    let four = try #require(
        model.exactMTPFour(
            MLXArray(fourInputs.map(Int32.init), [1, 4]),
            cache: fourCache
        )
    )
    var serialFourRows: [Gemma4MTPForward] = []
    for token in fourInputs {
        serialFourRows.append(
            try #require(
                model.fastMTPForward(
                    MLXArray([Int32(token)], [1, 1]),
                    cache: serialFourCache
                )
            )
        )
    }
    let serialFourLogits = concatenated(
        serialFourRows.map(\.logits),
        axis: 1
    )
    let serialFourHidden = concatenated(
        serialFourRows.map(\.lastHidden),
        axis: 1
    )
    let fourTokens = four.logits.asType(.float32).argMax(axis: -1)
    eval(fourTokens)
    #expect(
        fourTokens.asArray(Int32.self).map(Int.init)
            == Array(testCase.expectedTokens[1...4])
    )
    expectLogitBehaviorParity(
        four.logits,
        serialFourLogits,
        expectedRows: 4,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "exact-four logits"
    )
    expectHiddenStateNumericParity(
        four.lastHidden,
        serialFourHidden,
        expectedRows: 4,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "exact-four pre-norm hidden"
    )
    let fourOffset = testCase.promptTokens.count + 4
    expectCacheGeometryAndNumericParity(
        fourCache,
        serialFourCache,
        configuration: tensorConfiguration,
        expectedOffset: fourOffset,
        context: "exact-four full commit"
    )
    let serialFourLast = try #require(serialFourRows.last)
    let fullSharedShape = expectedKVShape(
        configuration: tensorConfiguration,
        layerType: Gemma4LayerType.full.rawValue,
        offset: fourOffset
    )
    let slidingSharedShape = expectedKVShape(
        configuration: tensorConfiguration,
        layerType: Gemma4LayerType.sliding.rawValue,
        offset: fourOffset
    )
    expectKVNumericParity(
        four.capturedSharedKV.fullAttention.0,
        serialFourLast.capturedSharedKV.fullAttention.0,
        expectedShape: fullSharedShape,
        context: "exact-four shared full-attention keys"
    )
    expectKVNumericParity(
        four.capturedSharedKV.fullAttention.1,
        serialFourLast.capturedSharedKV.fullAttention.1,
        expectedShape: fullSharedShape,
        context: "exact-four shared full-attention values"
    )
    expectKVNumericParity(
        four.capturedSharedKV.slidingAttention.0,
        serialFourLast.capturedSharedKV.slidingAttention.0,
        expectedShape: slidingSharedShape,
        context: "exact-four shared sliding-attention keys"
    )
    expectKVNumericParity(
        four.capturedSharedKV.slidingAttention.1,
        serialFourLast.capturedSharedKV.slidingAttention.1,
        expectedShape: slidingSharedShape,
        context: "exact-four shared sliding-attention values"
    )

    let firstDraft = testCase.expectedTokens[1]
    let pairInputs = MLXArray(
        [Int32(seed), Int32(firstDraft)],
        [1, 2]
    )
    let pair = try #require(
        model.exactMTPPair(pairInputs, cache: pairCache)
    )
    let rollbackPair = try #require(
        model.exactMTPPair(pairInputs, cache: rollbackCache)
    )
    let serial0 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(seed)], [1, 1]),
            cache: serialCache
        )
    )
    let serialPrefix0 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(seed)], [1, 1]),
            cache: serialPrefixCache
        )
    )
    let serial1 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: serialCache
        )
    )
    let pairTokens = pair.logits.asType(.float32).argMax(axis: -1)
    eval(pairTokens)
    #expect(
        pairTokens.asArray(Int32.self).map(Int.init)
            == Array(testCase.expectedTokens[1...2])
    )

    expectLogitBehaviorParity(
        pair.logits[0..., 0..<1, 0...],
        serial0.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "row-0 logits"
    )
    expectLogitBehaviorParity(
        pair.logits[0..., 1..<2, 0...],
        serial1.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "row-1 logits"
    )
    expectHiddenStateNumericParity(
        pair.lastHidden[0..., 0..<1, 0...],
        serial0.lastHidden,
        expectedRows: 1,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "row-0 pre-norm hidden"
    )
    expectHiddenStateNumericParity(
        pair.lastHidden[0..., 1..<2, 0...],
        serial1.lastHidden,
        expectedRows: 1,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "row-1 pre-norm hidden"
    )
    expectCacheGeometryAndNumericParity(
        pairCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + 2,
        context: "full pair commit"
    )

    let nextInput = testCase.expectedTokens[2]
    let nextDraft = testCase.expectedTokens[3]
    let pairAtNextOffset = try #require(
        model.exactMTPPair(
            MLXArray([Int32(nextInput), Int32(nextDraft)], [1, 2]),
            cache: pairCache
        )
    )
    let serial2 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(nextInput)], [1, 1]),
            cache: serialCache
        )
    )
    let serial3 = try #require(
        model.fastMTPForward(
            MLXArray([Int32(nextDraft)], [1, 1]),
            cache: serialCache
        )
    )
    let nextPairTokens = pairAtNextOffset.logits
        .asType(.float32)
        .argMax(axis: -1)
    eval(nextPairTokens)
    #expect(
        nextPairTokens.asArray(Int32.self).map(Int.init)
            == Array(testCase.expectedTokens[3...4])
    )
    expectLogitBehaviorParity(
        pairAtNextOffset.logits[0..., 0..<1, 0...],
        serial2.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "next-offset row-0 logits"
    )
    expectLogitBehaviorParity(
        pairAtNextOffset.logits[0..., 1..<2, 0...],
        serial3.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "next-offset row-1 logits"
    )
    expectHiddenStateNumericParity(
        pairAtNextOffset.lastHidden[0..., 0..<1, 0...],
        serial2.lastHidden,
        expectedRows: 1,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "next-offset row-0 pre-norm hidden"
    )
    expectHiddenStateNumericParity(
        pairAtNextOffset.lastHidden[0..., 1..<2, 0...],
        serial3.lastHidden,
        expectedRows: 1,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "next-offset row-1 pre-norm hidden"
    )
    expectCacheGeometryAndNumericParity(
        pairCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + 4,
        context: "next-offset full pair commit"
    )

    for cache in rollbackCache {
        #expect(cache.trim(1) == 1)
    }
    let rolledShared = Gemma4SharedKV.sliceTail(
        from: rollbackPair.capturedSharedKV,
        rejected: 1
    )
    let rollbackOffset = testCase.promptTokens.count + 1
    let rollbackFullShape = expectedKVShape(
        configuration: tensorConfiguration,
        layerType: Gemma4LayerType.full.rawValue,
        offset: rollbackOffset
    )
    let rollbackSlidingShape = expectedKVShape(
        configuration: tensorConfiguration,
        layerType: Gemma4LayerType.sliding.rawValue,
        offset: rollbackOffset
    )
    expectKVNumericParity(
        rolledShared.fullAttention.0,
        serialPrefix0.capturedSharedKV.fullAttention.0,
        expectedShape: rollbackFullShape,
        context: "rolled full-attention keys"
    )
    expectKVNumericParity(
        rolledShared.fullAttention.1,
        serialPrefix0.capturedSharedKV.fullAttention.1,
        expectedShape: rollbackFullShape,
        context: "rolled full-attention values"
    )
    expectKVNumericParity(
        rolledShared.slidingAttention.0,
        serialPrefix0.capturedSharedKV.slidingAttention.0,
        expectedShape: rollbackSlidingShape,
        context: "rolled sliding-attention keys"
    )
    expectKVNumericParity(
        rolledShared.slidingAttention.1,
        serialPrefix0.capturedSharedKV.slidingAttention.1,
        expectedShape: rollbackSlidingShape,
        context: "rolled sliding-attention values"
    )
    expectCacheGeometryAndNumericParity(
        rollbackCache,
        serialPrefixCache,
        configuration: tensorConfiguration,
        expectedOffset: rollbackOffset,
        context: "zero-acceptance rollback"
    )
    let resumedAfterRollback = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: rollbackCache
        )
    )
    let serialAfterPrefix = try #require(
        model.fastMTPForward(
            MLXArray([Int32(firstDraft)], [1, 1]),
            cache: serialPrefixCache
        )
    )
    expectLogitBehaviorParity(
        resumedAfterRollback.logits,
        serialAfterPrefix.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "post-rollback continuation logits"
    )
    expectHiddenStateNumericParity(
        resumedAfterRollback.lastHidden,
        serialAfterPrefix.lastHidden,
        expectedRows: 1,
        hiddenSize: tensorConfiguration.hiddenSize,
        context: "post-rollback continuation hidden"
    )
    expectCacheGeometryAndNumericParity(
        rollbackCache,
        serialPrefixCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + 2,
        context: "post-rollback continuation cache"
    )
}

/// Forces the distinct direct-four session branch through full, zero, and
/// partial acceptance. The dedicated counter proves the branch did not fall
/// back to pair composition; pair-segment and rollback counters retain their
/// pair-equivalent physical-row accounting semantics.
@Test
func exactFourSessionCountersCoverFullZeroAndPartialAcceptance() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let assistantPath = try #require(
        environment["MLXFAST_MTP_ASSISTANT_DIR"]
    )
    let goldenPath = try #require(
        environment["MLXFAST_MTP_PAIR_GOLDEN_PATH"]
    )
    let golden = try loadGoldenFixture(
        from: goldenPath,
        requiredSteps: 11,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    let testCase = try #require(golden.cases.first)
    let expected = testCase.expectedTokens
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()
    let tensorConfiguration = model.configuration
    let drafter = try await Gemma4AssistantDraftModel.load(
        from: URL(fileURLWithPath: assistantPath)
    )
    let session = try Gemma4TrainedMTPBlockSession(
        target: model,
        drafter: drafter,
        verificationMode: .exactPair
    )
    let seed = try session.begin(seedTokens: testCase.promptTokens)
    #expect(seed == expected[0])

    let serialCache = model.newCache(parameters: nil)
    _ = try #require(
        model.fastMTPForward(
            MLXArray(
                testCase.promptTokens.map(Int32.init),
                [1, testCase.promptTokens.count]
            ),
            cache: serialCache
        )
    )
    var committedInputRows = 0
    func forwardSerial(_ tokens: [Int]) throws {
        for token in tokens {
            _ = try #require(
                model.fastMTPForward(
                    MLXArray([Int32(token)], [1, 1]),
                    cache: serialCache
                )
            )
            committedInputRows += 1
        }
    }
    func mismatching(_ token: Int) -> Int {
        token == 0 ? 1 : 0
    }
    func expectDirectAccounting(_ context: String) {
        #expect(
            2 * session.exactPairSegmentCount
                - session.exactPairRollbackRowCount
                + session.serialVerificationRowCount
                == committedInputRows,
            Comment(rawValue: context)
        )
        expectCacheGeometryAndNumericParity(
            session.targetCache,
            serialCache,
            configuration: tensorConfiguration,
            expectedOffset: testCase.promptTokens.count + committedInputRows,
            context: context
        )
    }

    let full = try session.verifyWithExactPairs(
        previousToken: seed,
        drafts: Array(expected[1...3]),
        preferExactFour: true
    )
    #expect(full.committedTokens == Array(expected[1...4]))
    #expect(full.acceptedDrafts == 3)
    #expect(session.directExactFourInvocationCount == 1)
    #expect(session.exactPairSegmentCount == 2)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 0)
    try forwardSerial(Array(expected[0...3]))
    expectDirectAccounting("direct-four full acceptance")

    let zero = try session.verifyWithExactPairs(
        previousToken: expected[4],
        drafts: [
            mismatching(expected[5]),
            mismatching(expected[6]),
            mismatching(expected[7]),
        ],
        preferExactFour: true
    )
    #expect(zero.committedTokens == [expected[5]])
    #expect(zero.acceptedDrafts == 0)
    #expect(session.directExactFourInvocationCount == 2)
    #expect(session.exactPairSegmentCount == 4)
    #expect(session.exactPairRollbackRowCount == 3)
    #expect(session.serialVerificationRowCount == 0)
    try forwardSerial([expected[4]])
    expectDirectAccounting("direct-four zero acceptance")

    let partial = try session.verifyWithExactPairs(
        previousToken: expected[5],
        drafts: [
            expected[6],
            mismatching(expected[7]),
            mismatching(expected[8]),
        ],
        preferExactFour: true
    )
    #expect(partial.committedTokens == Array(expected[6...7]))
    #expect(partial.acceptedDrafts == 1)
    #expect(session.directExactFourInvocationCount == 3)
    #expect(session.exactPairSegmentCount == 6)
    #expect(session.exactPairRollbackRowCount == 5)
    #expect(session.serialVerificationRowCount == 0)
    try forwardSerial(Array(expected[5...6]))
    expectDirectAccounting("direct-four partial acceptance")

    let restartedSeed = try session.begin(seedTokens: testCase.promptTokens)
    #expect(restartedSeed == expected[0])
    #expect(session.directExactFourInvocationCount == 0)
    #expect(session.exactPairSegmentCount == 0)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 0)
    #expect(
        session.targetCache.allSatisfy {
            $0.offset == testCase.promptTokens.count
        }
    )
}

/// Forces zero, partial, full, and second-segment acceptance through the real
/// session verifier with fabricated drafts, so the seams do not depend on
/// assistant draft behavior. Every committed token, counter, cache offset,
/// state count, and physical tensor shape must match an independent
/// teacher-forced serial cache; floating KV values use the documented envelope.
@Test
func exactPairSessionForcedAcceptanceSeamsMatchSerial() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1" else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let assistantPath = try #require(
        environment["MLXFAST_MTP_ASSISTANT_DIR"]
    )
    let goldenPath = try #require(
        environment["MLXFAST_MTP_PAIR_GOLDEN_PATH"]
    )
    let golden = try loadGoldenFixture(
        from: goldenPath,
        requiredSteps: 17,
        requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
    )
    let testCase = try #require(golden.cases.first)
    let expected = testCase.expectedTokens
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()
    let tensorConfiguration = model.configuration
    let drafter = try await Gemma4AssistantDraftModel.load(
        from: URL(fileURLWithPath: assistantPath)
    )
    let session = try Gemma4TrainedMTPBlockSession(
        target: model,
        drafter: drafter,
        verificationMode: .exactPair
    )
    let seed = try session.begin(seedTokens: testCase.promptTokens)
    #expect(seed == expected[0])

    let serialCache = model.newCache(parameters: nil)
    _ = try #require(
        model.fastMTPForward(
            MLXArray(
                testCase.promptTokens.map(Int32.init),
                [1, testCase.promptTokens.count]
            ),
            cache: serialCache
        )
    )
    var serialInputs: [Int] = []
    func forwardSerial(_ tokens: [Int]) throws {
        for token in tokens {
            let row = try #require(
                model.fastMTPForward(
                    MLXArray([Int32(token)], [1, 1]),
                    cache: serialCache
                )
            )
            eval(row.logits)
            serialInputs.append(token)
        }
    }
    func mismatching(_ token: Int) -> Int {
        token == 0 ? 1 : 0
    }

    // Tail: K=1 is target-only, advances exactly one row, and never drafts.
    let targetOnlyTail = try session.generateBlock(
        previousCommittedToken: seed,
        maxBlockSize: 1
    )
    #expect(targetOnlyTail.tokens == [expected[1]])
    #expect(targetOnlyTail.acceptedDraftTokenCount == 0)
    #expect(!targetOnlyTail.usedDrafter)
    #expect(session.targetOnlyTailTokenCount == 1)
    #expect(session.exactPairSegmentCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial([seed])
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session target-only tail"
    )

    // Round A: K=2 full acceptance is exactly one two-row segment.
    let k2Block = try session.verifyWithExactPairs(
        previousToken: expected[1],
        drafts: [expected[2]],
        preferExactFour: false
    )
    #expect(k2Block.committedTokens == Array(expected[2...3]))
    #expect(k2Block.acceptedDrafts == 1)
    #expect(session.exactPairSegmentCount == 1)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[1...2]))
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session K=2 full acceptance"
    )

    // Round B: force K=4 through two exact-pair segments so the cumulative
    // segment counters below describe pair composition only.
    let fullBlock = try session.verifyWithExactPairs(
        previousToken: expected[3],
        drafts: Array(expected[4...6]),
        preferExactFour: false
    )
    #expect(fullBlock.committedTokens == Array(expected[4...7]))
    #expect(fullBlock.acceptedDrafts == 3)
    #expect(session.exactPairSegmentCount == 3)
    #expect(session.exactPairRollbackRowCount == 0)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[3...6]))
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session full acceptance"
    )

    // Round C: zero acceptance rejects row zero and must physically remove
    // row one before continuing.
    let zeroBlock = try session.verifyWithExactPairs(
        previousToken: expected[7],
        drafts: [
            mismatching(expected[8]),
            mismatching(expected[9]),
            mismatching(expected[10]),
        ],
        preferExactFour: false
    )
    #expect(zeroBlock.committedTokens == [expected[8]])
    #expect(zeroBlock.acceptedDrafts == 0)
    #expect(session.exactPairSegmentCount == 4)
    #expect(session.exactPairRollbackRowCount == 1)
    try forwardSerial([expected[7]])
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session zero acceptance rollback"
    )

    // Round D: the second segment rejects its first row after a fully
    // accepted first segment.
    let secondSegmentBlock = try session.verifyWithExactPairs(
        previousToken: expected[8],
        drafts: [
            expected[9],
            expected[10],
            mismatching(expected[11]),
        ],
        preferExactFour: false
    )
    #expect(
        secondSegmentBlock.committedTokens == Array(expected[9...11])
    )
    #expect(secondSegmentBlock.acceptedDrafts == 2)
    #expect(session.exactPairSegmentCount == 6)
    #expect(session.exactPairRollbackRowCount == 2)
    try forwardSerial(Array(expected[8...10]))
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session second-segment rejection"
    )

    // Round E: row-one rejection keeps both physical rows because both pair
    // inputs are committed tokens.
    let partialBlock = try session.verifyWithExactPairs(
        previousToken: expected[11],
        drafts: [
            expected[12],
            mismatching(expected[13]),
            mismatching(expected[14]),
        ],
        preferExactFour: false
    )
    #expect(partialBlock.committedTokens == Array(expected[12...13]))
    #expect(partialBlock.acceptedDrafts == 1)
    #expect(session.exactPairSegmentCount == 7)
    #expect(session.exactPairRollbackRowCount == 2)
    #expect(session.serialVerificationRowCount == 1)
    try forwardSerial(Array(expected[11...12]))
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session row-one rejection"
    )

    // Round F: K=3 uses one exact pair plus one serial bonus row.
    let k3TailBlock = try session.verifyWithExactPairs(
        previousToken: expected[13],
        drafts: Array(expected[14...15]),
        preferExactFour: false
    )
    #expect(k3TailBlock.committedTokens == Array(expected[14...16]))
    #expect(k3TailBlock.acceptedDrafts == 2)
    #expect(session.exactPairSegmentCount == 8)
    #expect(session.exactPairRollbackRowCount == 2)
    #expect(session.serialVerificationRowCount == 2)
    try forwardSerial(Array(expected[13...15]))
    expectCacheGeometryAndNumericParity(
        session.targetCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: testCase.promptTokens.count + serialInputs.count,
        context: "session K=3 serial bonus tail"
    )

    // Physical row accounting across all rounds.
    let committedTotal = targetOnlyTail.tokens.count
        + k2Block.committedTokens.count
        + fullBlock.committedTokens.count
        + zeroBlock.committedTokens.count
        + secondSegmentBlock.committedTokens.count
        + partialBlock.committedTokens.count
        + k3TailBlock.committedTokens.count
    #expect(
        2 * session.exactPairSegmentCount
            - session.exactPairRollbackRowCount
            + session.serialVerificationRowCount
            == committedTotal
    )
    #expect(serialInputs.count == committedTotal)
    #expect(
        session.targetCache.allSatisfy {
            $0.offset == testCase.promptTokens.count + committedTotal
        }
    )
}

/// Drives identical fixed BOS token streams through the exact-pair path and a
/// serial mirror across the deep decode seams: the 768-position combined
/// cache growth, the odd-offset sliding refusal before the 1,024 wrap, the
/// library's 1,024-key sliding kernel switch, and post-wrap serial decode.
/// Inputs are constant, so this isolates exact cache geometry, exact token
/// decisions, and numeric tensor parity rather than language quality, and it
/// needs no golden fixture.
@Test
func exactPairDeepOffsetsGrowthAndWrapMatchSerial() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_EXACT_PAIR_TESTS"] == "1",
          environment["MLXFAST_RUN_MTP_DEEP_OFFSET_TESTS"] == "1"
    else {
        return
    }
    let weightsPath = try #require(
        environment["MLXFAST_MTP_WEIGHTS_PATH"]
    )
    let config = try Gemma4Config.load(from: weightsPath)
    let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
    let weightCache = Gemma4RuntimeWeightCache(
        loader: loader,
        config: config
    )
    let model = try weightCache.requireLibraryModel()
    let tensorConfiguration = model.configuration

    let bos = Int32(2)
    let seedLength = MLXFastConstants.correctnessPromptTokens
    let prompt = MLXArray(
        Array(repeating: bos, count: seedLength),
        [1, seedLength]
    )
    let pairCache = model.newCache(parameters: nil)
    let serialCache = model.newCache(parameters: nil)
    _ = try #require(model.fastMTPForward(prompt, cache: pairCache))
    _ = try #require(model.fastMTPForward(prompt, cache: serialCache))

    let pairInput = MLXArray([bos, bos], [1, 2])
    let singleInput = MLXArray([bos], [1, 1])
    let slidingWindow = model.configuration.slidingWindow
    let checkpointOffsets: Set<Int> = [
        seedLength + 254,
        seedLength + 256,
        seedLength + 258,
        seedLength + 508,
    ]
    var offset = seedLength

    func serialRowPair() throws -> (Gemma4MTPForward, Gemma4MTPForward) {
        let first = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        let second = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        return (first, second)
    }

    // Even-offset pairs up to one row short of the sliding window boundary.
    while offset + 2 <= seedLength + 508 {
        let pair = try #require(
            model.exactMTPPair(pairInput, cache: pairCache)
        )
        let serial = try serialRowPair()
        offset += 2
        expectLogitBehaviorParity(
            pair.logits[0..., 0..<1, 0...],
            serial.0.logits,
            expectedRows: 1,
            vocabularySize: tensorConfiguration.vocabSize,
            context: "deep row-0 logits offset \(offset)"
        )
        expectLogitBehaviorParity(
            pair.logits[0..., 1..<2, 0...],
            serial.1.logits,
            expectedRows: 1,
            vocabularySize: tensorConfiguration.vocabSize,
            context: "deep row-1 logits offset \(offset)"
        )
        if checkpointOffsets.contains(offset) {
            expectCacheGeometryAndNumericParity(
                pairCache,
                serialCache,
                configuration: tensorConfiguration,
                expectedOffset: offset,
                context: "deep checkpoint offset \(offset)"
            )
        }
    }
    #expect(offset == seedLength + 508)

    // One serial row makes the next refusal happen at an odd offset just
    // before the sliding wrap.
    let oddRow = try #require(
        model.fastMTPForward(singleInput, cache: pairCache)
    )
    let oddSerial = try #require(
        model.fastMTPForward(singleInput, cache: serialCache)
    )
    offset += 1
    expectLogitBehaviorParity(
        oddRow.logits,
        oddSerial.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "odd serial row logits offset \(offset)"
    )
    let pairBeforeWrap = try #require(
        model.exactMTPPair(pairInput, cache: pairCache)
    )
    let serialBeforeWrap = try serialRowPair()
    offset += 2
    #expect(offset == slidingWindow - 1)
    expectLogitBehaviorParity(
        pairBeforeWrap.logits[0..., 1..<2, 0...],
        serialBeforeWrap.1.logits,
        expectedRows: 1,
        vocabularySize: tensorConfiguration.vocabSize,
        context: "pre-wrap pair row-1 logits"
    )
    expectCacheGeometryAndNumericParity(
        pairCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: offset,
        context: "pre-wrap cache cursor"
    )

    // The sliding caches can no longer retain a prefix view for a pair.
    #expect(model.exactMTPPair(pairInput, cache: pairCache) == nil)

    // Serial rows continue across the wrap and the 1,024-key kernel switch.
    for step in 0..<4 {
        let row = try #require(
            model.fastMTPForward(singleInput, cache: pairCache)
        )
        let serialRow = try #require(
            model.fastMTPForward(singleInput, cache: serialCache)
        )
        offset += 1
        expectLogitBehaviorParity(
            row.logits,
            serialRow.logits,
            expectedRows: 1,
            vocabularySize: tensorConfiguration.vocabSize,
            context: "post-wrap serial logits step \(step) offset \(offset)"
        )
    }
    expectCacheGeometryAndNumericParity(
        pairCache,
        serialCache,
        configuration: tensorConfiguration,
        expectedOffset: offset,
        context: "post-wrap cache state"
    )
    // The pair path stays unavailable after the wrap.
    #expect(model.exactMTPPair(pairInput, cache: pairCache) == nil)
}

private func expectNumericParity(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    expectedShape: [Int],
    expectedDType: DType,
    tolerance: ExactPairNumericTolerance,
    context: String
) {
    let shapeMatches =
        candidate.shape == expectedShape && reference.shape == expectedShape
    let dtypeMatches =
        candidate.dtype == expectedDType && reference.dtype == expectedDType
    #expect(
        shapeMatches,
        Comment(
            rawValue:
                "\(context) canonical shape \(expectedShape); "
                + "candidate=\(candidate.shape) reference=\(reference.shape)"
        )
    )
    #expect(
        dtypeMatches,
        Comment(
            rawValue:
                "\(context) canonical dtype \(expectedDType); "
                + "candidate=\(candidate.dtype) reference=\(reference.dtype)"
        )
    )
    guard shapeMatches, dtypeMatches else {
        return
    }

    let candidateFloat = candidate.asType(.float32)
    let referenceFloat = reference.asType(.float32)
    let candidateFiniteArray = all(MLX.isFinite(candidateFloat))
    let referenceFiniteArray = all(MLX.isFinite(referenceFloat))
    eval(candidateFiniteArray, referenceFiniteArray)
    let candidateFinite = candidateFiniteArray.item(Bool.self)
    let referenceFinite = referenceFiniteArray.item(Bool.self)
    #expect(candidateFinite, Comment(rawValue: "\(context) candidate finiteness"))
    #expect(referenceFinite, Comment(rawValue: "\(context) reference finiteness"))
    guard candidateFinite, referenceFinite else {
        return
    }

    let close = allClose(
        candidateFloat,
        referenceFloat,
        rtol: tolerance.relative,
        atol: tolerance.absolute
    )
    eval(close)
    let isClose = close.item(Bool.self)
    if !isClose {
        let maxAbsoluteError = MLX.abs(candidateFloat - referenceFloat)
            .max()
            .item(Float.self)
        #expect(
            isClose,
            Comment(
                rawValue:
                    "\(context) numeric envelope "
                    + "(rtol=\(tolerance.relative), atol=\(tolerance.absolute), "
                    + "max_abs_error=\(maxAbsoluteError))"
            )
        )
    }
}

private func expectLogitBehaviorParity(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    expectedRows: Int,
    vocabularySize: Int,
    context: String
) {
    expectNumericParity(
        candidate,
        reference,
        expectedShape: [1, expectedRows, vocabularySize],
        expectedDType: .float32,
        tolerance: exactPairLogitsTolerance,
        context: context
    )
    guard candidate.shape == reference.shape else {
        return
    }
    let candidateTokens = candidate.asType(.float32).argMax(axis: -1)
    let referenceTokens = reference.asType(.float32).argMax(axis: -1)
    let tokensEqual = arrayEqual(candidateTokens, referenceTokens)
    eval(tokensEqual)
    #expect(
        tokensEqual.item(Bool.self),
        Comment(rawValue: "\(context) exact argmax token IDs")
    )
}

private func expectHiddenStateNumericParity(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    expectedRows: Int,
    hiddenSize: Int,
    context: String
) {
    expectNumericParity(
        candidate,
        reference,
        expectedShape: [1, expectedRows, hiddenSize],
        expectedDType: .bfloat16,
        tolerance: exactPairHiddenStateTolerance,
        context: context
    )
}

private func expectKVNumericParity(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    expectedShape: [Int],
    context: String
) {
    expectNumericParity(
        candidate,
        reference,
        expectedShape: expectedShape,
        expectedDType: .bfloat16,
        tolerance: exactPairKVTolerance,
        context: context
    )
}

private func expectActivationNumericParity(
    _ candidate: MLXArray,
    _ reference: MLXArray,
    expectedShape: [Int],
    context: String
) {
    expectNumericParity(
        candidate,
        reference,
        expectedShape: expectedShape,
        expectedDType: .bfloat16,
        tolerance: exactPairAttentionTolerance,
        context: context
    )
}

private func expectedKVShape(
    configuration: Gemma4TextConfiguration,
    layerType: String,
    offset: Int
) -> [Int] {
    let isFull = layerType == Gemma4LayerType.full.rawValue
    let heads =
        isFull && configuration.attentionKeqV
        ? (configuration.numGlobalKeyValueHeads
            ?? configuration.numKeyValueHeads)
        : configuration.numKeyValueHeads
    let headDimension =
        isFull ? configuration.globalHeadDim : configuration.headDim
    let retainedLength =
        isFull ? offset : min(offset, configuration.slidingWindow)
    return [1, heads, retainedLength, headDimension]
}

private func expectedRotatingIndex(
    offset: Int,
    keep: Int,
    maxSize: Int
) -> Int {
    precondition(keep >= 0 && keep < maxSize)
    // Gemma4CombinedKVCache advances the cursor with every write. Before the
    // first wrap it equals the logical offset; at maxSize the next write resets
    // it to keep. Exact full-span boundaries leave the cursor at maxSize until
    // the following write performs that reset.
    if offset <= maxSize {
        return offset
    }
    let rotatingSpan = maxSize - keep
    let remainder = (offset - maxSize) % rotatingSpan
    return remainder == 0 ? maxSize : keep + remainder
}

private func expectCacheGeometryAndNumericParity(
    _ candidate: [KVCache],
    _ reference: [KVCache],
    configuration: Gemma4TextConfiguration,
    expectedOffset: Int,
    context: String
) {
    let expectedCacheCount =
        configuration.numHiddenLayers - configuration.numKvSharedLayers
    #expect(
        configuration.layerTypes.count == configuration.numHiddenLayers,
        Comment(rawValue: "\(context) canonical layer-type count")
    )
    #expect(
        candidate.count == expectedCacheCount,
        Comment(rawValue: "\(context) canonical candidate cache count")
    )
    #expect(
        reference.count == expectedCacheCount,
        Comment(rawValue: "\(context) canonical reference cache count")
    )
    guard configuration.layerTypes.count == configuration.numHiddenLayers,
          candidate.count == expectedCacheCount,
          reference.count == expectedCacheCount
    else {
        return
    }
    for index in candidate.indices {
        let layerType = configuration.layerTypes[index]
        let isFull = layerType == Gemma4LayerType.full.rawValue
        let expectedShape = expectedKVShape(
            configuration: configuration,
            layerType: layerType,
            offset: expectedOffset
        )
        #expect(
            candidate[index].offset == expectedOffset,
            Comment(rawValue: "\(context) layer \(index) candidate offset")
        )
        #expect(
            reference[index].offset == expectedOffset,
            Comment(rawValue: "\(context) layer \(index) reference offset")
        )
        #expect(
            candidate[index].maxSize
                == (isFull ? nil : configuration.slidingWindow),
            Comment(rawValue: "\(context) layer \(index) candidate max size")
        )
        #expect(
            reference[index].maxSize
                == (isFull ? nil : configuration.slidingWindow),
            Comment(rawValue: "\(context) layer \(index) reference max size")
        )

        let candidateMetadata = candidate[index].metaState
        let referenceMetadata = reference[index].metaState
        #expect(
            candidateMetadata == referenceMetadata,
            Comment(rawValue: "\(context) layer \(index) metadata")
        )
        if isFull {
            #expect(
                candidateMetadata == [""],
                Comment(rawValue: "\(context) layer \(index) full metadata")
            )
        } else {
            #expect(
                candidateMetadata.count == 5,
                Comment(rawValue: "\(context) layer \(index) rotating metadata count")
            )
            if candidateMetadata.count == 5 {
                #expect(candidateMetadata[0] == "0")
                #expect(Int(candidateMetadata[1]) == configuration.slidingWindow)
                #expect(Int(candidateMetadata[2]) == 256)
                #expect(Int(candidateMetadata[3]) == expectedOffset)
                #expect(
                    Int(candidateMetadata[4])
                        == expectedRotatingIndex(
                            offset: expectedOffset,
                            keep: 0,
                            maxSize: configuration.slidingWindow
                        ),
                    Comment(
                        rawValue:
                            "\(context) layer \(index) canonical rotating index"
                    )
                )
            }
        }

        let candidateState = candidate[index].state
        let referenceState = reference[index].state
        #expect(
            candidateState.count == 2,
            Comment(rawValue: "\(context) layer \(index) canonical candidate state count")
        )
        #expect(
            referenceState.count == 2,
            Comment(rawValue: "\(context) layer \(index) canonical reference state count")
        )
        guard candidateState.count == 2, referenceState.count == 2 else {
            continue
        }
        for stateIndex in candidateState.indices {
            expectKVNumericParity(
                candidateState[stateIndex],
                referenceState[stateIndex],
                expectedShape: expectedShape,
                context: "\(context) layer \(index) state \(stateIndex)"
            )
        }
    }
}

/// Load-path gate for the organizer-pinned QAT 4-bit assistant: proves the
/// pinned mlx-swift-lm revision materializes the quantized drafter
/// checkpoint (QuantizedLinear/QuantizedEmbedding construction from the
/// config's affine 4-bit group-64 quantization block) and that one drafter
/// forward over fabricated shared-KV produces a well-formed logits row.
/// Needs only the ~264 MB assistant sidecar, never the 17 GB target.
@Test
func qatAssistantCheckpointLoadsQuantizedAndForwards() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_MTP_ASSISTANT_LOAD_TESTS"] == "1" else {
        return
    }
    let assistantPath = try #require(
        environment["MLXFAST_MTP_ASSISTANT_DIR"]
    )
    let drafter = try await Gemma4AssistantDraftModel.load(
        from: URL(fileURLWithPath: assistantPath)
    )
    let text = drafter.config.textConfig
    // The QAT checkpoint declares its quantization at the top level of
    // config.json (the level Gemma4AssistantDraftModel.load consumes via
    // BaseConfiguration), not inside text_config.
    let base = try JSONDecoder().decode(
        BaseConfiguration.self,
        from: Data(
            contentsOf: URL(fileURLWithPath: assistantPath)
                .appendingPathComponent("config.json")
        )
    )
    let quantization = try #require(
        base.perLayerQuantization?.quantization
    )
    #expect(quantization.bits == 4)
    #expect(quantization.groupSize == 64)
    // The load-time quantization pass must swap every module that ships
    // `.scales` tensors: the projections and the tied token embedding
    // become their Quantized counterparts.
    #expect(drafter.preProjection is QuantizedLinear)
    #expect(drafter.postProjection is QuantizedLinear)
    #expect(drafter.model.embedTokens is QuantizedEmbedding)

    // One drafter forward over fabricated, fixed shared-KV (target-free):
    // shapes follow Gemma4SharedKV's documented [B, heads, T, headDim]
    // layout for the 31B-IT tower (4x512 full, 16x256 sliding).
    let backbone = drafter.config.backboneHiddenSize
    let sharedKV = Gemma4SharedKV(
        fullAttention: (
            MLXArray.zeros([1, 4, 8, 512], dtype: .bfloat16),
            MLXArray.zeros([1, 4, 8, 512], dtype: .bfloat16)
        ),
        slidingAttention: (
            MLXArray.zeros([1, 16, 8, 256], dtype: .bfloat16),
            MLXArray.zeros([1, 16, 8, 256], dtype: .bfloat16)
        )
    )
    let output = drafter(
        inputsEmbeds: MLXArray.zeros([1, 1, 2 * backbone], dtype: .bfloat16),
        sharedKV: sharedKV,
        positionOffset: Gemma4.PositionOffset.scalar(8)
    )
    eval(output.logits, output.lastHidden)
    #expect(output.logits.shape == [1, 1, text.vocabSize])
    #expect(output.lastHidden.shape == [1, 1, backbone])
    let draft = output.logits[0..., -1, 0...].argMax(axis: -1)
    let token = Int(draft.item(Int32.self))
    #expect(token >= 0 && token < text.vocabSize)
    #expect(
        output.logits.asType(.float32).sum().item(Float.self).isFinite
    )
}
