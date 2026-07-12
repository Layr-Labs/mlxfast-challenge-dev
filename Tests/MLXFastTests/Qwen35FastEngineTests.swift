import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func qwen35ProductionBackendRemainsLibraryOracleUntilBothGatesPass() {
    #expect(
        Qwen35FastPathReadiness.productionBackend == .libraryOracle
    )
    #expect(
        selectQwen35ExecutionBackend(
            realCheckpointParityPassed: false,
            productionActivationApproved: true
        ) == .libraryOracle
    )
    #expect(
        selectQwen35ExecutionBackend(
            realCheckpointParityPassed: true,
            productionActivationApproved: false
        ) == .libraryOracle
    )
    #expect(
        selectQwen35ExecutionBackend(
            realCheckpointParityPassed: true,
            productionActivationApproved: true
        ) == .customFastPath
    )
}

@Test
func qwen35BackendSelectionLoadsExactlyOneImplementation() throws {
    var events: [String] = []
    let library: Qwen35LoadedBackend<Int, Int> =
        try loadSelectedQwen35Backend(
            backend: .libraryOracle,
            loadLibrary: {
                events.append("library")
                return 1
            },
            loadFastPath: {
                events.append("fast")
                return 2
            }
        )
    guard case .library(1) = library else {
        Issue.record("library backend was not selected")
        return
    }
    #expect(events == ["library"])

    events.removeAll()
    let fast: Qwen35LoadedBackend<Int, Int> =
        try loadSelectedQwen35Backend(
            backend: .customFastPath,
            loadLibrary: {
                events.append("library")
                return 1
            },
            loadFastPath: {
                events.append("fast")
                return 2
            }
        )
    guard case .fastPath(2) = fast else {
        Issue.record("custom backend was not selected")
        return
    }
    #expect(events == ["fast"])
}

@Test
func qwen35UntiedLastTokenHeadMatchesSliceOfAllPositionLogits() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let engine = try tinyQwen35FastEngine()
    let inputIDs = MLXArray([Int32(0), 2, 4], [1, 3])
    let allLogits = try engine.allPositionLogits(inputIDs)
    let lastLogits = try engine.lastTokenLogits(inputIDs)
    let expectedLast = allLogits[
        0...,
        2..<3,
        0...
    ]
    let gap = qwen35MaximumAbsoluteDifference(
        lastLogits,
        expectedLast
    )

    #expect(allLogits.shape == [1, 3, 7])
    #expect(lastLogits.shape == [1, 1, 7])
    #expect(gap < 1e-6)
    #expect(engine.trunk.embedding.shape == [6, 24])
    #expect(engine.lmHead.weight.shape == [7, 24])
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35FastEngine(
            trunk: engine.trunk,
            lmHead: Qwen35LMHead(
                weight: Qwen35LinearWeight(MLXArray.zeros([7]))
            )
        )
    }
}

@Test
func qwen35FastTrunkPrefillMatchesCachedTokenSteps() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let engine = try tinyQwen35FastEngine()
    let inputIDs = MLXArray([Int32(1), 3, 5], [1, 3])
    let prefillCache = engine.newCache()
    let prefill = try engine.allPositionLogits(
        inputIDs,
        cache: prefillCache
    )

    let stepCache = engine.newCache()
    var stepLogits: [MLXArray] = []
    for position in 0..<3 {
        stepLogits.append(
            try engine.allPositionLogits(
                inputIDs[0..., position..<(position + 1)],
                cache: stepCache,
                positionOffset: position
            )
        )
    }
    let stepped = concatenated(stepLogits, axis: 1)
    let gap = qwen35MaximumAbsoluteDifference(prefill, stepped)

    #expect(prefillCache.expectedPositionOffset == 3)
    #expect(stepCache.expectedPositionOffset == 3)
    #expect(gap < 2e-5)
}

@Test
func qwen35FastEngineRollsBackLayerHeadAndOuterOffsets() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let engine = try tinyTransactionalQwen35FastEngine()
    let owner = Qwen35ModelCache(layerTypes: [.linear, .full])
    let cache = owner.cache(for: engine)
    let firstInput = MLXArray([Int32(1)], [1, 1])
    _ = try executeQwen35CachedForward(
        cache: owner,
        positionOffset: 0,
        inputLength: 1,
        validateLibraryCachePosition: {
            try owner.validatePosition(0, cache: cache)
        },
        forward: {
            try engine.allPositionLogits(
                firstInput,
                cache: cache,
                positionOffset: 0,
                ssmMask: MLXArray.ones([1, 1], dtype: .bool)
            )
        }
    )

    let delta = try #require(cache.layers[0].gatedDelta)
    let beforeConvolution = delta.convolution
    let beforeRecurrent = delta.recurrent
    let full = try #require(cache.layers[1].fullAttention)
    let beforeFullState = full.state
    #expect(owner.expectedPositionOffset == 1)
    #expect(cache.expectedPositionOffset == 1)
    #expect(full.offset == 1)

    for failurePoint in [
        Qwen35FastFailurePoint.afterLayer(1),
        Qwen35FastFailurePoint.afterHead,
    ] {
        #expect(throws: Error.self) {
            _ = try executeQwen35CachedForward(
                cache: owner,
                positionOffset: 1,
                inputLength: 1,
                validateLibraryCachePosition: {
                    try owner.validatePosition(1, cache: cache)
                },
                forward: {
                    try engine.allPositionLogits(
                        MLXArray([Int32(2)], [1, 1]),
                        cache: cache,
                        positionOffset: 1,
                        failurePoint: failurePoint
                    )
                }
            )
        }

        #expect(owner.expectedPositionOffset == 1)
        #expect(cache.expectedPositionOffset == 1)
        let restoredDelta = try #require(
            cache.layers[0].gatedDelta
        )
        #expect(
            qwen35MaximumAbsoluteDifference(
                restoredDelta.convolution,
                beforeConvolution
            ) == 0
        )
        #expect(
            qwen35MaximumAbsoluteDifference(
                restoredDelta.recurrent,
                beforeRecurrent
            ) == 0
        )
        let restoredFull = try #require(
            cache.layers[1].fullAttention
        )
        #expect(restoredFull.offset == 1)
        #expect(restoredFull.state.count == beforeFullState.count)
        for (actual, expected) in zip(
            restoredFull.state,
            beforeFullState
        ) {
            #expect(
                qwen35MaximumAbsoluteDifference(actual, expected) == 0
            )
        }
    }
}

private func tinyQwen35FastEngine() throws -> Qwen35FastEngine {
    let hiddenSize = 24
    let attentionSpec = try Qwen35AttentionSpec(
        hiddenSize: hiddenSize,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDimension: 24,
        rmsNormEps: 1e-6,
        ropeSpec: Qwen35RopeSpec(
            theta: 10_000_000,
            type: "default",
            partialRotaryFactor: 0.25,
            mropeInterleaved: true,
            mropeSection: [1, 1, 1]
        )
    )
    let zeroAttention = Qwen35AttentionWeights(
        queryProjection: Qwen35LinearWeight(
            MLXArray.zeros([
                attentionSpec.queryAndGateProjectionSize,
                hiddenSize,
            ])
        ),
        keyProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        valueProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        outputProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, attentionSpec.querySize])
        ),
        queryNorm: MLXArray.ones([attentionSpec.headDimension]),
        keyNorm: MLXArray.ones([attentionSpec.headDimension])
    )
    let zeroMLP = Qwen35MLPWeights(
        gateProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        ),
        upProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        ),
        downProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        )
    )
    let block = Qwen35BlockWeights(
        inputLayerNorm: MLXArray.ones([hiddenSize]),
        postAttentionLayerNorm: MLXArray.ones([hiddenSize]),
        mixer: .full(
            weights: zeroAttention,
            spec: attentionSpec
        ),
        mlp: zeroMLP
    )
    let trunk = try Qwen35FastTrunk(
        embedding: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: 6,
                columns: hiddenSize,
                scale: 0.02
            )
        ),
        blocks: [block],
        finalNorm: MLXArray.ones([hiddenSize]),
        hiddenSize: hiddenSize,
        rmsNormEps: 1e-6
    )
    return try Qwen35FastEngine(
        trunk: trunk,
        lmHead: Qwen35LMHead(
            weight: Qwen35LinearWeight(
                qwen35DeterministicMatrix(
                    rows: 7,
                    columns: hiddenSize,
                    scale: 0.03
                )
            )
        )
    )
}

private func tinyTransactionalQwen35FastEngine() throws
    -> Qwen35FastEngine
{
    let hiddenSize = 128
    let deltaSpec = try tinyQwen35GatedDeltaSpec()
    let attentionSpec = try Qwen35AttentionSpec(
        hiddenSize: hiddenSize,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        headDimension: 128,
        rmsNormEps: 1e-6,
        ropeSpec: Qwen35RopeSpec(
            theta: 10_000_000,
            type: "default",
            partialRotaryFactor: 0.25,
            mropeInterleaved: true,
            mropeSection: [6, 5, 5]
        )
    )
    let zeroAttention = Qwen35AttentionWeights(
        queryProjection: Qwen35LinearWeight(
            MLXArray.zeros([
                attentionSpec.queryAndGateProjectionSize,
                hiddenSize,
            ])
        ),
        keyProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        valueProjection: Qwen35LinearWeight(
            MLXArray.zeros([attentionSpec.keyValueSize, hiddenSize])
        ),
        outputProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, attentionSpec.querySize])
        ),
        queryNorm: MLXArray.ones([attentionSpec.headDimension]),
        keyNorm: MLXArray.ones([attentionSpec.headDimension])
    )
    let zeroMLP = Qwen35MLPWeights(
        gateProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        ),
        upProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        ),
        downProjection: Qwen35LinearWeight(
            MLXArray.zeros([hiddenSize, hiddenSize])
        )
    )
    let blocks = [
        Qwen35BlockWeights(
            inputLayerNorm: MLXArray.ones([hiddenSize]),
            postAttentionLayerNorm: MLXArray.ones([hiddenSize]),
            mixer: .linear(
                weights: tinyQwen35GatedDeltaWeights(spec: deltaSpec),
                spec: deltaSpec
            ),
            mlp: zeroMLP
        ),
        Qwen35BlockWeights(
            inputLayerNorm: MLXArray.ones([hiddenSize]),
            postAttentionLayerNorm: MLXArray.ones([hiddenSize]),
            mixer: .full(
                weights: zeroAttention,
                spec: attentionSpec
            ),
            mlp: zeroMLP
        ),
    ]
    let trunk = try Qwen35FastTrunk(
        embedding: Qwen35LinearWeight(
            qwen35DeterministicMatrix(
                rows: 6,
                columns: hiddenSize,
                scale: 0.01
            )
        ),
        blocks: blocks,
        finalNorm: MLXArray.ones([hiddenSize]),
        hiddenSize: hiddenSize,
        rmsNormEps: 1e-6
    )
    return try Qwen35FastEngine(
        trunk: trunk,
        lmHead: Qwen35LMHead(
            weight: Qwen35LinearWeight(
                qwen35DeterministicMatrix(
                    rows: 7,
                    columns: hiddenSize,
                    scale: 0.01
                )
            )
        )
    )
}
