import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

@Test
func qwen35PinnedGatedDeltaStateShapesAndDTypeMatchReference() throws {
    let spec = try Qwen35GatedDeltaSpec(
        hiddenSize: 5_120,
        numValueHeads: 48,
        numKeyHeads: 16,
        keyHeadDimension: 128,
        valueHeadDimension: 128,
        convolutionKernelSize: 4,
        rmsNormEps: 1e-6
    )
    #expect(spec.convolutionDimension == 10_240)
    #expect(spec.convolutionStateShape(batchSize: 2) == [2, 3, 10_240])
    #expect(
        spec.recurrentStateShape(batchSize: 2)
            == [2, 48, 128, 128]
    )

    guard qwen35MLXTestsEnabled() else {
        return
    }
    let state = try Qwen35GatedDeltaState(
        batchSize: 2,
        spec: spec,
        activationDType: .bfloat16
    )
    #expect(state.convolution.dtype == .bfloat16)
    #expect(state.recurrent.dtype == .float32)
}

@Test
func qwen35GatedDeltaSpecRejectsDerivedDimensionOverflow() {
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35GatedDeltaSpec(
            hiddenSize: 8,
            numValueHeads: Int.max,
            numKeyHeads: 1,
            keyHeadDimension: 2,
            valueHeadDimension: 2,
            convolutionKernelSize: 4,
            rmsNormEps: 1e-6
        )
    }
}

@Test
func qwen35GatedDeltaRejectsConvolutionStateDTypeMismatch() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }
    let spec = try tinyQwen35GatedDeltaSpec()
    let weights = tinyQwen35GatedDeltaWeights(spec: spec)
    let state = try Qwen35GatedDeltaState(
        batchSize: 1,
        spec: spec,
        activationDType: .bfloat16
    )
    #expect(throws: MLXFastError.self) {
        _ = try Qwen35GatedDeltaNet.forward(
            MLXArray.zeros([1, 1, spec.hiddenSize], dtype: .float32),
            weights: weights,
            spec: spec,
            state: state
        )
    }
}

@Test
func qwen35GatedDeltaPrefillMatchesStepRecurrence() throws {
    guard qwen35MLXTestsEnabled() else {
        return
    }

    let spec = try tinyQwen35GatedDeltaSpec()
    let weights = tinyQwen35GatedDeltaWeights(spec: spec)
    let input = qwen35DeterministicMatrix(
        rows: 3,
        columns: spec.hiddenSize,
        scale: 0.01
    ).reshaped(1, 3, spec.hiddenSize)

    let prefillState = try Qwen35GatedDeltaState(
        batchSize: 1,
        spec: spec,
        activationDType: input.dtype
    )
    let prefill = try Qwen35GatedDeltaNet.forward(
        input,
        weights: weights,
        spec: spec,
        state: prefillState
    )

    let stepState = try Qwen35GatedDeltaState(
        batchSize: 1,
        spec: spec,
        activationDType: input.dtype
    )
    var stepOutputs: [MLXArray] = []
    for position in 0..<3 {
        stepOutputs.append(
            try Qwen35GatedDeltaNet.forward(
                input[
                    0...,
                    position..<(position + 1),
                    0...
                ],
                weights: weights,
                spec: spec,
                state: stepState
            )
        )
    }
    let stepped = concatenated(stepOutputs, axis: 1)
    let outputGap = qwen35MaximumAbsoluteDifference(prefill, stepped)
    let convolutionGap = qwen35MaximumAbsoluteDifference(
        prefillState.convolution,
        stepState.convolution
    )
    let recurrentGap = qwen35MaximumAbsoluteDifference(
        prefillState.recurrent,
        stepState.recurrent
    )

    #expect(prefill.shape == [1, 3, spec.hiddenSize])
    #expect(prefillState.convolution.shape == [1, 3, 384])
    #expect(prefillState.recurrent.shape == [1, 1, 128, 128])
    #expect(prefillState.recurrent.dtype == .float32)
    #expect(stepState.recurrent.dtype == .float32)
    #expect(outputGap < 2e-4)
    #expect(convolutionGap < 1e-6)
    #expect(recurrentGap < 2e-4)
}

func tinyQwen35GatedDeltaSpec() throws -> Qwen35GatedDeltaSpec {
    try Qwen35GatedDeltaSpec(
        hiddenSize: 128,
        numValueHeads: 1,
        numKeyHeads: 1,
        keyHeadDimension: 128,
        valueHeadDimension: 128,
        convolutionKernelSize: 4,
        rmsNormEps: 1e-6
    )
}

func tinyQwen35GatedDeltaWeights(
    spec: Qwen35GatedDeltaSpec
) -> Qwen35GatedDeltaWeights {
    Qwen35GatedDeltaWeights(
        inputQKVProjection: Qwen35LinearWeight(
            qwen35RepeatedIdentity(
                repeats: 3,
                dimensions: spec.hiddenSize,
                scale: 0.25
            )
        ),
        inputZProjection: Qwen35LinearWeight(
            qwen35Identity(
                rows: spec.valueSize,
                columns: spec.hiddenSize,
                scale: 0.5
            )
        ),
        inputBProjection: Qwen35LinearWeight(
            MLXArray.zeros([spec.numValueHeads, spec.hiddenSize])
        ),
        inputAProjection: Qwen35LinearWeight(
            MLXArray.zeros([spec.numValueHeads, spec.hiddenSize])
        ),
        convolution: qwen35DepthwiseCurrentTokenKernel(
            channels: spec.convolutionDimension,
            kernelSize: spec.convolutionKernelSize
        ),
        timeStepBias: MLXArray.zeros([spec.numValueHeads]),
        aLog: MLXArray.zeros([spec.numValueHeads]),
        outputNorm: MLXArray.ones([spec.valueHeadDimension]),
        outputProjection: Qwen35LinearWeight(
            qwen35Identity(
                rows: spec.hiddenSize,
                columns: spec.valueSize,
                scale: 0.5
            )
        )
    )
}
