@testable import MLXFastHarness
@testable import MLXFastModel
import Testing

@Test
func qwen35WorkerDiscardsAffectedSessionAfterCaughtRequestFailure() {
    var state = RuntimeWorkerState(
        correctnessCache: Qwen35ModelCache(layerTypes: [.full]),
        correctnessPromptTokenCount: 8,
        correctnessStep: 2,
        decodeCache: Qwen35ModelCache(layerTypes: [.full]),
        decodeSeedTokenCount: 16,
        decodeStep: 3
    )

    #expect(throws: Qwen35WorkerStateTestError.self) {
        let _: Int = try executeRuntimeWorkerRequest(
            kind: "decode_step",
            state: &state
        ) { requestState in
            requestState.decodeStep = 4
            throw Qwen35WorkerStateTestError.injected
        }
    }
    #expect(state.decodeCache == nil)
    #expect(state.decodeSeedTokenCount == 0)
    #expect(state.decodeStep == 0)
    #expect(state.correctnessCache != nil)

    #expect(throws: Qwen35WorkerStateTestError.self) {
        let _: Int = try executeRuntimeWorkerRequest(
            kind: "correctness_step",
            state: &state
        ) { _ in
            throw Qwen35WorkerStateTestError.injected
        }
    }
    #expect(state.correctnessCache == nil)
    #expect(state.correctnessPromptTokenCount == 0)
    #expect(state.correctnessStep == 0)

    state.correctnessCache = Qwen35ModelCache(layerTypes: [.full])
    state.decodeCache = Qwen35ModelCache(layerTypes: [.full])
    state.discardSession(afterFailedRequest: "malformed")
    #expect(state.correctnessCache == nil)
    #expect(state.decodeCache == nil)
}

private enum Qwen35WorkerStateTestError: Error {
    case injected
}
