import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import Tokenizers

// A single model-step surface shared by the in-process and worker measurement
// paths. The orchestration (timing, correctness, comparisons) is written once
// against this protocol; the only thing that differs between the two paths is
// how a step is executed:
//   - InProcessInferenceSession calls DeepSeekModel.logits(...) directly.
//   - RuntimeWorkerClient sends the same request to the sandboxed subprocess.
// The method set is exactly the worker message set, so RuntimeWorkerClient
// already conforms; the in-process implementation below is lifted verbatim from
// what handleWorkerRequest used to do per message, so the worker subprocess and
// the in-process path now share one copy of the step logic.
protocol RuntimeInferenceSession: AnyObject {
    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse
    func beginTeacherForcedCorrectness(promptTokens: [Int]) throws -> RuntimeWorkerResponse
    func teacherForcedCorrectnessBatch(
        promptTokens: [Int],
        expectedTokens: [Int],
        steps: Int
    ) throws -> RuntimeWorkerResponse
    func teacherForcedCorrectnessStep(previousToken: Int) throws -> RuntimeWorkerResponse
    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse
    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse
    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse
}

// RuntimeWorkerClient already exposes exactly these methods returning
// RuntimeWorkerResponse, so conformance is free.
extension RuntimeWorkerClient: RuntimeInferenceSession {}

// The in-process backend. Holds the weight cache plus the same per-session
// state the worker kept in RuntimeWorkerState. Each method body is the logic
// that previously lived in the matching handleWorkerRequest case.
final class InProcessInferenceSession: RuntimeInferenceSession {
    let weightCache: DeepSeekRuntimeWeightCache

    private var correctnessCache: DeepSeekModelCache?
    private var correctnessPromptTokenCount = 0
    private var correctnessStep = 0
    private var decodeCache: DeepSeekModelCache?
    private var decodeSeedTokenCount = 0
    private var decodeStep = 0

    init(weightCache: DeepSeekRuntimeWeightCache) {
        self.weightCache = weightCache
    }

    private func response(
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        topLogitRows: [[CorrectnessTraceLogit]]? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        seconds: Double? = nil
    ) -> RuntimeWorkerResponse {
        RuntimeWorkerResponse(
            id: 0,
            nonce: nil,
            ok: true,
            token: token,
            topLogits: topLogits,
            topLogitRows: topLogitRows,
            seedToken: seedToken,
            tokens: tokens,
            seconds: seconds,
            expertStats: DeepSeekRuntime.expertStats(from: weightCache),
            peakRamGB: DeepSeekRuntime.currentResidentMemoryGB()
        )
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        let tokens = try DeepSeekRuntime.generateGreedyCached(
            promptTokens: promptTokens,
            steps: steps,
            weightCache: weightCache
        )
        return response(tokens: tokens)
    }

    func beginTeacherForcedCorrectness(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        let cache = DeepSeekModelCache(config: weightCache.config)
        let start = DispatchTime.now().uptimeNanoseconds
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray(promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        let token = try DeepSeekCorrectness.greedyToken(from: logits)
        correctnessCache = cache
        correctnessPromptTokenCount = promptTokens.count
        correctnessStep = 0
        let elapsed = DeepSeekRuntime.secondsSince(start)
        return response(
            token: token,
            topLogits: try DeepSeekRuntime.topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
            seconds: elapsed
        )
    }

    func teacherForcedCorrectnessBatch(
        promptTokens: [Int],
        expectedTokens: [Int],
        steps: Int
    ) throws -> RuntimeWorkerResponse {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("batched teacher-forced prompt_tokens must not be empty")
        }
        guard steps > 0 else {
            throw MLXFastError.invalidInput("batched teacher-forced steps must be positive")
        }
        guard expectedTokens.count >= steps else {
            throw MLXFastError.invalidInput(
                "batched teacher-forced expected_tokens has \(expectedTokens.count) tokens; expected at least \(steps)"
            )
        }
        let teacherForcedInput = promptTokens + Array(expectedTokens.prefix(max(steps - 1, 0)))
        let cache = DeepSeekModelCache(config: weightCache.config)
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray(teacherForcedInput),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        var tokens: [Int] = []
        var topLogitRows: [[CorrectnessTraceLogit]] = []
        tokens.reserveCapacity(steps)
        topLogitRows.reserveCapacity(steps)
        let firstLogitRow = promptTokens.count - 1
        for step in 0..<steps {
            let rowLogits = try DeepSeekRuntime.topLogits(
                from: logits,
                row: firstLogitRow + step,
                topK: MLXFastConstants.correctnessTopLogits
            )
            guard let token = rowLogits.first?.token else {
                throw MLXFastError.invalidInput("batched teacher-forced top logits missing token")
            }
            tokens.append(token)
            topLogitRows.append(rowLogits)
        }
        return response(topLogitRows: topLogitRows, tokens: tokens)
    }

    func teacherForcedCorrectnessStep(previousToken: Int) throws -> RuntimeWorkerResponse {
        guard let cache = correctnessCache else {
            throw MLXFastError.invalidInput("teacher-forced correctness step before begin")
        }
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray([previousToken]),
            weightCache: weightCache,
            cache: cache,
            positionOffset: correctnessPromptTokenCount + correctnessStep
        )
        let token = try DeepSeekCorrectness.greedyToken(from: logits)
        correctnessStep += 1
        return response(
            token: token,
            topLogits: try DeepSeekRuntime.topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        let cache = DeepSeekModelCache(config: weightCache.config)
        let start = DispatchTime.now().uptimeNanoseconds
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray(promptTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        eval(logits)
        let token = try DeepSeekCorrectness.greedyToken(from: logits)
        let elapsed = DeepSeekRuntime.secondsSince(start)
        Memory.clearCache()
        return response(token: token, seconds: elapsed)
    }

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        let warmupCache = DeepSeekModelCache(config: weightCache.config)
        let warmupLogits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: warmupCache,
            positionOffset: 0
        )
        _ = try DeepSeekCorrectness.greedyToken(from: warmupLogits)
        Memory.clearCache()

        let cache = DeepSeekModelCache(config: weightCache.config)
        let start = DispatchTime.now().uptimeNanoseconds
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray(seedTokens),
            weightCache: weightCache,
            cache: cache,
            positionOffset: 0
        )
        let token = try DeepSeekCorrectness.greedyToken(from: logits)
        cache.materializeCachedState()
        decodeCache = cache
        decodeSeedTokenCount = seedTokens.count
        decodeStep = 0
        let elapsed = DeepSeekRuntime.secondsSince(start)
        return response(seedToken: token, seconds: elapsed)
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        guard let cache = decodeCache else {
            throw MLXFastError.invalidInput("decode_step before decode_begin")
        }
        let validationDelayMS = try DeepSeekRuntime.submissionValidationDelayMilliseconds()
        let start = DispatchTime.now().uptimeNanoseconds
        let logits = try DeepSeekModel.logits(
            inputIDs: DeepSeekRuntime.inputIDsArray([inputToken]),
            weightCache: weightCache,
            cache: cache,
            positionOffset: decodeSeedTokenCount + decodeStep
        )
        let token = try DeepSeekCorrectness.greedyToken(from: logits)
        if validationDelayMS > 0 {
            Thread.sleep(forTimeInterval: Double(validationDelayMS) / 1_000.0)
        }
        let elapsed = DeepSeekRuntime.secondsSince(start)
        decodeStep += 1
        return response(token: token, seconds: elapsed)
    }
}
