import Foundation
import MLX
import MLXFastCore

public struct CorrectnessTokenComparison: Equatable {
    public let passed: Bool
    public let checkedSteps: Int
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?

    public init(
        passed: Bool,
        checkedSteps: Int,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?
    ) {
        self.passed = passed
        self.checkedSteps = checkedSteps
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
    }
}

public enum DeepSeekCorrectness {
    public static func generateGreedyNoCache(
        promptTokens: [Int],
        steps: Int = MLXFastConstants.correctnessSteps,
        nextToken: (_ contextTokens: [Int]) throws -> Int
    ) throws -> [Int] {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard steps >= 0 else {
            throw MLXFastError.invalidInput("greedy correctness steps must be non-negative")
        }

        var context = promptTokens
        var generated: [Int] = []
        generated.reserveCapacity(steps)
        for _ in 0..<steps {
            let token = try nextToken(context)
            generated.append(token)
            context.append(token)
        }
        return generated
    }

    public static func greedyToken(from logits: MLXArray) throws -> Int {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("greedy logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        let last = rows[-1]
        return Int(last.argMax().item(Int32.self))
    }

    public static func compareTokens(
        expected: [Int],
        actual: [Int],
        steps: Int = MLXFastConstants.correctnessSteps
    ) -> CorrectnessTokenComparison {
        let count = min(steps, expected.count)
        for step in 0..<count {
            let actualToken = step < actual.count ? actual[step] : nil
            if actualToken != expected[step] {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expected[step],
                    actualToken: actualToken
                )
            }
        }

        if actual.count < count {
            return CorrectnessTokenComparison(
                passed: false,
                checkedSteps: actual.count + 1,
                firstFailingStep: actual.count,
                expectedToken: expected[actual.count],
                actualToken: nil
            )
        }

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: count,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }
}
