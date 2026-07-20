import Foundation
import MLXFastCore
import Tokenizers

// GemmaRuntime is split across GemmaRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension GemmaRuntime {
    static let semanticGPQAMaxAnswerDocumentByteCount = 1 * 1024 * 1024

    struct SemanticGPQAAnswerDocument: Encodable {
        let version: Int
        let cases: [SemanticGPQAAnswerCase]
    }

    struct SemanticGPQAAnswerCase: Encodable {
        let id: String
        let domain: String?
        let subdomain: String?
        let prompt: String
        let answerKey: String?
        let referenceAnswer: String
        let candidateAnswer: String
        let candidateTokens: [Int]
        let maxNewTokens: Int

        enum CodingKeys: String, CodingKey {
            case id
            case domain
            case subdomain
            case prompt
            case answerKey = "answer_key"
            case referenceAnswer = "reference_answer"
            case candidateAnswer = "candidate_answer"
            case candidateTokens = "candidate_tokens"
            case maxNewTokens = "max_new_tokens"
        }
    }

    struct PendingSemanticGPQAAnswer {
        let id: String
        let domain: String?
        let subdomain: String?
        let prompt: String
        let answerKey: String?
        let referenceAnswer: String
        let candidateTokens: [Int]
        let maxNewTokens: Int
    }

    static func pendingSemanticAnswerCase(
        behavior: GoldenBehaviorCase,
        generatedTokens: [Int],
        maxNewTokens: Int
    ) throws -> PendingSemanticGPQAAnswer? {
        guard let prompt = trimmedNonEmpty(behavior.semanticPrompt),
              let referenceAnswer = trimmedNonEmpty(
                  behavior.semanticReferenceAnswer
              )
        else {
            return nil
        }
        let candidateTokens = Array(generatedTokens.prefix(maxNewTokens))
        guard !candidateTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(behavior.name) semantic GPQA candidate token list is empty"
            )
        }
        return PendingSemanticGPQAAnswer(
            id: behavior.name,
            domain: trimmedNonEmpty(behavior.semanticDomain),
            subdomain: trimmedNonEmpty(behavior.semanticSubdomain),
            prompt: prompt,
            answerKey: trimmedNonEmpty(behavior.semanticAnswerKey),
            referenceAnswer: referenceAnswer,
            candidateTokens: candidateTokens,
            maxNewTokens: maxNewTokens
        )
    }

    static func semanticAnswerCase(
        pending: PendingSemanticGPQAAnswer,
        tokenizer: any Tokenizer
    ) -> SemanticGPQAAnswerCase {
        let candidateAnswer = tokenizer.decode(
            tokens: pending.candidateTokens,
            skipSpecialTokens: true
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        return SemanticGPQAAnswerCase(
            id: pending.id,
            domain: pending.domain,
            subdomain: pending.subdomain,
            prompt: pending.prompt,
            answerKey: pending.answerKey,
            referenceAnswer: pending.referenceAnswer,
            candidateAnswer: candidateAnswer,
            candidateTokens: pending.candidateTokens,
            maxNewTokens: pending.maxNewTokens
        )
    }

    static func semanticAnswerCase(
        behavior: GoldenBehaviorCase,
        generatedTokens: [Int],
        tokenizer: any Tokenizer,
        maxNewTokens: Int
    ) throws -> SemanticGPQAAnswerCase? {
        guard let pending = try pendingSemanticAnswerCase(
            behavior: behavior,
            generatedTokens: generatedTokens,
            maxNewTokens: maxNewTokens
        ) else {
            return nil
        }
        return semanticAnswerCase(pending: pending, tokenizer: tokenizer)
    }

    static func materializeSemanticGPQAAnswers(
        _ pendingAnswers: [PendingSemanticGPQAAnswer],
        tokenizerPath: String
    ) throws -> [SemanticGPQAAnswerCase] {
        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        return pendingAnswers.map {
            semanticAnswerCase(pending: $0, tokenizer: tokenizer)
        }
    }

    static func writeSemanticGPQAAnswers(
        _ answers: [SemanticGPQAAnswerCase],
        to path: String,
        permissions: Int = 0o600
    ) throws {
        guard permissions == 0o600 else {
            throw MLXFastError.invalidInput(
                "semantic GPQA answer output permissions must be 0600"
            )
        }
        let data = try semanticGPQAAnswerDocumentData(answers)
        try PrivateFileWriter.writeAtomically(
            data,
            to: path,
            maximumByteCount: semanticGPQAMaxAnswerDocumentByteCount
        )
    }

    static func semanticGPQAAnswerDocumentData(
        _ answers: [SemanticGPQAAnswerCase]
    ) throws -> Data {
        let document = SemanticGPQAAnswerDocument(version: 1, cases: answers)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(document)
        guard data.count <= semanticGPQAMaxAnswerDocumentByteCount else {
            throw MLXFastError.invalidInput(
                "semantic GPQA answer document exceeds "
                    + "\(semanticGPQAMaxAnswerDocumentByteCount) bytes"
            )
        }
        return data
    }

}
