import Foundation

public struct GoldenCase: Codable, Equatable {
    public let name: String
    public let promptTokens: [Int]
    public let expectedTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
        case expectedTokens = "expected_tokens"
    }

    public init(name: String, promptTokens: [Int], expectedTokens: [Int]) {
        self.name = name
        self.promptTokens = promptTokens
        self.expectedTokens = expectedTokens
    }
}

private struct GoldenFile: Decodable {
    let version: Int?
    let cases: [GoldenCase]
}

public func loadGoldenCases(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps
) throws -> [GoldenCase] {
    try requireFile(path, description: "correctness golden file")

    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decoded = try JSONDecoder().decode(GoldenFile.self, from: data)
    guard !decoded.cases.isEmpty else {
        throw MLXFastError.invalidInput("correctness golden file must contain at least one case")
    }

    for testCase in decoded.cases {
        if testCase.promptTokens.isEmpty {
            throw MLXFastError.invalidInput("\(testCase.name).prompt_tokens must not be empty")
        }
        if testCase.expectedTokens.count < requiredSteps {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need at least \(requiredSteps)"
            )
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
        try validateTokens(testCase.expectedTokens, field: "\(testCase.name).expected_tokens")
    }

    return decoded.cases
}

private func validateTokens(_ tokens: [Int], field: String) throws {
    for (index, token) in tokens.enumerated() {
        if token < 0 || token >= MLXFastConstants.vocabSize {
            throw MLXFastError.invalidInput(
                "\(field)[\(index)]=\(token) is outside DeepSeek vocab range 0..<\(MLXFastConstants.vocabSize)"
            )
        }
    }
}
