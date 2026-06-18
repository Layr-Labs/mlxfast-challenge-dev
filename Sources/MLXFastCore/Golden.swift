import CryptoKit
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

public struct GoldenFixture: Equatable {
    public let cases: [GoldenCase]
    public let sha256: String

    public init(cases: [GoldenCase], sha256: String) {
        self.cases = cases
        self.sha256 = sha256
    }
}

public func loadGoldenCases(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps
) throws -> [GoldenCase] {
    try loadGoldenFixture(from: path, requiredSteps: requiredSteps).cases
}

public func loadGoldenFixture(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps
) throws -> GoldenFixture {
    guard requiredSteps > 0 else {
        throw MLXFastError.invalidInput("correctness required steps must be positive")
    }
    try requireFile(path, description: "correctness golden file")

    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decoded = try JSONDecoder().decode(GoldenFile.self, from: data)
    guard decoded.version == 1 else {
        throw MLXFastError.invalidInput("correctness golden file version must be 1")
    }
    guard !decoded.cases.isEmpty else {
        throw MLXFastError.invalidInput("correctness golden file must contain at least one case")
    }

    var names = Set<String>()
    for testCase in decoded.cases {
        let caseNameDescription = String(reflecting: testCase.name)
        let trimmedName = testCase.name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedName.isEmpty {
            throw MLXFastError.invalidInput("correctness golden case name must not be empty")
        }
        if testCase.name != trimmedName {
            throw MLXFastError.invalidInput(
                "correctness golden case name \(caseNameDescription) must not have leading or trailing whitespace"
            )
        }
        if testCase.name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) {
            throw MLXFastError.invalidInput(
                "correctness golden case name \(caseNameDescription) must not contain control characters"
            )
        }
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness golden case name \(testCase.name)")
        }
        if testCase.promptTokens.isEmpty {
            throw MLXFastError.invalidInput("\(testCase.name).prompt_tokens must not be empty")
        }
        if testCase.expectedTokens.count != requiredSteps {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need exactly \(requiredSteps)"
            )
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
        try validateTokens(testCase.expectedTokens, field: "\(testCase.name).expected_tokens")
    }

    let digest = SHA256.hash(data: data)
    let hash = digest.map { String(format: "%02x", $0) }.joined()
    return GoldenFixture(cases: decoded.cases, sha256: hash)
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
