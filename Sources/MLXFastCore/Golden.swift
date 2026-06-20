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

public struct BenchmarkGolden: Codable, Equatable {
    public let name: String
    public let prefillPromptTokens: [Int]
    public let expectedPrefillToken: Int
    public let decodeSeedTokens: [Int]
    public let expectedDecodeSeedToken: Int
    public let expectedDecodeTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case prefillPromptTokens = "prefill_prompt_tokens"
        case expectedPrefillToken = "expected_prefill_token"
        case decodeSeedTokens = "decode_seed_tokens"
        case expectedDecodeSeedToken = "expected_decode_seed_token"
        case expectedDecodeTokens = "expected_decode_tokens"
    }

    public init(
        name: String,
        prefillPromptTokens: [Int],
        expectedPrefillToken: Int,
        decodeSeedTokens: [Int],
        expectedDecodeSeedToken: Int,
        expectedDecodeTokens: [Int]
    ) {
        self.name = name
        self.prefillPromptTokens = prefillPromptTokens
        self.expectedPrefillToken = expectedPrefillToken
        self.decodeSeedTokens = decodeSeedTokens
        self.expectedDecodeSeedToken = expectedDecodeSeedToken
        self.expectedDecodeTokens = expectedDecodeTokens
    }
}

public struct GoldenFixturePayload: Codable, Equatable {
    public let version: Int
    public let cases: [GoldenCase]
    public let benchmark: BenchmarkGolden

    public init(version: Int = 1, cases: [GoldenCase], benchmark: BenchmarkGolden) {
        self.version = version
        self.cases = cases
        self.benchmark = benchmark
    }
}

private struct GoldenFile: Decodable {
    let version: Int?
    let cases: [GoldenCase]
    let benchmark: BenchmarkGolden?
}

public struct GoldenFixture: Equatable {
    public let cases: [GoldenCase]
    public let benchmark: BenchmarkGolden?
    public let sha256: String

    public init(cases: [GoldenCase], benchmark: BenchmarkGolden?, sha256: String) {
        self.cases = cases
        self.benchmark = benchmark
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
        try validateName(testCase.name, field: "correctness golden case name")
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
    if let benchmark = decoded.benchmark {
        try validateBenchmarkGolden(benchmark)
    }

    let digest = SHA256.hash(data: data)
    let hash = digest.map { String(format: "%02x", $0) }.joined()
    return GoldenFixture(cases: decoded.cases, benchmark: decoded.benchmark, sha256: hash)
}

private func validateBenchmarkGolden(_ benchmark: BenchmarkGolden) throws {
    try validateName(benchmark.name, field: "benchmark golden name")
    if benchmark.prefillPromptTokens.count != MLXFastConstants.benchmarkPrefillPromptTokens {
        throw MLXFastError.invalidInput(
            "\(benchmark.name).benchmark.prefill_prompt_tokens has \(benchmark.prefillPromptTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkPrefillPromptTokens)"
        )
    }
    if benchmark.decodeSeedTokens.count != MLXFastConstants.benchmarkDecodeSeedTokens {
        throw MLXFastError.invalidInput(
            "\(benchmark.name).benchmark.decode_seed_tokens has \(benchmark.decodeSeedTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkDecodeSeedTokens)"
        )
    }
    if benchmark.expectedDecodeTokens.count != MLXFastConstants.benchmarkDecodeSteps {
        throw MLXFastError.invalidInput(
            "\(benchmark.name).benchmark.expected_decode_tokens has \(benchmark.expectedDecodeTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkDecodeSteps)"
        )
    }
    try validateTokens(benchmark.prefillPromptTokens, field: "\(benchmark.name).benchmark.prefill_prompt_tokens")
    try validateToken(benchmark.expectedPrefillToken, field: "\(benchmark.name).benchmark.expected_prefill_token")
    try validateTokens(benchmark.decodeSeedTokens, field: "\(benchmark.name).benchmark.decode_seed_tokens")
    try validateToken(benchmark.expectedDecodeSeedToken, field: "\(benchmark.name).benchmark.expected_decode_seed_token")
    try validateTokens(benchmark.expectedDecodeTokens, field: "\(benchmark.name).benchmark.expected_decode_tokens")
}

private func validateName(_ name: String, field: String) throws {
    let nameDescription = String(reflecting: name)
    let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmedName.isEmpty {
        throw MLXFastError.invalidInput("\(field) must not be empty")
    }
    if name != trimmedName {
        throw MLXFastError.invalidInput(
            "\(field) \(nameDescription) must not have leading or trailing whitespace"
        )
    }
    if name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) {
        throw MLXFastError.invalidInput(
            "\(field) \(nameDescription) must not contain control characters"
        )
    }
}

private func validateTokens(_ tokens: [Int], field: String) throws {
    for (index, token) in tokens.enumerated() {
        try validateToken(token, field: "\(field)[\(index)]")
    }
}

private func validateToken(_ token: Int, field: String) throws {
    if token < 0 || token >= MLXFastConstants.vocabSize {
        throw MLXFastError.invalidInput(
            "\(field)=\(token) is outside DeepSeek vocab range 0..<\(MLXFastConstants.vocabSize)"
        )
    }
}
