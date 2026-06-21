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
    public let prefillPromptTokens: [Int]
    public let expectedPrefillToken: Int
    public let decodeSeedTokens: [Int]
    public let expectedDecodeSeedToken: Int
    public let expectedDecodeTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case prefillPromptTokens = "prefill_prompt_tokens"
        case expectedPrefillToken = "expected_prefill_token"
        case decodeSeedTokens = "decode_seed_tokens"
        case expectedDecodeSeedToken = "expected_decode_seed_token"
        case expectedDecodeTokens = "expected_decode_tokens"
    }

    public init(
        prefillPromptTokens: [Int],
        expectedPrefillToken: Int,
        decodeSeedTokens: [Int],
        expectedDecodeSeedToken: Int,
        expectedDecodeTokens: [Int]
    ) {
        self.prefillPromptTokens = prefillPromptTokens
        self.expectedPrefillToken = expectedPrefillToken
        self.decodeSeedTokens = decodeSeedTokens
        self.expectedDecodeSeedToken = expectedDecodeSeedToken
        self.expectedDecodeTokens = expectedDecodeTokens
    }
}

public struct GoldenDocument: Codable, Equatable {
    public let version: Int?
    public let cases: [GoldenCase]
    public let benchmark: BenchmarkGolden?

    public init(version: Int = 1, cases: [GoldenCase], benchmark: BenchmarkGolden?) {
        self.version = version
        self.cases = cases
        self.benchmark = benchmark
    }
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

public struct GoldenPromptCase: Codable, Equatable {
    public let name: String
    public let promptTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
    }

    public init(name: String, promptTokens: [Int]) {
        self.name = name
        self.promptTokens = promptTokens
    }
}

public struct BenchmarkPromptSpec: Codable, Equatable {
    public let name: String?
    public let promptTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
    }

    public init(name: String? = nil, promptTokens: [Int]) {
        self.name = name
        self.promptTokens = promptTokens
    }
}

public struct GoldenPromptManifest: Codable, Equatable {
    public let version: Int
    public let cases: [GoldenPromptCase]
    public let benchmark: BenchmarkPromptSpec
    public let maxOutputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case version
        case cases
        case benchmark
        case maxOutputTokens = "max_output_tokens"
    }

    public init(
        version: Int = 1,
        cases: [GoldenPromptCase],
        benchmark: BenchmarkPromptSpec,
        maxOutputTokens: Int? = nil
    ) {
        self.version = version
        self.cases = cases
        self.benchmark = benchmark
        self.maxOutputTokens = maxOutputTokens
    }
}

public struct BenchmarkTokenComparison: Equatable {
    public let passed: Bool
    public let label: String
    public let step: Int?
    public let expectedToken: Int?
    public let actualToken: Int?

    public init(
        passed: Bool,
        label: String,
        step: Int?,
        expectedToken: Int?,
        actualToken: Int?
    ) {
        self.passed = passed
        self.label = label
        self.step = step
        self.expectedToken = expectedToken
        self.actualToken = actualToken
    }
}

public func loadGoldenCases(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps,
    requiredPromptTokens: Int = MLXFastConstants.correctnessPromptTokens
) throws -> [GoldenCase] {
    try loadGoldenFixture(
        from: path,
        requiredSteps: requiredSteps,
        requiredPromptTokens: requiredPromptTokens
    ).cases
}

public func loadGoldenFixture(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps,
    requiredPromptTokens: Int = MLXFastConstants.correctnessPromptTokens
) throws -> GoldenFixture {
    guard requiredSteps > 0 else {
        throw MLXFastError.invalidInput("correctness required steps must be positive")
    }
    guard requiredPromptTokens > 0 else {
        throw MLXFastError.invalidInput("correctness required prompt tokens must be positive")
    }
    try requireFile(path, description: "correctness golden file")

    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decoded = try JSONDecoder().decode(GoldenDocument.self, from: data)
    guard decoded.version == 1 else {
        throw MLXFastError.invalidInput("correctness golden file version must be 1")
    }
    try validateGoldenCases(
        decoded.cases,
        requiredSteps: requiredSteps,
        requiredPromptTokens: requiredPromptTokens
    )
    if let benchmark = decoded.benchmark {
        try validateBenchmarkGolden(benchmark)
    }


    let digest = SHA256.hash(data: data)
    let hash = digest.map { String(format: "%02x", $0) }.joined()
    return GoldenFixture(cases: decoded.cases, benchmark: decoded.benchmark, sha256: hash)
}

public func loadGoldenPromptManifest(from path: String) throws -> GoldenPromptManifest {
    try requireFile(path, description: "golden prompt manifest")
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let manifest = try JSONDecoder().decode(GoldenPromptManifest.self, from: data)
    try validateGoldenPromptManifest(manifest)
    return manifest
}

public func validateGoldenPromptManifest(_ manifest: GoldenPromptManifest) throws {
    guard manifest.version == 1 else {
        throw MLXFastError.invalidInput("golden prompt manifest version must be 1")
    }
    guard !manifest.cases.isEmpty else {
        throw MLXFastError.invalidInput("golden prompt manifest must contain at least one case")
    }
    if let maxOutputTokens = manifest.maxOutputTokens {
        guard maxOutputTokens == MLXFastConstants.correctnessSteps else {
            throw MLXFastError.invalidInput(
                "golden prompt manifest max_output_tokens is \(maxOutputTokens); need exactly \(MLXFastConstants.correctnessSteps)"
            )
        }
    }

    var names = Set<String>()
    for testCase in manifest.cases {
        try validateCaseName(testCase.name, field: "golden prompt case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate golden prompt case name \(testCase.name)")
        }
        guard testCase.promptTokens.count == MLXFastConstants.correctnessPromptTokens else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).prompt_tokens has \(testCase.promptTokens.count) tokens; need exactly \(MLXFastConstants.correctnessPromptTokens)"
            )
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
    }

    if let name = manifest.benchmark.name {
        try validateCaseName(name, field: "benchmark prompt name")
    }
    let benchmarkPrompt = manifest.benchmark.promptTokens
    guard benchmarkPrompt.count >= MLXFastConstants.benchmarkPrefillPromptTokens else {
        throw MLXFastError.invalidInput(
            "benchmark.prompt_tokens has \(benchmarkPrompt.count) tokens; need at least \(MLXFastConstants.benchmarkPrefillPromptTokens)"
        )
    }
    try validateTokens(benchmarkPrompt, field: "benchmark.prompt_tokens")
}

public enum BenchmarkOutputValidator {
    public static func comparePrefillToken(
        expected: BenchmarkGolden,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        comparePrefillToken(
            expectedToken: expected.expectedPrefillToken,
            actualToken: actualToken
        )
    }

    public static func comparePrefillToken(
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareOne(
            label: "benchmark prefill token",
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeSeedToken(
        expected: BenchmarkGolden,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareDecodeSeedToken(
            expectedToken: expected.expectedDecodeSeedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeSeedToken(
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareOne(
            label: "benchmark decode seed token",
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeTokens(
        expected: BenchmarkGolden,
        actualTokens: [Int]
    ) -> BenchmarkTokenComparison {
        compareDecodeTokens(
            expectedTokens: expected.expectedDecodeTokens,
            actualTokens: actualTokens
        )
    }

    public static func compareDecodeTokens(
        expectedTokens: [Int],
        actualTokens: [Int]
    ) -> BenchmarkTokenComparison {
        let steps = max(expectedTokens.count, actualTokens.count)
        for step in 0..<steps {
            let expectedToken = step < expectedTokens.count ? expectedTokens[step] : nil
            let actualToken = step < actualTokens.count ? actualTokens[step] : nil
            if expectedToken != actualToken {
                return BenchmarkTokenComparison(
                    passed: false,
                    label: "benchmark decode token",
                    step: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
        }
        return BenchmarkTokenComparison(
            passed: true,
            label: "benchmark decode token",
            step: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    private static func compareOne(
        label: String,
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        if expectedToken == actualToken {
            return BenchmarkTokenComparison(
                passed: true,
                label: label,
                step: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return BenchmarkTokenComparison(
            passed: false,
            label: label,
            step: nil,
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }
}

public func validateBenchmarkGolden(_ benchmark: BenchmarkGolden) throws {
    guard benchmark.prefillPromptTokens.count == MLXFastConstants.benchmarkPrefillPromptTokens else {
        throw MLXFastError.invalidInput(
            "benchmark.prefill_prompt_tokens has \(benchmark.prefillPromptTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkPrefillPromptTokens)"
        )
    }
    guard benchmark.decodeSeedTokens.count == MLXFastConstants.benchmarkDecodeSeedTokens else {
        throw MLXFastError.invalidInput(
            "benchmark.decode_seed_tokens has \(benchmark.decodeSeedTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkDecodeSeedTokens)"
        )
    }
    guard benchmark.expectedDecodeTokens.count == MLXFastConstants.benchmarkDecodeSteps else {
        throw MLXFastError.invalidInput(
            "benchmark.expected_decode_tokens has \(benchmark.expectedDecodeTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkDecodeSteps)"
        )
    }
    try validateTokens(benchmark.prefillPromptTokens, field: "benchmark.prefill_prompt_tokens")
    try validateTokens([benchmark.expectedPrefillToken], field: "benchmark.expected_prefill_token")
    try validateTokens(benchmark.decodeSeedTokens, field: "benchmark.decode_seed_tokens")
    try validateTokens([benchmark.expectedDecodeSeedToken], field: "benchmark.expected_decode_seed_token")
    try validateTokens(benchmark.expectedDecodeTokens, field: "benchmark.expected_decode_tokens")
}

private func validateGoldenCases(
    _ cases: [GoldenCase],
    requiredSteps: Int,
    requiredPromptTokens: Int
) throws {
    guard !cases.isEmpty else {
        throw MLXFastError.invalidInput("correctness golden file must contain at least one case")
    }

    var names = Set<String>()
    for testCase in cases {
        try validateCaseName(testCase.name, field: "correctness golden case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness golden case name \(testCase.name)")
        }
        if testCase.promptTokens.count != requiredPromptTokens {
            throw MLXFastError.invalidInput(
                "\(testCase.name).prompt_tokens has \(testCase.promptTokens.count) tokens; need exactly \(requiredPromptTokens)"
            )
        }
        if testCase.expectedTokens.count != requiredSteps {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need exactly \(requiredSteps)"
            )
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
        try validateTokens(testCase.expectedTokens, field: "\(testCase.name).expected_tokens")
    }
}

private func validateCaseName(_ name: String, field: String) throws {
    let caseNameDescription = String(reflecting: name)
    let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmedName.isEmpty {
        throw MLXFastError.invalidInput("\(field) must not be empty")
    }
    if name != trimmedName {
        throw MLXFastError.invalidInput(
            "\(field) \(caseNameDescription) must not have leading or trailing whitespace"
        )
    }
    if name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) {
        throw MLXFastError.invalidInput(
            "\(field) \(caseNameDescription) must not contain control characters"
        )
    }
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
