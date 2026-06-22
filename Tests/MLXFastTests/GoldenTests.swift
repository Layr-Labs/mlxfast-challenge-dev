import Foundation
import CryptoKit
import Testing
@testable import MLXFastCore

@Test
func loadGoldenCasesAcceptsValidFixture() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let cases = try loadGoldenCases(from: path.path)

    #expect(cases.count == 1)
    #expect(cases[0].name == "hidden-0")
    #expect(cases[0].promptTokens == correctnessPrompt())
    #expect(cases[0].expectedTokens.count == MLXFastConstants.correctnessSteps)

    let fixture = try loadGoldenFixture(from: path.path)
    let digest = SHA256.hash(data: try Data(contentsOf: path))
    let expectedHash = digest.map { String(format: "%02x", $0) }.joined()
    #expect(fixture.cases == cases)
    #expect(fixture.sha256 == expectedHash)
}

@Test
func loadGoldenFixtureAcceptsBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let prefill = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(prefill),
        "expected_prefill_token": 4,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": \(decode)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark?.prefillPromptTokens == prefill)
    #expect(fixture.benchmark?.expectedPrefillToken == 4)
    #expect(fixture.benchmark?.decodeSeedTokens == seed)
    #expect(fixture.benchmark?.expectedDecodeSeedToken == 5)
    #expect(fixture.benchmark?.expectedDecodeTokens == decode)
}

@Test
func loadGoldenFixtureRejectsMalformedBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": [1],
        "expected_prefill_token": 4,
        "decode_seed_tokens": [2],
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": [3]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenPromptManifestAcceptsPrivatePromptSpec() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("prompts.json")
    let benchmarkPrompt = Array(repeating: 11, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let json = """
    {
      "version": 1,
      "max_output_tokens": \(MLXFastConstants.correctnessSteps),
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON())
        }
      ],
      "benchmark": {
        "name": "timed-hidden",
        "prompt_tokens": \(benchmarkPrompt)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let manifest = try loadGoldenPromptManifest(from: path.path)

    #expect(manifest.cases == [GoldenPromptCase(name: "hidden-0", promptTokens: correctnessPrompt())])
    #expect(manifest.benchmark.name == "timed-hidden")
    #expect(manifest.benchmark.promptTokens == benchmarkPrompt)
    #expect(manifest.maxOutputTokens == MLXFastConstants.correctnessSteps)
}

@Test
func loadGoldenPromptManifestRejectsShortBenchmarkPrompt() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("prompts.json")
    let json = """
    {
      "version": 1,
      "max_output_tokens": \(MLXFastConstants.correctnessSteps),
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON())
        }
      ],
      "benchmark": {
        "prompt_tokens": [1]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenPromptManifest(from: path.path)
    }
}

@Test
func loadGoldenPromptManifestRejectsShortCorrectnessPrompt() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("prompts.json")
    let benchmarkPrompt = Array(repeating: 11, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let json = """
    {
      "version": 1,
      "max_output_tokens": \(MLXFastConstants.correctnessSteps),
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": [1]
        }
      ],
      "benchmark": {
        "prompt_tokens": \(benchmarkPrompt)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenPromptManifest(from: path.path)
    }
}

@Test
func loadGoldenPromptManifestRejectsMissingMaxOutputTokens() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("prompts.json")
    let benchmarkPrompt = Array(repeating: 11, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON())
        }
      ],
      "benchmark": {
        "prompt_tokens": \(benchmarkPrompt)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenPromptManifest(from: path.path)
    }
}

@Test
func loadGoldenPromptManifestRejectsWrongMaxOutputTokens() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("prompts.json")
    let benchmarkPrompt = Array(repeating: 11, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let json = """
    {
      "version": 1,
      "max_output_tokens": \(MLXFastConstants.correctnessSteps - 1),
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON())
        }
      ],
      "benchmark": {
        "prompt_tokens": \(benchmarkPrompt)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenPromptManifest(from: path.path)
    }
}

@Test
func benchmarkOutputValidatorReportsTokenMismatches() {
    let oracle = BenchmarkGolden(
        prefillPromptTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
        expectedPrefillToken: 10,
        decodeSeedTokens: Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens),
        expectedDecodeSeedToken: 20,
        expectedDecodeTokens: [30, 31, 32]
    )

    let prefill = BenchmarkOutputValidator.comparePrefillToken(
        expected: oracle,
        actualToken: 11
    )
    #expect(!prefill.passed)
    #expect(prefill.expectedToken == 10)
    #expect(prefill.actualToken == 11)

    let seed = BenchmarkOutputValidator.compareDecodeSeedToken(
        expected: oracle,
        actualToken: 21
    )
    #expect(!seed.passed)
    #expect(seed.expectedToken == 20)
    #expect(seed.actualToken == 21)

    let decode = BenchmarkOutputValidator.compareDecodeTokens(
        expected: oracle,
        actualTokens: [30, 99, 32]
    )
    #expect(!decode.passed)
    #expect(decode.step == 1)
    #expect(decode.expectedToken == 31)
    #expect(decode.actualToken == 99)
}

@Test
func loadGoldenCasesRejectsOutOfRangeToken() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    var prompt = correctnessPrompt()
    prompt[0] = MLXFastConstants.vocabSize
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad",
          "prompt_tokens": \(arrayJSON(prompt)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsMissingVersion() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "cases": [
        {
          "name": "missing-version",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsDuplicateCaseNames() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        },
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON(2)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithSurroundingWhitespace() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": " ambiguous ",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithControlCharacters() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad\\nname",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongExpectedTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps + 1)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-count",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongPromptTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-prompt-count",
          "prompt_tokens": [1],
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredSteps() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = [7]
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredSteps: 0)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredPromptTokens() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-prompt-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredPromptTokens: 0)
    }
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func correctnessPrompt(_ token: Int = 1) -> [Int] {
    Array(repeating: token, count: MLXFastConstants.correctnessPromptTokens)
}

private func correctnessPromptJSON(_ token: Int = 1) -> String {
    arrayJSON(correctnessPrompt(token))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
