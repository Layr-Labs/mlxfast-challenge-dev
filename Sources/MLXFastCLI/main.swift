import Darwin
import Foundation
import MLXFastCore
import MLXFastDeepSeek
import MLXFastHarness
import MLXFastSubmission
import MLXFastTransform

let exitCode = MLXFastCLI.run(arguments: Array(CommandLine.arguments.dropFirst()))
exit(Int32(exitCode))

private enum MLXFastCLI {
    static func run(arguments: [String]) -> Int {
        guard let command = arguments.first, command != "help", command != "--help", command != "-h" else {
            printUsage()
            return 0
        }

        let options = ParsedOptions(Array(arguments.dropFirst()))

        do {
            switch command {
            case "transform":
                try runTransform(options)
                return 0
            case "verify-transform":
                try runVerifyTransform(options)
                return 0
            case "correctness":
                return try runCorrectness(options)
            case "preflight":
                try runPreflight(options)
                return 0
            case "benchmark":
                try runBenchmark(options)
                return 0
            case "make-golden":
                try runMakeGolden(options)
                return 0
            case "checkpoint-shards":
                try runCheckpointShards(options)
                return 0
            case "login":
                try runLogin(options)
                return 0
            case "clone":
                try runClone(options)
                return 0
            case "submit":
                try runSubmit(options)
                return 0
            default:
                fputs("mlxfast-swift: unknown command '\(command)'\n\n", stderr)
                printUsage()
                return 2
            }
        } catch {
            fputs("mlxfast-swift: \(error)\n", stderr)
            return 1
        }
    }

    private static func runTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--output"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let report = try SwiftTransform.run(
            TransformOptions(referencePath: referencePath, outputPath: outputPath)
        )
        print("reference: \(report.referencePath)")
        print("output: \(report.outputPath)")
        print("dense tensors: \(report.denseTensorCount) across \(report.denseShardCount) shard(s)")
        print("expert tensors: \(report.expertTensorCount)")
        print("expert manifest: \(report.manifestPath)")
    }

    private static func runVerifyTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--weights", "--tmp-parent", "--max-bytes"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let temporaryParentPath = options.value(for: "--tmp-parent", default: "")
        let maxBytesRaw = options.value(
            for: "--max-bytes",
            default: environmentValue(
                "MLXFAST_MAX_WEIGHTS_BYTES",
                fallback: "\(MLXFastConstants.defaultMaxTransformedWeightsBytes)"
            )
        )
        let maxByteCount = try parseMaxByteCount(
            maxBytesRaw,
            defaultByteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes,
            optionName: "--max-bytes"
        )
        let report = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: referencePath,
                weightsPath: weightsPath,
                temporaryParentPath: temporaryParentPath.isEmpty ? nil : temporaryParentPath,
                maxByteCount: maxByteCount
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runCorrectness(_ options: ParsedOptions) throws -> Int {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let report = try DeepSeekRuntime.runCorrectness(
            CorrectnessOptions(weightsPath: weightsPath, goldenPath: goldenPath)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
        return report.passed ? 0 : 1
    }

    private static func runPreflight(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let report = try BenchmarkPreflight.check(
            weightsPath: weightsPath,
            goldenPath: goldenPath
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runBenchmark(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden", "--score-path"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let scorePath = options.value(
            for: "--score-path",
            default: environmentValue(
                "MLXFAST_SCORE_PATH",
                fallback: MLXFastConstants.defaultScorePath
            )
        )
        let payload = DeepSeekRuntime.benchmark(
            BenchmarkOptions(weightsPath: weightsPath, goldenPath: goldenPath)
        )
        try writeScorePayload(payload, to: scorePath)
        print("wrote \(scorePath)")
    }

    private static func runMakeGolden(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--weights", "--output", "--prompt-file", "--name", "--prompt-tokens"]
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let outputPath = options.value(for: "--output", default: MLXFastConstants.defaultGoldenPath)
        let promptFile = options.value(for: "--prompt-file", default: "")
        let promptTokens = options.value(for: "--prompt-tokens", default: "")
        let name = options.value(for: "--name", default: "local")

        let manifest: GoldenPromptManifest
        if !promptFile.isEmpty {
            if !promptTokens.isEmpty || options.hasValue(for: "--name") {
                throw MLXFastError.invalidInput(
                    "--prompt-file cannot be combined with --name or --prompt-tokens"
                )
            }
            manifest = try loadGoldenPromptManifest(from: promptFile)
        } else {
            guard !promptTokens.isEmpty else {
                throw MLXFastError.invalidInput(
                    "make-golden requires --prompt-file PATH or --prompt-tokens TOKENS"
                )
            }
            let tokens = try parseTokenList(promptTokens)
            manifest = GoldenPromptManifest(
                cases: [
                    GoldenPromptCase(name: name, promptTokens: tokens),
                ],
                benchmark: BenchmarkPromptSpec(name: "benchmark", promptTokens: tokens)
            )
            try validateGoldenPromptManifest(manifest)
        }

        let document = try DeepSeekRuntime.generateGolden(
            GoldenGenerationOptions(weightsPath: weightsPath, promptManifest: manifest)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(document)
        let outputURL = URL(fileURLWithPath: outputPath)
        let parent = outputURL.deletingLastPathComponent()
        if !parent.path.isEmpty {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        try data.write(to: outputURL)
        print("wrote \(outputPath)")
    }

    private static func runCheckpointShards(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--index"])
        let indexPath = options.value(for: "--index", default: "")
        guard !indexPath.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint-shards requires --index PATH")
        }
        for shard in try CheckpointIndexTools.safetensorShardNames(from: indexPath) {
            print(shard)
        }
    }

    private static func runLogin(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--api-key"])
        let apiKey = options.value(for: "--api-key", default: "")
        let path = try SubmissionSupport.storeCredentials(apiKey: apiKey)
        print("credentials: \(path)")
    }

    private static func runClone(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--contract"])
        let contractPath = options.value(for: "--contract", default: "benchmark.json")
        let contract = try SubmissionSupport.ensureWorkspace(contractPath: contractPath)
        print("workspace: \(contract.name)")
        print("editable paths:")
        for path in contract.editablePaths {
            print("  \(path)")
        }
    }

    private static func runSubmit(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--contract", "--output", "--max-bytes"])
        let contractPath = options.value(for: "--contract", default: "benchmark.json")
        let outputPath = options.value(for: "--output", default: "mlxfast-submission.zip")
        let maxBytesRaw = options.value(
            for: "--max-bytes",
            default: environmentValue(
                "MLXFAST_MAX_SUBMISSION_BYTES",
                fallback: "\(MLXFastConstants.defaultMaxSubmissionSourceBytes)"
            )
        )
        let maxByteCount = try parseMaxByteCount(
            maxBytesRaw,
            defaultByteCount: MLXFastConstants.defaultMaxSubmissionSourceBytes,
            optionName: "--max-bytes"
        )
        let report = try SubmissionSupport.packageEditablePaths(
            contractPath: contractPath,
            outputPath: outputPath,
            maxByteCount: maxByteCount
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-swift transform [--reference PATH] [--output PATH]
              mlxfast-swift verify-transform [--reference PATH] [--weights PATH] [--tmp-parent PATH] [--max-bytes N]
              mlxfast-swift correctness [--weights PATH] [--golden PATH]
              mlxfast-swift preflight [--weights PATH] [--golden PATH]
              mlxfast-swift benchmark [--weights PATH] [--golden PATH] [--score-path PATH]
              mlxfast-swift make-golden [--weights PATH] [--output PATH] (--prompt-file PATH | --prompt-tokens TOKENS [--name NAME])
              mlxfast-swift checkpoint-shards --index PATH
              mlxfast-swift login --api-key KEY
              mlxfast-swift clone [--contract benchmark.json]
              mlxfast-swift submit [--contract benchmark.json] [--output mlxfast-submission.zip] [--max-bytes N]

            Swift-only DeepSeek V4 Flash harness entrypoint.
            """
        )
    }

    private static func environmentValue(_ name: String, fallback: String) -> String {
        let value = ProcessInfo.processInfo.environment[name] ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func parseMaxByteCount(
        _ raw: String,
        defaultByteCount: Int?,
        optionName: String
    ) throws -> Int? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return defaultByteCount
        }
        let lowercased = trimmed.lowercased()
        if lowercased == "0" || lowercased == "none" || lowercased == "unlimited" {
            return nil
        }
        guard let value = Int(trimmed), value > 0 else {
            throw MLXFastError.invalidInput(
                "\(optionName) must be a positive byte count, 0, none, or unlimited"
            )
        }
        return value
    }

    private static func parseTokenList(_ raw: String) throws -> [Int] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("--prompt-tokens must not be empty")
        }
        let stripped = trimmed
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        let parts = stripped.split { character in
            character == "," || character == " " || character == "\n" || character == "\t"
        }
        guard !parts.isEmpty else {
            throw MLXFastError.invalidInput("--prompt-tokens did not contain any token IDs")
        }
        return try parts.enumerated().map { index, part in
            guard let token = Int(part) else {
                throw MLXFastError.invalidInput("--prompt-tokens[\(index)] is not an integer: \(part)")
            }
            guard token >= 0, token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput(
                    "--prompt-tokens[\(index)]=\(token) is outside DeepSeek vocab range 0..<\(MLXFastConstants.vocabSize)"
                )
            }
            return token
        }
    }

}

private struct ParsedOptions {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []
    private var positionals: [String] = []
    private var duplicates: Set<String> = []

    init(_ arguments: [String]) {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument.hasPrefix("--") {
                if let separator = argument.firstIndex(of: "=") {
                    let key = String(argument[..<separator])
                    let value = String(argument[argument.index(after: separator)...])
                    recordOption(key)
                    values[key] = value
                    index += 1
                } else if index + 1 < arguments.count && !arguments[index + 1].hasPrefix("--") {
                    recordOption(argument)
                    values[argument] = arguments[index + 1]
                    index += 2
                } else {
                    recordOption(argument)
                    flags.insert(argument)
                    index += 1
                }
            } else {
                positionals.append(argument)
                index += 1
            }
        }
    }

    private mutating func recordOption(_ name: String) {
        if values[name] != nil || flags.contains(name) {
            duplicates.insert(name)
        }
    }

    func value(for name: String, default defaultValue: String) -> String {
        values[name] ?? defaultValue
    }

    func hasValue(for name: String) -> Bool {
        values[name] != nil
    }

    func validate(valueOptions: Set<String>, flagOptions: Set<String> = []) throws {
        if let duplicate = duplicates.first {
            throw MLXFastError.invalidInput("duplicate option \(duplicate)")
        }
        for name in values.keys where !valueOptions.contains(name) {
            throw MLXFastError.invalidInput("unknown option \(name)")
        }
        for (name, value) in values where value.isEmpty {
            throw MLXFastError.invalidInput("\(name) requires a non-empty value")
        }
        for flag in flags {
            if valueOptions.contains(flag) {
                throw MLXFastError.invalidInput("\(flag) requires a value")
            }
            if !flagOptions.contains(flag) {
                throw MLXFastError.invalidInput("unknown option \(flag)")
            }
        }
        if let positional = positionals.first {
            throw MLXFastError.invalidInput("unexpected argument \(positional)")
        }
    }
}
