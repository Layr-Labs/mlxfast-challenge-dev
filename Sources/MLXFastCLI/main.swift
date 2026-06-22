import Darwin
import Foundation
import MLXFastCore
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
            case "correctness-trace":
                try runCorrectnessTrace(options)
                return 0
            case "preflight":
                try runPreflight(options)
                return 0
            case "benchmark":
                try runBenchmark(options)
                return 0
            case "runtime-worker":
                try runRuntimeWorker(options)
                return 0
            case "checkpoint-shards":
                try runCheckpointShards(options)
                return 0
            case "login":
                try runLogin(options)
                return 0
            case "config":
                try runConfig(options)
                return 0
            case "clone":
                try runClone(options)
                return 0
            case "link":
                try runLink(options)
                return 0
            case "submit":
                try runSubmit(options)
                return 0
            case "submissions":
                try runSubmissions(options)
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
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let report = try DeepSeekRuntime.runCorrectness(
            CorrectnessOptions(weightsPath: weightsPath, goldenPath: goldenPath),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
        return report.passed ? 0 : 1
    }

    private static func runCorrectnessTrace(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden", "--case", "--step", "--top-k"])
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
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let stepRaw = options.value(for: "--step", default: "")
        guard let step = Int(stepRaw), step >= 0 else {
            throw MLXFastError.invalidInput("correctness-trace requires --step N with N >= 0")
        }
        let topKRaw = options.value(for: "--top-k", default: "8")
        guard let topK = Int(topKRaw), topK > 0 else {
            throw MLXFastError.invalidInput("--top-k must be a positive integer")
        }
        let caseName = options.value(for: "--case", default: "")
        let report = try DeepSeekRuntime.traceCorrectness(
            CorrectnessTraceOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                caseName: caseName.isEmpty ? nil : caseName,
                step: step,
                topK: topK
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
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
        try options.validate(
            valueOptions: ["--weights", "--golden", "--score-path"],
            flagOptions: ["--quick"]
        )
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
        let quick = options.hasFlag("--quick")
        let payload = DeepSeekRuntime.benchmark(
            BenchmarkOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                correctnessSteps: quick ? MLXFastConstants.quickCorrectnessSteps : MLXFastConstants.correctnessSteps,
                benchmarkDecodeSteps: quick ? MLXFastConstants.quickBenchmarkDecodeSteps : MLXFastConstants.benchmarkDecodeSteps
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )
        try writeScorePayload(payload, to: scorePath)
        if quick {
            try printScorePayload(at: scorePath)
        } else {
            print("wrote \(scorePath)")
        }
    }

    private static func printScorePayload(at path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        FileHandle.standardOutput.write(data)
        if data.last != 0x0a {
            print("")
        }
    }

    private static func runtimeWorkerOptions(blockedGoldenPath: String? = nil) throws -> RuntimeWorkerOptions? {
        let enabled = environmentValue("MLXFAST_USE_RUNTIME_WORKER", fallback: "1")
        guard enabled != "0" && enabled.lowercased() != "false" else {
            return nil
        }
        let executable = environmentValue(
            "MLXFAST_RUNTIME_WORKER_EXECUTABLE",
            fallback: CommandLine.arguments.first ?? ""
        )
        guard !executable.isEmpty else {
            return nil
        }
        let executablePath: String
        if executable.hasPrefix("/") {
            executablePath = executable
        } else {
            executablePath = URL(
                fileURLWithPath: executable,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            ).standardizedFileURL.path
        }
        var sandboxProfile = environmentValue("MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE", fallback: "")
        if sandboxProfile.isEmpty,
           environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") != "1",
           let blockedGoldenPath,
           !blockedGoldenPath.isEmpty
        {
            sandboxProfile = try writeRuntimeWorkerSandboxProfile(
                blockedGoldenPath: blockedGoldenPath,
                allowedExecutablePath: executablePath
            )
        }
        return RuntimeWorkerOptions(
            executablePath: executablePath,
            sandboxProfilePath: sandboxProfile.isEmpty ? nil : sandboxProfile
        )
    }

    private static func writeRuntimeWorkerSandboxProfile(
        blockedGoldenPath: String,
        allowedExecutablePath: String
    ) throws -> String {
        let sandboxExecutable = "/usr/bin/sandbox-exec"
        guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
            throw MLXFastError.invalidInput("sandbox-exec not found for runtime worker sandbox")
        }
        let profileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-runtime-worker-\(UUID().uuidString).sb")
        let absoluteGoldenPath = absolutePath(blockedGoldenPath)
        let absoluteExecutablePath = absolutePath(allowedExecutablePath)
        let profile = """
        (version 1)
        (allow default)
        (deny network*)
        (deny process-fork)
        (deny process-exec*)
        (allow process-exec (literal "\(seatbeltEscaped(absoluteExecutablePath))"))
        (deny file-write*)
        (deny file-read* (literal "\(seatbeltEscaped(absoluteGoldenPath))"))
        """
        try profile.write(to: profileURL, atomically: true, encoding: .utf8)
        return profileURL.path
    }

    private static func absolutePath(_ path: String) -> String {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL.path
        }
        return URL(
            fileURLWithPath: path,
            relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        ).standardizedFileURL.path
    }

    private static func seatbeltEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func runRuntimeWorker(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        try DeepSeekRuntime.runWorker(weightsPath: weightsPath)
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
        try options.validate(
            valueOptions: ["--api-key", "--api"],
            flagOptions: ["--no-verify"],
            allowPositionals: true
        )
        let positionalAPIKeys = options.positionalArguments()
        guard positionalAPIKeys.count <= 1 else {
            throw MLXFastError.invalidInput("login accepts at most one positional API key")
        }
        guard !(options.hasValue(for: "--api-key") && !positionalAPIKeys.isEmpty) else {
            throw MLXFastError.invalidInput("login accepts either --api-key KEY or KEY, not both")
        }
        let apiKey = options.value(
            for: "--api-key",
            default: positionalAPIKeys.first ?? environmentValue("MLXFAST_API_KEY", fallback: "")
        )
        let existingCredentials = try loadOptionalCredentials()
        let apiBaseURL = try options.value(
            for: "--api",
            default: SubmissionSupport.configuredAPIBaseURL(credentials: existingCredentials)
        )
        if !options.hasFlag("--no-verify") {
            do {
                let client = try YukonClient(apiBaseURL: apiBaseURL, apiKey: apiKey)
                _ = try client.me()
                print("account: verified")
            } catch let error as YukonAPIError where error.statusCode == 401 {
                throw MLXFastError.invalidInput(
                    "API key was not accepted by \(apiBaseURL); pass --api if the key belongs to another Yukon API"
                )
            }
        }
        let path = try SubmissionSupport.storeCredentials(apiKey: apiKey, apiBaseURL: apiBaseURL)
        print("credentials: \(path)")
        print("api: \(apiBaseURL)")
    }

    private static func runConfig(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: [])
        let credentials = try loadOptionalCredentials()
        let apiBaseURL = try SubmissionSupport.configuredAPIBaseURL(credentials: credentials)
        let hasToken = SubmissionSupport.configuredAPIKey(credentials: credentials) != nil
        print("api: \(apiBaseURL)")
        print("credentials: \(hasToken ? "configured" : "missing")")
    }

    private static func runClone(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--api", "--contract"],
            allowPositionals: true
        )
        let positionals = options.positionalArguments()
        guard positionals.count <= 2 else {
            throw MLXFastError.invalidInput("clone accepts BENCHMARK and optional DIRECTORY")
        }

        if positionals.isEmpty {
            let contractPath = options.value(for: "--contract", default: "benchmark.json")
            let contract = try SubmissionSupport.ensureWorkspace(contractPath: contractPath)
            print("workspace: \(contract.name)")
            print("editable paths:")
            for path in contract.editablePaths {
                print("  \(path)")
            }
            return
        }

        guard !options.hasValue(for: "--contract") else {
            throw MLXFastError.invalidInput("clone BENCHMARK does not use --contract")
        }
        let benchmarkRef = positionals[0]
        let client = try authenticatedYukonClient(apiOverride: options.value(for: "--api", default: ""))
        let response = try client.getBenchmark(benchmarkRef)
        let benchmark = response.benchmark
        let sourceURL = try requiredBenchmarkSourceURL(benchmark)
        let destinationPath = positionals.count == 2
            ? positionals[1]
            : defaultCloneDirectory(for: benchmark)

        _ = try runProcess("/usr/bin/git", arguments: ["clone", sourceURL, destinationPath])
        if let sourceRef = trimmedNonEmpty(benchmark.sourceRef) {
            _ = try runProcess("/usr/bin/git", arguments: ["-C", destinationPath, "checkout", sourceRef])
        }
        try writeYukonGitConfig(benchmark: benchmark, fallbackRef: benchmarkRef, repositoryPath: destinationPath)
        print("cloned: \(destinationPath)")
        print("benchmark: \(benchmark.id)")
    }

    private static func runLink(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--api", "--benchmark"],
            allowPositionals: true
        )
        let positionals = options.positionalArguments()
        guard positionals.count <= 1 else {
            throw MLXFastError.invalidInput("link accepts at most one positional benchmark")
        }
        guard !(options.hasValue(for: "--benchmark") && !positionals.isEmpty) else {
            throw MLXFastError.invalidInput("link accepts either --benchmark ID or positional benchmark, not both")
        }
        let benchmarkRef = try resolveBenchmarkRef(
            explicit: options.value(for: "--benchmark", default: positionals.first ?? "")
        )
        let client = try authenticatedYukonClient(apiOverride: options.value(for: "--api", default: ""))
        let response = try client.getBenchmark(benchmarkRef)
        try writeYukonGitConfig(
            benchmark: response.benchmark,
            fallbackRef: benchmarkRef,
            repositoryPath: FileManager.default.currentDirectoryPath
        )
        print("linked benchmark: \(response.benchmark.id)")
        if let sourceURL = trimmedNonEmpty(response.benchmark.sourceURL) {
            print("source: \(sourceURL)")
        }
        if let sourceRef = trimmedNonEmpty(response.benchmark.sourceRef) {
            print("source_ref: \(sourceRef)")
        }
    }

    private static func runSubmit(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: [
                "--api",
                "--base-ref",
                "--benchmark",
                "--claimed-score",
                "--contract",
                "--idempotency-key",
                "--max-bytes",
                "--note",
                "--note-file",
                "--output",
            ],
            flagOptions: ["--dry-run"],
            allowPositionals: true
        )
        let positionalBenchmarks = options.positionalArguments()
        guard positionalBenchmarks.count <= 1 else {
            throw MLXFastError.invalidInput("submit accepts at most one positional benchmark")
        }
        guard !(options.hasValue(for: "--benchmark") && !positionalBenchmarks.isEmpty) else {
            throw MLXFastError.invalidInput("submit accepts either --benchmark ID or positional benchmark, not both")
        }
        let contractPath = options.value(for: "--contract", default: "benchmark.json")
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
        let surfaceReport = try SubmissionSupport.enforceModifiableSurface(
            contractPath: contractPath,
            baseRef: options.value(for: "--base-ref", default: "")
        )
        let baseDescription = surfaceReport.baseRef ?? "unresolved"
        fputs(
            "submit: modifiable surface ok changed_files=\(surfaceReport.changedPaths.count) base=\(baseDescription)\n",
            stderr
        )

        let credentials = try loadOptionalCredentials()
        let apiKey = SubmissionSupport.configuredAPIKey(credentials: credentials)
        let uploadRequested = !options.hasFlag("--dry-run") && (
            apiKey != nil ||
                options.hasValue(for: "--api") ||
                options.hasValue(for: "--benchmark") ||
                options.hasValue(for: "--claimed-score") ||
                options.hasValue(for: "--note") ||
                options.hasValue(for: "--note-file") ||
                !positionalBenchmarks.isEmpty
        )
        if uploadRequested && apiKey == nil {
            throw MLXFastError.invalidInput(
                "submit upload requires login first; run mlxfast-swift login KEY or set MLXFAST_API_KEY"
            )
        }

        if uploadRequested {
            let benchmark = try resolveBenchmarkRef(
                explicit: options.value(
                    for: "--benchmark",
                    default: positionalBenchmarks.first ?? ""
                )
            )
            let note = try requiredSubmissionNote(from: options)
            let claimedScore = try parseOptionalDouble(options.value(for: "--claimed-score", default: ""))
            let idempotencyKey = options.value(for: "--idempotency-key", default: UUID().uuidString)
            let client = try authenticatedYukonClient(apiOverride: options.value(for: "--api", default: ""))
            let upload = try YukonSubmissionUploader.uploadEditablePaths(
                YukonLiveSubmissionOptions(
                    contractPath: contractPath,
                    benchmark: benchmark,
                    maxByteCount: maxByteCount,
                    note: note,
                    claimedScore: claimedScore,
                    idempotencyKey: idempotencyKey
                ),
                client: client
            )
            print("submission: \(upload.response.submission.id)")
            print("status: \(upload.response.submission.status)")
            if let job = upload.response.job {
                print("job: \(job.id)")
                print("job_status: \(job.status)")
            }
            print("idempotency_key: \(idempotencyKey)")
            print("archive_sha256: \(upload.archive.archiveSha256)")
        } else {
            let outputPath = options.value(for: "--output", default: "mlxfast-submission.zip")
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
    }

    private static func runSubmissions(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--api", "--benchmark"],
            allowPositionals: true
        )
        let positionals = options.positionalArguments()
        guard positionals.count <= 1 else {
            throw MLXFastError.invalidInput("submissions accepts at most one positional benchmark")
        }
        guard !(options.hasValue(for: "--benchmark") && !positionals.isEmpty) else {
            throw MLXFastError.invalidInput("submissions accepts either --benchmark ID or positional benchmark, not both")
        }
        let benchmarkRef = try resolveBenchmarkRef(
            explicit: options.value(for: "--benchmark", default: positionals.first ?? "")
        )
        let client = try authenticatedYukonClient(apiOverride: options.value(for: "--api", default: ""))
        let response = try client.listBenchmarkSubmissions(benchmarkRef)
        if response.submissions.isEmpty {
            print("no submissions")
            return
        }
        for submission in response.submissions {
            let officialScore = formatOptionalScore(submission.officialScore)
            let claimedScore = formatOptionalScore(submission.claimedScore)
            let improved = submission.improved.map { $0 ? "true" : "false" } ?? "-"
            let note = trimmedNonEmpty(submission.note)
                .map { " note=\"\(truncate($0, limit: 80))\"" } ?? ""
            print(
                "\(submission.id) status=\(submission.status) official=\(officialScore) claimed=\(claimedScore) improved=\(improved)\(note)"
            )
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-swift transform [--reference PATH] [--output PATH]
              mlxfast-swift verify-transform [--reference PATH] [--weights PATH] [--tmp-parent PATH] [--max-bytes N]
              mlxfast-swift correctness [--weights PATH] [--golden PATH]
              mlxfast-swift correctness-trace [--weights PATH] [--golden PATH] [--case NAME] --step N [--top-k N]
              mlxfast-swift preflight [--weights PATH] [--golden PATH]
              mlxfast-swift benchmark [--quick] [--weights PATH] [--golden PATH] [--score-path PATH]
              mlxfast-swift checkpoint-shards --index PATH
              mlxfast-swift login [--api-key KEY | KEY] [--api URL] [--no-verify]
              mlxfast-swift config
              mlxfast-swift clone [BENCHMARK [DIRECTORY]] [--api URL]
              mlxfast-swift link [BENCHMARK] [--benchmark ID] [--api URL]
              mlxfast-swift submit [BENCHMARK] [--benchmark ID] [--contract benchmark.json] [--base-ref REF] [--output mlxfast-submission.zip] [--max-bytes N] [--note TEXT | --note-file PATH] [--claimed-score N] [--idempotency-key KEY] [--dry-run]
              mlxfast-swift submissions [BENCHMARK] [--benchmark ID] [--api URL]

            Swift-only DeepSeek V4 Flash harness entrypoint.
            """
        )
    }

    private static func environmentValue(_ name: String, fallback: String) -> String {
        let value = ProcessInfo.processInfo.environment[name] ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func defaultCorrectnessGoldenPath() -> String {
        if FileManager.default.fileExists(atPath: MLXFastConstants.defaultGoldenPath) {
            return MLXFastConstants.defaultGoldenPath
        }
        let publicPath = environmentValue(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH",
            fallback: MLXFastConstants.defaultPublicCorrectnessGoldenPath
        )
        if FileManager.default.fileExists(atPath: publicPath) {
            return publicPath
        }
        return MLXFastConstants.defaultGoldenPath
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

    private static func parseOptionalDouble(_ raw: String) throws -> Double? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }
        guard let value = Double(trimmed), value.isFinite else {
            throw MLXFastError.invalidInput("--claimed-score must be finite")
        }
        return value
    }

    private static func parseNonNegativeInt(_ raw: String, optionName: String) throws -> Int {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), value >= 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a non-negative integer")
        }
        return value
    }

    private static func loadOptionalCredentials() throws -> StoredCredentials? {
        do {
            return try SubmissionSupport.loadCredentials()
        } catch MLXFastError.missingFile {
            return nil
        }
    }

    private static func authenticatedYukonClient(apiOverride: String) throws -> YukonClient {
        let credentials = try loadOptionalCredentials()
        guard let apiKey = SubmissionSupport.configuredAPIKey(credentials: credentials) else {
            throw MLXFastError.invalidInput(
                "Yukon command requires login first; run mlxfast-swift login KEY or set MLXFAST_API_KEY"
            )
        }
        let trimmedAPIOverride = apiOverride.trimmingCharacters(in: .whitespacesAndNewlines)
        let apiBaseURL: String
        if trimmedAPIOverride.isEmpty {
            apiBaseURL = try SubmissionSupport.configuredAPIBaseURL(credentials: credentials)
        } else {
            apiBaseURL = try SubmissionSupport.configuredAPIBaseURL(
                credentials: credentials,
                environment: ["MLXFAST_API_URL": trimmedAPIOverride]
            )
        }
        return try YukonClient(apiBaseURL: apiBaseURL, apiKey: apiKey)
    }

    private static func resolveBenchmarkRef(explicit: String) throws -> String {
        let explicit = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
        if !explicit.isEmpty {
            return explicit
        }
        if let value = nonEmptyEnvironmentValue("MLXFAST_BENCHMARK_REF") {
            return value
        }
        if let value = nonEmptyEnvironmentValue("YUKON_BENCHMARK_REF") {
            return value
        }
        if let value = gitConfigValue("yukon.benchmark-id") {
            return value
        }
        throw MLXFastError.invalidInput(
            "benchmark id is required; pass BENCHMARK, --benchmark ID, or set MLXFAST_BENCHMARK_REF"
        )
    }

    private static func submissionNote(from options: ParsedOptions) throws -> String? {
        guard !(options.hasValue(for: "--note") && options.hasValue(for: "--note-file")) else {
            throw MLXFastError.invalidInput("pass either --note or --note-file, not both")
        }
        if options.hasValue(for: "--note") {
            return options.value(for: "--note", default: "")
        }
        if options.hasValue(for: "--note-file") {
            let path = options.value(for: "--note-file", default: "")
            guard !path.isEmpty else {
                throw MLXFastError.invalidInput("--note-file requires a path")
            }
            return try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)
        }
        return nil
    }

    private static func requiredSubmissionNote(from options: ParsedOptions) throws -> String {
        guard let note = try submissionNote(from: options).flatMap(trimmedNonEmpty) else {
            throw MLXFastError.invalidInput("submit upload requires --note TEXT or --note-file PATH")
        }
        return note
    }

    private static func nonEmptyEnvironmentValue(_ name: String) -> String? {
        let value = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? nil : value
    }

    private static func gitConfigValue(_ key: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["config", "--get", key]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else {
            return nil
        }
        let value = String(
            data: output.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? nil : value
    }

    private static func writeYukonGitConfig(
        benchmark: YukonBenchmark,
        fallbackRef: String,
        repositoryPath: String
    ) throws {
        let repositoryPath = URL(fileURLWithPath: repositoryPath).standardizedFileURL.path
        let benchmarkId = trimmedNonEmpty(benchmark.id) ?? fallbackRef
        try setGitConfig("yukon.benchmark-id", value: benchmarkId, repositoryPath: repositoryPath)
        if let name = trimmedNonEmpty(benchmark.name) {
            try setGitConfig("yukon.benchmark-name", value: name, repositoryPath: repositoryPath)
        }
        if let sourceURL = trimmedNonEmpty(benchmark.sourceURL) {
            try setGitConfig("yukon.source-url", value: sourceURL, repositoryPath: repositoryPath)
        }
        if let sourceRef = trimmedNonEmpty(benchmark.sourceRef) {
            try setGitConfig("yukon.source-ref", value: sourceRef, repositoryPath: repositoryPath)
        }
    }

    private static func setGitConfig(_ key: String, value: String, repositoryPath: String) throws {
        _ = try runProcess("/usr/bin/git", arguments: ["-C", repositoryPath, "config", key, value])
    }

    private static func requiredBenchmarkSourceURL(_ benchmark: YukonBenchmark) throws -> String {
        guard let sourceURL = trimmedNonEmpty(benchmark.sourceURL) else {
            throw MLXFastError.invalidInput("Yukon benchmark \(benchmark.id) does not include sourceUrl")
        }
        return sourceURL
    }

    private static func defaultCloneDirectory(for benchmark: YukonBenchmark) -> String {
        let rawName = trimmedNonEmpty(benchmark.name)
            ?? trimmedNonEmpty(benchmark.sourceURL).flatMap(repositoryName(from:))
            ?? benchmark.id
        return sanitizePathComponent(rawName)
    }

    private static func repositoryName(from sourceURL: String) -> String? {
        guard let last = sourceURL.split(separator: "/").last else {
            return nil
        }
        let name = String(last)
        if name.hasSuffix(".git") {
            return String(name.dropLast(4))
        }
        return name
    }

    private static func sanitizePathComponent(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        var result = ""
        for scalar in trimmed.unicodeScalars {
            let character = Character(scalar)
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_" || scalar == "." {
                result.append(character)
            } else {
                result.append("-")
            }
        }
        let collapsed = result.split(separator: "-").joined(separator: "-")
        return collapsed.isEmpty ? "mlxfast-benchmark" : collapsed
    }

    private static func formatOptionalScore(_ value: Double?) -> String {
        guard let value else {
            return "-"
        }
        return String(format: "%.6g", value)
    }

    private static func truncate(_ value: String, limit: Int) -> String {
        guard value.count > limit else {
            return value
        }
        let index = value.index(value.startIndex, offsetBy: limit)
        return String(value[..<index]) + "..."
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    @discardableResult
    private static func runProcess(
        _ executable: String,
        arguments: [String]
    ) throws -> String {
        let process = Process()
        let output = Pipe()
        let errorPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = errorPipe
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let stderrData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard process.terminationStatus == 0 else {
            let stderr = String(data: stderrData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let detail = stderr.isEmpty ? stdout : stderr
            throw MLXFastError.invalidInput(
                "\(URL(fileURLWithPath: executable).lastPathComponent) failed with status \(process.terminationStatus): \(detail)"
            )
        }
        return stdout
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

    func hasFlag(_ name: String) -> Bool {
        flags.contains(name)
    }

    func positionalArguments() -> [String] {
        positionals
    }

    func validate(
        valueOptions: Set<String>,
        flagOptions: Set<String> = [],
        allowPositionals: Bool = false
    ) throws {
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
        if !allowPositionals, let positional = positionals.first {
            throw MLXFastError.invalidInput("unexpected argument \(positional)")
        }
    }
}
