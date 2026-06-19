import Foundation
import MLXFastCore
import MLXFastDeepSeek

public struct BenchmarkPreflightReport: Codable, Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let mactopPath: String
    public let weightsByteCount: Int
    public let maxWeightsByteCount: Int?

    public init(
        weightsPath: String,
        goldenPath: String,
        mactopPath: String,
        weightsByteCount: Int = 0,
        maxWeightsByteCount: Int? = MLXFastConstants.defaultMaxTransformedWeightsBytes
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.mactopPath = mactopPath
        self.weightsByteCount = weightsByteCount
        self.maxWeightsByteCount = maxWeightsByteCount
    }
}

public struct BenchmarkPromptPlan: Equatable {
    public let prefillTokens: [Int]
    public let expectedPrefillToken: Int
    public let decodeSeedTokens: [Int]
    public let expectedDecodeSeedToken: Int
    public let expectedDecodeTokens: [Int]

    public init(
        prefillTokens: [Int],
        expectedPrefillToken: Int,
        decodeSeedTokens: [Int],
        expectedDecodeSeedToken: Int,
        expectedDecodeTokens: [Int]
    ) {
        self.prefillTokens = prefillTokens
        self.expectedPrefillToken = expectedPrefillToken
        self.decodeSeedTokens = decodeSeedTokens
        self.expectedDecodeSeedToken = expectedDecodeSeedToken
        self.expectedDecodeTokens = expectedDecodeTokens
    }
}

public enum BenchmarkPrompt {
    public static func plan(from benchmark: BenchmarkGolden) throws -> BenchmarkPromptPlan {
        try validateBenchmarkGolden(benchmark)
        return BenchmarkPromptPlan(
            prefillTokens: benchmark.prefillPromptTokens,
            expectedPrefillToken: benchmark.expectedPrefillToken,
            decodeSeedTokens: benchmark.decodeSeedTokens,
            expectedDecodeSeedToken: benchmark.expectedDecodeSeedToken,
            expectedDecodeTokens: benchmark.expectedDecodeTokens
        )
    }
}

public enum BenchmarkPreflight {
    public static func check(
        weightsPath: String,
        goldenPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> BenchmarkPreflightReport {
        let requiredFiles = [
            ("\(weightsPath)/config.json", "transformed config"),
            ("\(weightsPath)/model.safetensors.index.json", "dense safetensors index"),
            ("\(weightsPath)/experts/manifest.json", "expert manifest"),
            (goldenPath, "correctness golden file"),
        ]
        for (path, description) in requiredFiles {
            try requireFile(path, description: description)
        }

        let maxWeightsByteCount = try transformedWeightsByteLimit(environment: environment)
        let weightsByteCount = try transformedWeightsByteCount(
            weightsPath: weightsPath,
            maxByteCount: maxWeightsByteCount
        )

        let golden = try loadGoldenFixture(from: goldenPath)
        guard let benchmark = golden.benchmark else {
            throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
        }
        _ = try BenchmarkPrompt.plan(from: benchmark)
        let config = try DeepSeekConfig.load(from: weightsPath)

        let denseStore = try DenseTensorStore(weightsPath: weightsPath)
        try denseStore.validateReadableByteRanges()

        let expertBank = try ExpertSlotBank(manifestPath: "\(weightsPath)/experts/manifest.json")
        try expertBank.validateReadableByteRanges()
        try DeepSeekWeightLoader(denseStore: denseStore, expertBank: expertBank)
            .validateRequiredMetadata(config: config)

        return BenchmarkPreflightReport(
            weightsPath: weightsPath,
            goldenPath: goldenPath,
            mactopPath: try MactopLocator.executablePath(environment: environment),
            weightsByteCount: weightsByteCount,
            maxWeightsByteCount: maxWeightsByteCount
        )
    }

    private static func transformedWeightsByteLimit(environment: [String: String]) throws -> Int? {
        let raw = environment["MLXFAST_MAX_WEIGHTS_BYTES"] ?? ""
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return MLXFastConstants.defaultMaxTransformedWeightsBytes
        }

        let lowercased = trimmed.lowercased()
        if lowercased == "0" || lowercased == "none" || lowercased == "unlimited" {
            return nil
        }
        guard let value = Int(trimmed), value > 0 else {
            throw MLXFastError.invalidInput(
                "MLXFAST_MAX_WEIGHTS_BYTES must be a positive byte count, 0, none, or unlimited"
            )
        }
        return value
    }

    private static func transformedWeightsByteCount(
        weightsPath: String,
        maxByteCount: Int?,
        fileManager: FileManager = .default
    ) throws -> Int {
        let root = URL(fileURLWithPath: weightsPath).standardizedFileURL
        let rootValues = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isSymbolicLink != true else {
            throw MLXFastError.invalidInput("transformed weights path must not be a symlink: \(root.path)")
        }
        guard rootValues.isDirectory == true else {
            throw MLXFastError.invalidInput("transformed weights path must be a directory: \(root.path)")
        }

        let rootPrefix = root.path + "/"
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("transformed weights directory not found at \(root.path)")
        }

        var byteCount = 0
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let path = standardized.path
            guard path.hasPrefix(rootPrefix) else {
                throw MLXFastError.invalidInput("transformed weights path escaped root: \(path)")
            }
            let relativePath = String(path.dropFirst(rootPrefix.count))
            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("transformed weights must not contain symlink \(relativePath)")
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput("transformed weights contains non-regular file \(relativePath)")
            }

            let size = try fileSizeByteCount(
                from: fileManager.attributesOfItem(atPath: standardized.path),
                path: standardized.path
            )
            guard byteCount <= Int.max - size else {
                throw MLXFastError.invalidInput("transformed weights byte count exceeds Int range")
            }
            byteCount += size
            if let maxByteCount, byteCount > maxByteCount {
                throw MLXFastError.invalidInput(
                    "transformed weights are \(byteCount) bytes, above limit \(maxByteCount)"
                )
            }
        }
        return byteCount
    }
}

public enum MactopLocator {
    public static func executablePath(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> String {
        if let override = environment["MLXFAST_MACTOP_BIN"], !override.isEmpty {
            guard fileManager.isExecutableFile(atPath: override) else {
                throw MLXFastError.missingFile(
                    "MLXFAST_MACTOP_BIN points to a non-executable file: \(override)"
                )
            }
            return override
        }

        let pathCandidates = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map { "\($0)/mactop" }
        let candidates = unique(pathCandidates + [
            "/opt/homebrew/bin/mactop",
            "/usr/local/bin/mactop",
            "/usr/bin/mactop",
        ])
        if let executable = candidates.first(where: { fileManager.isExecutableFile(atPath: $0) }) {
            return executable
        }
        throw MLXFastError.missingFile(
            "mactop not found; install it with Homebrew or set MLXFAST_MACTOP_BIN"
        )
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }
}

public struct MactopBandwidth: Equatable {
    public let samplesGBPerSecond: [Double]

    public init(samplesGBPerSecond: [Double]) {
        self.samplesGBPerSecond = samplesGBPerSecond
    }

    public static func parseSamples(from data: Data) -> [Double] {
        guard !data.isEmpty else {
            return []
        }
        if let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return array.compactMap(sampleValue)
        }

        guard let text = String(data: data, encoding: .utf8) else {
            return []
        }
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            guard let lineData = String(line).data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else {
                return nil
            }
            return sampleValue(object)
        }
    }

    public static func gigabytesPerToken(
        samples: [Double],
        idleGBPerSecond: Double,
        decodeElapsedSeconds: Double,
        decodedTokens: Int
    ) throws -> Double {
        guard decodedTokens > 0 else {
            throw MLXFastError.invalidInput("decoded token count must be positive")
        }
        let netSamples = samples
            .map { max($0 - idleGBPerSecond, 0) }
            .filter { $0 > 0 && $0.isFinite }
        guard !netSamples.isEmpty else {
            throw MLXFastError.invalidInput(
                "mactop produced no usable bandwidth samples after idle subtraction"
            )
        }
        let meanGBPerSecond = netSamples.reduce(0, +) / Double(netSamples.count)
        return meanGBPerSecond * decodeElapsedSeconds / Double(decodedTokens)
    }

    private static func sampleValue(_ object: [String: Any]) -> Double? {
        guard let socMetrics = object["soc_metrics"] as? [String: Any],
              let raw = socMetrics["dram_bw_combined_gbs"] else {
            return nil
        }
        switch raw {
        case let value as Double where value.isFinite && value >= 0:
            return value
        case let value as Int where value >= 0:
            return Double(value)
        case let value as NSNumber where value.doubleValue.isFinite && value.doubleValue >= 0:
            return value.doubleValue
        default:
            return nil
        }
    }
}

final class MactopSession: @unchecked Sendable {
    private let process: Process
    private let output = Pipe()
    private let errorOutput = Pipe()
    private let lock = NSLock()
    private var outputData = Data()
    private var errorData = Data()

    private init(process: Process) {
        self.process = process
    }

    static func measureIdleSamples(
        sampleCount: Int = 30,
        timeoutSeconds: TimeInterval = 45,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [Double] {
        precondition(sampleCount > 0)
        precondition(timeoutSeconds > 0)

        let process = try configuredProcess(
            arguments: [
                "--headless",
                "--interval", "100",
                "--format", "json",
            ],
            environment: environment
        )
        let session = MactopSession(process: process)
        process.standardOutput = session.output
        process.standardError = session.errorOutput
        session.capture(session.output.fileHandleForReading, intoErrorBuffer: false)
        session.capture(session.errorOutput.fileHandleForReading, intoErrorBuffer: true)
        try process.run()

        let sampleWindowSeconds = timeoutSeconds
        let deadline = Date().addingTimeInterval(sampleWindowSeconds)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: min(0.05, deadline.timeIntervalSinceNow))
        }
        let terminatedForSampling = process.isRunning
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        session.output.fileHandleForReading.readabilityHandler = nil
        session.errorOutput.fileHandleForReading.readabilityHandler = nil
        session.appendAvailable(session.output.fileHandleForReading, intoErrorBuffer: false)
        session.appendAvailable(session.errorOutput.fileHandleForReading, intoErrorBuffer: true)

        let samples = MactopBandwidth.parseSamples(from: session.outputSnapshot())
        if samples.count >= sampleCount {
            return Array(samples.prefix(sampleCount))
        }

        if terminatedForSampling {
            throw MLXFastError.invalidInput(
                "mactop idle measurement collected \(samples.count) usable samples in \(sampleWindowSeconds)s; expected \(sampleCount)"
            )
        }
        if process.terminationStatus != 0 {
            throw MLXFastError.invalidInput(
                "mactop idle measurement failed: \(String(data: session.errorSnapshot(), encoding: .utf8) ?? "")"
            )
        }
        return samples
    }

    static func start() throws -> MactopSession {
        let process = try configuredProcess(
            arguments: [
                "--headless",
                "--interval", "100",
                "--format", "json",
            ]
        )
        let session = MactopSession(process: process)
        process.standardOutput = session.output
        process.standardError = session.errorOutput
        session.capture(session.output.fileHandleForReading, intoErrorBuffer: false)
        session.capture(session.errorOutput.fileHandleForReading, intoErrorBuffer: true)
        try process.run()
        return session
    }

    func stop() throws -> [Double] {
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        output.fileHandleForReading.readabilityHandler = nil
        errorOutput.fileHandleForReading.readabilityHandler = nil

        appendAvailable(output.fileHandleForReading, intoErrorBuffer: false)
        appendAvailable(errorOutput.fileHandleForReading, intoErrorBuffer: true)

        lock.lock()
        let data = outputData
        let stderr = errorData
        lock.unlock()

        let samples = MactopBandwidth.parseSamples(from: data)
        guard !samples.isEmpty else {
            throw MLXFastError.invalidInput(
                "mactop produced no usable bandwidth samples. \(String(data: stderr, encoding: .utf8) ?? "")"
            )
        }
        return samples
    }

    private static func configuredProcess(
        arguments: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: try MactopLocator.executablePath(environment: environment))
        process.arguments = arguments
        return process
    }

    private func capture(_ handle: FileHandle, intoErrorBuffer: Bool) {
        handle.readabilityHandler = { [weak self] handle in
            self?.appendAvailable(handle, intoErrorBuffer: intoErrorBuffer)
        }
    }

    private func appendAvailable(_ handle: FileHandle, intoErrorBuffer: Bool) {
        let data = handle.availableData
        guard !data.isEmpty else {
            return
        }
        lock.lock()
        if intoErrorBuffer {
            errorData.append(data)
        } else {
            outputData.append(data)
        }
        lock.unlock()
    }

    private func outputSnapshot() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return outputData
    }

    private func errorSnapshot() -> Data {
        lock.lock()
        defer { lock.unlock() }
        return errorData
    }
}
