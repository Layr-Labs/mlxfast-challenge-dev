import Foundation
import MLXFastCore

public struct BenchmarkPreflightReport: Codable, Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let mactopPath: String

    public init(weightsPath: String, goldenPath: String, mactopPath: String) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.mactopPath = mactopPath
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

        _ = try loadGoldenCases(from: goldenPath)
        _ = try DeepSeekConfig.load(from: weightsPath)

        let denseStore = try DenseTensorStore(weightsPath: weightsPath)
        try denseStore.validateReadableByteRanges()

        let expertBank = try ExpertSlotBank(manifestPath: "\(weightsPath)/experts/manifest.json")
        try expertBank.validateReadableByteRanges()

        return BenchmarkPreflightReport(
            weightsPath: weightsPath,
            goldenPath: goldenPath,
            mactopPath: try MactopLocator.executablePath(environment: environment)
        )
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
        case let value as Double where value.isFinite && value > 0:
            return value
        case let value as Int where value > 0:
            return Double(value)
        case let value as NSNumber where value.doubleValue.isFinite && value.doubleValue > 0:
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

    static func measureIdleSamples(sampleCount: Int = 30) throws -> [Double] {
        let output = Pipe()
        let errorOutput = Pipe()
        let process = try configuredProcess(
            arguments: [
                "--headless",
                "--count", "\(sampleCount)",
                "--interval", "100",
                "--format", "json",
            ]
        )
        process.standardOutput = output
        process.standardError = errorOutput
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        if process.terminationStatus != 0 {
            let errorData = errorOutput.fileHandleForReading.readDataToEndOfFile()
            throw MLXFastError.invalidInput(
                "mactop idle measurement failed: \(String(data: errorData, encoding: .utf8) ?? "")"
            )
        }
        return MactopBandwidth.parseSamples(from: data)
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

    private static func configuredProcess(arguments: [String]) throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: try MactopLocator.executablePath())
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
}
