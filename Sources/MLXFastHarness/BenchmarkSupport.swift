import Foundation
import MLXFastCore
import MLXFastModel

public struct BenchmarkPreflightReport: Codable, Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let weightsByteCount: Int
    public let maxWeightsByteCount: Int?

    public init(
        weightsPath: String,
        goldenPath: String,
        weightsByteCount: Int = 0,
        maxWeightsByteCount: Int? = MLXFastConstants.defaultMaxTransformedWeightsBytes
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
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
