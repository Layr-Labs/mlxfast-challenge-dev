import Foundation

public struct ScorePayload: Codable, Equatable {
    public let score: Double?
    public let passed: Bool
    public let metrics: ScoreMetrics

    enum CodingKeys: String, CodingKey {
        case score
        case passed
        case metrics
    }

    public init(score: Double?, passed: Bool, metrics: ScoreMetrics) {
        self.score = score
        self.passed = passed
        self.metrics = metrics
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.score = try container.decodeIfPresent(Double.self, forKey: .score)
        self.passed = try container.decode(Bool.self, forKey: .passed)
        self.metrics = try container.decode(ScoreMetrics.self, forKey: .metrics)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let score {
            try container.encode(score, forKey: .score)
        } else {
            try container.encodeNil(forKey: .score)
        }
        try container.encode(passed, forKey: .passed)
        try container.encode(metrics, forKey: .metrics)
    }
}

public struct ScoreMetrics: Codable, Equatable {
    public let peakRamGB: Double
    public let bandwidthGBPerToken: Double
    public let decodeSecondsPerToken: Double
    public let prefillSecondsPerToken: Double
    public let passedCorrectness: Bool
    public let numLayers: Int
    public let firstFailingLayer: Int?
    public let firstFailingStep: Int?
    public let maxAbsDiff: Double
    public let bandwidthSource: String
    public let error: String
    public let commit: String
    public let timestamp: String
    public let harnessHash: String
    public let runtime: String

    enum CodingKeys: String, CodingKey {
        case peakRamGB = "peak_ram_gb"
        case bandwidthGBPerToken = "bandwidth_gb_per_token"
        case decodeSecondsPerToken = "decode_seconds_per_token"
        case prefillSecondsPerToken = "prefill_seconds_per_token"
        case passedCorrectness = "passed_correctness"
        case numLayers = "num_layers"
        case firstFailingLayer = "first_failing_layer"
        case firstFailingStep = "first_failing_step"
        case maxAbsDiff = "max_abs_diff"
        case bandwidthSource = "bandwidth_source"
        case error
        case commit
        case timestamp
        case harnessHash = "harness_hash"
        case runtime
    }

    public init(
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        passedCorrectness: Bool,
        numLayers: Int,
        firstFailingLayer: Int?,
        firstFailingStep: Int?,
        maxAbsDiff: Double,
        bandwidthSource: String,
        error: String,
        commit: String,
        timestamp: String,
        harnessHash: String,
        runtime: String
    ) {
        self.peakRamGB = peakRamGB
        self.bandwidthGBPerToken = bandwidthGBPerToken
        self.decodeSecondsPerToken = decodeSecondsPerToken
        self.prefillSecondsPerToken = prefillSecondsPerToken
        self.passedCorrectness = passedCorrectness
        self.numLayers = numLayers
        self.firstFailingLayer = firstFailingLayer
        self.firstFailingStep = firstFailingStep
        self.maxAbsDiff = maxAbsDiff
        self.bandwidthSource = bandwidthSource
        self.error = error
        self.commit = commit
        self.timestamp = timestamp
        self.harnessHash = harnessHash
        self.runtime = runtime
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(peakRamGB, forKey: .peakRamGB)
        try container.encode(bandwidthGBPerToken, forKey: .bandwidthGBPerToken)
        try container.encode(decodeSecondsPerToken, forKey: .decodeSecondsPerToken)
        try container.encode(prefillSecondsPerToken, forKey: .prefillSecondsPerToken)
        try container.encode(passedCorrectness, forKey: .passedCorrectness)
        try container.encode(numLayers, forKey: .numLayers)
        if let firstFailingLayer {
            try container.encode(firstFailingLayer, forKey: .firstFailingLayer)
        } else {
            try container.encodeNil(forKey: .firstFailingLayer)
        }
        if let firstFailingStep {
            try container.encode(firstFailingStep, forKey: .firstFailingStep)
        } else {
            try container.encodeNil(forKey: .firstFailingStep)
        }
        try container.encode(maxAbsDiff, forKey: .maxAbsDiff)
        try container.encode(bandwidthSource, forKey: .bandwidthSource)
        try container.encode(error, forKey: .error)
        try container.encode(commit, forKey: .commit)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(harnessHash, forKey: .harnessHash)
        try container.encode(runtime, forKey: .runtime)
    }
}

extension ScorePayload {
    public static func failed(
        error: String,
        commit: String = "",
        harnessHash: String = ""
    ) -> ScorePayload {
        ScorePayload(
            score: nil,
            passed: false,
            metrics: ScoreMetrics(
                peakRamGB: 0,
                bandwidthGBPerToken: 0,
                decodeSecondsPerToken: 0,
                prefillSecondsPerToken: 0,
                passedCorrectness: false,
                numLayers: MLXFastConstants.numHiddenLayers,
                firstFailingLayer: nil,
                firstFailingStep: nil,
                maxAbsDiff: 0,
                bandwidthSource: "",
                error: error,
                commit: commit,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash,
                runtime: "swift"
            )
        )
    }
}

public func writeScorePayload(_ payload: ScorePayload, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    let parent = url.deletingLastPathComponent()
    if !parent.path.isEmpty {
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(payload)
    try data.write(to: url)
}
