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
    public let checkedSteps: Int
    public let caseCount: Int
    public let expertCacheHits: UInt64
    public let expertCacheMisses: UInt64
    public let expertCacheEvictions: UInt64
    public let expertBytesRead: UInt64
    public let expertReadSeconds: Double
    public let expertPeakCachedTensors: UInt64
    public let expertHitRate: Double
    public let firstFailingLayer: Int?
    public let firstFailingCase: String?
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?
    public let maxAbsDiff: Double
    public let goldenHash: String
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
        case checkedSteps = "checked_steps"
        case caseCount = "case_count"
        case expertCacheHits = "expert_cache_hits"
        case expertCacheMisses = "expert_cache_misses"
        case expertCacheEvictions = "expert_cache_evictions"
        case expertBytesRead = "expert_bytes_read"
        case expertReadSeconds = "expert_read_seconds"
        case expertPeakCachedTensors = "expert_peak_cached_tensors"
        case expertHitRate = "expert_hit_rate"
        case firstFailingLayer = "first_failing_layer"
        case firstFailingCase = "first_failing_case"
        case firstFailingStep = "first_failing_step"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case maxAbsDiff = "max_abs_diff"
        case goldenHash = "golden_hash"
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
        checkedSteps: Int,
        caseCount: Int,
        expertCacheHits: UInt64 = 0,
        expertCacheMisses: UInt64 = 0,
        expertCacheEvictions: UInt64 = 0,
        expertBytesRead: UInt64 = 0,
        expertReadSeconds: Double = 0,
        expertPeakCachedTensors: UInt64 = 0,
        expertHitRate: Double = 0,
        firstFailingLayer: Int?,
        firstFailingCase: String?,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        maxAbsDiff: Double,
        goldenHash: String,
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
        self.checkedSteps = checkedSteps
        self.caseCount = caseCount
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.expertCacheEvictions = expertCacheEvictions
        self.expertBytesRead = expertBytesRead
        self.expertReadSeconds = expertReadSeconds
        self.expertPeakCachedTensors = expertPeakCachedTensors
        self.expertHitRate = expertHitRate
        self.firstFailingLayer = firstFailingLayer
        self.firstFailingCase = firstFailingCase
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
        self.maxAbsDiff = maxAbsDiff
        self.goldenHash = goldenHash
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
        try container.encode(checkedSteps, forKey: .checkedSteps)
        try container.encode(caseCount, forKey: .caseCount)
        try container.encode(expertCacheHits, forKey: .expertCacheHits)
        try container.encode(expertCacheMisses, forKey: .expertCacheMisses)
        try container.encode(expertCacheEvictions, forKey: .expertCacheEvictions)
        try container.encode(expertBytesRead, forKey: .expertBytesRead)
        try container.encode(expertReadSeconds, forKey: .expertReadSeconds)
        try container.encode(expertPeakCachedTensors, forKey: .expertPeakCachedTensors)
        try container.encode(expertHitRate, forKey: .expertHitRate)
        if let firstFailingLayer {
            try container.encode(firstFailingLayer, forKey: .firstFailingLayer)
        } else {
            try container.encodeNil(forKey: .firstFailingLayer)
        }
        if let firstFailingCase {
            try container.encode(firstFailingCase, forKey: .firstFailingCase)
        } else {
            try container.encodeNil(forKey: .firstFailingCase)
        }
        if let firstFailingStep {
            try container.encode(firstFailingStep, forKey: .firstFailingStep)
        } else {
            try container.encodeNil(forKey: .firstFailingStep)
        }
        if let expectedToken {
            try container.encode(expectedToken, forKey: .expectedToken)
        } else {
            try container.encodeNil(forKey: .expectedToken)
        }
        if let actualToken {
            try container.encode(actualToken, forKey: .actualToken)
        } else {
            try container.encodeNil(forKey: .actualToken)
        }
        try container.encode(maxAbsDiff, forKey: .maxAbsDiff)
        try container.encode(goldenHash, forKey: .goldenHash)
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
                checkedSteps: 0,
                caseCount: 0,
                firstFailingLayer: nil,
                firstFailingCase: nil,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil,
                maxAbsDiff: 0,
                goldenHash: "",
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
