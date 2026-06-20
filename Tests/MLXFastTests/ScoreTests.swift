import Foundation
import Testing
@testable import MLXFastCore

@Test
func writeScorePayloadEmitsDarkbloomShape() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("score.json")

    try writeScorePayload(
        .failed(error: "runtime unavailable"),
        to: path.path
    )

    let data = try Data(contentsOf: path)
    let raw = String(decoding: data, as: UTF8.self)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(raw.contains("\"score\" : null"))
    #expect(decoded.score == nil)
    #expect(decoded.passed == false)
    #expect(decoded.metrics.passedCorrectness == false)
    #expect(decoded.metrics.checkedSteps == 0)
    #expect(decoded.metrics.caseCount == 0)
    #expect(decoded.metrics.expertCacheHits == 0)
    #expect(decoded.metrics.expertCacheMisses == 0)
    #expect(decoded.metrics.expertCacheEvictions == 0)
    #expect(decoded.metrics.expertBytesRead == 0)
    #expect(decoded.metrics.expertReadSeconds == 0)
    #expect(decoded.metrics.expertPeakCachedTensors == 0)
    #expect(decoded.metrics.expertHitRate == 0)
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingCase == nil)
    #expect(decoded.metrics.firstFailingStep == nil)
    #expect(decoded.metrics.expectedToken == nil)
    #expect(decoded.metrics.actualToken == nil)
    #expect(decoded.metrics.goldenHash == "")
    #expect(decoded.metrics.error == "runtime unavailable")
    #expect(decoded.metrics.runtime == "swift")
}

@Test
func writeScorePayloadKeepsTokenStepSeparateFromLayerFailures() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("score.json")

    try writeScorePayload(
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
                checkedSteps: 13,
                caseCount: 2,
                expertCacheHits: 3,
                expertCacheMisses: 5,
                expertCacheEvictions: 2,
                expertBytesRead: 1024,
                expertReadSeconds: 0.25,
                expertPeakCachedTensors: 4,
                expertHitRate: 0.375,
                firstFailingLayer: nil,
                firstFailingCase: "case-b",
                firstFailingStep: 12,
                expectedToken: 42,
                actualToken: 17,
                maxAbsDiff: 0,
                goldenHash: "golden-hash",
                bandwidthSource: "",
                error: "generated token mismatch",
                commit: "abc123",
                timestamp: "2026-06-18T00:00:00Z",
                harnessHash: "hash",
                runtime: "swift"
            )
        ),
        to: path.path
    )

    let data = try Data(contentsOf: path)
    let raw = String(decoding: data, as: UTF8.self)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(raw.contains("\"first_failing_layer\" : null"))
    #expect(raw.contains("\"first_failing_case\" : \"case-b\""))
    #expect(raw.contains("\"first_failing_step\" : 12"))
    #expect(raw.contains("\"expected_token\" : 42"))
    #expect(raw.contains("\"actual_token\" : 17"))
    #expect(raw.contains("\"checked_steps\" : 13"))
    #expect(raw.contains("\"case_count\" : 2"))
    #expect(raw.contains("\"expert_cache_hits\" : 3"))
    #expect(raw.contains("\"expert_cache_misses\" : 5"))
    #expect(raw.contains("\"expert_cache_evictions\" : 2"))
    #expect(raw.contains("\"expert_bytes_read\" : 1024"))
    #expect(raw.contains("\"expert_read_seconds\" : 0.25"))
    #expect(raw.contains("\"expert_peak_cached_tensors\" : 4"))
    #expect(raw.contains("\"expert_hit_rate\" : 0.375"))
    #expect(raw.contains("\"golden_hash\" : \"golden-hash\""))
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingCase == "case-b")
    #expect(decoded.metrics.firstFailingStep == 12)
    #expect(decoded.metrics.expectedToken == 42)
    #expect(decoded.metrics.actualToken == 17)
    #expect(decoded.metrics.checkedSteps == 13)
    #expect(decoded.metrics.caseCount == 2)
    #expect(decoded.metrics.expertCacheHits == 3)
    #expect(decoded.metrics.expertCacheMisses == 5)
    #expect(decoded.metrics.expertCacheEvictions == 2)
    #expect(decoded.metrics.expertBytesRead == 1024)
    #expect(decoded.metrics.expertReadSeconds == 0.25)
    #expect(decoded.metrics.expertPeakCachedTensors == 4)
    #expect(decoded.metrics.expertHitRate == 0.375)
    #expect(decoded.metrics.goldenHash == "golden-hash")
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
