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
    #expect(decoded.metrics.weightsHash == "")
    #expect(decoded.metrics.weightsByteCount == 0)
    #expect(decoded.metrics.weightsFileCount == 0)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
    #expect(decoded.metrics.processResidentMemoryGB == 0)
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
                benchmarkWallSeconds: 11,
                preflightSeconds: 1,
                correctnessSeconds: 2,
                timedBenchmarkSeconds: 8,
                processResidentMemoryGB: 3.5,
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
                weightsHash: "weights-hash",
                weightsByteCount: 4096,
                weightsFileCount: 7,
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
    #expect(raw.contains("\"weights_hash\" : \"weights-hash\""))
    #expect(raw.contains("\"weights_byte_count\" : 4096"))
    #expect(raw.contains("\"weights_file_count\" : 7"))
    #expect(raw.contains("\"benchmark_wall_seconds\" : 11"))
    #expect(raw.contains("\"preflight_seconds\" : 1"))
    #expect(raw.contains("\"correctness_seconds\" : 2"))
    #expect(raw.contains("\"timed_benchmark_seconds\" : 8"))
    #expect(raw.contains("\"process_resident_memory_gb\" : 3.5"))
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
    #expect(decoded.metrics.weightsHash == "weights-hash")
    #expect(decoded.metrics.weightsByteCount == 4096)
    #expect(decoded.metrics.weightsFileCount == 7)
    #expect(decoded.metrics.benchmarkWallSeconds == 11)
    #expect(decoded.metrics.preflightSeconds == 1)
    #expect(decoded.metrics.correctnessSeconds == 2)
    #expect(decoded.metrics.timedBenchmarkSeconds == 8)
    #expect(decoded.metrics.processResidentMemoryGB == 3.5)
}

@Test
func scoreMetricsDecodeOlderPayloadWithoutWeightsIntegrityFields() throws {
    let data = """
    {
      "score": null,
      "passed": false,
      "metrics": {
        "peak_ram_gb": 0,
        "bandwidth_gb_per_token": 0,
        "decode_seconds_per_token": 0,
        "prefill_seconds_per_token": 0,
        "passed_correctness": false,
        "num_layers": \(MLXFastConstants.numHiddenLayers),
        "checked_steps": 0,
        "case_count": 0,
        "expert_cache_hits": 0,
        "expert_cache_misses": 0,
        "expert_cache_evictions": 0,
        "expert_bytes_read": 0,
        "expert_read_seconds": 0,
        "expert_peak_cached_tensors": 0,
        "expert_hit_rate": 0,
        "first_failing_layer": null,
        "first_failing_case": null,
        "first_failing_step": null,
        "expected_token": null,
        "actual_token": null,
        "max_abs_diff": 0,
        "golden_hash": "",
        "bandwidth_source": "",
        "error": "old payload",
        "commit": "",
        "timestamp": "2026-06-18T00:00:00Z",
        "harness_hash": "",
        "runtime": "swift"
      }
    }
    """.data(using: .utf8)!

    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(decoded.metrics.weightsHash == "")
    #expect(decoded.metrics.weightsByteCount == 0)
    #expect(decoded.metrics.weightsFileCount == 0)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
    #expect(decoded.metrics.processResidentMemoryGB == 0)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
