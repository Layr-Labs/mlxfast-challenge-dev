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
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingStep == nil)
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
                firstFailingLayer: nil,
                firstFailingStep: 12,
                maxAbsDiff: 0,
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
    #expect(raw.contains("\"first_failing_step\" : 12"))
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingStep == 12)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
