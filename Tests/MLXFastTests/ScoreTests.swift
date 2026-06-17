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
    #expect(decoded.metrics.error == "runtime unavailable")
    #expect(decoded.metrics.runtime == "swift")
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
