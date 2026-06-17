import Foundation
@testable import MLXFastCore
@testable import MLXFastDeepSeek
import Testing

@Test
func mactopBandwidthParsesJSONArraySamples() {
    let data = """
    [
      {"soc_metrics": {"dram_bw_combined_gbs": 10.5}},
      {"soc_metrics": {"dram_bw_combined_gbs": 11}},
      {"soc_metrics": {"ignored": 99}},
      {"other": true}
    ]
    """.data(using: .utf8)!

    #expect(MactopBandwidth.parseSamples(from: data) == [10.5, 11.0])
}

@Test
func mactopBandwidthParsesNDJSONSamples() {
    let data = """
    {"soc_metrics": {"dram_bw_combined_gbs": 3.25}}
    {"soc_metrics": {"dram_bw_combined_gbs": 4.75}}
    not json
    {"soc_metrics": {"dram_bw_combined_gbs": 0}}
    """.data(using: .utf8)!

    #expect(MactopBandwidth.parseSamples(from: data) == [3.25, 4.75])
}

@Test
func mactopBandwidthComputesIdleSubtractedGigabytesPerToken() throws {
    let value = try MactopBandwidth.gigabytesPerToken(
        samples: [10, 12, 8],
        idleGBPerSecond: 2,
        decodeElapsedSeconds: 4,
        decodedTokens: 8
    )

    #expect(abs(value - 4.0) < 1e-9)
}

@Test
func mactopBandwidthRejectsNoUsableNetSamples() {
    #expect(throws: MLXFastError.self) {
        _ = try MactopBandwidth.gigabytesPerToken(
            samples: [1, 2],
            idleGBPerSecond: 3,
            decodeElapsedSeconds: 1,
            decodedTokens: 1
        )
    }
}

@Test
func mactopLocatorUsesExplicitExecutableOverride() throws {
    let directory = try temporaryDirectory()
    let executable = directory.appendingPathComponent("mactop")
    try "#!/bin/sh\nexit 0\n".write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: executable.path
    )

    let resolved = try MactopLocator.executablePath(environment: [
        "MLXFAST_MACTOP_BIN": executable.path,
    ])

    #expect(resolved == executable.path)
}

@Test
func benchmarkPreflightAcceptsRequiredArtifacts() throws {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    let experts = weights.appendingPathComponent("experts", isDirectory: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
    try "{}".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(
        to: weights.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(to: experts.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
    let golden = directory.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let executable = directory.appendingPathComponent("mactop")
    try "#!/bin/sh\nexit 0\n".write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: executable.path
    )

    let report = try BenchmarkPreflight.check(
        weightsPath: weights.path,
        goldenPath: golden.path,
        environment: ["MLXFAST_MACTOP_BIN": executable.path]
    )

    #expect(report.weightsPath == weights.path)
    #expect(report.goldenPath == golden.path)
    #expect(report.mactopPath == executable.path)
}

@Test
func benchmarkPreflightRejectsMissingExpertManifest() throws {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(
        to: weights.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )
    let golden = directory.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let executable = directory.appendingPathComponent("mactop")
    try "#!/bin/sh\nexit 0\n".write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: executable.path
    )

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: weights.path,
            goldenPath: golden.path,
            environment: ["MLXFAST_MACTOP_BIN": executable.path]
        )
    }
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
