import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastModel

@Test
func expertResidencyPolicyHonorsEnvironmentOverrideAndMemoryThreshold() {
    // Explicit override wins in both directions regardless of machine size.
    #expect(ExpertResidencyPolicy.fullResidencyEnabled(
        environment: ["MLXFAST_EXPERT_RESIDENT": "1"],
        physicalMemory: 8 << 30
    ))
    #expect(!ExpertResidencyPolicy.fullResidencyEnabled(
        environment: ["MLXFAST_EXPERT_RESIDENT": "0"],
        physicalMemory: 512 << 30
    ))
    #expect(ExpertResidencyPolicy.fullResidencyEnabled(
        environment: ["MLXFAST_EXPERT_RESIDENT": " true "],
        physicalMemory: 8 << 30
    ))

    // Unset (or unparseable) means auto by physical memory: the official
    // M3 Ultra 256 GB contract is on, laptops are off.
    #expect(ExpertResidencyPolicy.fullResidencyEnabled(
        environment: [:],
        physicalMemory: 256 << 30
    ))
    #expect(ExpertResidencyPolicy.fullResidencyEnabled(
        environment: [:],
        physicalMemory: ExpertResidencyPolicy.minimumPhysicalMemoryBytes
    ))
    #expect(!ExpertResidencyPolicy.fullResidencyEnabled(
        environment: [:],
        physicalMemory: ExpertResidencyPolicy.minimumPhysicalMemoryBytes - 1
    ))
    #expect(!ExpertResidencyPolicy.fullResidencyEnabled(
        environment: ["MLXFAST_EXPERT_RESIDENT": "maybe"],
        physicalMemory: 48 << 30
    ))
}

@Test
func expertStreamingConfigDefaultsKeepTestsStreamingAndEnvironmentAppliesPolicy() {
    // Memberwise default stays false so fixture-driven tests exercise the
    // deterministic streaming fallback on every machine.
    #expect(!ExpertStreamingConfig().fullResidency)

    // fromEnvironment applies the policy: forced on/off via the env var.
    #expect(ExpertStreamingConfig.fromEnvironment(["MLXFAST_EXPERT_RESIDENT": "1"]).fullResidency)
    #expect(!ExpertStreamingConfig.fromEnvironment(["MLXFAST_EXPERT_RESIDENT": "0"]).fullResidency)
}

@Test
func residentAllExpertsStoreServesStackedSlicesByteIdenticallyToBank() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = root.appendingPathComponent("weights/experts", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let shard = reference.appendingPathComponent("model-00001.safetensors")
    try Data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).write(to: shard)

    let stackedName = "model.layers.0.ffn.switch_mlp.gate_proj.weight"
    let scalesName = "model.layers.0.ffn.switch_mlp.gate_proj.scales"
    let flatName = "model.layers.0.ffn.switch_mlp.route_bias"
    let manifestURL = experts.appendingPathComponent("manifest.json")
    try manifestJSON(
        referencePath: reference.path,
        records: [
            record(name: stackedName, shard: shard.lastPathComponent, offset: 0, length: 6, shape: [3, 2]),
            record(name: scalesName, shard: shard.lastPathComponent, offset: 6, length: 3, shape: [3, 1]),
            // Non-stacked (1-D) records cannot live in the store; they must be
            // skipped without disabling residency for everything else.
            record(name: flatName, shard: shard.lastPathComponent, offset: 9, length: 1, shape: [1]),
        ]
    ).write(to: manifestURL, atomically: true, encoding: .utf8)

    let store = try #require(ResidentExpertTensors(
        allExpertsFromManifest: manifestURL.path,
        metrics: nil
    ))
    #expect(store.isResident(name: stackedName))
    #expect(store.isResident(name: scalesName))
    #expect(!store.isResident(name: flatName))

    // Slices must be byte-identical to the slot bank's own reads.
    let bank = try ExpertSlotBank(manifestPath: manifestURL.path, capacity: 4)
    for index in 0..<3 {
        let resident = try #require(store.materializedTensor(named: stackedName, firstAxisIndex: index))
        let banked = try bank.materializedTensor(named: stackedName, firstAxisIndex: index)
        #expect(try resident.uint8Values() == banked.uint8Values())
        #expect(resident.shape == banked.shape)
    }
    let wholeResident = try #require(store.materializedTensor(named: scalesName, firstAxisIndex: nil))
    #expect(try wholeResident.uint8Values() == bank.materializedTensor(named: scalesName).uint8Values())
}

@Test
func fullyResidentBandwidthDiagnosticsLabelZeroReadsInsteadOfFailing() throws {
    let zeroReads = ExpertStreamingStats()

    // Streaming run observing zero decode reads is a bypass signal: throws.
    #expect(throws: MLXFastError.self) {
        _ = try DeepSeekRuntime.expertStreamingBandwidthGBPerToken(
            before: nil,
            after: zeroReads,
            decodedTokens: 128
        )
    }

    // Resident run legitimately reads nothing during decode: labeled, not fatal.
    let resident = try DeepSeekRuntime.expertStreamingBandwidthGBPerToken(
        before: nil,
        after: zeroReads,
        decodedTokens: 128,
        fullyResidentExperts: true
    )
    #expect(resident.gbPerToken == 0)
    #expect(resident.source == ExpertResidencyPolicy.residentBandwidthSource)

    // Real reads keep the streaming source label either way.
    let streaming = try DeepSeekRuntime.expertStreamingBandwidthGBPerToken(
        before: nil,
        after: ExpertStreamingStats(bytesRead: 128 << 30),
        decodedTokens: 128,
        fullyResidentExperts: true
    )
    #expect(streaming.source == ExpertStreamingMetrics.bandwidthSource)
    #expect(streaming.gbPerToken == 1)

    // The local-iterate variant mirrors the same labeling.
    let localResident = DeepSeekRuntime.localIterateBandwidthGBPerToken(
        bytesRead: 0,
        decodedTokens: 16,
        fullyResidentExperts: true
    )
    #expect(localResident.source == ExpertResidencyPolicy.residentBandwidthSource)
    let localStreaming = DeepSeekRuntime.localIterateBandwidthGBPerToken(
        bytesRead: 0,
        decodedTokens: 16
    )
    #expect(localStreaming.source == ExpertStreamingMetrics.bandwidthSource)
}

@Test
func benchmarkHarnessDecidesResidencyParentSideAndSkipsSeedGateOnlyThen() throws {
    let benchmark = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntimeBenchmark.swift",
        encoding: .utf8
    )
    let localIterate = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntimeLocalIterate.swift",
        encoding: .utf8
    )

    // The trusted parent computes residency from its OWN environment and
    // machine (same host as the worker), so a submission cannot claim
    // residency to dodge the seed-read plausibility gate on a streaming run.
    #expect(
        benchmark.components(
            separatedBy: "fullyResidentExperts: ExpertResidencyPolicy.fullResidencyEnabled()"
        ).count >= 3
    )
    #expect(benchmark.contains("if !fullyResidentExperts {"))
    #expect(benchmark.contains("try requirePlausibleSeedForwardExpertReads("))
    #expect(
        localIterate.components(
            separatedBy: "fullyResidentExperts: ExpertResidencyPolicy.fullResidencyEnabled()"
        ).count >= 3
    )
}

private func manifestJSON(referencePath: String, records: [String]) -> String {
    """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(referencePath)",
      "expert_tensors": [
        \(records.joined(separator: ",\n        "))
      ]
    }
    """
}

private func record(
    name: String,
    shard: String,
    offset: Int,
    length: Int,
    shape: [Int]
) -> String {
    """
    {
      "name": "\(name)",
      "shard": "\(shard)",
      "dtype": "U8",
      "shape": [\(shape.map(String.init).joined(separator: ","))],
      "data_offsets": [0, \(length)],
      "byte_offset": \(offset),
      "byte_length": \(length)
    }
    """
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
