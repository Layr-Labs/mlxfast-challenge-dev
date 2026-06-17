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
    let fixture = try makePreflightFixture()

    let report = try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
    )

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
    #expect(report.mactopPath == fixture.mactop.path)
}

@Test
func benchmarkPreflightRejectsMissingExpertManifest() throws {
    let fixture = try makePreflightFixture(writeManifest: false)

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightRejectsMalformedGolden() throws {
    let fixture = try makePreflightFixture(goldenContents: "{}")

    #expect(throws: Error.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightRejectsUnreadableExpertByteRange() throws {
    let fixture = try makePreflightFixture(expertByteLengthOverride: 1_000_000)

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

private struct PreflightFixture {
    let weights: URL
    let golden: URL
    let mactop: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func makePreflightFixture(
    writeManifest: Bool = true,
    goldenContents: String? = nil,
    expertByteLengthOverride: Int? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    let reference = directory.appendingPathComponent("reference", isDirectory: true)
    let experts = weights.appendingPathComponent("experts", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    try minimalDeepSeekConfigJSON().write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let denseTensor = TensorFixture(
        name: "model.embed_tokens.weight",
        dtype: "U8",
        shape: [1],
        data: Data([1])
    )
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: [denseTensor])
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: [denseTensor],
        shardName: denseShard
    )

    let expertTensor = TensorFixture(
        name: "model.layers.0.ffn.switch_mlp.0.gate_proj.weight",
        dtype: "U8",
        shape: [1],
        data: Data([9])
    )
    let expertShard = "expert-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(expertShard), tensors: [expertTensor])
    if writeManifest {
        try writeExpertManifest(
            experts.appendingPathComponent("manifest.json"),
            referencePath: reference.path,
            shardName: expertShard,
            tensor: expertTensor,
            expertByteLengthOverride: expertByteLengthOverride
        )
    }

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try (goldenContents ?? validGoldenJSON()).write(to: golden, atomically: true, encoding: .utf8)

    let mactop = directory.appendingPathComponent("mactop")
    try "#!/bin/sh\nexit 0\n".write(to: mactop, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: mactop.path
    )

    return PreflightFixture(weights: weights, golden: golden, mactop: mactop)
}

private func minimalDeepSeekConfigJSON() -> String {
    """
    {
      "model_type": "deepseek_v4",
      "vocab_size": \(MLXFastConstants.vocabSize),
      "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
      "n_routed_experts": \(MLXFastConstants.routedExperts),
      "num_experts_per_tok": \(MLXFastConstants.expertsPerToken)
    }
    """
}

private func validGoldenJSON() -> String {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": [1],
          "expected_tokens": \(expected)
        }
      ]
    }
    """
}

private func writeIndex(_ path: URL, tensors: [TensorFixture], shardName: String) throws {
    let entries = tensors.map { #""\#($0.name)": "\#(shardName)""# }.joined(separator: ",")
    try """
    {
      "weight_map": {
        \(entries)
      }
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

private func writeExpertManifest(
    _ path: URL,
    referencePath: String,
    shardName: String,
    tensor: TensorFixture,
    expertByteLengthOverride: Int?
) throws {
    let header = try Safetensors.readHeader(URL(fileURLWithPath: referencePath).appendingPathComponent(shardName))
    let info = try #require(header.tensors[tensor.name])
    try """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(referencePath)",
      "expert_tensors": [
        {
          "name": "\(tensor.name)",
          "shard": "\(shardName)",
          "dtype": "\(tensor.dtype)",
          "shape": \(arrayJSON(tensor.shape)),
          "data_offsets": [\(info.dataStart), \(info.dataEnd)],
          "byte_offset": \(Int(header.dataBaseOffset) + info.dataStart),
          "byte_length": \(expertByteLengthOverride ?? info.byteCount)
        }
      ]
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
