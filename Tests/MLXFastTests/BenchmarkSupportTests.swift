import Foundation
@testable import MLXFastCore
@testable import MLXFastDeepSeek
@testable import MLXFastDeepSeekHarness
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

    #expect(MactopBandwidth.parseSamples(from: data) == [3.25, 4.75, 0])
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
func decodeTimingPlanStartsAfterSeedPrefill() throws {
    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    var offsets: [Int] = []
    for step in 0..<plan.decodeSteps {
        offsets.append(try plan.positionOffset(forDecodedStep: step))
    }

    #expect(offsets == [32, 33, 34, 35])
}

@Test
func decodeTimingPlanRejectsInvalidRanges() throws {
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 0, decodeSteps: 4)
    }
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 0)
    }

    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    #expect(throws: MLXFastError.self) {
        _ = try plan.positionOffset(forDecodedStep: 4)
    }
}

@Test
func benchmarkPromptPlanUsesHiddenBenchmarkOracle() throws {
    let prefill = Array(0..<MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(0..<MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps)
    let plan = try BenchmarkPrompt.plan(from: BenchmarkGolden(
        prefillPromptTokens: prefill,
        expectedPrefillToken: 17,
        decodeSeedTokens: seed,
        expectedDecodeSeedToken: 23,
        expectedDecodeTokens: decode
    ))

    #expect(plan.prefillTokens == prefill)
    #expect(plan.expectedPrefillToken == 17)
    #expect(plan.decodeSeedTokens == seed)
    #expect(plan.expectedDecodeSeedToken == 23)
    #expect(plan.expectedDecodeTokens == decode)
}

@Test
func benchmarkPromptPlanRejectsMalformedBenchmarkOracle() {
    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPrompt.plan(from: BenchmarkGolden(
            prefillPromptTokens: [1],
            expectedPrefillToken: 7,
            decodeSeedTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkDecodeSeedTokens),
            expectedDecodeSeedToken: 7,
            expectedDecodeTokens: Array(repeating: 7, count: MLXFastConstants.benchmarkDecodeSteps)
        ))
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
func mactopIdleMeasurementStopsAfterEnoughSamples() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let executable = try writeExecutableScript(
        directory.appendingPathComponent("mactop"),
        contents: """
        #!/bin/sh
        printf '%s\\n' '{"soc_metrics":{"dram_bw_combined_gbs":6}}'
        printf '%s\\n' '{"soc_metrics":{"dram_bw_combined_gbs":7}}'
        sleep 5
        """
    )

    let samples = try MactopSession.measureIdleSamples(
        sampleCount: 2,
        timeoutSeconds: 1,
        environment: ["MLXFAST_MACTOP_BIN": executable.path]
    )

    #expect(samples == [6, 7])
}

@Test
func mactopIdleMeasurementTimesOutWithoutSamples() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let executable = try writeExecutableScript(
        directory.appendingPathComponent("mactop"),
        contents: """
        #!/bin/sh
        sleep 5
        """
    )

    #expect(throws: MLXFastError.self) {
        _ = try MactopSession.measureIdleSamples(
            sampleCount: 1,
            timeoutSeconds: 0.2,
            environment: ["MLXFAST_MACTOP_BIN": executable.path]
        )
    }
}

@Test
func benchmarkPreflightAcceptsRequiredArtifacts() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
    )

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
    #expect(report.mactopPath == fixture.mactop.path)
    #expect(report.weightsByteCount > 0)
    #expect(report.maxWeightsByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
}

@Test
func benchmarkPreflightRejectsWeightsAboveDefaultByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("oversized.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("large-but-allowed.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )
    let override = MLXFastConstants.defaultMaxTransformedWeightsBytes * 2

    let report = try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: [
            "MLXFAST_MACTOP_BIN": fixture.mactop.path,
            "MLXFAST_MAX_WEIGHTS_BYTES": "\(override)",
        ]
    )

    #expect(report.weightsByteCount > MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.maxWeightsByteCount == override)
}

@Test
func benchmarkPreflightRejectsMissingExpertManifest() throws {
    let fixture = try makePreflightFixture(writeManifest: false)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

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
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: Error.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightRejectsShortBenchmarkPrompt() throws {
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(benchmarkPromptTokens: [1]))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightRejectsMissingBenchmarkOracle() throws {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let fixture = try makePreflightFixture(goldenContents: """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

@Test
func benchmarkPreflightRejectsMissingSemanticTensor() throws {
    let fixture = try makePreflightFixture(omitDenseTensorName: DeepSeekWeightNames.finalNorm[0])
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
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
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try BenchmarkPreflight.check(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            environment: ["MLXFAST_MACTOP_BIN": fixture.mactop.path]
        )
    }
}

private struct PreflightFixture {
    let root: URL
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
    omitDenseTensorName: String? = nil,
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

    var denseTensors = requiredDenseTensorFixtures()
    if let omitDenseTensorName {
        denseTensors.removeAll { $0.name == omitDenseTensorName }
    }
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let expertTensors = requiredStackedExpertTensorFixtures()
    let expertShard = "expert-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(expertShard), tensors: expertTensors)
    if writeManifest {
        try writeExpertManifest(
            experts.appendingPathComponent("manifest.json"),
            referencePath: reference.path,
            shardName: expertShard,
            tensors: expertTensors,
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

    return PreflightFixture(root: directory, weights: weights, golden: golden, mactop: mactop)
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

private func validGoldenJSON(
    correctnessPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens),
    benchmarkPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
) -> String {
    let correctnessPrompt = arrayJSON(correctnessPromptTokens)
    let benchmarkPrompt = arrayJSON(benchmarkPromptTokens)
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let seed = arrayJSON(Array(benchmarkPromptTokens.prefix(MLXFastConstants.benchmarkDecodeSeedTokens)))
    let decode = arrayJSON(Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps))
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(correctnessPrompt),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(benchmarkPrompt),
        "expected_prefill_token": 8,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 7,
        "expected_decode_tokens": \(decode)
      }
    }
    """
}

private func requiredDenseTensorFixtures() -> [TensorFixture] {
    var tensors: [String: TensorFixture] = [:]
    func add(_ candidates: [String], shape: [Int]) {
        let name = candidates[0]
        tensors[name] = TensorFixture(
            name: name,
            dtype: "U8",
            shape: shape,
            data: Data([UInt8((tensors.count % 251) + 1)])
        )
    }

    let hidden = MLXFastConstants.hiddenSize
    let vocab = MLXFastConstants.vocabSize
    let layers = MLXFastConstants.numHiddenLayers
    let routedExperts = MLXFastConstants.routedExperts
    let moe = MLXFastConstants.moeIntermediateSize
    let heads = MLXFastConstants.attentionHeads
    let headDim = 512
    let qLoraRank = 1_024
    let outputGroups = 8
    let outputLoraRank = 1_024
    let groupedInput = heads * headDim / outputGroups
    let hcMult = 4
    let hcMix = (2 + hcMult) * hcMult
    let indexHeads = 64
    let indexHeadDim = 128
    let compressRatios = DeepSeekConfig.defaultCompressRatios(layerCount: layers)

    add(DeepSeekWeightNames.embedTokens, shape: [vocab, hidden])
    add(DeepSeekWeightNames.finalNorm, shape: [hidden])
    add(DeepSeekWeightNames.hcHeadFn, shape: [hcMult, hcMult * hidden])
    add(DeepSeekWeightNames.hcHeadBase, shape: [hcMult])
    add(DeepSeekWeightNames.hcHeadScale, shape: [1])
    add(DeepSeekWeightNames.lmHead, shape: [vocab, hidden])

    for layerIndex in 0..<layers {
        add(DeepSeekWeightNames.attentionNorm(layerIndex), shape: [hidden])
        add(DeepSeekWeightNames.feedForwardNorm(layerIndex), shape: [hidden])
        for block in [DeepSeekHyperConnectionBlock.attention, .feedForward] {
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .fn),
                shape: [hcMix, hcMult * hidden]
            )
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .base),
                shape: [hcMix]
            )
            add(
                DeepSeekWeightNames.hyperConnection(layerIndex: layerIndex, block: block, component: .scale),
                shape: [3]
            )
        }

        add(DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"), shape: [qLoraRank, hidden])
        add(DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"), shape: [qLoraRank])
        add(DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"), shape: [heads * headDim, qLoraRank])
        add(DeepSeekWeightNames.attention(layerIndex, "wkv.weight"), shape: [headDim, hidden])
        add(DeepSeekWeightNames.attention(layerIndex, "kv_norm.weight"), shape: [headDim])
        add(
            DeepSeekWeightNames.attention(layerIndex, "wo_a.weight"),
            shape: [outputGroups, outputLoraRank, groupedInput]
        )
        add(
            DeepSeekWeightNames.attention(layerIndex, "wo_b.weight"),
            shape: [hidden, outputGroups * outputLoraRank]
        )

        let ratio = compressRatios[layerIndex]
        if ratio != 0 {
            let outDim = headDim * (ratio == 4 ? 2 : 1)
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.wkv.weight"), shape: [outDim, hidden])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.wgate.weight"), shape: [outDim, hidden])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.ape"), shape: [ratio, outDim])
            add(DeepSeekWeightNames.attention(layerIndex, "compressor.norm.weight"), shape: [headDim])

            if ratio == 4 {
                let indexOutDim = indexHeadDim * 2
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.wq_b.weight"),
                    shape: [indexHeads * indexHeadDim, qLoraRank]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.weights_proj.weight"),
                    shape: [indexHeads, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wkv.weight"),
                    shape: [indexOutDim, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wgate.weight"),
                    shape: [indexOutDim, hidden]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.ape"),
                    shape: [ratio, indexOutDim]
                )
                add(
                    DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.norm.weight"),
                    shape: [indexHeadDim]
                )
            }
        }

        add(DeepSeekWeightNames.feedForward(layerIndex, "gate.weight"), shape: [routedExperts, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"), shape: [moe, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"), shape: [moe, hidden])
        add(DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"), shape: [hidden, moe])
    }

    return tensors.values.sorted { $0.name < $1.name }
}

private func requiredStackedExpertTensorFixtures() -> [TensorFixture] {
    var tensors: [TensorFixture] = []
    let routedExperts = MLXFastConstants.routedExperts
    let hidden = MLXFastConstants.hiddenSize
    let moe = MLXFastConstants.moeIntermediateSize
    for layerIndex in 0..<MLXFastConstants.numHiddenLayers {
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .gate)[0],
                dtype: "U8",
                shape: [routedExperts, moe, hidden],
                data: Data([1])
            )
        )
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .up)[0],
                dtype: "U8",
                shape: [routedExperts, moe, hidden],
                data: Data([2])
            )
        )
        tensors.append(
            TensorFixture(
                name: DeepSeekWeightNames.routedExpert(layerIndex: layerIndex, expertIndex: 0, projection: .down)[0],
                dtype: "U8",
                shape: [routedExperts, hidden, moe],
                data: Data([3])
            )
        )
    }
    return tensors
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
    tensors: [TensorFixture],
    expertByteLengthOverride: Int?
) throws {
    let header = try Safetensors.readHeader(URL(fileURLWithPath: referencePath).appendingPathComponent(shardName))
    let overrideName = tensors.first?.name
    let records = try tensors.map { tensor in
        let info = try #require(header.tensors[tensor.name])
        let byteLength = tensor.name == overrideName ? (expertByteLengthOverride ?? info.byteCount) : info.byteCount
        return """
        {
          "name": "\(tensor.name)",
          "shard": "\(shardName)",
          "dtype": "\(tensor.dtype)",
          "shape": \(arrayJSON(tensor.shape)),
          "data_offsets": [\(info.dataStart), \(info.dataEnd)],
          "byte_offset": \(Int(header.dataBaseOffset) + info.dataStart),
          "byte_length": \(byteLength)
        }
        """
    }.joined(separator: ",\n")
    try """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(referencePath)",
      "expert_tensors": [
        \(records)
      ]
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        let byteCount = try expectedTensorByteCount(
            name: tensor.name,
            dtype: TensorDType.parse(tensor.dtype),
            shape: tensor.shape
        )
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + byteCount],
        ]
        cursor += byteCount
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    try output.write(to: path)

    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(output.count + cursor))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func writeExecutableScript(_ path: URL, contents: String) throws -> URL {
    try contents.write(to: path, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: path.path
    )
    return path
}

private func writeSparseFile(_ path: URL, byteCount: Int) throws {
    _ = FileManager.default.createFile(atPath: path.path, contents: nil)
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
