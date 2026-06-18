import Foundation
import MLX
import Testing
@testable import MLXFastCore
@testable import MLXFastModel

@Test
func deepSeekWeightLoaderResolvesDenseTensorNamesAndShapes() throws {
    let fixture = try makeWeightLoaderFixture(
        denseTensors: [
            TensorFixture(
                name: "model.embed_tokens.weight",
                dtype: "U8",
                shape: [2, 2],
                data: Data([1, 2, 3, 4])
            ),
            TensorFixture(
                name: "language_model.lm_head.weight",
                dtype: "U8",
                shape: [2, 2],
                data: Data([5, 6, 7, 8])
            ),
        ],
        expertTensors: [
            TensorFixture(
                name: "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                dtype: "U8",
                shape: [1],
                data: Data([9])
            )
        ]
    )

    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)

    #expect(try loader.resolveDenseName(DeepSeekWeightNames.embedTokens) == "model.embed_tokens.weight")
    #expect(try loader.resolveDenseName(DeepSeekWeightNames.lmHead) == "language_model.lm_head.weight")

    let embed = try loader.materializedDenseTensor(
        candidates: DeepSeekWeightNames.embedTokens,
        expectedShape: [2, 2]
    )
    #expect(try embed.uint8Values() == [1, 2, 3, 4])

    #expect(throws: MLXFastError.self) {
        _ = try loader.materializedDenseTensor(
            candidates: DeepSeekWeightNames.embedTokens,
            expectedShape: [4, 1]
        )
    }
}

@Test
func deepSeekWeightLoaderReadsExpertManifestTensor() throws {
    let expertName = "model.layers.0.ffn.switch_mlp.gate_proj.weight"
    let fixture = try makeWeightLoaderFixture(
        denseTensors: [
            TensorFixture(
                name: "model.embed_tokens.weight",
                dtype: "U8",
                shape: [1],
                data: Data([1])
            )
        ],
        expertTensors: [
            TensorFixture(
                name: expertName,
                dtype: "U8",
                shape: [3],
                data: Data([9, 8, 7])
            )
        ]
    )

    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let expert = try loader.materializedExpertTensor(named: expertName, expectedShape: [3])

    #expect(expert.name == expertName)
    #expect(try expert.uint8Values() == [9, 8, 7])
}

@Test
func deepSeekWeightLoaderBuildsAffineQuantizedDenseLinearWeight() throws {
    let fixture = try makeWeightLoaderFixture(
        denseTensors: [
            TensorFixture(
                name: "model.layers.0.attn.wq_a.weight",
                dtype: "U32",
                shape: [2, 1],
                data: uint32Bytes([1, 2])
            ),
            TensorFixture(
                name: "model.layers.0.attn.wq_a.scales",
                dtype: "BF16",
                shape: [2, 2],
                data: Data(repeating: 0, count: 8)
            ),
            TensorFixture(
                name: "model.layers.0.attn.wq_a.biases",
                dtype: "BF16",
                shape: [2, 2],
                data: Data(repeating: 0, count: 8)
            ),
        ],
        expertTensors: [
            TensorFixture(
                name: "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                dtype: "U8",
                shape: [1],
                data: Data([1])
            )
        ]
    )

    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let weight = try loader.denseLinearWeight(
        candidates: DeepSeekWeightNames.attention(0, "wq_a.weight"),
        expectedShape: [2, 8]
    )

    #expect(weight.isQuantized)
    #expect(weight.shape == [2, 8])
    #expect(weight.weight.shape == [2, 1])
    #expect(weight.scales?.shape == [2, 2])
    #expect(weight.biases?.shape == [2, 2])
    #expect(weight.bits == 4)
    #expect(weight.groupSize == 4)
    #expect(weight.mode == .affine)
}

@Test
func deepSeekWeightLoaderSlicesStackedQuantizedRoutedExpert() throws {
    let fixture = try makeWeightLoaderFixture(
        denseTensors: [
            TensorFixture(
                name: "model.embed_tokens.weight",
                dtype: "U8",
                shape: [1],
                data: Data([1])
            )
        ],
        expertTensors: [
            TensorFixture(
                name: "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                dtype: "U32",
                shape: [3, 2, 1],
                data: uint32Bytes([1, 2, 3, 4, 5, 6])
            ),
            TensorFixture(
                name: "model.layers.0.ffn.switch_mlp.gate_proj.scales",
                dtype: "U8",
                shape: [3, 2, 2],
                data: Data(0..<12)
            ),
        ]
    )

    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let weight = try loader.expertLinearWeight(
        candidates: DeepSeekWeightNames.routedExpert(
            layerIndex: 0,
            expertIndex: 1,
            projection: .gate
        ),
        expectedShape: [2, 8],
        expertIndex: 1
    )

    #expect(weight.isQuantized)
    #expect(weight.shape == [2, 8])
    #expect(weight.weight.shape == [2, 1])
    #expect(weight.scales?.shape == [2, 2])
    #expect(weight.biases == nil)
    #expect(weight.bits == 4)
    #expect(weight.groupSize == 4)
    #expect(weight.mode == .mxfp4)
}

@Test
func deepSeekWeightLoaderBuildsSemanticRuntimeWeights() throws {
    let tensor = { (name: String, shape: [Int]) in
        TensorFixture(
            name: name,
            dtype: "U8",
            shape: shape,
            data: Data(repeating: 1, count: shape.reduce(1, *))
        )
    }
    let fixture = try makeWeightLoaderFixture(
        denseTensors: [
            tensor("model.layers.0.attn.wq_a.weight", [2, 2]),
            tensor("model.layers.0.attn.q_norm.weight", [2]),
            tensor("model.layers.0.attn.wq_b.weight", [2, 2]),
            tensor("model.layers.0.attn.wkv.weight", [2, 2]),
            tensor("model.layers.0.attn.kv_norm.weight", [2]),
            tensor("model.layers.0.attn.wo_a.weight", [1, 2, 2]),
            tensor("model.layers.0.attn.wo_b.weight", [2, 2]),
            tensor("model.layers.0.attn.attn_sink", [1]),
            tensor("model.layers.0.attn_norm.weight", [2]),
            tensor("model.layers.0.ffn_norm.weight", [2]),
            tensor("model.layers.0.attn_hc.fn", [8, 4]),
            tensor("model.layers.0.attn_hc.base", [8]),
            tensor("model.layers.0.attn_hc.scale", [3]),
            tensor("model.layers.0.ffn_hc.fn", [8, 4]),
            tensor("model.layers.0.ffn_hc.base", [8]),
            tensor("model.layers.0.ffn_hc.scale", [3]),
            tensor("model.layers.0.ffn.shared_experts.gate_proj.weight", [2, 2]),
            tensor("model.layers.0.ffn.shared_experts.up_proj.weight", [2, 2]),
            tensor("model.layers.0.ffn.shared_experts.down_proj.weight", [2, 2]),
        ],
        expertTensors: [
            TensorFixture(
                name: "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                dtype: "U8",
                shape: [1],
                data: Data([1])
            )
        ]
    )

    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let attentionSpec = DeepSeekLocalAttentionSpec(
        numAttentionHeads: 1,
        headDim: 2,
        outputGroups: 1,
        qkRopeHeadDim: 2,
        ropeTheta: 10_000,
        maxPositionEmbeddings: 1_024,
        rmsNormEps: 0
    )
    let attention = try loader.localAttentionWeights(
        layerIndex: 0,
        hiddenSize: 2,
        qLoraRank: 2,
        outputLoraRank: 2,
        spec: attentionSpec
    )
    #expect(attention.wqA.shape == [2, 2])
    #expect(attention.woA.shape == [1, 2, 2])
    #expect(attention.attentionSink?.shape == [1])

    let blockSpec = DeepSeekBlockSpec(
        hcMult: 2,
        hcSinkhornIters: 2,
        hcEps: 0,
        rmsNormEps: 0
    )
    let block = try loader.blockWeights(
        layerIndex: 0,
        hiddenSize: 2,
        spec: blockSpec
    )
    #expect(block.attentionNorm.shape == [2])
    #expect(block.attentionHyperConnection.fn.shape == [8, 4])
    #expect(block.feedForwardHyperConnection.scale.shape == [3])

    let shared = try loader.sharedMLPWeights(
        layerIndex: 0,
        hiddenSize: 2,
        intermediateSize: 2
    )
    #expect(shared.gate.shape == [2, 2])
    #expect(shared.down.shape == [2, 2])
}

private struct WeightLoaderFixture {
    let weights: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func makeWeightLoaderFixture(
    denseTensors: [TensorFixture],
    expertTensors: [TensorFixture]
) throws -> WeightLoaderFixture {
    let root = try temporaryDirectory()
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = weights.appendingPathComponent("experts", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let expertShard = "expert-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(expertShard), tensors: expertTensors)
    let header = try Safetensors.readHeader(reference.appendingPathComponent(expertShard))
    let records = try expertTensors.sorted(by: { $0.name < $1.name }).map { tensor in
        let info = try #require(header.tensors[tensor.name])
        return """
        {
          "name": "\(tensor.name)",
          "shard": "\(expertShard)",
          "dtype": "\(tensor.dtype)",
          "shape": \(arrayJSON(tensor.shape)),
          "data_offsets": [\(info.dataStart), \(info.dataEnd)],
          "byte_offset": \(Int(header.dataBaseOffset) + info.dataStart),
          "byte_length": \(info.byteCount)
        }
        """
    }
    try """
    {
      "version": 1,
      "source": "safetensors",
      "reference_path": "\(reference.path)",
      "expert_tensors": [
        \(records.joined(separator: ","))
      ]
    }
    """.write(
        to: experts.appendingPathComponent("manifest.json"),
        atomically: true,
        encoding: .utf8
    )

    return WeightLoaderFixture(weights: weights)
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

private func uint32Bytes(_ values: [UInt32]) -> Data {
    var data = Data()
    for value in values {
        var littleEndian = value.littleEndian
        data.append(Data(bytes: &littleEndian, count: 4))
    }
    return data
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
