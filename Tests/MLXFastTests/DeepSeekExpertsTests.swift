import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastDeepSeek
import Testing

@Test
func deepSeekRoutedExpertsRunsSelectedExpertsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let fixture = try makeExpertFixture(
        expertTensors: [
            floatTensor(
                name: "model.layers.0.ffn.switch_mlp.0.gate_proj.weight",
                shape: [2, 2],
                values: [0, 0, 0, 0]
            ),
            floatTensor(
                name: "model.layers.0.ffn.switch_mlp.0.up_proj.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.switch_mlp.0.down_proj.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w1.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w3.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w2.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
        ]
    )
    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let output = try DeepSeekRoutedExperts.forward(
        MLXArray([Float(1), 2], [1, 1, 2]),
        expertIndices: MLXArray([Int32(0), Int32(1)], [1, 1, 2]),
        loader: loader,
        spec: DeepSeekRoutedExpertSpec(
            layerIndex: 0,
            hiddenSize: 2,
            intermediateSize: 2,
            swigluLimit: 0
        )
    )

    #expect(output.shape == [1, 1, 2, 2])
    let values = output.asArray(Float.self)
    #expect(values[0] == 0)
    #expect(values[1] == 0)
    #expect(abs(values[2] - 0.7310586) < 1e-5)
    #expect(abs(values[3] - 3.5231884) < 1e-5)
}

@Test
func deepSeekMoEForwardRoutesAndRunsExpertsWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let fixture = try makeExpertFixture(
        denseTensors: [
            floatTensor(
                name: "model.layers.0.ffn.gate.weight",
                shape: [2, 2],
                values: [0, 0, 1, 0]
            ),
            floatTensor(
                name: "model.layers.0.ffn.shared_experts.gate_proj.weight",
                shape: [2, 2],
                values: [0, 0, 0, 0]
            ),
            floatTensor(
                name: "model.layers.0.ffn.shared_experts.up_proj.weight",
                shape: [2, 2],
                values: [0, 0, 0, 0]
            ),
            floatTensor(
                name: "model.layers.0.ffn.shared_experts.down_proj.weight",
                shape: [2, 2],
                values: [0, 0, 0, 0]
            ),
        ],
        expertTensors: [
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w1.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w3.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            floatTensor(
                name: "model.layers.0.ffn.experts.1.w2.weight",
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
        ]
    )
    let loader = try DeepSeekWeightLoader(weightsPath: fixture.weights.path)
    let weights = try loader.moeWeights(
        layerIndex: 0,
        hiddenSize: 2,
        routedExperts: 2,
        vocabSize: 4,
        expertsPerToken: 1,
        sharedIntermediateSize: 2,
        isHashLayer: false
    )
    let output = try DeepSeekMoE.forward(
        MLXArray([Float(1), 2], [1, 1, 2]),
        inputIDs: nil,
        weights: weights,
        loader: loader,
        spec: DeepSeekMoESpec(
            routedExperts: DeepSeekRoutedExpertSpec(
                layerIndex: 0,
                hiddenSize: 2,
                intermediateSize: 2,
                swigluLimit: 0
            ),
            expertsPerToken: 1,
            routedScalingFactor: 1,
            normTopKProb: true,
            scoring: .sigmoid
        )
    )

    #expect(output.shape == [1, 1, 2])
    let values = output.asArray(Float.self)
    #expect(abs(values[0] - 0.7310586) < 1e-5)
    #expect(abs(values[1] - 3.5231884) < 1e-5)
}

@Test
func expertTensorRecordNormalizesLegacyProjectionNames() {
    let record = ExpertTensorRecord(
        name: "model.layers.0.ffn.experts.7.w2.weight",
        shard: "model.safetensors",
        dtype: "F32",
        shape: [2, 2],
        dataOffsets: [0, 16],
        byteOffset: 8,
        byteLength: 16
    )

    #expect(record.layerIndex == 0)
    #expect(record.expertIndex == 7)
    #expect(record.projection == "down_proj")
}

private struct ExpertFixture {
    let weights: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func floatTensor(name: String, shape: [Int], values: [Float]) -> TensorFixture {
    var data = Data()
    for value in values {
        var little = value.bitPattern.littleEndian
        data.append(Data(bytes: &little, count: MemoryLayout<UInt32>.size))
    }
    return TensorFixture(name: name, dtype: "F32", shape: shape, data: data)
}

private func makeExpertFixture(
    denseTensors: [TensorFixture] = [],
    expertTensors: [TensorFixture]
) throws -> ExpertFixture {
    let root = try temporaryDirectory()
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let experts = weights.appendingPathComponent("experts", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

    let denseShard = "model-00001.safetensors"
    let allDenseTensors = denseTensors + [
            TensorFixture(
                name: "model.embed_tokens.weight",
                dtype: "U8",
                shape: [1],
                data: Data([0])
            )
    ]
    try writeSafetensors(
        weights.appendingPathComponent(denseShard),
        tensors: allDenseTensors
    )
    let denseMap = allDenseTensors
        .map { #""\#($0.name)": "\#(denseShard)""# }
        .joined(separator: ",\n        ")
    try """
    {
      "weight_map": {
        \(denseMap)
      }
    }
    """.write(
        to: weights.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let expertShard = "experts.safetensors"
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

    return ExpertFixture(weights: weights)
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
