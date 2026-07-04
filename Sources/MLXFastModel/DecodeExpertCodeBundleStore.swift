import Foundation
import MLX
import MLXFastCore

/// Optional decode-only overlay produced by the Swift transform.
///
/// Each bundle is one safetensors U8 record containing the gate/up/down code
/// slices for a single (layer, expert). Reading one bundle through ExpertSlotBank
/// keeps expert-read metrics truthful while reducing decode's per-expert code
/// reads from three separate projection preads to one contiguous pread.
final class DecodeExpertCodeBundleStore {
    private struct Root: Decodable {
        let decodeCodeBundles: Index?

        enum CodingKeys: String, CodingKey {
            case decodeCodeBundles = "decode_code_bundles"
        }
    }

    private struct Index: Decodable {
        let version: Int
        let layers: [Layer]
    }

    private struct Layer: Decodable {
        let layerIndex: Int
        let expertCount: Int
        let bundleByteLength: Int
        let records: [ProjectionRecord]

        enum CodingKeys: String, CodingKey {
            case layerIndex = "layer_index"
            case expertCount = "expert_count"
            case bundleByteLength = "bundle_byte_length"
            case records
        }
    }

    private struct ProjectionRecord: Decodable {
        let name: String
        let dtype: String
        let sliceShape: [Int]
        let sliceByteLength: Int
        let bundleOffset: Int

        enum CodingKeys: String, CodingKey {
            case name
            case dtype
            case sliceShape = "slice_shape"
            case sliceByteLength = "slice_byte_length"
            case bundleOffset = "bundle_offset"
        }
    }

    private let bank: ExpertSlotBank
    private let layersByIndex: [Int: Layer]

    init?(manifestPath: String, metrics: ExpertStreamingMetrics?) {
        guard FileManager.default.fileExists(atPath: manifestPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: manifestPath)),
              let root = try? JSONDecoder().decode(Root.self, from: data),
              let index = root.decodeCodeBundles,
              index.version == 1,
              !index.layers.isEmpty,
              let bank = try? ExpertSlotBank(
                  manifestPath: manifestPath,
                  capacity: 0,
                  metrics: metrics
              )
        else {
            return nil
        }
        self.bank = bank
        self.layersByIndex = Dictionary(uniqueKeysWithValues: index.layers.map { ($0.layerIndex, $0) })
    }

    func containsLayer(_ layerIndex: Int) -> Bool {
        layersByIndex[layerIndex] != nil
    }

    func prefetch(
        layerIndex: Int,
        expertIndices: [Int],
        residentScales: ResidentExpertTensors?,
        bridge: MLXArrayTensorBridge
    ) -> [String: StagedExpertCode]? {
        guard let layer = layersByIndex[layerIndex], !expertIndices.isEmpty else {
            return nil
        }
        var uniqueExperts: [Int] = []
        uniqueExperts.reserveCapacity(expertIndices.count)
        var seen = Set<Int>()
        for expertIndex in expertIndices
            where expertIndex >= 0 && expertIndex < layer.expertCount && seen.insert(expertIndex).inserted
        {
            uniqueExperts.append(expertIndex)
        }
        guard !uniqueExperts.isEmpty else {
            return nil
        }

        var results = [[(String, StagedExpertCode)]?](repeating: nil, count: uniqueExperts.count)
        results.withUnsafeMutableBufferPointer { buffer in
            let sink = DecodeExpertBundleSink(buffer: buffer)
            DispatchQueue.concurrentPerform(iterations: uniqueExperts.count) { index in
                let expertIndex = uniqueExperts[index]
                guard let bundle = try? bank.materializedTensor(
                    named: Self.bundleName(layerIndex: layer.layerIndex, expertIndex: expertIndex)
                ), bundle.bytes.count == layer.bundleByteLength
                else {
                    return
                }

                var staged: [(String, StagedExpertCode)] = []
                staged.reserveCapacity(layer.records.count)
                for record in layer.records {
                    let end = record.bundleOffset + record.sliceByteLength
                    guard record.bundleOffset >= 0, end <= bundle.bytes.count,
                          let dtype = try? TensorDType.parse(record.dtype)
                    else {
                        return
                    }
                    let bytes = bundle.bytes[
                        bundle.bytes.startIndex + record.bundleOffset..<bundle.bytes.startIndex + end
                    ]
                    guard
                        let tensor = try? MaterializedTensor(
                            name: "\(record.name)[\(expertIndex)]",
                            dtype: dtype,
                            shape: record.sliceShape,
                            bytes: bytes
                        ),
                        let array = try? bridge.makeArray(from: tensor)
                    else {
                        return
                    }
                    let scalesArray = DeepSeekWeightLoader.residentScalesArray(
                        residentScales: residentScales,
                        bridge: bridge,
                        codeName: record.name,
                        expertIndex: expertIndex
                    )
                    staged.append((
                        DeepSeekWeightLoader.decodePrefetchKey(record.name, expertIndex),
                        StagedExpertCode(
                            tensor: tensor,
                            array: array,
                            scalesArray: scalesArray
                        )
                    ))
                }
                sink.buffer[index] = staged
            }
        }

        var map: [String: StagedExpertCode] = [:]
        map.reserveCapacity(uniqueExperts.count * 3)
        for staged in results {
            for (key, code) in staged ?? [] {
                map[key] = code
            }
        }
        return map.isEmpty ? nil : map
    }

    static func bundleName(layerIndex: Int, expertIndex: Int) -> String {
        "mlxfast.decode_code_bundle.layer_\(layerIndex).expert_\(expertIndex)"
    }
}

private struct DecodeExpertBundleSink: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<[(String, StagedExpertCode)]?>
}
