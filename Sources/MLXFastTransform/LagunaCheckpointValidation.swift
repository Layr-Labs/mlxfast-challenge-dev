import CoreFoundation
import Foundation
import MLXFastCore

/// Quantization expectations parsed from Poolside Laguna's matching
/// `quantization` and `quantization_config` blocks. The intended checkpoint
/// is exactly NVFP4 4-bit group-16 with no per-tensor overrides.
struct LagunaTransformQuantizationSpec: Equatable {
    struct Override: Equatable {
        let groupSize: Int
        let bits: Int
    }

    let groupSize: Int
    let bits: Int
    let mode: String
    let overrides: [String: Override]

    /// Expected quantization for a quantized tensor, resolved from the
    /// per-tensor overrides with fallback to the global spec. `stem` is the
    /// source tensor name without the trailing `.weight`.
    func expected(forTensorStem stem: String) -> (groupSize: Int, bits: Int) {
        if let override = overrides[stem] {
            return (override.groupSize, override.bits)
        }
        return (groupSize, bits)
    }
}

/// Transform-side structural validation of the Laguna text-tower tensor set
/// against `docs/laguna-weight-contract.md`. The source
/// checkpoint (`poolside/Laguna-XS-2.1-NVFP4-mlx`) is already MLX
/// NVFP4-quantized, so the transform passes tensors through unchanged; this
/// pass fails fast -- before the multi-GB copy -- when the set it would copy
/// cannot satisfy the runtime loader (`LagunaWeightLoader`):
///
/// - every recognized expert projection stored as packed U32 codes must ship
///   U8 `.scales` with matching leading dimensions and no affine `.biases`;
/// - the SwitchGLU expert tensors (`mlp.switch_mlp.{gate,up,down}_proj`)
///   must keep the stacked 3D layout (leading experts axis; never split per
///   expert), while every other projection -- attention q/k/v/o and the
///   per-head `g_proj` gate, the layer-0 dense MLP, the shared expert, the
///   router, the embedding, and the untied `lm_head` -- is 2D;
/// - each tensor's packed width must match the group size and bit width the
///   emitted config.json declares (NVFP4 4-bit group-16);
/// - each raw BF16 router pairs with an F32 correction vector, and the
///   stacked expert tensors agree with the router on expert count.
///
/// Unquantized tensors are passed through without NVFP4 packing checks; the
/// runtime validates their full BF16/F32 dtype and exact geometry against
/// `LagunaConfig` before the first forward.
enum LagunaCheckpointValidation {
    struct RecognizedQuantizedStem: Equatable {
        enum Kind: Equatable {
            /// Ordinary 2D projection: `[rows, in]`.
            case matrix
            /// SwitchGLU stacked expert projection: 3D
            /// `[experts, rows, in]`.
            case stackedExperts
        }

        let kind: Kind
        let layerIndex: Int?
    }

    private static let layerPrefix = "model.layers."

    /// Recognizes the quantized-projection stems of the Laguna text tower.
    /// Norm weights (`*_layernorm`, `q_norm`, `k_norm`, `model.norm`) and
    /// the router correction bias are raw tensors and are not returned here.
    static func recognizedQuantizedStem(forWeightName name: String) -> RecognizedQuantizedStem? {
        guard name.hasSuffix(".weight") else {
            return nil
        }
        guard name.hasPrefix(layerPrefix) else {
            return nil
        }
        let remainder = name.dropFirst(layerPrefix.count)
        guard let separator = remainder.firstIndex(of: "."),
              let layerIndex = Int(remainder[..<separator])
        else {
            return nil
        }
        switch String(remainder[separator...]) {
        case ".mlp.shared_expert.gate_proj.weight",
             ".mlp.shared_expert.up_proj.weight",
             ".mlp.shared_expert.down_proj.weight":
            return RecognizedQuantizedStem(kind: .matrix, layerIndex: layerIndex)
        case ".mlp.switch_mlp.gate_proj.weight",
             ".mlp.switch_mlp.up_proj.weight",
             ".mlp.switch_mlp.down_proj.weight":
            return RecognizedQuantizedStem(kind: .stackedExperts, layerIndex: layerIndex)
        default:
            return nil
        }
    }

    /// Parses the two explicit, byte-contract quantization blocks carried by
    /// Poolside's config. Both blocks are required and must agree exactly.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> LagunaTransformQuantizationSpec {
        func requiredBlock(_ key: String) throws -> LagunaTransformQuantizationSpec {
            guard let value = root[key] else {
                throw MLXFastError.invalidInput("Laguna config is missing required \(key)")
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput("Laguna config \(key) must be an object")
            }

            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys)
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Laguna config \(key) contains unsupported fields: \(unexpectedKeys.sorted().joined(separator: ", "))"
                )
            }
            guard block["group_size"] != nil, block["bits"] != nil, block["mode"] != nil else {
                throw MLXFastError.invalidInput(
                    "Laguna config \(key) must explicitly define group_size, bits, and mode"
                )
            }

            let groupSize = try intField("group_size", in: block, defaultValue: 0)
            let bits = try intField("bits", in: block, defaultValue: 0)
            let mode = try stringField("mode", in: block, defaultValue: "")
            guard mode == "nvfp4",
                  groupSize == 16,
                  bits == 4
            else {
                throw MLXFastError.invalidInput(
                    "Poolside Laguna quantization must be NVFP4 4-bit group_size 16"
                )
            }
            return LagunaTransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                overrides: [:]
            )
        }

        let quantization = try requiredBlock("quantization")
        let quantizationConfig = try requiredBlock("quantization_config")
        guard quantization == quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Laguna config quantization and quantization_config must match exactly"
            )
        }
        return quantization
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: LagunaTransformQuantizationSpec
    ) throws {
        var routerExpertsByLayer: [Int: Int] = [:]
        var stackedExpertsByLayer: [Int: Int] = [:]

        for name in selectedKeys.sorted() {
            guard let recognized = recognizedQuantizedStem(forWeightName: name) else {
                continue
            }
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            let hasScales = selectedKeys.contains(scalesName)
            let hasBiases = selectedKeys.contains(biasesName)
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  hasScales,
                  !hasBiases
            else {
                throw MLXFastError.invalidInput(
                    "Poolside NVFP4 expert \(name) must use U32 packed weights, U8 scales, and no affine biases"
                )
            }
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let expectedRank = recognized.kind == .stackedExperts ? 3 : 2
            guard scalesInfo.dtype == TensorDType.u8.rawValue,
                  weightInfo.shape.count == expectedRank,
                  scalesInfo.shape.count == expectedRank,
                  weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Poolside NVFP4 expert \(stem) has incompatible weight or scale metadata"
                )
            }

            let expected = quantization.expected(forTensorStem: stem)
            let packedWidth = weightInfo.shape[expectedRank - 1]
            let groupCount = scalesInfo.shape[expectedRank - 1]
            let (inputFeatures, inputOverflow) = groupCount.multipliedReportingOverflow(
                by: expected.groupSize
            )
            guard !inputOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) input dimension overflows Int"
                )
            }
            let (packedBits, packedOverflow) = packedWidth.multipliedReportingOverflow(by: 32)
            let (expectedPackedBits, expectedOverflow) = inputFeatures.multipliedReportingOverflow(
                by: expected.bits
            )
            guard !packedOverflow, !expectedOverflow else {
                throw MLXFastError.invalidInput(
                    "Laguna projection \(stem) packed width overflows Int"
                )
            }
            guard packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Laguna projection \(stem) stored width \(packedWidth) does not "
                        + "match config quantization group_size \(expected.groupSize) "
                        + "bits \(expected.bits) for input dimension \(inputFeatures)"
                )
            }

            switch recognized.kind {
            case .matrix:
                break
            case .stackedExperts:
                if let layerIndex = recognized.layerIndex {
                    let experts = weightInfo.shape[0]
                    if let existing = stackedExpertsByLayer[layerIndex], existing != experts {
                        throw MLXFastError.invalidInput(
                            "Laguna layer \(layerIndex) stacked expert tensors disagree on the "
                                + "expert count (\(existing) vs \(experts))"
                        )
                    }
                    stackedExpertsByLayer[layerIndex] = experts
                }
            }
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".mlp.gate.weight") {
            guard let layerIndex = layerIndex(from: name) else {
                throw MLXFastError.invalidInput("invalid Poolside Laguna router tensor name \(name)")
            }
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count == 2,
                  weightInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Poolside Laguna router \(name) must be a non-empty BF16 matrix"
                )
            }
            try validateRouterCorrectionBias(
                routerName: name,
                routerRows: weightInfo.shape[0],
                selectedKeys: selectedKeys,
                index: index,
                headers: headers
            )
            routerExpertsByLayer[layerIndex] = weightInfo.shape[0]
        }

        for (layerIndex, experts) in stackedExpertsByLayer.sorted(by: { $0.key < $1.key }) {
            guard let routerExperts = routerExpertsByLayer[layerIndex] else {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) has NVFP4 expert stacks but no BF16 router"
                )
            }
            if routerExperts != experts {
                throw MLXFastError.invalidInput(
                    "Laguna layer \(layerIndex) router rows (\(routerExperts)) do not match the "
                        + "stacked expert count (\(experts))"
                )
            }
        }
    }

    /// A raw BF16 router (`<layer>.mlp.gate.weight`) always pairs with the
    /// `<layer>.mlp.gate.e_score_correction_bias` vector: F32, one
    /// entry per routed expert.
    private static func validateRouterCorrectionBias(
        routerName: String,
        routerRows: Int,
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let routerStem = String(routerName.dropLast(".weight".count))
        let biasName = routerStem + ".e_score_correction_bias"
        guard selectedKeys.contains(biasName) else {
            throw MLXFastError.invalidInput(
                "Laguna router \(routerStem) is missing its e_score_correction_bias tensor"
            )
        }
        let biasInfo = try tensorInfo(named: biasName, index: index, headers: headers)
        guard biasInfo.dtype == TensorDType.f32.rawValue,
              biasInfo.shape == [routerRows]
        else {
            throw MLXFastError.invalidInput(
                "Laguna router correction bias \(biasName) must be F32 with one entry per expert"
            )
        }
    }

    private static func layerIndex(from name: String) -> Int? {
        guard name.hasPrefix(layerPrefix) else { return nil }
        let remainder = name.dropFirst(layerPrefix.count)
        guard let separator = remainder.firstIndex(of: ".") else { return nil }
        return Int(remainder[..<separator])
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func validateGroupAndBits(
        groupSize: Int,
        bits: Int,
        context: String
    ) throws {
        guard groupSize > 0, [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput(
                "Laguna \(context) must use a positive group size and bits in {2, 4, 8}"
            )
        }
    }

    private static func intField(
        _ key: String,
        in object: [String: Any],
        defaultValue: Int
    ) throws -> Int {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a finite integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String,
        in object: [String: Any],
        defaultValue: String
    ) throws -> String {
        guard let value = object[key], !(value is NSNull) else {
            return defaultValue
        }
        guard let string = value as? String else {
            throw MLXFastError.invalidInput(
                "Laguna quantization field \(key) must be a string"
            )
        }
        return string
    }
}
