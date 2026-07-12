import MLX
import MLXFastCore

// Zero-dup combined projections authored by GPT 5.6 Sol through Gaj's OpenCode Harness.

enum Gemma4CombinedProjectionProfile {
    static let metadataKey = "mlxfast_combined_projection_profile"
    static let value = "gemma4-31b-zero-dup-attention-v1"
    static let shardName = "mlxfast-combined-attention-v1.safetensors"
    static let physicalPrefix = "mlxfast.combined_attention.layers."
    static let rowOrder = "q-k-v;q-k"
    static let tensorByteCount = 3_034_644_480

    struct Component {
        let diskStem: String
        let runtimeStem: String
        let rows: Int
    }

    struct Group {
        let layerIndex: Int
        let physicalStem: String
        let components: [Component]

        var rows: Int { components.reduce(0) { $0 + $1.rows } }
    }

    static let groups: [Group] = (0..<60).map { layerIndex in
        let diskPrefix = "language_model.model.layers.\(layerIndex)"
        let runtimePrefix = "model.layers.\(layerIndex)"
        let physicalPrefix = "\(Self.physicalPrefix)\(layerIndex)"
        if layerIndex % 6 == 5 {
            return Group(
                layerIndex: layerIndex,
                physicalStem: "\(physicalPrefix).self_attn.qk",
                components: [
                    Component(
                        diskStem: "\(diskPrefix).self_attn.q_proj",
                        runtimeStem: "\(runtimePrefix).self_attn.q_proj",
                        rows: 16_384
                    ),
                    Component(
                        diskStem: "\(diskPrefix).self_attn.k_proj",
                        runtimeStem: "\(runtimePrefix).self_attn.k_proj",
                        rows: 2_048
                    ),
                ]
            )
        }
        return Group(
            layerIndex: layerIndex,
            physicalStem: "\(physicalPrefix).self_attn.qkv",
            components: [
                Component(
                    diskStem: "\(diskPrefix).self_attn.q_proj",
                    runtimeStem: "\(runtimePrefix).self_attn.q_proj",
                    rows: 8_192
                ),
                Component(
                    diskStem: "\(diskPrefix).self_attn.k_proj",
                    runtimeStem: "\(runtimePrefix).self_attn.k_proj",
                    rows: 4_096
                ),
                Component(
                    diskStem: "\(diskPrefix).self_attn.v_proj",
                    runtimeStem: "\(runtimePrefix).self_attn.v_proj",
                    rows: 4_096
                ),
            ]
        )
    }

}

/// Returns a contiguous output-row view into a transform-backed parent tensor.
/// The parent must already be materialized so MLX can preserve its backing and
/// represent the result as an offset view rather than a gather/copy operation.
func gemma4CombinedProjectionRowView(
    _ parent: MLXArray,
    rows: Range<Int>
) -> MLXArray {
    precondition(parent.ndim == 2)
    precondition(rows.lowerBound >= 0 && rows.upperBound <= parent.dim(0))
    return parent[rows, 0...]
}

struct CombinedAttentionPrefillProjection: @unchecked Sendable {
    private let combined: FastQuantizedProjection
    private let q: FastQuantizedProjection
    private let k: FastQuantizedProjection
    private let v: FastQuantizedProjection?
    private let qRows: Int
    private let kRows: Int
    private let verifyBits: Bool

    init?(
        combined: FastQuantizedProjection,
        q: FastQuantizedProjection,
        k: FastQuantizedProjection,
        v: FastQuantizedProjection?,
        verifyBits: Bool
    ) {
        guard let combinedBiases = combined.biases,
              combined.groupSize == 64,
              combined.bits == 4,
              combined.weight.dtype == .uint32,
              combined.weight.ndim == 2,
              combined.weight.dim(1) == 672,
              combined.scales.dtype == .bfloat16,
              combined.scales.shape == [combined.weight.dim(0), 84],
              combinedBiases.dtype == .bfloat16,
              combinedBiases.shape == combined.scales.shape,
              combined.weight.dim(0)
                == q.weight.dim(0) + k.weight.dim(0) + (v?.weight.dim(0) ?? 0)
        else {
            return nil
        }
        self.combined = combined
        self.q = q
        self.k = k
        self.v = v
        self.qRows = q.weight.dim(0)
        self.kRows = k.weight.dim(0)
        self.verifyBits = verifyBits
    }

    func callAsFunction(
        _ input: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray?) {
        precondition(input.dtype == .bfloat16)
        precondition(input.ndim == 3 && input.dim(0) == 1 && input.dim(1) > 1)
        precondition(input.dim(2) == 5_376)
        let projected = combined(input)
        let qEnd = qRows
        let kEnd = qEnd + kRows
        let queries = projected[.ellipsis, 0..<qEnd]
        let keys = projected[.ellipsis, qEnd..<kEnd]
        let values = v.map { projection in
            projected[.ellipsis, kEnd..<(kEnd + projection.weight.dim(0))]
        }

        if verifyBits {
            var comparisons = [
                arrayEqual(queries.view(dtype: .uint16), q(input).view(dtype: .uint16)),
                arrayEqual(keys.view(dtype: .uint16), k(input).view(dtype: .uint16)),
            ]
            if let v, let values {
                comparisons.append(arrayEqual(
                    values.view(dtype: .uint16),
                    v(input).view(dtype: .uint16)
                ))
            }
            eval(comparisons)
            precondition(
                comparisons.allSatisfy { $0.item(Bool.self) },
                "combined attention prefill raw projection differs from component views"
            )
        }
        return (queries, keys, values)
    }
}

struct RuntimeCombinedProjectionSet: @unchecked Sendable {
    let attentionByLayer: [Int: FastQuantizedProjection]
    let retainedTensorBytes: Int

    static let empty = RuntimeCombinedProjectionSet(
        attentionByLayer: [:],
        retainedTensorBytes: 0
    )
}

func expandCombinedRuntimeWeights(
    _ loaded: [String: MLXArray],
    profile: String?
) throws -> (weights: [String: MLXArray], projections: RuntimeCombinedProjectionSet) {
    let reservedNames = Set(loaded.keys.filter {
        $0.hasPrefix(Gemma4CombinedProjectionProfile.physicalPrefix)
    })
    guard let profile else {
        guard reservedNames.isEmpty else {
            throw MLXFastError.invalidInput(
                "combined projection arrays require an index profile"
            )
        }
        return (loaded, .empty)
    }
    guard profile == Gemma4CombinedProjectionProfile.value else {
        throw MLXFastError.invalidInput(
            "unsupported combined projection profile: \(profile)"
        )
    }

    let expectedNames = Set(Gemma4CombinedProjectionProfile.groups.flatMap { group in
        ["weight", "scales", "biases"].map { "\(group.physicalStem).\($0)" }
    })
    guard reservedNames == expectedNames else {
        throw MLXFastError.invalidInput(
            "runtime combined projection inventory does not match its profile"
        )
    }

    let parents = expectedNames.sorted().compactMap { loaded[$0] }
    eval(parents)
    var weights = loaded
    for name in expectedNames {
        weights.removeValue(forKey: name)
    }

    var attentionByLayer: [Int: FastQuantizedProjection] = [:]
    var retainedTensorBytes = 0
    var aliases: [MLXArray] = []
    for group in Gemma4CombinedProjectionProfile.groups {
        let weightName = "\(group.physicalStem).weight"
        let scalesName = "\(group.physicalStem).scales"
        let biasesName = "\(group.physicalStem).biases"
        guard let parentWeight = loaded[weightName],
              let parentScales = loaded[scalesName],
              let parentBiases = loaded[biasesName],
              parentWeight.dtype == .uint32,
              parentWeight.shape == [group.rows, 672],
              parentScales.dtype == .bfloat16,
              parentScales.shape == [group.rows, 84],
              parentBiases.dtype == .bfloat16,
              parentBiases.shape == parentScales.shape
        else {
            throw MLXFastError.invalidInput(
                "runtime combined projection has invalid arrays: \(group.physicalStem)"
            )
        }
        let projection = FastQuantizedProjection(
            weight: parentWeight,
            scales: parentScales,
            biases: parentBiases,
            groupSize: 64,
            bits: 4
        )
        attentionByLayer[group.layerIndex] = projection
        for parent in [parentWeight, parentScales, parentBiases] {
            let (next, overflow) = retainedTensorBytes.addingReportingOverflow(parent.nbytes)
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "runtime combined projection byte count overflows Int"
                )
            }
            retainedTensorBytes = next
        }

        var startRow = 0
        for component in group.components {
            let rows = startRow..<(startRow + component.rows)
            for (suffix, parent) in [
                ("weight", parentWeight),
                ("scales", parentScales),
                ("biases", parentBiases),
            ] {
                let componentName = "\(component.runtimeStem).\(suffix)"
                guard weights[componentName] == nil else {
                    throw MLXFastError.invalidInput(
                        "combined projection profile mixes parent and component array \(componentName)"
                    )
                }
                let view = gemma4CombinedProjectionRowView(parent, rows: rows)
                weights[componentName] = view
                aliases.append(view)
            }
            startRow += component.rows
        }
    }
    eval(aliases)
    guard retainedTensorBytes == Gemma4CombinedProjectionProfile.tensorByteCount else {
        throw MLXFastError.invalidInput(
            "runtime combined projection byte count does not match its profile"
        )
    }
    return (
        weights,
        RuntimeCombinedProjectionSet(
            attentionByLayer: attentionByLayer,
            retainedTensorBytes: retainedTensorBytes
        )
    )
}
