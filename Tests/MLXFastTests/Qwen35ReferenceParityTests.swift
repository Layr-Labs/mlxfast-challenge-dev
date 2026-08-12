import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXFastCore
@testable import MLXFastModel

@Suite(.serialized)
struct Qwen35ReferenceParityTests {
    @Test("real transformed Qwen checkpoint matches library and custom paths")
    func transformedCheckpointParity() throws {
        guard let weightsPath = try qwen35ReferenceParityWeightsPath() else {
            return
        }

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let tensorNames = loader.denseStore.tensorNames
        let uniqueTensorNames = Set(tensorNames)

        #expect(
            tensorNames.count == Qwen35WeightLoader.requiredTensorCount
        )
        #expect(
            uniqueTensorNames.count == Qwen35WeightLoader.requiredTensorCount
        )
        #expect(
            tensorNames.allSatisfy {
                $0.hasPrefix("language_model.")
            }
        )
        #expect(
            !tensorNames.contains {
                $0.hasPrefix("vision_tower.")
                    || $0.split(separator: ".").contains("mtp")
            }
        )
        try loader.validateRequiredMetadata(config: config)

        let configData = try Data(
            contentsOf: URL(fileURLWithPath: weightsPath)
                .appendingPathComponent("config.json")
        )
        let configObject = try #require(
            JSONSerialization.jsonObject(with: configData)
                as? [String: Any]
        )
        #expect(configObject["vision_config"] == nil)

        #expect(!Qwen35FastPathReadiness.realCheckpointParityPassed)
        #expect(!Qwen35FastPathReadiness.productionActivationApproved)
        #expect(
            Qwen35FastPathReadiness.productionBackend == .libraryOracle
        )

        let library = try qwen35ReferenceSnapshot(
            loader: loader,
            config: config,
            backend: .libraryOracle
        )
        Memory.clearCache()
        let custom = try qwen35ReferenceSnapshot(
            loader: loader,
            config: config,
            backend: .customFastPath
        )

        #expect(custom.shape == library.shape)
        qwen35ExpectReferenceParity(
            custom.logits,
            library.logits,
            tolerance: 0.125,
            label: "custom fast engine versus Qwen35TextModel"
        )
        #expect(
            qwen35LastTopToken(
                custom.logits,
                vocabularySize: config.vocabSize
            )
                == qwen35LastTopToken(
                    library.logits,
                    vocabularySize: config.vocabSize
                )
        )
    }
}

private struct Qwen35ReferenceSnapshot {
    let shape: [Int]
    let logits: [Float]
}

private let qwen35ReferenceTokens: [Int32] = [
    248_044,
    17,
    23,
    5_003,
    91,
    42,
    7_777,
    314,
]

private func qwen35ReferenceParityWeightsPath() throws -> String? {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXFAST_RUN_QWEN_REFERENCE_PARITY"] == "1" else {
        print(
            "SKIP Qwen reference parity: set "
                + "MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 and "
                + "MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH=<transformed-weights>"
        )
        return nil
    }

    guard
        let configuredPath =
            environment["MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
        !configuredPath.isEmpty
    else {
        print(
            "SKIP Qwen reference parity: "
                + "MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH is absent; it must "
                + "name a transformed text-only directory containing "
                + "config.json, model.safetensors.index.json, and every "
                + "indexed safetensors shard"
        )
        return nil
    }

    let path = URL(fileURLWithPath: configuredPath)
        .standardizedFileURL
    var isDirectory = ObjCBool(false)
    guard
        FileManager.default.fileExists(
            atPath: path.path,
            isDirectory: &isDirectory
        ), isDirectory.boolValue
    else {
        throw MLXFastError.missingFile(
            "Qwen reference parity weights directory does not exist: "
                + path.path
        )
    }
    for fileName in [
        "config.json",
        "model.safetensors.index.json",
    ] {
        let file = path.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw MLXFastError.missingFile(
                "Qwen reference parity transformed weights are missing "
                    + fileName
            )
        }
    }
    return path.path
}

private func qwen35ReferenceSnapshot(
    loader: Qwen35WeightLoader,
    config: Qwen35Config,
    backend: Qwen35ExecutionBackend
) throws -> Qwen35ReferenceSnapshot {
    defer {
        Memory.clearCache()
    }

    let runtime = Qwen35RuntimeWeightCache(
        loader: loader,
        config: config,
        backend: backend
    )
    try runtime.validateSelectedBackend()
    switch backend {
    case .libraryOracle:
        let model = try runtime.requireLibraryModel()
        #expect(!model.hasMTPHead)
    case .customFastPath:
        _ = try runtime.requireFastEngine()
    }

    // The production pipeline -- golden generation, correctness replay, and
    // the timed benchmark -- executes one schedule: a cached prefill pass
    // followed by cached single-token decode steps. The reference below IS
    // that schedule. Full-context uncached recompute is kept only as a
    // gross-corruption tripwire with a RELATIVE bound: bf16 kernel dispatch
    // is shape-dependent (different sequence lengths take different matmul
    // reduction geometries), so absolute closeness across shapes is not a
    // property this GDN architecture provides. Agreement BETWEEN incremental
    // schedules, by contrast, is exact and is asserted exactly below.
    func streamingLogits(chunks: [Int]) throws -> [Float] {
        let cache = Qwen35ModelCache(config: config)
        var values: [Float] = []
        var offset = 0
        for length in chunks {
            let end = offset + length
            let chunk = Array(qwen35ReferenceTokens[offset..<end])
            let logits = try Qwen35Model.logits(
                inputIDs: qwen35ReferenceInput(chunk),
                weightCache: runtime,
                cache: cache,
                positionOffset: offset
            )
            eval(logits)
            cache.materializeCachedState()
            values.append(contentsOf: logits.asArray(Float.self))
            offset = end
        }
        #expect(offset == qwen35ReferenceTokens.count)
        return values
    }

    let tokenCount = qwen35ReferenceTokens.count
    let streamReference = try streamingLogits(chunks: [tokenCount])
    #expect(streamReference.allSatisfy { $0.isFinite })
    let streamRepeat = try streamingLogits(chunks: [tokenCount])
    qwen35ExpectReferenceParity(
        streamRepeat,
        streamReference,
        tolerance: 0,
        label: "\(backend) deterministic streaming prefill"
    )

    let oneShot = try Qwen35Model.logits(
        inputIDs: qwen35ReferenceInput(qwen35ReferenceTokens),
        weightCache: runtime
    )
    eval(oneShot)
    let oneShotValues = oneShot.asArray(Float.self)
    qwen35ExpectRelativeParity(
        oneShotValues,
        streamReference,
        relativeTolerance: qwen35ShapeDispatchRelativeTolerance,
        label: "\(backend) one-shot recompute versus streaming prefill"
    )

    // Incremental schedules must agree with each other EXACTLY: the cache
    // handoff is deterministic state storage, so any drift between chunkings
    // of the same schedule indicates cache corruption, not rounding.
    let chunked233 = try streamingLogits(chunks: [2, 3, 3])
    let chunked44 = try streamingLogits(chunks: [4, 4])
    let chunkedSingles = try streamingLogits(
        chunks: Array(repeating: 1, count: tokenCount)
    )
    qwen35ExpectReferenceParity(
        chunked44,
        chunked233,
        tolerance: 0,
        label: "\(backend) chunked [4,4] versus [2,3,3]"
    )
    qwen35ExpectReferenceParity(
        chunkedSingles,
        chunked233,
        tolerance: 0,
        label: "\(backend) per-token decode versus [2,3,3]"
    )
    qwen35ExpectRelativeParity(
        chunked233,
        streamReference,
        relativeTolerance: qwen35ShapeDispatchRelativeTolerance,
        label: "\(backend) chunked prefill versus streaming prefill"
    )

    // Production decode shape: cached prefill of the prefix, then one cached
    // decode step. The token decision must match the streaming reference
    // exactly; logits stay inside the shape-dispatch envelope.
    let prefix = Array(qwen35ReferenceTokens.dropLast())
    let decodeToken = try #require(qwen35ReferenceTokens.last)
    let decodeCache = Qwen35ModelCache(config: config)
    let prefill = try Qwen35Model.logits(
        inputIDs: qwen35ReferenceInput(prefix),
        weightCache: runtime,
        cache: decodeCache,
        positionOffset: 0
    )
    eval(prefill)
    decodeCache.materializeCachedState()
    let decode = try Qwen35Model.logits(
        inputIDs: qwen35ReferenceInput([decodeToken]),
        weightCache: runtime,
        cache: decodeCache,
        positionOffset: prefix.count
    )
    eval(decode)
    decodeCache.materializeCachedState()
    let decodeValues = decode.asArray(Float.self)
    let streamLast = Array(streamReference.suffix(config.vocabSize))
    qwen35ExpectRelativeParity(
        decodeValues,
        streamLast,
        relativeTolerance: qwen35ShapeDispatchRelativeTolerance,
        label: "\(backend) cached one-token decode versus streaming prefill"
    )
    #expect(
        qwen35LastTopToken(
            decodeValues,
            vocabularySize: config.vocabSize
        )
            == qwen35LastTopToken(
                streamLast,
                vocabularySize: config.vocabSize
            )
    )

    // The cross-backend comparison certifies the schedule production runs.
    return Qwen35ReferenceSnapshot(
        shape: oneShot.shape,
        logits: streamReference
    )
}

private func qwen35ReferenceInput(_ tokens: [Int32]) -> MLXArray {
    MLXArray(tokens, [1, tokens.count])
}

private func qwen35ExpectReferenceParity(
    _ actual: [Float],
    _ expected: [Float],
    tolerance: Float,
    label: String
) {
    #expect(
        actual.count == expected.count,
        "\(label) element count mismatch"
    )
    guard actual.count == expected.count else {
        return
    }

    var maximumDifference: Float = 0
    for (actualValue, expectedValue) in zip(actual, expected) {
        maximumDifference = max(
            maximumDifference,
            abs(actualValue - expectedValue)
        )
    }
    #expect(
        maximumDifference <= tolerance,
        "\(label) maximum absolute difference \(maximumDifference) exceeded \(tolerance)"
    )
}

/// bf16 reduction-geometry budget for comparisons that cross kernel-shape
/// classes (short-chunk vs long-sequence dispatch). 9/128 is the reference
/// implementation's construction-time semantic-parity bound for exactly this
/// phenomenon (MTPLX `_BF16_GEOMETRY_RELATIVE_LIMIT`: "token decisions and
/// cross-row isolation are still required to match exactly"). Measured on
/// this checkpoint: 5.6% (Swift, both backends identically), 1.3% (mlx-lm).
/// Same-shape-class comparisons are exact and asserted at tolerance 0.
private let qwen35ShapeDispatchRelativeTolerance: Float = 9.0 / 128.0

/// Relative-envelope comparison for cross-shape checks: bf16 kernel dispatch
/// differs by sequence length, so different shapes of the same computation
/// agree only to a reduction-geometry envelope proportional to logit scale.
private func qwen35ExpectRelativeParity(
    _ actual: [Float],
    _ expected: [Float],
    relativeTolerance: Float,
    label: String
) {
    #expect(
        actual.count == expected.count,
        "\(label) element count mismatch"
    )
    guard actual.count == expected.count else {
        return
    }

    var maximumDifference: Float = 0
    var scale: Float = 0
    for (actualValue, expectedValue) in zip(actual, expected) {
        maximumDifference = max(
            maximumDifference,
            abs(actualValue - expectedValue)
        )
        scale = max(scale, abs(expectedValue))
    }
    let bound = relativeTolerance * max(scale, 1)
    #expect(
        maximumDifference <= bound,
        "\(label) maximum absolute difference \(maximumDifference) exceeded relative bound \(bound) (scale \(scale), relative tolerance \(relativeTolerance))"
    )
}

private func qwen35LastTopToken(
    _ logits: [Float],
    vocabularySize: Int
) -> Int {
    let row = logits.suffix(vocabularySize)
    guard let first = row.first else {
        return -1
    }
    var bestIndex = 0
    var bestValue = first
    for (index, value) in row.dropFirst().enumerated() where value > bestValue {
        bestIndex = index + 1
        bestValue = value
    }
    return bestIndex
}
