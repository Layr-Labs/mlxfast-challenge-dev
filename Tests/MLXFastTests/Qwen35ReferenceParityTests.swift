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

    let oneShot = try Qwen35Model.logits(
        inputIDs: qwen35ReferenceInput(qwen35ReferenceTokens),
        weightCache: runtime
    )
    let repeated = try Qwen35Model.logits(
        inputIDs: qwen35ReferenceInput(qwen35ReferenceTokens),
        weightCache: runtime
    )
    eval(oneShot, repeated)
    let oneShotValues = oneShot.asArray(Float.self)
    let repeatedValues = repeated.asArray(Float.self)
    #expect(oneShotValues.allSatisfy { $0.isFinite })
    qwen35ExpectReferenceParity(
        repeatedValues,
        oneShotValues,
        tolerance: 0,
        label: "\(backend) deterministic synthetic-token prefill"
    )

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
    let oneShotLast = Array(oneShotValues.suffix(config.vocabSize))
    qwen35ExpectReferenceParity(
        decodeValues,
        oneShotLast,
        tolerance: 0.05,
        label: "\(backend) cached one-token decode versus full context"
    )
    #expect(
        qwen35LastTopToken(
            decodeValues,
            vocabularySize: config.vocabSize
        )
            == qwen35LastTopToken(
                oneShotLast,
                vocabularySize: config.vocabSize
            )
    )

    let chunkCache = Qwen35ModelCache(config: config)
    var chunkedValues: [Float] = []
    var positionOffset = 0
    for chunkLength in [2, 3, 3] {
        let end = positionOffset + chunkLength
        let chunk = Array(
            qwen35ReferenceTokens[positionOffset..<end]
        )
        let logits = try Qwen35Model.logits(
            inputIDs: qwen35ReferenceInput(chunk),
            weightCache: runtime,
            cache: chunkCache,
            positionOffset: positionOffset
        )
        eval(logits)
        chunkCache.materializeCachedState()
        chunkedValues.append(contentsOf: logits.asArray(Float.self))
        positionOffset = end
    }
    #expect(positionOffset == qwen35ReferenceTokens.count)
    qwen35ExpectReferenceParity(
        chunkedValues,
        oneShotValues,
        tolerance: 0.05,
        label: "\(backend) chunked versus one-shot prefill"
    )

    return Qwen35ReferenceSnapshot(
        shape: oneShot.shape,
        logits: oneShotValues
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
