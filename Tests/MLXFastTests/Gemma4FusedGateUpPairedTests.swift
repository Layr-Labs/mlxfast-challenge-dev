import Foundation
import MLX
@testable import MLXFastCore
@testable import MLXFastModel
import Testing

/// Deterministic 64-bit LCG so synthetic projection contents are stable
/// across runs and machines.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

private let gemma4GateUpRows = 21_504
private let gemma4GateUpInputWidth = 5_376
private let gemma4GateUpWeightWords = 672
private let gemma4GateUpGroups = 84

private struct SyntheticGateUpProjection {
    let projection: FastQuantizedProjection
    let metadata: IndexedAffineMetadata
}

/// Build a synthetic production-geometry 4-bit projection whose affine
/// metadata draws from a bounded pair set, so the packed12/co-tiled payload
/// eligibility (LUT <= 4096) always holds.
private func makeSyntheticGateUpProjection(
    seed: UInt64,
    uniquePairCount: Int = 1_024
) -> SyntheticGateUpProjection {
    var generator = SplitMix64(seed: seed)

    var weightWords = [UInt32]()
    weightWords.reserveCapacity(gemma4GateUpRows * gemma4GateUpWeightWords)
    for _ in 0..<(gemma4GateUpRows * gemma4GateUpWeightWords) {
        weightWords.append(UInt32(truncatingIfNeeded: generator.next()))
    }

    var scaleValues = [Float]()
    var biasValues = [Float]()
    scaleValues.reserveCapacity(uniquePairCount)
    biasValues.reserveCapacity(uniquePairCount)
    for pair in 0..<uniquePairCount {
        scaleValues.append(0.0005 * Float(pair + 1))
        biasValues.append(0.001 * Float(pair) - 0.25)
    }

    var scales = [Float]()
    var biases = [Float]()
    scales.reserveCapacity(gemma4GateUpRows * gemma4GateUpGroups)
    biases.reserveCapacity(gemma4GateUpRows * gemma4GateUpGroups)
    for _ in 0..<(gemma4GateUpRows * gemma4GateUpGroups) {
        let pair = Int(generator.next() % UInt64(uniquePairCount))
        scales.append(scaleValues[pair])
        biases.append(biasValues[pair])
    }

    let weight = MLXArray(weightWords, [gemma4GateUpRows, gemma4GateUpWeightWords])
    let scalesArray = MLXArray(scales, [gemma4GateUpRows, gemma4GateUpGroups])
        .asType(.bfloat16)
    let biasesArray = MLXArray(biases, [gemma4GateUpRows, gemma4GateUpGroups])
        .asType(.bfloat16)
    eval(weight, scalesArray, biasesArray)

    let projection = FastQuantizedProjection(
        weight: weight,
        scales: scalesArray,
        biases: biasesArray,
        groupSize: 64,
        bits: 4
    )
    let metadata = makeIndexedAffineMetadata(
        scales: scalesArray,
        biases: biasesArray
    )
    return SyntheticGateUpProjection(projection: projection, metadata: metadata)
}

/// The paired-load co-tiled gate/up payload must be a pure re-ordering of the
/// co-tiled payload's pair-tile weight words: same values, `[evenodd, proj,
/// row, lane]` -> `[proj, row, lane, evenodd]`, with metadata words and the
/// 256-input tail tile untouched.
@Test
func gemma4PairedCoTiledGateUpPayloadIsAPureReordering() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    var generator = SplitMix64(seed: 0x51ed_c0de)
    var gateWeightValues = [UInt32]()
    var upWeightValues = [UInt32]()
    gateWeightValues.reserveCapacity(gemma4GateUpRows * gemma4GateUpWeightWords)
    upWeightValues.reserveCapacity(gemma4GateUpRows * gemma4GateUpWeightWords)
    for _ in 0..<(gemma4GateUpRows * gemma4GateUpWeightWords) {
        gateWeightValues.append(UInt32(truncatingIfNeeded: generator.next()))
        upWeightValues.append(UInt32(truncatingIfNeeded: generator.next()))
    }
    var gateIndexValues = [UInt16]()
    var upIndexValues = [UInt16]()
    gateIndexValues.reserveCapacity(gemma4GateUpRows * gemma4GateUpGroups)
    upIndexValues.reserveCapacity(gemma4GateUpRows * gemma4GateUpGroups)
    for _ in 0..<(gemma4GateUpRows * gemma4GateUpGroups) {
        gateIndexValues.append(UInt16(generator.next() % 4_096))
        upIndexValues.append(UInt16(generator.next() % 4_096))
    }

    let gateWeight = MLXArray(
        gateWeightValues, [gemma4GateUpRows, gemma4GateUpWeightWords])
    let upWeight = MLXArray(
        upWeightValues, [gemma4GateUpRows, gemma4GateUpWeightWords])
    let gateIndices = MLXArray(
        gateIndexValues, [gemma4GateUpRows, gemma4GateUpGroups])
    let upIndices = MLXArray(
        upIndexValues, [gemma4GateUpRows, gemma4GateUpGroups])

    let coTiled = try #require(gemma4MakeCoTiledFixed12GateUpPayload(
        gateWeight: gateWeight,
        upWeight: upWeight,
        gateIndices: gateIndices,
        upIndices: upIndices
    ))
    let paired = try #require(gemma4MakePairedCoTiledFixed12GateUpPayload(
        coTiledWords: coTiled
    ))

    #expect(paired.shape == coTiled.shape)
    let coTiledWords = coTiled.asArray(UInt32.self)
    let pairedWords = paired.asArray(UInt32.self)

    let wordsPerThreadgroup = 5_632
    let wordsPerPairTile = 536
    let pairTiles = 10
    var mismatches = 0
    for threadgroup in [0, 1, 2_687, 5_375] {
        let base = threadgroup * wordsPerThreadgroup
        for tile in 0..<pairTiles {
            let tileBase = base + tile * wordsPerPairTile
            for evenOdd in 0..<2 {
                for projection in 0..<2 {
                    for row in 0..<4 {
                        for lane in 0..<32 {
                            let coTiledIndex = tileBase
                                + evenOdd * 256
                                + projection * 128
                                + row * 32
                                + lane
                            let pairedIndex = tileBase
                                + projection * 256
                                + row * 64
                                + lane * 2
                                + evenOdd
                            if coTiledWords[coTiledIndex]
                                != pairedWords[pairedIndex]
                            {
                                mismatches += 1
                            }
                        }
                    }
                }
            }
            for metadataWord in 512..<536 {
                if coTiledWords[tileBase + metadataWord]
                    != pairedWords[tileBase + metadataWord]
                {
                    mismatches += 1
                }
            }
        }
        for tailWord in (pairTiles * wordsPerPairTile)..<wordsPerThreadgroup {
            if coTiledWords[base + tailWord] != pairedWords[base + tailWord] {
                mismatches += 1
            }
        }
    }
    #expect(mismatches == 0)
}

/// On-device raw BF16 equality between the shipped co-tiled fixed12 gate/up
/// activation kernel and the paired-load variant, over synthetic projections
/// at the exact production geometry.
@Test
func gemma4PairedGateUpActivationKernelMatchesCoTiledBits() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    let gate = makeSyntheticGateUpProjection(seed: 0x6a7e)
    let up = makeSyntheticGateUpProjection(seed: 0x0b57_ac1e)
    #expect((1...4_096).contains(gate.metadata.lut.size))
    #expect((1...4_096).contains(up.metadata.lut.size))

    var generator = SplitMix64(seed: 0x1235_813)
    var inputValues = [Float]()
    inputValues.reserveCapacity(gemma4GateUpInputWidth)
    for _ in 0..<gemma4GateUpInputWidth {
        let raw = Double(generator.next() >> 11) * 0x1p-53
        inputValues.append(Float(raw * 2.0 - 1.0))
    }
    let input = MLXArray(inputValues, [1, 1, gemma4GateUpInputWidth])
        .asType(.bfloat16)
    eval(input)

    let runners = try #require(gemma4MakeCoTiledAndPairedGateUpActivationRunners(
        gate: gate.projection,
        up: up.projection,
        gateMetadata: gate.metadata,
        upMetadata: up.metadata
    ))
    let coTiledOutput = runners.coTiled(input)
    let pairedOutput = runners.paired(input)
    let matches = arrayEqual(
        coTiledOutput.view(dtype: .uint16),
        pairedOutput.view(dtype: .uint16)
    )
    eval(matches)
    #expect(matches.item(Bool.self))

    // Also require the outputs to be non-trivial (not all zeros), so an
    // accidentally empty kernel cannot pass the equality check.
    let magnitude = abs(pairedOutput.asType(.float32)).max()
    eval(magnitude)
    #expect(magnitude.item(Float.self) > 0)
}

/// DIRECTIONAL local timing of the shipped co-tiled kernel versus the paired
/// variant. Never a ranked signal (different silicon); opt-in via
/// MLXFAST_RUN_KERNEL_TIMING=1 on top of the runtime-tests flag.
@Test
func gemma4PairedGateUpActivationKernelTiming() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
          ProcessInfo.processInfo.environment["MLXFAST_RUN_KERNEL_TIMING"] == "1"
    else {
        return
    }

    let gate = makeSyntheticGateUpProjection(seed: 0x6a7e)
    let up = makeSyntheticGateUpProjection(seed: 0x0b57_ac1e)
    var generator = SplitMix64(seed: 0x7357)
    var inputValues = [Float]()
    inputValues.reserveCapacity(gemma4GateUpInputWidth)
    for _ in 0..<gemma4GateUpInputWidth {
        let raw = Double(generator.next() >> 11) * 0x1p-53
        inputValues.append(Float(raw * 2.0 - 1.0))
    }
    let input = MLXArray(inputValues, [1, 1, gemma4GateUpInputWidth])
        .asType(.bfloat16)
    eval(input)

    let runners = try #require(gemma4MakeCoTiledAndPairedGateUpActivationRunners(
        gate: gate.projection,
        up: up.projection,
        gateMetadata: gate.metadata,
        upMetadata: up.metadata
    ))

    func timeVariant(_ run: (MLXArray) -> MLXArray) -> Double {
        for _ in 0..<20 {
            eval(run(input))
        }
        let iterations = 100
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            eval(run(input))
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        return Double(elapsed) / Double(iterations) / 1.0e6
    }

    // Interleave measurement order to reduce thermal bias.
    var coTiledSamples = [Double]()
    var pairedSamples = [Double]()
    for _ in 0..<3 {
        coTiledSamples.append(timeVariant(runners.coTiled))
        pairedSamples.append(timeVariant(runners.paired))
    }
    let coTiledBest = coTiledSamples.min() ?? 0
    let pairedBest = pairedSamples.min() ?? 0
    print(
        "gate/up activation QMV timing (best of 3x100): co-tiled "
            + String(format: "%.4f", coTiledBest)
            + " ms vs paired "
            + String(format: "%.4f", pairedBest)
            + " ms per invocation (DIRECTIONAL, local machine only)"
    )
}
