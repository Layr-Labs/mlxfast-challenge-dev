import Foundation
import MLX

/// Losslessly permutes affine-W4 gate bytes from row-major into
/// [rowTile16][KBlock256][row16][128-byte block] order.
func gemma4PackGateKMajor16(
    weightWords: [UInt32],
    rows: Int = 21_504,
    packedWordsPerRow: Int = 672
) -> [UInt32]? {
    guard rows > 0, rows.isMultiple(of: 16), packedWordsPerRow > 0,
          packedWordsPerRow.isMultiple(of: 32),
          weightWords.count == rows * packedWordsPerRow else { return nil }
    let blocks = packedWordsPerRow / 32
    var result = Array(repeating: UInt32(0), count: weightWords.count)
    for tile in 0..<(rows / 16) {
        for block in 0..<blocks {
            for row in 0..<16 {
                let source = (tile * 16 + row) * packedWordsPerRow + block * 32
                let destination = ((tile * blocks + block) * 16 + row) * 32
                result[destination..<(destination + 32)] = weightWords[source..<(source + 32)]
            }
        }
    }
    return result
}

func gemma4InversePackGateKMajor16(
    packedWords: [UInt32],
    rows: Int = 21_504,
    packedWordsPerRow: Int = 672
) -> [UInt32]? {
    guard rows > 0, rows.isMultiple(of: 16), packedWordsPerRow > 0,
          packedWordsPerRow.isMultiple(of: 32),
          packedWords.count == rows * packedWordsPerRow else { return nil }
    let blocks = packedWordsPerRow / 32
    var result = Array(repeating: UInt32(0), count: packedWords.count)
    for tile in 0..<(rows / 16) {
        for block in 0..<blocks {
            for row in 0..<16 {
                let source = ((tile * blocks + block) * 16 + row) * 32
                let destination = (tile * 16 + row) * packedWordsPerRow + block * 32
                result[destination..<(destination + 32)] = packedWords[source..<(source + 32)]
            }
        }
    }
    return result
}

func supportsKMajorGate(
    input: MLXArray,
    gate: FastQuantizedProjection,
    metadata: IndexedAffineMetadata
) -> Bool {
    input.dtype == .bfloat16 && input.shape == [1, 1, 5_376]
        && gate.groupSize == 64 && gate.bits == 4
        && gate.weight.dtype == .uint32 && gate.weight.shape == [21_504, 672]
        && gate.scales.shape == [21_504, 84]
        && gate.biases?.shape == [21_504, 84]
        && metadata.indices.dtype == .uint16 && metadata.indices.shape == [21_504, 84]
        && metadata.lut.dtype == .uint32 && metadata.lut.ndim == 1
}

struct KMajorGateProjection: @unchecked Sendable {
    let original: FastQuantizedProjection
    let metadata: IndexedAffineMetadata
    let packedWeight: MLXArray

    init?(original: FastQuantizedProjection, metadata: IndexedAffineMetadata) {
        guard original.weight.dtype == .uint32,
              original.weight.shape == [21_504, 672] else { return nil }
        let words = original.weight.asArray(UInt32.self)
        guard let packed = gemma4PackGateKMajor16(weightWords: words) else { return nil }
        self.original = original
        self.metadata = metadata
        self.packedWeight = MLXArray(packed, [21_504, 672])
        eval(self.packedWeight)
    }
}
