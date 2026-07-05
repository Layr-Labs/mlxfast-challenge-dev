import Foundation
import MLX

/// GPU palette decode for the packed e8m0 expert scales.
///
/// The offline transform stores each routed-expert U8 scales tensor as 4-bit
/// palette indices (two elements per byte, low nibble = even element) plus a
/// 16-entry palette. This helper rebuilds the exact U8 scales MLXArray with a
/// tiny lazy graph of exact integer ops — nibble split, interleave, palette
/// gather — instead of a CPU LUT loop plus a full-size host copy:
///
///     low  = nibbles & 0x0F          // even elements' palette indices
///     high = nibbles >> 4            // odd elements' palette indices
///     idx  = stacked([low, high], axis: -1).reshaped(decodedShape)
///     out  = take(palette, idx)      // uint8, byte-identical to CPU decode
///
/// Every op preserves `.uint8` (operator forms coerce scalars to the lhs
/// dtype; `take` returns the palette's dtype), so the `mode == .mxfp4`
/// derivation and the quantizedMM kernel see exactly the bytes the CPU decode
/// would produce. Host->GPU traffic is the packed nibbles — half the bytes of
/// the decoded tensor — and the expansion happens at eval on the GPU.
enum DeepSeekScaleDecode {
    /// Builds the decoded U8 scales array from a packed view. Returns nil when
    /// the view is malformed (callers then fall back to the CPU byte decode,
    /// which reproduces existing behavior exactly).
    static func scalesArray(from view: ResidentExpertTensors.PackedScaleView) -> MLXArray? {
        let shape = view.decodedShape
        guard
            let groups = shape.last,
            groups > 0,
            groups % 2 == 0,
            view.palette.count == 16
        else {
            return nil
        }
        var packedShape = shape
        packedShape[packedShape.count - 1] = groups / 2
        let packedCount = packedShape.reduce(1, *)
        guard packedCount == view.nibbles.count else {
            return nil
        }

        let nibbles = MLXArray(view.nibbles, packedShape, dtype: .uint8)
        let low = nibbles & 0x0F
        let high = nibbles >> 4
        let indices = stacked([low, high], axis: -1).reshaped(shape)
        let palette = MLXArray(view.palette)
        return take(palette, indices)
    }
}
