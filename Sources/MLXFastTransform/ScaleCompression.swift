import Foundation
import MLXFastCore

/// Offline lossless 4-bit palette compression for routed-expert e8m0 scale
/// tensors.
///
/// The DeepSeek V4 Flash 4-bit routed experts store their mxfp4 group scales as
/// U8 e8m0 exponents. Empirically every routed `*.scales` tensor uses only a
/// handful of distinct exponent values (4-9 out of 256), so each tensor packs
/// losslessly into 4-bit palette indices (two per byte) plus a tiny per-tensor
/// palette. That halves the ~8.66 GiB of scales to ~4.3 GiB.
///
/// This is a pure transform-time artifact: it writes
/// `experts/scales_packed.safetensors` (the packed nibble bytes) and
/// `experts/scale_compression.json` (palette + original shape per tensor) into
/// the transformed `weights/` tree. The expert manifest is left untouched — its
/// `*.scales` records still describe the original U8 tensors in the reference
/// checkpoint — so `ExpertSlotBank` validation and the streaming fallback path
/// are unchanged. Only `ResidentExpertTensors` opts into the packed overlay,
/// decoding slices back to byte-identical U8 on demand.
///
/// Output is fully deterministic (sorted tensor names, ascending palettes,
/// sorted JSON keys) so `verify-transform` reproduces it byte-for-byte.
public enum ScaleCompression {
    public static let packedShardName = "scales_packed.safetensors"
    public static let sidecarName = "scale_compression.json"
    public static let formatVersion = 1
    private static let maxPaletteSize = 16

    /// A packed scale tensor ready to serialize.
    private struct Packed {
        let name: String
        let palette: [UInt8]
        let shape: [Int]
        let elementCount: Int
        let packed: [UInt8]
    }

    /// Packs every U8 `*.scales` routed-expert tensor listed in the manifest at
    /// `expertsDirectory/manifest.json`, reading the source bytes from the
    /// reference checkpoint. Returns the number of tensors packed (0 when there
    /// is nothing packable, in which case no files are written).
    @discardableResult
    public static func writePackedScales(
        referenceDirectory: URL,
        expertsDirectory: URL
    ) throws -> Int {
        let manifestPath = expertsDirectory.appendingPathComponent("manifest.json").path
        let manifest = try ExpertManifest.load(from: manifestPath)

        let scaleRecords = manifest.expertTensors
            .filter { $0.name.hasSuffix(".scales") && $0.dtype == "U8" }
            .sorted { $0.name < $1.name }
        guard !scaleRecords.isEmpty else {
            return 0
        }

        var packedTensors: [Packed] = []
        packedTensors.reserveCapacity(scaleRecords.count)
        for record in scaleRecords {
            guard
                let firstDimension = record.shape.first,
                record.shape.count >= 2,
                firstDimension > 0
            else {
                continue
            }
            let elementCount = record.shape.reduce(1, *)
            guard elementCount > 0, elementCount == record.byteLength else {
                continue
            }
            // Whole-tensor flatten packing requires each first-axis slice to
            // begin on an even flat-element boundary so a slice maps to whole
            // packed bytes. That holds iff the per-slice element count is even.
            let sliceElementCount = elementCount / firstDimension
            guard sliceElementCount % 2 == 0 else {
                continue
            }

            let bytes = try readTensorBytes(referenceDirectory: referenceDirectory, record: record)
            guard let (palette, packed) = packNibbles(bytes) else {
                // More than 16 distinct values: leave this tensor to the bank.
                continue
            }
            packedTensors.append(
                Packed(
                    name: record.name,
                    palette: palette,
                    shape: record.shape,
                    elementCount: elementCount,
                    packed: packed
                )
            )
        }

        guard !packedTensors.isEmpty else {
            return 0
        }

        try writeSafetensorsShard(
            packedTensors,
            to: expertsDirectory.appendingPathComponent(packedShardName)
        )
        try writeSidecar(
            packedTensors,
            to: expertsDirectory.appendingPathComponent(sidecarName)
        )
        return packedTensors.count
    }

    /// Lossless 4-bit palette pack. Returns nil when the tensor has more than
    /// 16 distinct byte values. The palette is the ascending list of distinct
    /// values, so indices and layout are deterministic.
    static func packNibbles(_ data: Data) -> (palette: [UInt8], packed: [UInt8])? {
        data.withUnsafeBytes { raw -> (palette: [UInt8], packed: [UInt8])? in
            let source = raw.bindMemory(to: UInt8.self)
            let count = source.count
            guard count > 0 else {
                return ([], [])
            }

            var present = [Bool](repeating: false, count: 256)
            for index in 0..<count {
                present[Int(source[index])] = true
            }
            var palette: [UInt8] = []
            palette.reserveCapacity(maxPaletteSize)
            var lut = [UInt8](repeating: 0, count: 256)
            for value in 0..<256 where present[value] {
                if palette.count >= maxPaletteSize {
                    return nil
                }
                lut[value] = UInt8(palette.count)
                palette.append(UInt8(value))
            }

            var packed = [UInt8](repeating: 0, count: (count + 1) / 2)
            packed.withUnsafeMutableBufferPointer { output in
                var readIndex = 0
                var writeIndex = 0
                let pairEnd = count - (count % 2)
                while readIndex < pairEnd {
                    let low = lut[Int(source[readIndex])]
                    let high = lut[Int(source[readIndex + 1])]
                    output[writeIndex] = low | (high << 4)
                    readIndex += 2
                    writeIndex += 1
                }
                if readIndex < count {
                    output[writeIndex] = lut[Int(source[readIndex])]
                }
            }
            return (palette, packed)
        }
    }

    private static func readTensorBytes(
        referenceDirectory: URL,
        record: ExpertTensorRecord
    ) throws -> Data {
        let shardURL = referenceDirectory.appendingPathComponent(record.shard)
        let handle = try FileHandle(forReadingFrom: shardURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(record.byteOffset))
        var data = Data()
        data.reserveCapacity(record.byteLength)
        var remaining = record.byteLength
        while remaining > 0 {
            let chunk = handle.readData(ofLength: min(remaining, 32 * 1024 * 1024))
            if chunk.isEmpty {
                break
            }
            data.append(chunk)
            remaining -= chunk.count
        }
        guard data.count == record.byteLength else {
            throw MLXFastError.invalidInput(
                "scale compression could not read \(record.byteLength) bytes for \(record.name) (got \(data.count))"
            )
        }
        return data
    }

    private static func writeSafetensorsShard(_ tensors: [Packed], to url: URL) throws {
        let ordered = tensors.sorted { $0.name < $1.name }
        var header: [String: Any] = [:]
        var cursor = 0
        for tensor in ordered {
            let length = tensor.packed.count
            header[tensor.name] = [
                "dtype": "U8",
                "shape": [length],
                "data_offsets": [cursor, cursor + length],
            ]
            cursor += length
        }

        var headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        while headerData.count % 8 != 0 {
            headerData.append(0x20)
        }

        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        // Data().write gives a real throwing create under macOS Seatbelt where
        // FileManager.createFile can silently fail (mirrors Safetensors.copySubset).
        try Data().write(to: url, options: [])
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }

        var headerLength = UInt64(headerData.count).littleEndian
        output.write(Data(bytes: &headerLength, count: 8))
        output.write(headerData)
        for tensor in ordered {
            tensor.packed.withUnsafeBufferPointer { buffer in
                output.write(Data(buffer: buffer))
            }
        }
    }

    private static func writeSidecar(_ tensors: [Packed], to url: URL) throws {
        var tensorEntries: [String: Any] = [:]
        for tensor in tensors {
            tensorEntries[tensor.name] = [
                "palette": tensor.palette.map { Int($0) },
                "shape": tensor.shape,
                "elementCount": tensor.elementCount,
                "packedByteLength": tensor.packed.count,
            ]
        }
        let object: [String: Any] = [
            "version": formatVersion,
            "shard": packedShardName,
            "tensors": tensorEntries,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: url)
    }
}
