import Foundation
import MLXFastCore

/// RAM-resident copies of selected routed-expert tensors, serving
/// byte-identical stand-ins for the slot bank's whole and firstAxisIndex
/// materializations.
///
/// Two instances back the runtime:
///
/// - **Scales** (all layers): e8m0 scales are only ~6% of streamed
///   expert bytes but half of all bank round-trips — every expert
///   materialization pays a second open/fstat/pread/close for its small
///   scales slice. When the offline transform emitted the palette-packed
///   scale overlay (`experts/scales_packed.safetensors` +
///   `scale_compression.json`), the store keeps the scales in their compact
///   4-bit form (~4.3 GiB instead of ~8.66 GiB) and decodes each requested
///   slice back to byte-identical U8 on demand. That halves the resident
///   scale footprint, freeing unified memory for expert-code page cache and
///   staging on the memory-constrained runner.
/// - **Hash-layer codes** (layers routed by token id, only on
///   machines with the official 48 GB or more): the expert working set
///   cycles through far more bytes than any cache, and cyclic scans defeat
///   LRU — pinning converts those layers' reads into guaranteed RAM hits
///   every token instead of probabilistic page-cache hits.
///
/// Loads happen once in the (untimed) loader constructor. Raw entries come
/// through a dedicated capacity-0 ExpertSlotBank that shares the loader's
/// metrics; packed scale entries are read directly from the transform's
/// compact overlay shard. In both cases the bytes handed to the runtime are
/// identical to what the bank's firstAxisIndex read would return, by the
/// bank's own slice arithmetic (byteLength / firstDimension).
public final class ResidentExpertTensors {
    /// The compact packed-scale form: 4-bit palette indices (two per byte)
    /// plus the per-tensor palette, padded to 16 entries so a nibble index is
    /// always in range.
    private struct PackedScale {
        let nibbles: Data
        let palette: [UInt8]
    }

    private struct Entry {
        let dtype: TensorDType
        let shape: [Int]
        let sliceByteLength: Int
        let elementCount: Int
        /// Exactly one of `rawBytes` / `packed` is populated.
        let rawBytes: Data?
        let packed: PackedScale?

        func decodedWhole() -> Data {
            if let rawBytes {
                return rawBytes
            }
            guard let packed else {
                return Data()
            }
            return ResidentExpertTensors.decode(
                nibbles: packed.nibbles,
                palette: packed.palette,
                count: elementCount
            )
        }

        func decodedSlice(_ index: Int) -> Data {
            if let rawBytes {
                let start = rawBytes.startIndex + index * sliceByteLength
                return rawBytes[start..<(start + sliceByteLength)]
            }
            guard let packed else {
                return Data()
            }
            let nibblesPerSlice = sliceByteLength / 2
            let start = packed.nibbles.startIndex + index * nibblesPerSlice
            let slice = packed.nibbles[start..<(start + nibblesPerSlice)]
            return ResidentExpertTensors.decode(
                nibbles: slice,
                palette: packed.palette,
                count: sliceByteLength
            )
        }
    }

    private let entries: [String: Entry]

    public var residentTensorCount: Int {
        entries.count
    }

    /// Builds the store by reading every manifest record accepted by the
    /// filter. Returns nil (callers fall back to streaming) when nothing
    /// matches or any read fails — behavior is then identical to the
    /// pre-residency runtime.
    public init?(
        manifestPath: String,
        metrics: ExpertStreamingMetrics?,
        recordFilter: (ExpertTensorRecord) -> Bool
    ) {
        guard let bank = try? ExpertSlotBank(
            manifestPath: manifestPath,
            capacity: 0,
            metrics: metrics
        ) else {
            return nil
        }

        var loaded: [String: Entry] = [:]
        for record in bank.manifest.expertTensors where recordFilter(record) {
            guard
                let firstDimension = record.shape.first,
                record.shape.count >= 2,
                firstDimension > 0,
                record.byteLength % firstDimension == 0,
                let tensor = try? bank.materializedTensor(named: record.name)
            else {
                return nil
            }
            loaded[record.name] = Entry(
                dtype: tensor.dtype,
                shape: tensor.shape,
                sliceByteLength: record.byteLength / firstDimension,
                elementCount: tensor.shape.reduce(1, *),
                rawBytes: tensor.bytes,
                packed: nil
            )
        }
        guard !loaded.isEmpty else {
            return nil
        }
        self.entries = loaded
    }

    /// Builds the store directly from prebuilt entries (used by the packed
    /// scale loader). Fails when empty so callers fall back to streaming.
    private init?(entries: [String: Entry]) {
        guard !entries.isEmpty else {
            return nil
        }
        self.entries = entries
    }

    public func isResident(name: String) -> Bool {
        entries[name] != nil
    }

    /// True when the entry serves whole-tensor requests as zero-copy raw
    /// bytes. Packed entries return false: their whole-tensor materialization
    /// pays a full palette expansion, so bulk consumers (the prefill staging
    /// planner) should prefer streaming those bytes with the layer instead.
    /// Per-slice decode stays cheap either way (worker-side, overlapped).
    public func servesZeroCopyBytes(name: String) -> Bool {
        entries[name]?.rawBytes != nil
    }

    /// Byte-identical stand-in for the slot bank's materializedTensor calls.
    /// Returns nil when the tensor is not resident or the request does not
    /// match the stacked layout, so the caller can fall back to the bank.
    public func materializedTensor(named name: String, firstAxisIndex: Int?) -> MaterializedTensor? {
        guard let entry = entries[name] else {
            return nil
        }
        guard let firstAxisIndex else {
            return try? MaterializedTensor(
                name: name,
                dtype: entry.dtype,
                shape: entry.shape,
                bytes: entry.decodedWhole()
            )
        }
        guard
            let firstDimension = entry.shape.first,
            firstAxisIndex >= 0,
            firstAxisIndex < firstDimension
        else {
            return nil
        }
        return try? MaterializedTensor(
            name: "\(name)[\(firstAxisIndex)]",
            dtype: entry.dtype,
            shape: Array(entry.shape.dropFirst()),
            bytes: entry.decodedSlice(firstAxisIndex)
        )
    }

    /// Decodes `count` U8 values from packed 4-bit palette indices. The palette
    /// must have 16 entries so every nibble index is in range.
    ///
    /// The output buffer is allocated uninitialized (no zero-fill) and handed to
    /// `Data` without a copy — the decode loop writes every byte, so the whole
    /// buffer is defined before use.
    fileprivate static func decode(nibbles: Data, palette: [UInt8], count: Int) -> Data {
        guard count > 0, let destination = malloc(count)?.assumingMemoryBound(to: UInt8.self) else {
            return Data()
        }
        nibbles.withUnsafeBytes { rawInput in
            guard let source = rawInput.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                // Cannot happen for a non-empty slice, but keep the buffer defined.
                memset(destination, 0, count)
                return
            }
            palette.withUnsafeBufferPointer { lut in
                var writeIndex = 0
                var readIndex = 0
                let pairEnd = count - (count % 2)
                while writeIndex < pairEnd {
                    let byte = source[readIndex]
                    destination[writeIndex] = lut[Int(byte & 0x0F)]
                    destination[writeIndex + 1] = lut[Int(byte >> 4)]
                    writeIndex += 2
                    readIndex += 1
                }
                if writeIndex < count {
                    destination[writeIndex] = lut[Int(source[readIndex] & 0x0F)]
                }
            }
        }
        return Data(bytesNoCopy: destination, count: count, deallocator: .free)
    }
}

extension ResidentExpertTensors {
    /// Filenames of the offline scale-compression overlay. These MUST match
    /// `ScaleCompression` in MLXFastTransform (a small cross-module contract;
    /// MLXFastModel does not depend on MLXFastTransform).
    private static let packedScaleShardName = "scales_packed.safetensors"
    private static let packedScaleSidecarName = "scale_compression.json"
    private static let packedScaleVersion = 1

    /// All `*.scales` records — the store behind resident expert scales. Uses
    /// the compact palette-packed overlay emitted by the transform when it is
    /// present and valid; otherwise falls back to the uncompressed bank load.
    public convenience init?(scalesFromManifest manifestPath: String, metrics: ExpertStreamingMetrics?) {
        if let entries = ResidentExpertTensors.loadPackedScaleEntries(
            manifestPath: manifestPath,
            metrics: metrics
        ) {
            self.init(entries: entries)
        } else {
            self.init(manifestPath: manifestPath, metrics: metrics) { record in
                record.name.hasSuffix(".scales")
            }
        }
    }

    /// The packed U32 code tensors of the first `hashLayerCount` layers.
    public convenience init?(
        hashLayerCodesFromManifest manifestPath: String,
        hashLayerCount: Int,
        metrics: ExpertStreamingMetrics?
    ) {
        guard hashLayerCount > 0 else {
            return nil
        }
        self.init(manifestPath: manifestPath, metrics: metrics) { record in
            record.dtype == "U32"
                && Self.layerIndex(fromRecordName: record.name).map { $0 < hashLayerCount } == true
        }
    }

    static func layerIndex(fromRecordName name: String) -> Int? {
        let components = name.split(separator: ".")
        guard
            let layersPosition = components.firstIndex(of: "layers"),
            components.index(after: layersPosition) < components.endIndex
        else {
            return nil
        }
        return Int(components[components.index(after: layersPosition)])
    }

    /// Loads resident scale entries from the transform's palette-packed overlay
    /// next to the manifest. Returns nil (caller falls back to the bank load)
    /// when the overlay is absent, malformed, or inconsistent with the manifest.
    private static func loadPackedScaleEntries(
        manifestPath: String,
        metrics: ExpertStreamingMetrics?
    ) -> [String: Entry]? {
        let manifestURL = URL(fileURLWithPath: manifestPath)
        let expertsDirectory = manifestURL.deletingLastPathComponent()
        let sidecarURL = expertsDirectory.appendingPathComponent(packedScaleSidecarName)
        let shardURL = expertsDirectory.appendingPathComponent(packedScaleShardName)

        guard
            FileManager.default.fileExists(atPath: sidecarURL.path),
            FileManager.default.fileExists(atPath: shardURL.path),
            let sidecarData = try? Data(contentsOf: sidecarURL),
            let sidecarObject = (try? JSONSerialization.jsonObject(with: sidecarData)) as? [String: Any],
            (sidecarObject["version"] as? Int) == packedScaleVersion,
            let packedMeta = sidecarObject["tensors"] as? [String: Any],
            let header = try? Safetensors.readHeader(shardURL),
            let bank = try? ExpertSlotBank(manifestPath: manifestPath, capacity: 0, metrics: metrics),
            let shardHandle = try? FileHandle(forReadingFrom: shardURL)
        else {
            return nil
        }
        defer { try? shardHandle.close() }

        var loaded: [String: Entry] = [:]
        for record in bank.manifest.expertTensors where record.name.hasSuffix(".scales") {
            guard
                let firstDimension = record.shape.first,
                record.shape.count >= 2,
                firstDimension > 0,
                record.byteLength % firstDimension == 0
            else {
                return nil
            }

            if let entry = packedEntry(
                record: record,
                firstDimension: firstDimension,
                packedMeta: packedMeta,
                header: header,
                shardHandle: shardHandle
            ) {
                loaded[record.name] = entry
                continue
            }

            // Any scale tensor missing from the overlay (or inconsistent with
            // it) is served raw from the bank so the store stays complete.
            guard let tensor = try? bank.materializedTensor(named: record.name) else {
                return nil
            }
            loaded[record.name] = Entry(
                dtype: tensor.dtype,
                shape: tensor.shape,
                sliceByteLength: record.byteLength / firstDimension,
                elementCount: tensor.shape.reduce(1, *),
                rawBytes: tensor.bytes,
                packed: nil
            )
        }

        // Require that at least one tensor actually came from the packed
        // overlay; otherwise there is no benefit over the plain bank load.
        guard !loaded.isEmpty, loaded.values.contains(where: { $0.packed != nil }) else {
            return nil
        }
        return loaded
    }

    private static func packedEntry(
        record: ExpertTensorRecord,
        firstDimension: Int,
        packedMeta: [String: Any],
        header: SafetensorsHeader,
        shardHandle: FileHandle
    ) -> Entry? {
        guard
            record.dtype == "U8",
            let meta = packedMeta[record.name] as? [String: Any],
            let paletteValues = meta["palette"] as? [Int],
            let shape = meta["shape"] as? [Int],
            let elementCount = meta["elementCount"] as? Int,
            let packedByteLength = meta["packedByteLength"] as? Int,
            let info = header.tensors[record.name],
            shape == record.shape,
            elementCount == record.shape.reduce(1, *),
            elementCount == record.byteLength,
            (elementCount / firstDimension) % 2 == 0,
            packedByteLength == (elementCount + 1) / 2,
            info.byteCount == packedByteLength,
            paletteValues.count <= 16,
            !paletteValues.isEmpty,
            paletteValues.allSatisfy({ $0 >= 0 && $0 <= 255 })
        else {
            return nil
        }

        let absoluteOffset = header.dataBaseOffset + UInt64(info.dataStart)
        guard let nibbles = readShardBytes(
            shardHandle,
            offset: absoluteOffset,
            length: packedByteLength
        ) else {
            return nil
        }

        // Pad the palette to 16 entries. Only indices 0..<paletteValues.count
        // are ever produced by the packer, so the padding value is irrelevant;
        // it exists solely so a raw nibble (0..15) can index without bounds risk.
        var palette = [UInt8](repeating: 0, count: 16)
        for index in 0..<paletteValues.count {
            palette[index] = UInt8(paletteValues[index])
        }

        return Entry(
            dtype: .u8,
            shape: shape,
            sliceByteLength: elementCount / firstDimension,
            elementCount: elementCount,
            rawBytes: nil,
            packed: PackedScale(nibbles: nibbles, palette: palette)
        )
    }

    private static func readShardBytes(_ handle: FileHandle, offset: UInt64, length: Int) -> Data? {
        do {
            try handle.seek(toOffset: offset)
        } catch {
            return nil
        }
        var data = Data()
        data.reserveCapacity(length)
        var remaining = length
        while remaining > 0 {
            let chunk = handle.readData(ofLength: min(remaining, 32 * 1024 * 1024))
            if chunk.isEmpty {
                break
            }
            data.append(chunk)
            remaining -= chunk.count
        }
        return data.count == length ? data : nil
    }
}

/// Process-wide registry so every DeepSeekWeightLoader for the same manifest
/// shares ONE copy of each resident store. The trusted benchmark harness holds
/// two loaders alive at once (a correctness loader and a benchmark loader);
/// without sharing, the resident scales and pinned codes would be
/// duplicated and could exceed the 48 GB runner budget before scoring starts.
/// Stores are immutable after construction, so sharing is safe; nil results
/// are cached too so a failed load is not retried per loader.
public enum ResidentExpertStoreRegistry {
    private static let scalesCache = LockedCache<String, ResidentExpertTensors?>()
    private static let pinnedCodesCache = LockedCache<String, ResidentExpertTensors?>()

    public static func scales(
        manifestPath: String,
        metrics: ExpertStreamingMetrics?
    ) -> ResidentExpertTensors? {
        scalesCache.value(for: manifestPath) {
            ResidentExpertTensors(scalesFromManifest: manifestPath, metrics: metrics)
        }
    }

    public static func pinnedHashLayerCodes(
        manifestPath: String,
        hashLayerCount: Int,
        metrics: ExpertStreamingMetrics?
    ) -> ResidentExpertTensors? {
        pinnedCodesCache.value(for: "\(hashLayerCount)|\(manifestPath)") {
            ResidentExpertTensors(
                hashLayerCodesFromManifest: manifestPath,
                hashLayerCount: hashLayerCount,
                metrics: metrics
            )
        }
    }
}
