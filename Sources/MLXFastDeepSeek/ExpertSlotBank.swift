import Darwin
import Foundation
import MLXFastCore

public struct ExpertTensorRecord: Codable, Equatable {
    public let name: String
    public let shard: String
    public let dtype: String
    public let shape: [Int]
    public let dataOffsets: [Int]
    public let byteOffset: Int
    public let byteLength: Int

    enum CodingKeys: String, CodingKey {
        case name
        case shard
        case dtype
        case shape
        case dataOffsets = "data_offsets"
        case byteOffset = "byte_offset"
        case byteLength = "byte_length"
    }

    public var layerIndex: Int? {
        parseInt(after: ".layers.")
    }

    public var expertIndex: Int? {
        if let index = parseInt(after: ".ffn.switch_mlp.") {
            return index
        }
        return parseInt(after: ".ffn.experts.")
    }

    public var projection: String? {
        for candidate in ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"] {
            if name.contains(".\(candidate).") {
                switch candidate {
                case "w1":
                    return "gate_proj"
                case "w2":
                    return "down_proj"
                case "w3":
                    return "up_proj"
                default:
                    return candidate
                }
            }
        }
        return nil
    }

    private func parseInt(after marker: String) -> Int? {
        guard let markerRange = name.range(of: marker) else {
            return nil
        }
        let suffix = name[markerRange.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        guard !digits.isEmpty else {
            return nil
        }
        return Int(digits)
    }
}

public struct ExpertManifest: Codable, Equatable {
    public let version: Int
    public let source: String
    public let referencePath: String
    public let expertTensors: [ExpertTensorRecord]

    enum CodingKeys: String, CodingKey {
        case version
        case source
        case referencePath = "reference_path"
        case expertTensors = "expert_tensors"
    }

    public static func load(from path: String) throws -> ExpertManifest {
        try requireFile(path, description: "expert manifest")
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let manifest = try JSONDecoder().decode(ExpertManifest.self, from: data)
        try manifest.validate()
        return manifest
    }

    private func validate() throws {
        guard version == 1 else {
            throw MLXFastError.invalidInput("unsupported expert manifest version \(version)")
        }
        guard source == "safetensors" else {
            throw MLXFastError.invalidInput("unsupported expert manifest source \(source)")
        }
        guard !expertTensors.isEmpty else {
            throw MLXFastError.invalidInput("expert manifest contains no tensors")
        }

        var names = Set<String>()
        for record in expertTensors {
            guard !record.name.isEmpty else {
                throw MLXFastError.invalidInput("expert manifest contains an empty tensor name")
            }
            guard names.insert(record.name).inserted else {
                throw MLXFastError.invalidInput("duplicate expert tensor in manifest: \(record.name)")
            }
            guard record.byteOffset >= 0, record.byteLength > 0 else {
                throw MLXFastError.invalidInput("invalid byte range for expert tensor \(record.name)")
            }
            guard record.dataOffsets.count == 2, record.dataOffsets[1] >= record.dataOffsets[0] else {
                throw MLXFastError.invalidInput("invalid data_offsets for expert tensor \(record.name)")
            }
        }
    }
}

public final class ExpertSlotBank {
    public let manifest: ExpertManifest
    public let capacity: Int
    public let metrics: ExpertStreamingMetrics?

    private var recordsByName: [String: ExpertTensorRecord]
    private var cache: [String: Data] = [:]
    private var lru: [String] = []

    public init(
        manifestPath: String,
        capacity: Int = ExpertStreamingConfig.defaultTensorCacheCapacity,
        metrics: ExpertStreamingMetrics? = nil
    ) throws {
        self.manifest = try ExpertManifest.load(from: manifestPath)
        self.capacity = max(0, capacity)
        self.metrics = metrics
        self.recordsByName = Dictionary(
            uniqueKeysWithValues: manifest.expertTensors.map { ($0.name, $0) }
        )
    }

    public func record(named name: String) -> ExpertTensorRecord? {
        recordsByName[name]
    }

    public func tensorBytes(named name: String) throws -> Data {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("expert tensor not found in manifest: \(name)")
        }
        return try tensorBytes(for: record)
    }

    public func tensorBytes(for record: ExpertTensorRecord) throws -> Data {
        if let cached = cache[record.name] {
            touch(record.name)
            metrics?.recordCacheHit()
            return cached
        }

        let start = DispatchTime.now().uptimeNanoseconds
        let data = try readBytes(for: record)
        metrics?.recordCacheMiss(
            bytes: record.byteLength,
            nanoseconds: DispatchTime.now().uptimeNanoseconds - start
        )
        insert(data, for: record.name)
        return data
    }

    public func materializedTensor(named name: String) throws -> MaterializedTensor {
        guard let record = recordsByName[name] else {
            throw MLXFastError.invalidInput("expert tensor not found in manifest: \(name)")
        }
        return try materializeTensor(
            name: record.name,
            dtype: record.dtype,
            shape: record.shape,
            bytes: tensorBytes(for: record)
        )
    }

    public func validateReadableByteRanges(fileManager: FileManager = .default) throws {
        let baseURL = URL(fileURLWithPath: manifest.referencePath)
        let recordsByShard = Dictionary(grouping: manifest.expertTensors) { $0.shard }
        for shard in recordsByShard.keys.sorted() {
            let shardPath = baseURL.appendingPathComponent(shard).path
            let attributes = try fileManager.attributesOfItem(atPath: shardPath)
            guard let fileSize = attributes[.size] as? NSNumber else {
                throw MLXFastError.invalidInput("expert shard size is unavailable: \(shardPath)")
            }
            let byteCount = fileSize.intValue
            for record in recordsByShard[shard, default: []] {
                let end = record.byteOffset + record.byteLength
                guard record.byteOffset >= 0, record.byteLength > 0, end <= byteCount else {
                    throw MLXFastError.invalidInput(
                        "expert tensor \(record.name) byte range \(record.byteOffset)..<\(end) exceeds shard size \(byteCount)"
                    )
                }
            }
        }
    }

    public var cachedTensorNames: [String] {
        lru
    }

    private func insert(_ data: Data, for name: String) {
        guard capacity > 0 else {
            return
        }
        cache[name] = data
        touch(name)
        while lru.count > capacity {
            let evicted = lru.removeFirst()
            cache.removeValue(forKey: evicted)
        }
    }

    private func touch(_ name: String) {
        lru.removeAll { $0 == name }
        lru.append(name)
    }

    private func readBytes(for record: ExpertTensorRecord) throws -> Data {
        let shardPath = URL(fileURLWithPath: manifest.referencePath)
            .appendingPathComponent(record.shard)
            .path
        let fd = open(shardPath, O_RDONLY)
        guard fd >= 0 else {
            throw MLXFastError.missingFile(
                "failed to open expert shard \(shardPath): \(String(cString: strerror(errno)))"
            )
        }
        defer {
            close(fd)
        }

        var output = Data(count: record.byteLength)
        let bytesRead = output.withUnsafeMutableBytes { buffer -> Int in
            guard let base = buffer.baseAddress else {
                return 0
            }
            return pread(fd, base, record.byteLength, off_t(record.byteOffset))
        }
        guard bytesRead == record.byteLength else {
            let reason = bytesRead < 0 ? String(cString: strerror(errno)) : "short read \(bytesRead)/\(record.byteLength)"
            throw MLXFastError.invalidInput(
                "failed to read expert tensor \(record.name) from \(record.shard): \(reason)"
            )
        }
        return output
    }
}
