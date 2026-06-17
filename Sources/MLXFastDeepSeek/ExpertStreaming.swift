import Foundation

public struct ExpertStreamingConfig: Equatable, Sendable {
    public enum Mode: String, Equatable, Sendable {
        case directNVMe = "direct_nvme"
    }

    public static let defaultTensorCacheCapacity = 128

    public let mode: Mode
    public let tensorCacheCapacity: Int
    public let recordsMetrics: Bool

    public init(
        mode: Mode = .directNVMe,
        tensorCacheCapacity: Int = Self.defaultTensorCacheCapacity,
        recordsMetrics: Bool = false
    ) {
        self.mode = mode
        self.tensorCacheCapacity = max(0, tensorCacheCapacity)
        self.recordsMetrics = recordsMetrics
    }

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ExpertStreamingConfig {
        let tensorCapacity =
            parsePositiveInt(environment["MLXFAST_EXPERT_CACHE_TENSORS"])
            ?? parsePositiveInt(environment["MLXFAST_EXPERT_CACHE_EXPERTS"]).map { $0 * 3 }
            ?? Self.defaultTensorCacheCapacity
        return ExpertStreamingConfig(
            tensorCacheCapacity: tensorCapacity,
            recordsMetrics: parseBool(environment["MLXFAST_EXPERT_STREAM_METRICS"]) ?? false
        )
    }

    private static func parsePositiveInt(_ value: String?) -> Int? {
        guard let value, let parsed = Int(value), parsed >= 0 else {
            return nil
        }
        return parsed
    }

    private static func parseBool(_ value: String?) -> Bool? {
        guard let normalized = value?.lowercased() else {
            return nil
        }
        switch normalized {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }
}

public final class ExpertStreamingMetrics: @unchecked Sendable {
    public struct Snapshot: Equatable, Sendable {
        public let cacheHits: UInt64
        public let cacheMisses: UInt64
        public let bytesRead: UInt64
        public let readNanoseconds: UInt64

        public var totalLookups: UInt64 {
            cacheHits + cacheMisses
        }

        public var hitRate: Double {
            totalLookups == 0 ? 0 : Double(cacheHits) / Double(totalLookups)
        }
    }

    private let lock = NSLock()
    private var cacheHits: UInt64 = 0
    private var cacheMisses: UInt64 = 0
    private var bytesRead: UInt64 = 0
    private var readNanoseconds: UInt64 = 0

    public init() {}

    public func recordCacheHit() {
        lock.lock()
        cacheHits += 1
        lock.unlock()
    }

    public func recordCacheMiss(bytes: Int, nanoseconds: UInt64) {
        lock.lock()
        cacheMisses += 1
        bytesRead += UInt64(max(0, bytes))
        readNanoseconds += nanoseconds
        lock.unlock()
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer {
            lock.unlock()
        }
        return Snapshot(
            cacheHits: cacheHits,
            cacheMisses: cacheMisses,
            bytesRead: bytesRead,
            readNanoseconds: readNanoseconds
        )
    }
}
