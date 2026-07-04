import Darwin
import Foundation
import MLXFastCore

/// Keeps selected expert shard pages mapped and, when the OS permits it,
/// wired in memory. The model still reads all tensor bytes through
/// ExpertSlotBank; these mappings only bias macOS's unified file cache for
/// hot structural expert code ranges.
public final class ExpertPageLockStore: @unchecked Sendable {
    private struct Mapping: @unchecked Sendable {
        let pointer: UnsafeMutableRawPointer
        let length: Int
        let locked: Bool
    }

    public let mappedTensorBytes: Int
    public let lockedMappingBytes: Int

    private let mappings: [Mapping]

    public init?(
        manifestPath: String,
        layerRange: Range<Int>,
        maxTensorBytes: Int
    ) {
        guard maxTensorBytes > 0,
              !layerRange.isEmpty,
              let manifest = try? ExpertManifest.load(from: manifestPath)
        else {
            return nil
        }

        let referenceBaseURL = URL(fileURLWithPath: manifest.referencePath)
            .standardizedFileURL
        let records = manifest.expertTensors
            .filter { record in
                record.dtype == "U32"
                    && record.layerIndex.map(layerRange.contains) == true
            }
            .sorted { ($0.shard, $0.byteOffset) < ($1.shard, $1.byteOffset) }

        var remainingBytes = maxTensorBytes
        var nextMappings: [Mapping] = []
        var nextMappedTensorBytes = 0
        var nextLockedMappingBytes = 0
        for record in records where remainingBytes > 0 {
            guard record.byteLength <= remainingBytes,
                  let mapping = Self.mapAndLock(record: record, referenceBaseURL: referenceBaseURL)
            else {
                continue
            }
            nextMappings.append(mapping)
            nextMappedTensorBytes += record.byteLength
            if mapping.locked {
                nextLockedMappingBytes += mapping.length
            }
            remainingBytes -= record.byteLength
        }

        guard !nextMappings.isEmpty else {
            return nil
        }
        self.mappings = nextMappings
        self.mappedTensorBytes = nextMappedTensorBytes
        self.lockedMappingBytes = nextLockedMappingBytes
    }

    deinit {
        for mapping in mappings {
            if mapping.locked {
                munlock(mapping.pointer, mapping.length)
            }
            munmap(mapping.pointer, mapping.length)
        }
    }

    private static func mapAndLock(
        record: ExpertTensorRecord,
        referenceBaseURL: URL
    ) -> Mapping? {
        guard record.byteOffset >= 0, record.byteLength > 0 else {
            return nil
        }
        let shardURL = referenceBaseURL
            .appendingPathComponent(record.shard)
            .standardizedFileURL
        guard shardURL.path.hasPrefix(referenceBaseURL.path + "/") else {
            return nil
        }

        let fd = open(shardURL.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            return nil
        }
        defer { close(fd) }

        var status = stat()
        guard fstat(fd, &status) == 0,
              (status.st_mode & S_IFMT) == S_IFREG,
              Int64(record.byteOffset) + Int64(record.byteLength) <= Int64(status.st_size)
        else {
            return nil
        }

        let pageSize = max(Int(getpagesize()), 4096)
        let alignedOffset = (record.byteOffset / pageSize) * pageSize
        let offsetDelta = record.byteOffset - alignedOffset
        let mappedLength = roundUpToPage(offsetDelta + record.byteLength, pageSize: pageSize)
        guard mappedLength > 0 else {
            return nil
        }

        advise(fd: fd, offset: record.byteOffset, length: record.byteLength)
        let pointer = mmap(
            nil,
            mappedLength,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            off_t(alignedOffset)
        )
        guard pointer != MAP_FAILED else {
            return nil
        }

        _ = madvise(pointer, mappedLength, MADV_WILLNEED)
        let locked = mlock(pointer, mappedLength) == 0
        return Mapping(pointer: pointer!, length: mappedLength, locked: locked)
    }

    private static func roundUpToPage(_ value: Int, pageSize: Int) -> Int {
        let remainder = value % pageSize
        return remainder == 0 ? value : value + pageSize - remainder
    }

    private static func advise(fd: Int32, offset: Int, length: Int) {
        guard length > 0, length <= Int(Int32.max) else {
            return
        }
        var advisory = radvisory(
            ra_offset: off_t(offset),
            ra_count: Int32(length)
        )
        _ = withUnsafeMutablePointer(to: &advisory) { pointer in
            fcntl(fd, F_RDADVISE, pointer)
        }
    }
}

public enum ExpertPageLockRegistry {
    private static let cache = LockedCache<String, ExpertPageLockStore?>()

    public static func expertCodePages(
        manifestPath: String,
        firstLayer: Int,
        layerCount: Int,
        maxTensorBytes: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ExpertPageLockStore? {
        guard layerCount > 0,
              maxTensorBytes > 0,
              environment["MLXFAST_EXPERT_PAGE_LOCK"].map({
                  !["0", "false", "no", "off"].contains($0.lowercased())
              }) ?? true
        else {
            return nil
        }

        let upperLayer = firstLayer + layerCount
        let key = "\(firstLayer)..<\(upperLayer)|\(maxTensorBytes)|\(manifestPath)"
        return cache.value(for: key) {
            ExpertPageLockStore(
                manifestPath: manifestPath,
                layerRange: firstLayer..<upperLayer,
                maxTensorBytes: maxTensorBytes
            )
        }
    }
}
