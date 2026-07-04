import Darwin
import Foundation
import MLXFastCore

/// Best-effort page-cache residency for structural expert code ranges.
///
/// This is intentionally not a decoded-weight cache: inference still reads
/// every requested tensor through `ExpertSlotBank` and records the same byte
/// ranges. The mappings only ask macOS to keep selected file pages resident,
/// avoiding a second app-owned copy for the first non-pinned hash layer.
final class ExpertFilePageLocker {
    private struct Mapping {
        let address: UnsafeMutableRawPointer
        let length: Int
    }

    private let mappings: [Mapping]

    init?(
        manifestPath: String,
        layerRange: Range<Int>
    ) {
        guard !layerRange.isEmpty,
              let manifest = try? ExpertManifest.load(from: manifestPath)
        else {
            return nil
        }
        let referenceBaseURL = URL(fileURLWithPath: manifest.referencePath).standardizedFileURL
        let pageSize = max(Int(getpagesize()), 1)
        var locked: [Mapping] = []

        for record in manifest.expertTensors {
            guard record.dtype == "U32",
                  let layerIndex = record.layerIndex,
                  layerRange.contains(layerIndex)
            else {
                continue
            }
            guard let mapping = Self.lock(
                record: record,
                referenceBaseURL: referenceBaseURL,
                pageSize: pageSize
            ) else {
                continue
            }
            locked.append(mapping)
        }

        guard !locked.isEmpty else {
            return nil
        }
        self.mappings = locked
    }

    deinit {
        for mapping in mappings {
            munlock(mapping.address, mapping.length)
            munmap(mapping.address, mapping.length)
        }
    }

    private static func lock(
        record: ExpertTensorRecord,
        referenceBaseURL: URL,
        pageSize: Int
    ) -> Mapping? {
        let shardURL = referenceBaseURL
            .appendingPathComponent(record.shard)
            .standardizedFileURL
        let fd = open(shardURL.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            return nil
        }
        defer {
            close(fd)
        }

        var status = stat()
        guard fstat(fd, &status) == 0,
              (status.st_mode & S_IFMT) == S_IFREG
        else {
            return nil
        }

        let end = record.byteOffset + record.byteLength
        guard record.byteOffset >= 0,
              record.byteLength > 0,
              end <= Int(status.st_size)
        else {
            return nil
        }

        let alignedOffset = (record.byteOffset / pageSize) * pageSize
        let delta = record.byteOffset - alignedOffset
        let mappedLength = roundUp(delta + record.byteLength, to: pageSize)
        guard mappedLength > 0 else {
            return nil
        }

        guard let address = mmap(
            nil,
            mappedLength,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            off_t(alignedOffset)
        ), address != MAP_FAILED else {
            return nil
        }

        guard mlock(address, mappedLength) == 0 else {
            munmap(address, mappedLength)
            return nil
        }
        return Mapping(address: address, length: mappedLength)
    }

    private static func roundUp(_ value: Int, to alignment: Int) -> Int {
        guard alignment > 1 else {
            return value
        }
        let remainder = value % alignment
        return remainder == 0 ? value : value + alignment - remainder
    }
}

enum ExpertFilePageLockRegistry {
    private static let cache = LockedCache<String, ExpertFilePageLocker?>()

    static func hashLayerCodes(
        manifestPath: String,
        layerRange: Range<Int>
    ) -> ExpertFilePageLocker? {
        cache.value(for: "\(layerRange.lowerBound)..<\(layerRange.upperBound)|\(manifestPath)") {
            ExpertFilePageLocker(
                manifestPath: manifestPath,
                layerRange: layerRange
            )
        }
    }
}
