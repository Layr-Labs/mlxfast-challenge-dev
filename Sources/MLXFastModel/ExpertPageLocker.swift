import Darwin
import Foundation
import MLXFastCore

/// Best-effort OS page-cache steering for structural expert-code pages.
///
/// This does not surface tensor bytes to model code and does not bypass
/// ExpertSlotBank: all runtime reads still go through the trusted pread path.
/// The mapping is retained only so macOS keeps the selected file pages wired.
final class ExpertPageLocker {
    private struct LockedRegion {
        let address: UnsafeMutableRawPointer
        let length: Int
    }

    private var regions: [LockedRegion] = []

    init?(
        manifestPath: String,
        startLayer: Int,
        layerCount: Int,
        maximumBytes: Int
    ) {
        guard layerCount > 0, maximumBytes > 0 else {
            return nil
        }
        guard let manifest = try? ExpertManifest.load(from: manifestPath) else {
            return nil
        }
        let baseURL = URL(fileURLWithPath: manifest.referencePath).standardizedFileURL
        let records = manifest.expertTensors
            .filter { record in
                guard record.dtype == "U32",
                      record.name.hasSuffix(".weight"),
                      record.name.contains(".ffn.switch_mlp."),
                      let layerIndex = ResidentExpertTensors.layerIndex(fromRecordName: record.name)
                else {
                    return false
                }
                return layerIndex >= startLayer && layerIndex < startLayer + layerCount
            }
            .sorted { ($0.shard, $0.byteOffset) < ($1.shard, $1.byteOffset) }

        var lockedBytes = 0
        for record in records {
            guard lockedBytes + record.byteLength <= maximumBytes,
                  lock(record: record, baseURL: baseURL)
            else {
                continue
            }
            lockedBytes += record.byteLength
        }

        if regions.isEmpty {
            return nil
        }
    }

    deinit {
        for region in regions {
            munlock(region.address, region.length)
            munmap(region.address, region.length)
        }
    }

    private func lock(record: ExpertTensorRecord, baseURL: URL) -> Bool {
        let path = baseURL.appendingPathComponent(record.shard).path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            return false
        }
        defer { close(fd) }

        let pageSize = max(Int(sysconf(_SC_PAGESIZE)), 4096)
        let alignedOffset = (record.byteOffset / pageSize) * pageSize
        let delta = record.byteOffset - alignedOffset
        let mappedLength = record.byteLength + delta
        guard mappedLength > 0 else {
            return false
        }

        let address = mmap(
            nil,
            mappedLength,
            PROT_READ,
            MAP_PRIVATE,
            fd,
            off_t(alignedOffset)
        )
        guard address != MAP_FAILED, let address else {
            return false
        }
        guard mlock(address, mappedLength) == 0 else {
            munmap(address, mappedLength)
            return false
        }
        regions.append(LockedRegion(address: address, length: mappedLength))
        return true
    }
}

enum ExpertPageLockRegistry {
    private static let cache = LockedCache<String, ExpertPageLocker?>()

    static func prefixCodePages(
        manifestPath: String,
        startLayer: Int,
        layerCount: Int,
        maximumBytes: Int
    ) -> ExpertPageLocker? {
        let key = "\(startLayer)|\(layerCount)|\(maximumBytes)|\(manifestPath)"
        return cache.value(for: key) {
            ExpertPageLocker(
                manifestPath: manifestPath,
                startLayer: startLayer,
                layerCount: layerCount,
                maximumBytes: maximumBytes
            )
        }
    }
}
