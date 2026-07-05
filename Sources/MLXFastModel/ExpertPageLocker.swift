import Darwin
import Foundation
import MLXFastCore

/// Process-lifetime wiring of selected expert weight file pages.
///
/// This does not surface bytes to the model and does not bypass
/// ExpertSlotBank: all tensor materialization still uses the trusted `pread`
/// path and records the same read metrics. The mapping only asks the kernel
/// to keep structural expert-code file pages resident so those preads hit the
/// page cache instead of storage.
public final class ExpertPageLocker {
    private final class Mapping {
        let address: UnsafeMutableRawPointer
        let byteLength: Int

        init(address: UnsafeMutableRawPointer, byteLength: Int) {
            self.address = address
            self.byteLength = byteLength
        }

        deinit {
            munlock(address, byteLength)
            munmap(address, byteLength)
        }
    }

    private let mappings: [Mapping]
    public let lockedBytes: Int

    public var mappingCount: Int {
        mappings.count
    }

    public init?(
        referencePath: String,
        records: [ExpertTensorRecord],
        maxBytes: Int
    ) {
        guard maxBytes > 0, !records.isEmpty else {
            return nil
        }

        let referenceURL = URL(fileURLWithPath: referencePath).standardizedFileURL
        let pageSize = max(1, Int(sysconf(Int32(_SC_PAGESIZE))))
        var locked: [Mapping] = []
        var totalBytes = 0

        for record in records {
            guard record.dtype == "U32" else {
                continue
            }
            let range = Self.pageAlignedRange(
                offset: record.byteOffset,
                length: record.byteLength,
                pageSize: pageSize
            )
            guard range.length > 0, totalBytes + range.length <= maxBytes else {
                break
            }
            guard let mapping = Self.lockRange(
                referenceURL: referenceURL,
                shard: record.shard,
                offset: range.offset,
                length: range.length
            ) else {
                continue
            }
            locked.append(mapping)
            totalBytes += range.length
        }

        guard !locked.isEmpty else {
            return nil
        }
        self.mappings = locked
        self.lockedBytes = totalBytes
    }

    private static func pageAlignedRange(
        offset: Int,
        length: Int,
        pageSize: Int
    ) -> (offset: Int, length: Int) {
        guard offset >= 0, length > 0 else {
            return (offset, 0)
        }
        let start = (offset / pageSize) * pageSize
        let end = ((offset + length + pageSize - 1) / pageSize) * pageSize
        return (start, end - start)
    }

    private static func lockRange(
        referenceURL: URL,
        shard: String,
        offset: Int,
        length: Int
    ) -> Mapping? {
        let shardURL = referenceURL.appendingPathComponent(shard).standardizedFileURL
        let fd = open(shardURL.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            return nil
        }
        defer {
            close(fd)
        }

        var status = stat()
        guard fstat(fd, &status) == 0, (status.st_mode & S_IFMT) == S_IFREG else {
            return nil
        }
        guard offset >= 0, length > 0, offset + length <= Int(status.st_size) else {
            return nil
        }

        guard let mapped = mmap(nil, length, PROT_READ, MAP_SHARED, fd, off_t(offset)),
              mapped != MAP_FAILED
        else {
            return nil
        }
        guard mlock(mapped, length) == 0 else {
            munmap(mapped, length)
            return nil
        }
        return Mapping(address: mapped, byteLength: length)
    }
}

public enum ExpertPageLockRegistry {
    private static let minimumPhysicalMemoryBytes: UInt64 = 40 << 30
    private static let lockedStageableLayerCount = 3
    private static let maxLockedExpertCodeBytes = 10 << 30
    private static let cache = LockedCache<String, ExpertPageLocker?>()

    @discardableResult
    public static func lockInitialExpertCodePages(
        loader: DeepSeekWeightLoader,
        config: DeepSeekConfig
    ) -> ExpertPageLocker? {
        guard ProcessInfo.processInfo.physicalMemory >= minimumPhysicalMemoryBytes else {
            return nil
        }
        let cacheKey = "\(loader.expertBank.manifest.referencePath)|\(lockedStageableLayerCount)|\(maxLockedExpertCodeBytes)"
        return cache.value(for: cacheKey) {
            let records = initialExpertCodeRecords(loader: loader, config: config)
            return ExpertPageLocker(
                referencePath: loader.expertBank.manifest.referencePath,
                records: records,
                maxBytes: maxLockedExpertCodeBytes
            )
        }
    }

    private static func initialExpertCodeRecords(
        loader: DeepSeekWeightLoader,
        config: DeepSeekConfig
    ) -> [ExpertTensorRecord] {
        var records: [ExpertTensorRecord] = []
        var selectedLayers = 0
        for layerIndex in 0..<config.numHiddenLayers {
            guard selectedLayers < lockedStageableLayerCount else {
                break
            }
            guard let plan = loader.stagedExpertLayerPlan(layerIndex: layerIndex) else {
                continue
            }
            let layerRecords = plan.recordNames.compactMap { name -> ExpertTensorRecord? in
                guard let record = loader.expertBank.record(named: name), record.dtype == "U32" else {
                    return nil
                }
                return record
            }
            guard !layerRecords.isEmpty else {
                continue
            }
            records.append(contentsOf: layerRecords)
            selectedLayers += 1
        }
        return records
    }
}
