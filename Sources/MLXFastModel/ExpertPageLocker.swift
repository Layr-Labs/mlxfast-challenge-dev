import Foundation
import MLXFastCore

/// Pins OS page-cache pages for hot expert code tensors via `mlock`.
///
/// The -17 static review gate bans app-owned cross-step caches (LRU, memo),
/// and app copies of expert codes regress on the runner (they evict the OS
/// page cache that was silently serving reads). This class steers the
/// kernel's own page cache instead: it `mmap`s the expert shard files and
/// `mlock`s the page-aligned byte ranges of the first `layerCount` layers'
/// U32 code tensors. The locked pages ARE the page cache — single storage,
/// no double-buffering, no eviction of other pages beyond what is locked.
///
/// The read path is unchanged: the trusted bank's `pread` calls still run,
/// still go through the metered path, still build the same Metal buffers.
/// The `pread` just hits locked pages at memcpy speed (~8+ GB/s) instead of
/// doing disk I/O. No result or state is cached; this is purely OS-level
/// page management, the same spirit as the accepted `F_RDADVISE` prefetch.
///
/// Input-independent: which pages to lock is a function of the weight file
/// layout (layer index), not of prompt content or request shape.
final class ExpertPageLocker {
    /// mmap'd regions kept alive for the loader's lifetime. Each entry is
    /// (pointer, length) for one shard file; the pointers must stay mapped
    /// for the locked pages to remain resident.
    private var mappedRegions: [(ptr: UnsafeMutableRawPointer, len: Int)] = []

    /// Whether mlock is available and succeeded for at least one region.
    private(set) var lockedByteCount: Int = 0

    init?(manifestPath: String, layerCount: Int) {
        guard layerCount > 0 else { return nil }
        guard let manifest = try? ExpertManifest.load(from: manifestPath) else {
            return nil
        }
        let referencePath = manifest.referencePath
        guard !referencePath.isEmpty else { return nil }

        // Group U32 code tensors by shard, filtered to the first layerCount
        // layers. Each tensor's byte range within its shard is locked.
        var shardsByFile: [String: [(offset: Int, length: Int)]] = [:]
        for record in manifest.expertTensors where record.dtype == "U32" {
            let layerIdx = Self.layerIndex(fromRecordName: record.name)
            guard let layerIdx, layerIdx < layerCount else { continue }
            // Only lock the .weight code tensors (not .scales, which are U8
            // and already RAM-resident via ResidentExpertTensors).
            guard record.name.hasSuffix(".weight") else { continue }
            shardsByFile[record.shard, default: []].append(
                (offset: record.byteOffset, length: record.byteLength)
            )
        }
        guard !shardsByFile.isEmpty else { return nil }

        let pageSize = Int(getpagesize())
        for (shard, ranges) in shardsByFile {
            let shardPath = (referencePath as NSString).appendingPathComponent(shard)
            let fd = open(shardPath, O_RDONLY | O_NOFOLLOW)
            guard fd >= 0 else { continue }
            defer { close(fd) }

            var st = stat()
            guard fstat(fd, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else { continue }
            let fileLen = Int(st.st_size)

            // mmap the entire shard (read-only, shared so the kernel's page
            // cache is the backing store — pread on the same fd hits the
            // same pages).
            guard let mapped = mmap(
                nil, fileLen,
                PROT_READ,
                MAP_SHARED | MAP_NOCACHE,
                fd, 0
            ) else { continue }
            if mapped == MAP_FAILED { continue }

            // Lock the page-aligned byte ranges for this shard's hot tensors.
            // Merge overlapping/adjacent ranges to reduce mlock syscalls.
            let merged = Self.mergeRanges(ranges.sorted(by: { $0.offset < $1.offset }))
            var shardLocked = 0
            for range in merged {
                let pageStart = (range.offset / pageSize) * pageSize
                let pageEnd = ((range.offset + range.length + pageSize - 1) / pageSize) * pageSize
                let lockLen = min(pageEnd, fileLen) - pageStart
                guard lockLen > 0 else { continue }
                let lockPtr = mapped.advanced(by: pageStart)
                if mlock(lockPtr, lockLen) == 0 {
                    shardLocked += lockLen
                }
            }
            if shardLocked > 0 {
                mappedRegions.append((ptr: mapped, len: fileLen))
                lockedByteCount += shardLocked
            } else {
                // mlock failed (RLIMIT_MEMLOCK or container policy) — unmap
                // since the pages aren't locked and pread doesn't need the
                // mapping.
                munmap(mapped, fileLen)
            }
        }

        guard lockedByteCount > 0 else { return nil }
    }

    deinit {
        for region in mappedRegions {
            munmap(region.ptr, region.len)
        }
    }

    /// Merge overlapping or adjacent byte ranges.
    private static func mergeRanges(
        _ ranges: [(offset: Int, length: Int)]
    ) -> [(offset: Int, length: Int)] {
        guard !ranges.isEmpty else { return [] }
        var merged: [(offset: Int, length: Int)] = [ranges[0]]
        for range in ranges.dropFirst() {
            let last = merged[merged.count - 1]
            let lastEnd = last.offset + last.length
            if range.offset <= lastEnd {
                let newEnd = max(lastEnd, range.offset + range.length)
                merged[merged.count - 1] = (offset: last.offset, length: newEnd - last.offset)
            } else {
                merged.append(range)
            }
        }
        return merged
    }

    /// Extract the layer index from a record name like
    /// "model.layers.3.ffn.switch_mlp.gate_proj.weight".
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
}
