import Darwin
import Foundation
import Compression
import MLXFastCore

public final class CompressedExpertCodeStore: @unchecked Sendable {
    public struct Key: Hashable {
        public let name: String
        public let expertIndex: Int

        public init(name: String, expertIndex: Int) {
            self.name = name
            self.expertIndex = expertIndex
        }
    }

    private struct Manifest: Decodable {
        let version: Int
        let algorithm: String
        let entries: [Entry]
    }

    private struct Entry: Decodable {
        let name: String
        let expertIndex: Int
        let dtype: String
        let shape: [Int]
        let originalByteLength: Int
        let pack: String
        let offset: Int64
        let compressedByteLength: Int

        enum CodingKeys: String, CodingKey {
            case name
            case expertIndex = "expert_index"
            case dtype
            case shape
            case originalByteLength = "original_byte_length"
            case pack
            case offset
            case compressedByteLength = "compressed_byte_length"
        }
    }

    private let expertsURL: URL
    private let entriesByKey: [Key: Entry]

    public init?(weightsPath: String) {
        let expertsURL = URL(fileURLWithPath: weightsPath)
            .appendingPathComponent("experts", isDirectory: true)
            .standardizedFileURL
        let manifestURL = expertsURL.appendingPathComponent("codepack_manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
              manifest.version == 1,
              manifest.algorithm == "lz4",
              !manifest.entries.isEmpty
        else {
            return nil
        }

        var dictionary: [Key: Entry] = [:]
        dictionary.reserveCapacity(manifest.entries.count)
        for entry in manifest.entries where entry.dtype == "U32" {
            guard entry.originalByteLength > 0,
                  entry.compressedByteLength > 0,
                  entry.offset >= 0,
                  !entry.pack.isEmpty,
                  !entry.pack.contains("/"),
                  !entry.pack.contains("\\")
            else {
                continue
            }
            dictionary[Key(name: entry.name, expertIndex: entry.expertIndex)] = entry
        }
        guard !dictionary.isEmpty else {
            return nil
        }
        self.expertsURL = expertsURL
        self.entriesByKey = dictionary
    }

    public func contains(recordName: String, expertIndex: Int) -> Bool {
        entriesByKey[Key(name: recordName, expertIndex: expertIndex)] != nil
    }

    public func materializedTensor(recordName: String, expertIndex: Int) -> MaterializedTensor? {
        guard let entry = entriesByKey[Key(name: recordName, expertIndex: expertIndex)] else {
            return nil
        }
        guard let compressed = readCompressedBytes(entry) else {
            return nil
        }
        var decoded = Data(count: entry.originalByteLength)
        let decodedLength = decoded.withUnsafeMutableBytes { destinationBuffer in
            compressed.withUnsafeBytes { sourceBuffer in
                compression_decode_buffer(
                    destinationBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    entry.originalByteLength,
                    sourceBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    entry.compressedByteLength,
                    nil,
                    COMPRESSION_LZ4
                )
            }
        }
        guard decodedLength == entry.originalByteLength else {
            return nil
        }
        return try? MaterializedTensor(
            name: "\(recordName)[\(expertIndex)]",
            dtype: .u32,
            shape: entry.shape,
            bytes: decoded
        )
    }

    private func readCompressedBytes(_ entry: Entry) -> Data? {
        let packURL = expertsURL.appendingPathComponent(entry.pack).standardizedFileURL
        let fd = open(packURL.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            return nil
        }
        defer { close(fd) }

        var status = stat()
        guard fstat(fd, &status) == 0,
              (status.st_mode & S_IFMT) == S_IFREG,
              entry.offset + Int64(entry.compressedByteLength) <= Int64(status.st_size)
        else {
            return nil
        }

        var output = Data(count: entry.compressedByteLength)
        let bytesRead = output.withUnsafeMutableBytes { buffer -> Int in
            guard let base = buffer.baseAddress else {
                return 0
            }
            return pread(fd, base, entry.compressedByteLength, off_t(entry.offset))
        }
        guard bytesRead == entry.compressedByteLength else {
            return nil
        }
        return output
    }
}
