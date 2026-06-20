import Foundation

public struct SafetensorInfo: Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataStart: Int
    public let dataEnd: Int

    public var byteCount: Int {
        dataEnd - dataStart
    }

    public init(name: String, dtype: String, shape: [Int], dataStart: Int, dataEnd: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dataStart = dataStart
        self.dataEnd = dataEnd
    }
}

public struct SafetensorsHeader: Equatable {
    public let headerLength: Int
    public let metadata: [String: String]
    public let tensors: [String: SafetensorInfo]

    public var dataBaseOffset: UInt64 {
        UInt64(8 + headerLength)
    }

    public init(headerLength: Int, metadata: [String: String], tensors: [String: SafetensorInfo]) {
        self.headerLength = headerLength
        self.metadata = metadata
        self.tensors = tensors
    }
}

public enum Safetensors {
    public static func readHeader(_ path: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: path)
        defer {
            try? handle.close()
        }

        let prefix = handle.readData(ofLength: 8)
        guard prefix.count == 8 else {
            throw MLXFastError.invalidInput("safetensors file is too small: \(path.path)")
        }
        let headerLength = Int(prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        })
        guard headerLength > 0 else {
            throw MLXFastError.invalidInput("safetensors header is empty: \(path.path)")
        }

        let headerData = handle.readData(ofLength: headerLength)
        guard headerData.count == headerLength else {
            throw MLXFastError.invalidInput("truncated safetensors header: \(path.path)")
        }

        let object = try JSONSerialization.jsonObject(with: headerData)
        guard let dictionary = object as? [String: Any] else {
            throw MLXFastError.invalidInput("safetensors header must be a JSON object: \(path.path)")
        }

        var metadata: [String: String] = [:]
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary {
            if name == "__metadata__" {
                if let raw = value as? [String: String] {
                    metadata = raw
                }
                continue
            }
            guard let tensor = value as? [String: Any] else {
                continue
            }
            guard
                let dtype = tensor["dtype"] as? String,
                let shape = tensor["shape"] as? [Int],
                let offsets = tensor["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw MLXFastError.invalidInput("invalid tensor header for \(name) in \(path.path)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0] else {
                throw MLXFastError.invalidInput("invalid data_offsets for \(name) in \(path.path)")
            }
            tensors[name] = SafetensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataStart: offsets[0],
                dataEnd: offsets[1]
            )
        }

        return SafetensorsHeader(
            headerLength: headerLength,
            metadata: metadata,
            tensors: tensors
        )
    }

    public static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String]
    ) throws -> Int {
        let header = try readHeader(source)
        var selected: [SafetensorInfo] = []
        selected.reserveCapacity(tensorNames.count)
        for name in tensorNames {
            guard let tensor = header.tensors[name] else {
                throw MLXFastError.invalidInput(
                    "tensor \(name) requested from \(source.lastPathComponent) but missing from safetensors header"
                )
            }
            selected.append(tensor)
        }
        selected.sort { $0.name < $1.name }
        guard !selected.isEmpty else {
            return 0
        }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        FileManager.default.createFile(atPath: destination.path, contents: nil)

        let input = try FileHandle(forReadingFrom: source)
        let output = try FileHandle(forWritingTo: destination)
        defer {
            try? input.close()
            try? output.close()
        }

        let outputHeader = try makeHeaderData(
            tensors: selected,
            metadata: header.metadata
        )
        var headerLength = UInt64(outputHeader.count).littleEndian
        let prefix = Data(bytes: &headerLength, count: 8)
        output.write(prefix)
        output.write(outputHeader)

        for tensor in selected {
            try copyBytes(
                from: input,
                to: output,
                offset: header.dataBaseOffset + UInt64(tensor.dataStart),
                count: tensor.byteCount
            )
        }

        return selected.count
    }

    private static func makeHeaderData(
        tensors: [SafetensorInfo],
        metadata: [String: String]
    ) throws -> Data {
        var object: [String: Any] = [:]
        if !metadata.isEmpty {
            object["__metadata__"] = metadata
        }

        var cursor = 0
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            object[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, cursor + tensor.byteCount],
            ]
            cursor += tensor.byteCount
        }

        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while data.count % 8 != 0 {
            data.append(0x20)
        }
        return data
    }

    private static func copyBytes(
        from input: FileHandle,
        to output: FileHandle,
        offset: UInt64,
        count: Int
    ) throws {
        try input.seek(toOffset: offset)
        var remaining = count
        let chunkSize = 8 * 1024 * 1024
        while remaining > 0 {
            let data = input.readData(ofLength: min(chunkSize, remaining))
            if data.isEmpty {
                throw MLXFastError.invalidInput("unexpected EOF while copying safetensors tensor data")
            }
            output.write(data)
            remaining -= data.count
        }
    }
}
