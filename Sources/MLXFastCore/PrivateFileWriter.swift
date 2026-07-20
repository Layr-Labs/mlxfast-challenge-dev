import Darwin
import Foundation

/// Writes private trusted artifacts without ever creating a broader-mode
/// intermediate file. The temporary sibling starts as 0600 and is atomically
/// renamed over the destination after all bytes are flushed.
public enum PrivateFileWriter {
    public static func writeAtomically(
        _ data: Data,
        to path: String,
        maximumByteCount: Int
    ) throws {
        guard maximumByteCount > 0,
              !data.isEmpty,
              data.count <= maximumByteCount
        else {
            throw MLXFastError.invalidInput(
                "private output is empty or exceeds its fixed byte cap"
            )
        }

        let fileManager = FileManager.default
        let outputURL = URL(fileURLWithPath: path).standardizedFileURL
        let parentURL = outputURL.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: parentURL,
            withIntermediateDirectories: true
        )
        let existingValues = try? outputURL.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        if existingValues?.isSymbolicLink == true {
            throw MLXFastError.invalidInput(
                "private output must not be a symlink"
            )
        }
        if fileManager.fileExists(atPath: outputURL.path),
           existingValues?.isRegularFile != true
        {
            throw MLXFastError.invalidInput(
                "private output must be a regular file"
            )
        }

        let temporaryURL = parentURL.appendingPathComponent(
            ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp"
        )
        let descriptor = Darwin.open(
            temporaryURL.path,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
            S_IRUSR | S_IWUSR
        )
        guard descriptor >= 0 else {
            throw MLXFastError.invalidInput(
                "failed to create private output staging file"
            )
        }
        var descriptorIsOpen = true
        defer {
            if descriptorIsOpen {
                _ = Darwin.close(descriptor)
            }
            try? fileManager.removeItem(at: temporaryURL)
        }

        try data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                throw MLXFastError.invalidInput(
                    "private output has no readable bytes"
                )
            }
            var offset = 0
            while offset < bytes.count {
                let result = Darwin.write(
                    descriptor,
                    baseAddress.advanced(by: offset),
                    bytes.count - offset
                )
                if result < 0 {
                    if errno == EINTR {
                        continue
                    }
                    throw MLXFastError.invalidInput(
                        "failed to write private output staging file"
                    )
                }
                guard result > 0 else {
                    throw MLXFastError.invalidInput(
                        "private output staging write made no progress"
                    )
                }
                offset += result
            }
        }
        guard Darwin.fsync(descriptor) == 0 else {
            throw MLXFastError.invalidInput(
                "failed to flush private output staging file"
            )
        }
        guard Darwin.close(descriptor) == 0 else {
            throw MLXFastError.invalidInput(
                "failed to close private output staging file"
            )
        }
        descriptorIsOpen = false

        guard Darwin.rename(temporaryURL.path, outputURL.path) == 0 else {
            throw MLXFastError.invalidInput(
                "failed to atomically publish private output"
            )
        }
        guard Darwin.chmod(outputURL.path, S_IRUSR | S_IWUSR) == 0 else {
            throw MLXFastError.invalidInput(
                "failed to enforce mode 0600 on private output"
            )
        }
        let attributes = try fileManager.attributesOfItem(
            atPath: outputURL.path
        )
        let permissions = (attributes[.posixPermissions] as? NSNumber)?
            .intValue
        guard permissions.map({ $0 & 0o777 }) == 0o600 else {
            throw MLXFastError.invalidInput(
                "private output mode is not 0600"
            )
        }
    }
}
