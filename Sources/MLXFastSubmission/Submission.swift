import CryptoKit
import Foundation
import MLXFastCore

public struct ChallengeContract: Decodable, Equatable {
    public let schemaVersion: Int
    public let name: String
    public let editablePaths: [String]
}

public struct SubmissionArchiveReport: Codable, Equatable {
    public let contractPath: String
    public let archivePath: String
    public let editablePaths: [String]
    public let fileCount: Int
    public let byteCount: Int
    public let archiveSha256: String
}

public enum SubmissionSupport {
    public static func ensureWorkspace(contractPath: String) throws -> ChallengeContract {
        let contract = try loadContract(at: contractPath)
        _ = try editableFiles(from: contract, contractPath: contractPath)
        return contract
    }

    public static func packageEditablePaths(
        contractPath: String,
        outputPath: String,
        maxByteCount: Int? = MLXFastConstants.defaultMaxSubmissionSourceBytes
    ) throws -> SubmissionArchiveReport {
        let contract = try loadContract(at: contractPath)
        let files = try editableFiles(from: contract, contractPath: contractPath)
        guard !files.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json editablePaths did not select any files")
        }

        let archiveURL = try validateArchiveOutputPath(
            outputPath,
            contractPath: contractPath,
            editablePaths: contract.editablePaths
        )
        let byteCount = try totalByteCount(files)
        if let maxByteCount, byteCount > maxByteCount {
            throw MLXFastError.invalidInput(
                "submission source files total \(byteCount) bytes; limit is \(maxByteCount)"
            )
        }

        if let parent = archiveURL.deletingLastPathComponentIfPresent() {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        if FileManager.default.fileExists(atPath: archiveURL.path) {
            let values = try archiveURL.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory != true else {
                throw MLXFastError.invalidInput("submission archive output is a directory: \(archiveURL.path)")
            }
            try FileManager.default.removeItem(at: archiveURL)
        }

        let contractRoot = URL(fileURLWithPath: contractPath).standardizedFileURL
            .deletingLastPathComponent()
        try runZip(
            relativeFiles: files.map(\.relativePath),
            archivePath: archiveURL.path,
            workingDirectory: contractRoot
        )
        let archiveSha256 = try fileSHA256(archiveURL)

        return SubmissionArchiveReport(
            contractPath: URL(fileURLWithPath: contractPath).standardizedFileURL.path,
            archivePath: archiveURL.path,
            editablePaths: contract.editablePaths,
            fileCount: files.count,
            byteCount: byteCount,
            archiveSha256: archiveSha256
        )
    }

    public static func storeCredentials(apiKey: String) throws -> String {
        let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("login requires a non-empty API key")
        }
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("mlxfast", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("credentials.json")
        let object: [String: Any] = [
            "api_key": trimmed,
            "stored_at": Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: path, options: [.atomic])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        return path.path
    }

    private struct EditableFile: Equatable {
        let relativePath: String
        let absoluteURL: URL
    }

    private static let reservedSubmissionPathPrefixes: Set<String> = [
        ".build",
        ".git",
        ".github",
        ".swiftpm",
        "correctness_golden.json",
        "mlxfast-submission.zip",
        "reference_weights",
        "score.json",
        "weights",
    ]

    private static func loadContract(at path: String) throws -> ChallengeContract {
        try requireFile(path, description: "benchmark contract")
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let contract = try JSONDecoder().decode(ChallengeContract.self, from: data)
        guard contract.schemaVersion == 1 else {
            throw MLXFastError.invalidInput("benchmark.json schemaVersion must be 1")
        }
        guard !contract.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json name must not be empty")
        }
        guard !contract.editablePaths.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json editablePaths must not be empty")
        }

        var seen: Set<String> = []
        for path in contract.editablePaths {
            try validateRelativeContractPath(path, field: "editablePaths")
            guard seen.insert(path).inserted else {
                throw MLXFastError.invalidInput("benchmark.json editablePaths contains duplicate path \(path)")
            }
        }
        return contract
    }

    private static func editableFiles(
        from contract: ChallengeContract,
        contractPath: String
    ) throws -> [EditableFile] {
        let contractURL = URL(fileURLWithPath: contractPath).standardizedFileURL
        let root = contractURL.deletingLastPathComponent()
        var files: [EditableFile] = []
        for editablePath in contract.editablePaths {
            let rootURL = root.appendingPathComponent(editablePath).standardizedFileURL
            try ensureInside(rootURL, root: root, relativePath: editablePath)
            guard FileManager.default.fileExists(atPath: rootURL.path) else {
                throw MLXFastError.missingFile("editable path \(editablePath) not found")
            }
            let values = try rootURL.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("editable path \(editablePath) must not be a symlink")
            }
            if shouldIgnoreMetadataFile(editablePath) {
                continue
            }
            if values.isRegularFile == true {
                files.append(EditableFile(relativePath: editablePath, absoluteURL: rootURL))
                continue
            }
            guard values.isDirectory == true else {
                throw MLXFastError.invalidInput("editable path \(editablePath) must be a file or directory")
            }
            files.append(contentsOf: try regularFilesUnder(rootURL, root: root))
        }
        let sortedFiles = files.sorted { $0.relativePath < $1.relativePath }
        var seenFiles: Set<String> = []
        for file in sortedFiles {
            guard seenFiles.insert(file.relativePath).inserted else {
                throw MLXFastError.invalidInput(
                    "benchmark.json editablePaths selects duplicate file \(file.relativePath)"
                )
            }
        }
        return sortedFiles
    }

    private static func regularFilesUnder(_ directory: URL, root: URL) throws -> [EditableFile] {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("editable directory not found at \(directory.path)")
        }

        var files: [EditableFile] = []
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let relativePath = try relativePathForURL(standardized, root: root)
            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput(
                    "editable path \(relativePath) must not be a symlink"
                )
            }
            if values.isDirectory == true {
                if shouldIgnoreMetadataDirectory(relativePath) {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput(
                    "editable path \(relativePath) must be a regular file"
                )
            }
            if shouldIgnoreMetadataFile(relativePath) {
                continue
            }
            files.append(EditableFile(relativePath: relativePath, absoluteURL: standardized))
        }
        return files
    }

    private static func validateRelativeContractPath(_ path: String, field: String) throws {
        guard !path.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json \(field) contains an empty path")
        }
        guard !path.hasPrefix("/") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) must be relative")
        }
        guard !path.contains("\0"), !path.contains("\n"), !path.contains("\\") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) contains invalid characters")
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains(""), !components.contains("."), !components.contains("..") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) is not normalized")
        }
        guard !isReservedSubmissionPath(path) else {
            throw MLXFastError.invalidInput(
                "benchmark.json \(field) path \(path) selects generated or repository metadata"
            )
        }
    }

    private static func relativePathForURL(_ url: URL, root: URL) throws -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else {
            throw MLXFastError.invalidInput("editable path escaped repository root: \(path)")
        }
        let relativePath = String(path.dropFirst(rootPath.count + 1))
        try validateRelativeContractPath(relativePath, field: "editable file")
        return relativePath
    }

    private static func ensureInside(_ url: URL, root: URL, relativePath: String) throws {
        let resolved = url.standardizedFileURL.path
        let rootPath = root.standardizedFileURL.path
        guard resolved == rootPath || resolved.hasPrefix(rootPath + "/") else {
            throw MLXFastError.invalidInput(
                "benchmark.json editable path \(relativePath) escapes repository root"
            )
        }
    }

    private static func validateArchiveOutputPath(
        _ outputPath: String,
        contractPath: String,
        editablePaths: [String]
    ) throws -> URL {
        let archiveURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        guard archiveURL.pathExtension.lowercased() == "zip" else {
            throw MLXFastError.invalidInput("submission archive output must end in .zip")
        }

        let root = URL(fileURLWithPath: contractPath).standardizedFileURL.deletingLastPathComponent()
        let archivePath = archiveURL.path
        for editablePath in editablePaths {
            let editableURL = root.appendingPathComponent(editablePath).standardizedFileURL
            let editableRootPath = editableURL.path
            guard archivePath != editableRootPath,
                  !archivePath.hasPrefix(editableRootPath + "/") else {
                throw MLXFastError.invalidInput(
                    "submission archive output must not be inside editable path \(editablePath)"
                )
            }
        }
        return archiveURL
    }

    private static func totalByteCount(_ files: [EditableFile]) throws -> Int {
        var total = 0
        for file in files {
            let size = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: file.absoluteURL.path),
                path: file.absoluteURL.path
            )
            guard total <= Int.max - size else {
                throw MLXFastError.invalidInput("submission source files exceed Int range")
            }
            total += size
        }
        return total
    }

    private static func isReservedSubmissionPath(_ path: String) -> Bool {
        for reserved in reservedSubmissionPathPrefixes {
            if path == reserved || path.hasPrefix(reserved + "/") {
                return true
            }
        }
        return false
    }

    private static func shouldIgnoreMetadataDirectory(_ path: String) -> Bool {
        URL(fileURLWithPath: path).lastPathComponent == "__MACOSX"
    }

    private static func shouldIgnoreMetadataFile(_ path: String) -> Bool {
        let name = URL(fileURLWithPath: path).lastPathComponent
        return name == ".DS_Store" || name.hasPrefix("._")
    }

    private static func runZip(
        relativeFiles: [String],
        archivePath: String,
        workingDirectory: URL
    ) throws {
        let candidates = ["/usr/bin/zip", "/opt/homebrew/bin/zip", "/usr/local/bin/zip"]
        guard let zip = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw MLXFastError.missingFile("zip executable not found")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: zip)
        process.arguments = ["-q", "-X", archivePath, "-@"]
        process.currentDirectoryURL = workingDirectory
        let input = Pipe()
        let errorPipe = Pipe()
        process.standardInput = input
        process.standardError = errorPipe

        try process.run()
        let list = relativeFiles.joined(separator: "\n") + "\n"
        input.fileHandleForWriting.write(Data(list.utf8))
        try input.fileHandleForWriting.close()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let error = String(
                data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw MLXFastError.invalidInput("zip failed with status \(process.terminationStatus): \(error)")
        }
    }

    private static func fileSHA256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }

        var hasher = SHA256()
        let chunkSize = 8 * 1024 * 1024
        while true {
            let data = handle.readData(ofLength: chunkSize)
            if data.isEmpty {
                return hasher.finalize().map { String(format: "%02x", $0) }.joined()
            }
            hasher.update(data: data)
        }
    }
}

private extension URL {
    func deletingLastPathComponentIfPresent() -> URL? {
        let parent = deletingLastPathComponent()
        return parent.path.isEmpty ? nil : parent
    }
}
