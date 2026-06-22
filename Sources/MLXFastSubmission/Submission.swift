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

public struct ModifiableSurfaceReport: Codable, Equatable {
    public let contractPath: String
    public let repositoryRoot: String
    public let baseRef: String?
    public let editablePaths: [String]
    public let changedPaths: [String]

    public init(
        contractPath: String,
        repositoryRoot: String,
        baseRef: String?,
        editablePaths: [String],
        changedPaths: [String]
    ) {
        self.contractPath = contractPath
        self.repositoryRoot = repositoryRoot
        self.baseRef = baseRef
        self.editablePaths = editablePaths
        self.changedPaths = changedPaths
    }
}

public struct StoredCredentials: Codable, Equatable {
    public let apiKey: String
    public let apiBaseURL: String?
    public let storedAt: Double

    public init(apiKey: String, apiBaseURL: String? = nil, storedAt: Double) {
        self.apiKey = apiKey
        self.apiBaseURL = apiBaseURL
        self.storedAt = storedAt
    }

    private enum CodingKeys: String, CodingKey {
        case apiKey = "api_key"
        case apiBaseURL = "api_base_url"
        case storedAt = "stored_at"
    }
}

public enum SubmissionSupport {
    public static func ensureWorkspace(contractPath: String) throws -> ChallengeContract {
        let contract = try loadContract(at: contractPath)
        _ = try editableFiles(from: contract, contractPath: contractPath)
        return contract
    }

    public static func enforceModifiableSurface(
        contractPath: String,
        baseRef explicitBaseRef: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ModifiableSurfaceReport {
        let contract = try loadContract(at: contractPath)
        let contractURL = URL(fileURLWithPath: contractPath).standardizedFileURL
        let contractRoot = contractURL.deletingLastPathComponent()
        let repositoryRoot = try gitRepositoryRoot(startingAt: contractRoot)
        let contractPrefix = try gitRelativePath(for: contractRoot, repositoryRoot: repositoryRoot)
        let allowedPaths = contract.editablePaths.map {
            contractPrefix.isEmpty ? $0 : "\(contractPrefix)/\($0)"
        }
        let resolvedBaseRef = try resolveLocalSubmissionBaseRef(
            explicitBaseRef,
            repositoryRoot: repositoryRoot,
            environment: environment
        )
        guard let baseRef = resolvedBaseRef else {
            throw MLXFastError.invalidInput(
                "submit base ref not found; run mlxfast-swift link/clone or pass --base-ref REF"
            )
        }

        var changedPaths = Set<String>()
        changedPaths.formUnion(
            try gitPathList(
                ["diff", "--name-only", "-z", baseRef, "HEAD", "--"],
                repositoryRoot: repositoryRoot
            )
        )
        changedPaths.formUnion(
            try gitPathList(["diff", "--name-only", "-z", "--"], repositoryRoot: repositoryRoot)
        )
        changedPaths.formUnion(
            try gitPathList(["diff", "--cached", "--name-only", "-z", "--"], repositoryRoot: repositoryRoot)
        )
        changedPaths.formUnion(
            try gitPathList(["ls-files", "--others", "--exclude-standard", "-z"], repositoryRoot: repositoryRoot)
        )

        let sortedChangedPaths = changedPaths
            .filter { !isIgnoredLocalGitChange($0) }
            .sorted()
        let rejectedPaths = sortedChangedPaths.filter { !path($0, isAllowedBy: allowedPaths) }
        guard rejectedPaths.isEmpty else {
            let listed = rejectedPaths.prefix(12).joined(separator: ", ")
            let suffix = rejectedPaths.count > 12 ? ", ..." : ""
            throw MLXFastError.invalidInput(
                "local submit includes changes outside benchmark.json editablePaths: \(listed)\(suffix)"
            )
        }

        return ModifiableSurfaceReport(
            contractPath: contractURL.path,
            repositoryRoot: repositoryRoot.path,
            baseRef: resolvedBaseRef,
            editablePaths: contract.editablePaths,
            changedPaths: sortedChangedPaths
        )
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

    public static func credentialsPath(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        try credentialsDirectory(homeDirectory: homeDirectory, environment: environment)
            .appendingPathComponent("credentials")
    }

    public static func legacyCredentialsPath(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        try credentialsDirectory(homeDirectory: homeDirectory, environment: environment)
            .appendingPathComponent("credentials.json")
    }

    public static func storeCredentials(
        apiKey: String,
        apiBaseURL: String? = nil,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        storedAt: Double = Date().timeIntervalSince1970
    ) throws -> String {
        let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("login requires a non-empty API key")
        }
        let normalizedAPIBaseURL = try apiBaseURL.map(normalizeAPIBaseURL)
        let directory = try credentialsDirectory(homeDirectory: homeDirectory, environment: environment)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("credentials")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(
            StoredCredentials(apiKey: trimmed, apiBaseURL: normalizedAPIBaseURL, storedAt: storedAt)
        )
        try data.write(to: path, options: [.atomic])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        return path.path
    }

    public static func loadCredentials(
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> StoredCredentials {
        let path = try credentialsPath(homeDirectory: homeDirectory, environment: environment)
        let legacyPath = try legacyCredentialsPath(homeDirectory: homeDirectory, environment: environment)
        let selectedPath = FileManager.default.fileExists(atPath: path.path) ? path : legacyPath
        guard FileManager.default.fileExists(atPath: selectedPath.path) else {
            throw MLXFastError.missingFile("credentials not found at \(path.path)")
        }
        let credentials = try JSONDecoder().decode(
            StoredCredentials.self,
            from: Data(contentsOf: selectedPath)
        )
        let trimmed = credentials.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("stored credentials contain an empty API key")
        }
        let normalizedAPIBaseURL = try credentials.apiBaseURL.map(normalizeAPIBaseURL)
        return StoredCredentials(
            apiKey: trimmed,
            apiBaseURL: normalizedAPIBaseURL,
            storedAt: credentials.storedAt
        )
    }

    public static func configuredAPIBaseURL(
        credentials: StoredCredentials? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> String {
        if let value = nonEmptyEnvironmentValue("MLXFAST_API_URL", in: environment) {
            return try normalizeAPIBaseURL(value)
        }
        if let value = nonEmptyEnvironmentValue("YUKON_API_URL", in: environment) {
            return try normalizeAPIBaseURL(value)
        }
        if let value = credentials?.apiBaseURL {
            return try normalizeAPIBaseURL(value)
        }
        return "https://yukon-api-dev.fly.dev"
    }

    public static func configuredAPIKey(
        credentials: StoredCredentials? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let value = nonEmptyEnvironmentValue("MLXFAST_API_KEY", in: environment) {
            return value
        }
        if let value = nonEmptyEnvironmentValue("MLXFAST_API_TOKEN", in: environment) {
            return value
        }
        if let value = nonEmptyEnvironmentValue("YUKON_API_TOKEN", in: environment) {
            return value
        }
        return credentials?.apiKey
    }

    public static func packageEditablePathsTarGzip(
        contractPath: String,
        outputPath: String,
        maxByteCount: Int? = MLXFastConstants.defaultMaxSubmissionSourceBytes
    ) throws -> SubmissionArchiveReport {
        let contract = try loadContract(at: contractPath)
        let files = try editableFiles(from: contract, contractPath: contractPath)
        guard !files.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json editablePaths did not select any files")
        }

        let archiveURL = try validateTarGzipOutputPath(
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
        try runTarGzip(
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

    private struct EditableFile: Equatable {
        let relativePath: String
        let absoluteURL: URL
    }

    private static let reservedSubmissionPathPrefixes: Set<String> = [
        ".build",
        ".cache",
        ".git",
        ".github",
        ".swiftpm",
        "correctness_golden.json",
        "golden_prompt",
        "mlxfast-submission.zip",
        "reference_weights",
        "score.json",
        "weights",
    ]

    private static let ignoredLocalGitChangePrefixes: Set<String> = [
        ".build",
        ".cache",
        ".mlxfast-cache",
        ".swiftpm",
        "benchmark-integrity.json",
        "correctness_golden.json",
        "golden_prompt",
        "local_correctness_golden.json",
        "m5_artifacts",
        "mlxfast-submission.tar.gz",
        "mlxfast-submission.tgz",
        "mlxfast-submission.zip",
        "private_prompts.json",
        "reference_weights",
        "results.tsv",
        "score.json",
        "score.json.sha256",
        "weights",
    ]

    private static func credentialsDirectory(
        homeDirectory: URL,
        environment: [String: String]
    ) throws -> URL {
        if let configHome = nonEmptyEnvironmentValue("MLXFAST_CONFIG_HOME", in: environment) {
            return try absoluteDirectory(from: configHome, environmentName: "MLXFAST_CONFIG_HOME")
                .appendingPathComponent("mlxfast", isDirectory: true)
        }
        if let xdgConfigHome = nonEmptyEnvironmentValue("XDG_CONFIG_HOME", in: environment) {
            return try absoluteDirectory(from: xdgConfigHome, environmentName: "XDG_CONFIG_HOME")
                .appendingPathComponent("mlxfast", isDirectory: true)
        }
        return homeDirectory
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("mlxfast", isDirectory: true)
    }

    private static func nonEmptyEnvironmentValue(
        _ name: String,
        in environment: [String: String]
    ) -> String? {
        let value = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? nil : value
    }

    private static func resolveLocalSubmissionBaseRef(
        _ explicitBaseRef: String?,
        repositoryRoot: URL,
        environment: [String: String]
    ) throws -> String? {
        if let explicit = explicitBaseRef?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            guard gitRefExists(explicit, repositoryRoot: repositoryRoot) else {
                throw MLXFastError.invalidInput("submit base ref not found: \(explicit)")
            }
            return explicit
        }

        var candidates: [String] = []
        if let value = nonEmptyEnvironmentValue("MLXFAST_SUBMISSION_BASE_REF", in: environment) {
            candidates.append(value)
        }
        if let value = nonEmptyEnvironmentValue("YUKON_SOURCE_REF", in: environment) {
            candidates.append(value)
        }
        if let value = gitConfigValue("yukon.source-ref", repositoryRoot: repositoryRoot) {
            candidates.append(value)
        }
        candidates.append("origin/main")

        var seen: Set<String> = []
        for candidate in candidates {
            guard seen.insert(candidate).inserted else {
                continue
            }
            if gitRefExists(candidate, repositoryRoot: repositoryRoot) {
                return candidate
            }
        }
        return nil
    }

    private static func gitRepositoryRoot(startingAt directory: URL) throws -> URL {
        do {
            let output = try runGit(["rev-parse", "--show-toplevel"], repositoryRoot: directory)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !output.isEmpty else {
                throw MLXFastError.invalidInput("git did not return a repository root")
            }
            return URL(fileURLWithPath: output, isDirectory: true).standardizedFileURL
        } catch {
            throw MLXFastError.invalidInput(
                "submit must run inside a git checkout so local editable-path enforcement can compare changes"
            )
        }
    }

    private static func gitRelativePath(for url: URL, repositoryRoot: URL) throws -> String {
        let rootPath = repositoryRoot.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        if path == rootPath {
            return ""
        }
        guard path.hasPrefix(rootPath + "/") else {
            throw MLXFastError.invalidInput("benchmark contract is outside the git repository")
        }
        return String(path.dropFirst(rootPath.count + 1))
    }

    private static func gitPathList(_ arguments: [String], repositoryRoot: URL) throws -> [String] {
        try runGit(arguments, repositoryRoot: repositoryRoot)
            .split(separator: "\0")
            .map(String.init)
            .filter { !$0.isEmpty }
    }

    private static func gitConfigValue(_ key: String, repositoryRoot: URL) -> String? {
        guard let output = try? runGit(["config", "--get", key], repositoryRoot: repositoryRoot) else {
            return nil
        }
        let value = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    private static func gitRefExists(_ ref: String, repositoryRoot: URL) -> Bool {
        (try? runGit(["rev-parse", "--verify", "--quiet", "\(ref)^{commit}"], repositoryRoot: repositoryRoot)) != nil
    }

    private static func path(_ path: String, isAllowedBy allowedPaths: [String]) -> Bool {
        for allowedPath in allowedPaths {
            if path == allowedPath || path.hasPrefix(allowedPath + "/") {
                return true
            }
        }
        return false
    }

    private static func isIgnoredLocalGitChange(_ path: String) -> Bool {
        if shouldIgnoreMetadataFile(path) {
            return true
        }
        for ignoredPath in ignoredLocalGitChangePrefixes {
            if path == ignoredPath || path.hasPrefix(ignoredPath + "/") {
                return true
            }
        }
        return false
    }

    private static func runGit(_ arguments: [String], repositoryRoot: URL) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", repositoryRoot.path] + arguments
        let output = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = output
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let error = String(
                data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw MLXFastError.invalidInput("git \(arguments.joined(separator: " ")) failed: \(error)")
        }

        return String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private static func absoluteDirectory(from rawPath: String, environmentName: String) throws -> URL {
        let expandedPath = (rawPath as NSString).expandingTildeInPath
        guard (expandedPath as NSString).isAbsolutePath else {
            throw MLXFastError.invalidInput("\(environmentName) must be an absolute path")
        }
        return URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
    }

    private static func normalizeAPIBaseURL(_ rawValue: String) throws -> String {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("API URL must not be empty")
        }
        guard let components = URLComponents(string: trimmed),
              let scheme = components.scheme?.lowercased(),
              scheme == "https" || scheme == "http",
              components.host?.isEmpty == false else {
            throw MLXFastError.invalidInput("API URL must be an absolute http(s) URL")
        }
        return String(trimmed.drop { $0 == " " || $0 == "\n" || $0 == "\t" })
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

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
        try validateArchiveNotInsideEditablePath(
            archiveURL,
            contractPath: contractPath,
            editablePaths: editablePaths
        )
        return archiveURL
    }

    private static func validateTarGzipOutputPath(
        _ outputPath: String,
        contractPath: String,
        editablePaths: [String]
    ) throws -> URL {
        let archiveURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        guard archiveURL.path.hasSuffix(".tar.gz") || archiveURL.path.hasSuffix(".tgz") else {
            throw MLXFastError.invalidInput("submission archive output must end in .tar.gz or .tgz")
        }
        try validateArchiveNotInsideEditablePath(
            archiveURL,
            contractPath: contractPath,
            editablePaths: editablePaths
        )
        return archiveURL
    }

    private static func validateArchiveNotInsideEditablePath(
        _ archiveURL: URL,
        contractPath: String,
        editablePaths: [String]
    ) throws {
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

    private static func runTarGzip(
        relativeFiles: [String],
        archivePath: String,
        workingDirectory: URL
    ) throws {
        let candidates = ["/usr/bin/tar", "/opt/homebrew/bin/tar", "/usr/local/bin/tar"]
        guard let tar = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw MLXFastError.missingFile("tar executable not found")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: tar)
        process.arguments = ["-czf", archivePath] + relativeFiles
        process.currentDirectoryURL = workingDirectory
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["COPYFILE_DISABLE": "1"]
        ) { _, new in new }
        let errorPipe = Pipe()
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let error = String(
                data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw MLXFastError.invalidInput("tar failed with status \(process.terminationStatus): \(error)")
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
