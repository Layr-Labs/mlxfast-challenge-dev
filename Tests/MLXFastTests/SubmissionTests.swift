import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastSubmission

@Test
func submissionPackagesEditableFilesAndSkipsMacMetadata() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "local metadata".write(
        to: sources.appendingPathComponent(".DS_Store"),
        atomically: true,
        encoding: .utf8
    )
    try "appledouble".write(
        to: sources.appendingPathComponent("._A.swift"),
        atomically: true,
        encoding: .utf8
    )
    let macosx = sources.appendingPathComponent("__MACOSX", isDirectory: true)
    try FileManager.default.createDirectory(at: macosx, withIntermediateDirectories: true)
    try "ignored".write(
        to: macosx.appendingPathComponent("Ignored.swift"),
        atomically: true,
        encoding: .utf8
    )

    let archive = root.appendingPathComponent("submission.zip")
    let report = try SubmissionSupport.packageEditablePaths(
        contractPath: root.appendingPathComponent("benchmark.json").path,
        outputPath: archive.path
    )

    #expect(report.fileCount == 1)
    #expect(report.byteCount == "public enum A {}\n".utf8.count)
    #expect(report.archiveSha256.count == 64)
    #expect(FileManager.default.fileExists(atPath: archive.path))
}

@Test
func submissionRejectsGeneratedArtifactsInEditablePaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["weights"])
    try FileManager.default.createDirectory(
        at: root.appendingPathComponent("weights", isDirectory: true),
        withIntermediateDirectories: true
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsDuplicateEditablePaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources", "Sources"])

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsOverlappingEditablePathsThatSelectSameFile() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources", "Sources/A.swift"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsBackslashContractPaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources\\A.swift"])

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsSymlinkInsideEditableDirectory() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    let target = sources.appendingPathComponent("A.swift")
    try "public enum A {}\n".write(to: target, atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(
        at: sources.appendingPathComponent("Alias.swift"),
        withDestinationURL: target
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsArchiveOutputInsideEditablePath() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: sources.appendingPathComponent("submission.zip").path
        )
    }
}

@Test
func submissionRejectsNonZipArchiveOutput() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: root.appendingPathComponent("submission.txt").path
        )
    }
}

@Test
func submissionRejectsSourceByteCountAboveLimit() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "1234".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: root.appendingPathComponent("submission.zip").path,
            maxByteCount: 3
        )
    }
}

@Test
func submissionAllowsUnlimitedSourceByteCountWhenExplicitlyConfigured() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "1234".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SubmissionSupport.packageEditablePaths(
        contractPath: root.appendingPathComponent("benchmark.json").path,
        outputPath: root.appendingPathComponent("submission.zip").path,
        maxByteCount: nil
    )

    #expect(report.byteCount == 4)
    #expect(report.archiveSha256.count == 64)
}

private func makeSubmissionWorkspace(editablePaths: [String]) throws -> URL {
    let root = try temporarySubmissionDirectory()
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
    try writeSubmissionContract(
        root.appendingPathComponent("benchmark.json"),
        editablePaths: editablePaths
    )
    return root
}

private func writeSubmissionContract(_ path: URL, editablePaths: [String]) throws {
    let encodedPaths = editablePaths
        .map { #""\#($0.replacingOccurrences(of: "\\", with: "\\\\"))""# }
        .joined(separator: ", ")
    let json = """
    {
      "schemaVersion": 1,
      "name": "fixture",
      "editablePaths": [\(encodedPaths)]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)
}

private func temporarySubmissionDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlxfast-submission-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
