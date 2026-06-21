import Foundation
import MLXFastCore

public struct YukonLiveSubmissionOptions: Equatable {
    public let contractPath: String
    public let benchmark: String
    public let maxByteCount: Int?
    public let note: String
    public let claimedScore: Double?
    public let idempotencyKey: String

    public init(
        contractPath: String,
        benchmark: String,
        maxByteCount: Int? = MLXFastConstants.defaultMaxSubmissionSourceBytes,
        note: String,
        claimedScore: Double? = nil,
        idempotencyKey: String = UUID().uuidString
    ) {
        self.contractPath = contractPath
        self.benchmark = benchmark
        self.maxByteCount = maxByteCount
        self.note = note
        self.claimedScore = claimedScore
        self.idempotencyKey = idempotencyKey
    }
}

public struct YukonLiveSubmissionReport: Equatable {
    public let archive: SubmissionArchiveReport
    public let response: YukonSubmissionResponse

    public init(archive: SubmissionArchiveReport, response: YukonSubmissionResponse) {
        self.archive = archive
        self.response = response
    }
}

public enum YukonSubmissionUploader {
    public static func uploadEditablePaths(
        _ options: YukonLiveSubmissionOptions,
        client: YukonClient
    ) throws -> YukonLiveSubmissionReport {
        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-submit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryURL, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }

        let archiveURL = temporaryURL.appendingPathComponent("submission.tar.gz")
        let archive = try SubmissionSupport.packageEditablePathsTarGzip(
            contractPath: options.contractPath,
            outputPath: archiveURL.path,
            maxByteCount: options.maxByteCount
        )
        let response = try client.createSubmission(
            YukonSubmissionOptions(
                benchmark: options.benchmark,
                archivePath: archive.archivePath,
                idempotencyKey: options.idempotencyKey,
                note: options.note,
                claimedScore: options.claimedScore
            )
        )
        return YukonLiveSubmissionReport(archive: archive, response: response)
    }
}
