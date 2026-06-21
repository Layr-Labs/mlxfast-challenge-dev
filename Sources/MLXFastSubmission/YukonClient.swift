import Foundation
import MLXFastCore

public struct YukonAccount: Decodable, Equatable {
    public let id: String
    public let email: String
    public let username: String?
}

public struct YukonMeResponse: Decodable, Equatable {
    public let account: YukonAccount
}

public struct YukonSubmission: Decodable, Equatable {
    public let id: String
    public let benchmarkId: String
    public let status: String
    public let note: String?
    public let claimedScore: Double?
    public let officialScore: Double?
    public let improved: Bool?
    public let createdAt: String?
    public let updatedAt: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case benchmarkId
        case status
        case note
        case claimedScore
        case officialScore
        case improved
        case createdAt
        case updatedAt
    }

    public init(
        id: String,
        benchmarkId: String = "",
        status: String,
        note: String? = nil,
        claimedScore: Double? = nil,
        officialScore: Double? = nil,
        improved: Bool? = nil,
        createdAt: String? = nil,
        updatedAt: String? = nil
    ) {
        self.id = id
        self.benchmarkId = benchmarkId
        self.status = status
        self.note = note
        self.claimedScore = claimedScore
        self.officialScore = officialScore
        self.improved = improved
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.benchmarkId = try container.decodeIfPresent(String.self, forKey: .benchmarkId) ?? ""
        self.status = try container.decodeIfPresent(String.self, forKey: .status) ?? ""
        self.note = try container.decodeIfPresent(String.self, forKey: .note)
        self.claimedScore = try container.decodeIfPresent(Double.self, forKey: .claimedScore)
        self.officialScore = try container.decodeIfPresent(Double.self, forKey: .officialScore)
        self.improved = try container.decodeIfPresent(Bool.self, forKey: .improved)
        self.createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt)
        self.updatedAt = try container.decodeIfPresent(String.self, forKey: .updatedAt)
    }
}

public struct YukonJob: Decodable, Equatable {
    public let id: String
    public let status: String
}

public struct YukonSubmissionResponse: Decodable, Equatable {
    public let submission: YukonSubmission
    public let job: YukonJob?
}

public struct YukonSubmissionListResponse: Decodable, Equatable {
    public let submissions: [YukonSubmission]

    private enum CodingKeys: String, CodingKey {
        case submissions
        case items
        case data
    }

    public init(submissions: [YukonSubmission]) {
        self.submissions = submissions
    }

    public init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: CodingKeys.self) {
            if let submissions = try container.decodeIfPresent([YukonSubmission].self, forKey: .submissions) {
                self.submissions = submissions
                return
            }
            if let submissions = try container.decodeIfPresent([YukonSubmission].self, forKey: .items) {
                self.submissions = submissions
                return
            }
            if let submissions = try container.decodeIfPresent([YukonSubmission].self, forKey: .data) {
                self.submissions = submissions
                return
            }
        }
        self.submissions = try [YukonSubmission](from: decoder)
    }
}

public struct YukonBenchmark: Decodable, Equatable {
    public let id: String
    public let name: String?
    public let status: String?
    public let category: String?
    public let direction: String?
    public let sourceURL: String?
    public let sourceRef: String?
    public let scorePath: String?
    public let setupCommand: [String]?
    public let benchmarkCommand: [String]?
    public let currentBestScore: Double?
    public let baselineScore: Double?
    public let closesAt: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case status
        case category
        case direction
        case sourceURL = "sourceUrl"
        case sourceRef
        case scorePath
        case setupCommand
        case benchmarkCommand
        case currentBestScore
        case baselineScore
        case closesAt
    }

    public init(
        id: String,
        name: String? = nil,
        status: String? = nil,
        category: String? = nil,
        direction: String? = nil,
        sourceURL: String? = nil,
        sourceRef: String? = nil,
        scorePath: String? = nil,
        setupCommand: [String]? = nil,
        benchmarkCommand: [String]? = nil,
        currentBestScore: Double? = nil,
        baselineScore: Double? = nil,
        closesAt: String? = nil
    ) {
        self.id = id
        self.name = name
        self.status = status
        self.category = category
        self.direction = direction
        self.sourceURL = sourceURL
        self.sourceRef = sourceRef
        self.scorePath = scorePath
        self.setupCommand = setupCommand
        self.benchmarkCommand = benchmarkCommand
        self.currentBestScore = currentBestScore
        self.baselineScore = baselineScore
        self.closesAt = closesAt
    }
}

public struct YukonBenchmarkResponse: Decodable, Equatable {
    public let benchmark: YukonBenchmark

    private enum CodingKeys: String, CodingKey {
        case benchmark
    }

    public init(benchmark: YukonBenchmark) {
        self.benchmark = benchmark
    }

    public init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: CodingKeys.self),
           let benchmark = try container.decodeIfPresent(YukonBenchmark.self, forKey: .benchmark) {
            self.benchmark = benchmark
        } else {
            self.benchmark = try YukonBenchmark(from: decoder)
        }
    }
}

public struct YukonSubmissionOptions: Equatable {
    public let benchmark: String
    public let archivePath: String
    public let idempotencyKey: String
    public let note: String?
    public let claimedScore: Double?

    public init(
        benchmark: String,
        archivePath: String,
        idempotencyKey: String = UUID().uuidString,
        note: String? = nil,
        claimedScore: Double? = nil
    ) {
        self.benchmark = benchmark
        self.archivePath = archivePath
        self.idempotencyKey = idempotencyKey
        self.note = note
        self.claimedScore = claimedScore
    }
}

public struct YukonAPIError: Error, CustomStringConvertible, Equatable {
    public let statusCode: Int
    public let responseBody: String
    public let code: String?
    public let message: String?

    public init(statusCode: Int, responseBody: String) {
        self.statusCode = statusCode
        self.responseBody = responseBody
        let parsed = Self.parseErrorBody(responseBody)
        self.code = parsed.code
        self.message = parsed.message
    }

    public var description: String {
        if let message, !message.isEmpty {
            if let code, !code.isEmpty {
                return "Yukon API request failed with HTTP \(statusCode) [\(code)]: \(message)"
            }
            return "Yukon API request failed with HTTP \(statusCode): \(message)"
        }
        if responseBody.isEmpty {
            return "Yukon API request failed with HTTP \(statusCode)"
        }
        return "Yukon API request failed with HTTP \(statusCode): \(responseBody)"
    }

    private static func parseErrorBody(_ body: String) -> (code: String?, message: String?) {
        guard let data = body.data(using: .utf8), !data.isEmpty else {
            return (nil, nil)
        }
        do {
            let decoded = try JSONDecoder().decode(YukonErrorEnvelope.self, from: data)
            return (
                decoded.error?.code ?? decoded.code,
                decoded.error?.message ?? decoded.message
            )
        } catch {
            return (nil, nil)
        }
    }
}

public struct YukonClient {
    public typealias Transport = (URLRequest) throws -> (Data, HTTPURLResponse)

    public static let maxSubmissionArchiveBytes = 25 * 1024 * 1024
    public static let maxSubmissionNoteBytes = 10 * 1024

    private let apiBaseURL: URL
    private let apiKey: String
    private let transport: Transport

    public init(apiBaseURL: String, apiKey: String) throws {
        try self.init(apiBaseURL: apiBaseURL, apiKey: apiKey, transport: URLSessionTransport.perform)
    }

    public init(apiBaseURL: String, apiKey: String, transport: @escaping Transport) throws {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            throw MLXFastError.invalidInput("Yukon API key must not be empty")
        }
        let normalizedBaseURL = try SubmissionSupport.configuredAPIBaseURL(
            credentials: StoredCredentials(apiKey: trimmedKey, apiBaseURL: apiBaseURL, storedAt: 0),
            environment: [:]
        )
        guard let url = URL(string: normalizedBaseURL) else {
            throw MLXFastError.invalidInput("Yukon API URL is invalid: \(normalizedBaseURL)")
        }
        self.apiBaseURL = url
        self.apiKey = trimmedKey
        self.transport = transport
    }

    public func me() throws -> YukonMeResponse {
        var request = try authorizedRequest(path: "/api/me")
        request.httpMethod = "GET"
        let (data, _) = try perform(request)
        return try JSONDecoder().decode(YukonMeResponse.self, from: data)
    }

    public func getBenchmark(_ benchmark: String) throws -> YukonBenchmarkResponse {
        let benchmark = benchmark.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !benchmark.isEmpty else {
            throw MLXFastError.invalidInput("benchmark id or name must not be empty")
        }
        var request = try authorizedRequest(path: "/api/benchmarks/\(urlPathEncode(benchmark))")
        request.httpMethod = "GET"
        let (data, _) = try perform(request)
        return try JSONDecoder().decode(YukonBenchmarkResponse.self, from: data)
    }

    public func listBenchmarkSubmissions(_ benchmark: String) throws -> YukonSubmissionListResponse {
        let benchmark = benchmark.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !benchmark.isEmpty else {
            throw MLXFastError.invalidInput("benchmark id or name must not be empty")
        }
        var request = try authorizedRequest(path: "/api/benchmarks/\(urlPathEncode(benchmark))/submissions")
        request.httpMethod = "GET"
        let (data, _) = try perform(request)
        return try JSONDecoder().decode(YukonSubmissionListResponse.self, from: data)
    }

    public func createSubmission(_ options: YukonSubmissionOptions) throws -> YukonSubmissionResponse {
        let benchmark = options.benchmark.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !benchmark.isEmpty else {
            throw MLXFastError.invalidInput("submit upload requires a benchmark id or name")
        }
        let idempotencyKey = options.idempotencyKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !idempotencyKey.isEmpty else {
            throw MLXFastError.invalidInput("Yukon idempotency key must not be empty")
        }
        let archiveURL = URL(fileURLWithPath: options.archivePath)
        let archiveBytes = try Data(contentsOf: archiveURL)
        guard archiveBytes.count <= Self.maxSubmissionArchiveBytes else {
            throw MLXFastError.invalidInput(
                "submission archive must be at most \(Self.maxSubmissionArchiveBytes) bytes"
            )
        }
        let note = try normalizedNote(options.note)

        var queryItems: [URLQueryItem] = []
        if let claimedScore = options.claimedScore {
            guard claimedScore.isFinite else {
                throw MLXFastError.invalidInput("--claimed-score must be finite")
            }
            queryItems.append(URLQueryItem(name: "claimedScore", value: "\(claimedScore)"))
        }

        var request = try authorizedRequest(
            path: "/api/benchmarks/\(urlPathEncode(benchmark))/submissions",
            queryItems: queryItems
        )
        request.httpMethod = "POST"
        request.setValue(idempotencyKey, forHTTPHeaderField: "idempotency-key")
        if let note {
            let boundary = "----mlxfast-yukon-\(UUID().uuidString)"
            request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
            request.httpBody = multipartBody(archiveBytes: archiveBytes, note: note, boundary: boundary)
        } else {
            request.setValue("application/gzip", forHTTPHeaderField: "content-type")
            request.httpBody = archiveBytes
        }

        let (data, _) = try perform(request)
        return try JSONDecoder().decode(YukonSubmissionResponse.self, from: data)
    }

    private func authorizedRequest(
        path: String,
        queryItems: [URLQueryItem] = []
    ) throws -> URLRequest {
        guard var components = URLComponents(url: apiBaseURL, resolvingAgainstBaseURL: false) else {
            throw MLXFastError.invalidInput("Yukon API URL is invalid: \(apiBaseURL.absoluteString)")
        }
        let basePath = components.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let requestPath = path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components.percentEncodedPath = "/" + [basePath, requestPath]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw MLXFastError.invalidInput("Yukon request URL is invalid")
        }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
        request.setValue("application/json", forHTTPHeaderField: "accept")
        return request
    }

    private func perform(_ request: URLRequest) throws -> (Data, HTTPURLResponse) {
        let (data, response) = try transport(request)
        guard (200..<300).contains(response.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw YukonAPIError(statusCode: response.statusCode, responseBody: body)
        }
        return (data, response)
    }
}

private enum URLSessionTransport {
    static func perform(_ request: URLRequest) throws -> (Data, HTTPURLResponse) {
        let semaphore = DispatchSemaphore(value: 0)
        let resultBox = URLSessionResultBox()
        URLSession.shared.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let error {
                resultBox.store(.failure(error))
                return
            }
            guard let response = response as? HTTPURLResponse else {
                resultBox.store(
                    .failure(MLXFastError.invalidInput("Yukon API did not return an HTTP response"))
                )
                return
            }
            resultBox.store(.success((data ?? Data(), response)))
        }.resume()
        semaphore.wait()
        guard let result = resultBox.load() else {
            throw MLXFastError.invalidInput("Yukon API request did not complete")
        }
        return try result.get()
    }
}

private final class URLSessionResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<(Data, HTTPURLResponse), Error>?

    func store(_ result: Result<(Data, HTTPURLResponse), Error>) {
        lock.lock()
        self.result = result
        lock.unlock()
    }

    func load() -> Result<(Data, HTTPURLResponse), Error>? {
        lock.lock()
        defer { lock.unlock() }
        return result
    }
}

private struct YukonErrorEnvelope: Decodable {
    let error: YukonErrorDetails?
    let code: String?
    let message: String?
}

private struct YukonErrorDetails: Decodable {
    let code: String?
    let message: String?
}

private func normalizedNote(_ note: String?) throws -> String? {
    guard let note else {
        return nil
    }
    let trimmed = note.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        return nil
    }
    guard Data(trimmed.utf8).count <= YukonClient.maxSubmissionNoteBytes else {
        throw MLXFastError.invalidInput(
            "submission note must be at most \(YukonClient.maxSubmissionNoteBytes) bytes"
        )
    }
    return trimmed
}

private func multipartBody(archiveBytes: Data, note: String, boundary: String) -> Data {
    var body = Data()
    body.appendUTF8("--\(boundary)\r\n")
    body.appendUTF8("Content-Disposition: form-data; name=\"archive\"; filename=\"submission.tar.gz\"\r\n")
    body.appendUTF8("Content-Type: application/gzip\r\n\r\n")
    body.append(archiveBytes)
    body.appendUTF8("\r\n")
    body.appendUTF8("--\(boundary)\r\n")
    body.appendUTF8("Content-Disposition: form-data; name=\"note\"\r\n\r\n")
    body.appendUTF8(note)
    body.appendUTF8("\r\n")
    body.appendUTF8("--\(boundary)--\r\n")
    return body
}

private func urlPathEncode(_ value: String) -> String {
    var allowed = CharacterSet.urlPathAllowed
    allowed.remove(charactersIn: "/")
    return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
}

private extension Data {
    mutating func appendUTF8(_ string: String) {
        append(Data(string.utf8))
    }
}
