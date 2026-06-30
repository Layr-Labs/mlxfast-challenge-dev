import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import Tokenizers

// DeepSeekRuntime is split across DeepSeekRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension DeepSeekRuntime {
    public static func runWorker(weightsPath: String) throws {
        let config = try DeepSeekConfig.load(from: weightsPath)
        let loader = try DeepSeekWeightLoader(
            weightsPath: weightsPath,
            expertStreamingConfig: ExpertStreamingConfig.fromEnvironment(recordsMetricsDefault: true)
        )
        let weightCache = DeepSeekRuntimeWeightCache(loader: loader, config: config)
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        // Keep protocol I/O off fd 0/1 so submitted model code cannot read
        // future request nonces or spoof JSON responses with normal stdio.
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))
        let session = InProcessInferenceSession(weightCache: weightCache)

        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                do {
                    response = try handleWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        session: session
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(id: -1, nonce: sessionNonce, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            try protocolIO.writeLine(data)
        }
    }

    static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        session: InProcessInferenceSession
    ) throws -> RuntimeWorkerResponse {
        let result: RuntimeWorkerResponse
        switch request.kind {
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            result = try session.generateCorrectness(promptTokens: promptTokens, steps: steps)

        case "correctness_begin":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing prompt_tokens")
            }
            result = try session.beginTeacherForcedCorrectness(promptTokens: promptTokens)

        case "correctness_teacher_forced_batch":
            guard let promptTokens = request.promptTokens,
                  let expectedTokens = request.expectedTokens,
                  let steps = request.steps
            else {
                throw MLXFastError.invalidInput(
                    "runtime worker batched teacher-forced request missing prompt_tokens, expected_tokens, or steps"
                )
            }
            result = try session.teacherForcedCorrectnessBatch(
                promptTokens: promptTokens,
                expectedTokens: expectedTokens,
                steps: steps
            )

        case "correctness_step":
            guard let previousToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing token")
            }
            result = try session.teacherForcedCorrectnessStep(previousToken: previousToken)

        case "prefill":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker prefill request missing prompt_tokens")
            }
            result = try session.prefill(promptTokens: promptTokens)

        case "decode_begin":
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode_begin request missing seed_tokens")
            }
            result = try session.beginDecode(seedTokens: seedTokens)

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            result = try session.decodeStep(inputToken: inputToken)

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
        return result.withEnvelope(id: request.id, nonce: sessionNonce)
    }

}

struct RuntimeWorkerRequest: Codable {
    let id: Int
    let kind: String
    let promptTokens: [Int]?
    let expectedTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let steps: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case expectedTokens = "expected_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
    }
}

struct RuntimeWorkerResponse: Codable {
    let id: Int
    let nonce: String?
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let topLogitRows: [[CorrectnessTraceLogit]]?
    let seedToken: Int?
    let tokens: [Int]?
    let seconds: Double?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?

    init(
        id: Int,
        nonce: String? = nil,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        topLogitRows: [[CorrectnessTraceLogit]]? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        seconds: Double? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil
    ) {
        self.id = id
        self.nonce = nonce
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.topLogitRows = topLogitRows
        self.seedToken = seedToken
        self.tokens = tokens
        self.seconds = seconds
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
    }

    enum CodingKeys: String, CodingKey {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case topLogitRows = "top_logit_rows"
        case seedToken = "seed_token"
        case tokens
        case seconds
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
    }
}

extension RuntimeWorkerResponse {
    func withEnvelope(id: Int, nonce: String) -> RuntimeWorkerResponse {
        RuntimeWorkerResponse(
            id: id,
            nonce: nonce,
            ok: ok,
            error: error,
            token: token,
            topLogits: topLogits,
            topLogitRows: topLogitRows,
            seedToken: seedToken,
            tokens: tokens,
            seconds: seconds,
            expertStats: expertStats,
            peakRamGB: peakRamGB
        )
    }
}

final class RuntimeWorkerProtocolIO {
    private let input: FileHandle
    private let output: FileHandle

    private init(inputDescriptor: Int32, outputDescriptor: Int32) {
        self.input = FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        self.output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
    }

    static func isolatingStandardIO() throws -> RuntimeWorkerProtocolIO {
        let inputFD = try duplicatePrivateDescriptor(STDIN_FILENO, label: "stdin")
        let outputFD = try duplicatePrivateDescriptor(STDOUT_FILENO, label: "stdout")
        do {
            try redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY, label: "stdin")
            try redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY, label: "stdout")
        } catch {
            close(inputFD)
            close(outputFD)
            throw error
        }
        return RuntimeWorkerProtocolIO(inputDescriptor: inputFD, outputDescriptor: outputFD)
    }

    func readLine() throws -> String? {
        var data = Data()
        while true {
            let byte = input.readData(ofLength: 1)
            if byte.isEmpty {
                if data.isEmpty {
                    return nil
                }
                break
            }
            if byte[byte.startIndex] == 0x0a {
                break
            }
            data.append(byte)
        }
        guard let line = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput("runtime worker received non-UTF8 protocol input")
        }
        return line
    }

    func writeLine(_ data: Data) throws {
        try output.write(contentsOf: data)
        try output.write(contentsOf: Data([0x0a]))
    }
}

func duplicatePrivateDescriptor(_ descriptor: Int32, label: String) throws -> Int32 {
    let lowerBound = Int32(64 + Int(arc4random_uniform(449)))
    let duplicatedFD = fcntl(descriptor, F_DUPFD_CLOEXEC, lowerBound)
    guard duplicatedFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to duplicate \(label) for protocol I/O")
    }
    return duplicatedFD
}

func redirectDescriptorToDevNull(_ descriptor: Int32, flags: Int32, label: String) throws {
    let devNullFD = open("/dev/null", flags)
    guard devNullFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to open /dev/null for \(label) redirection")
    }
    defer {
        close(devNullFD)
    }
    guard dup2(devNullFD, descriptor) >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to redirect \(label) away from protocol I/O")
    }
}

final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let errorOutput: FileHandle
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var sessionNonce = ""
    private var nextID = 1
    private var closed = false

    init(options: RuntimeWorkerOptions, weightsPath: String) throws {
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = [
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        }
        var environment = ProcessInfo.processInfo.environment
        environment["MLXFAST_USE_RUNTIME_WORKER"] = "0"
        for key in [
            "ANTHROPIC_API_KEY",
            "MLXFAST_CORRECTNESS_GOLDEN_PATH",
            "MLXFAST_CORRECTNESS_GOLDEN_URL",
            "MLXFAST_CORRECTNESS_GOLDEN_AUTH_HEADER",
            "MLXFAST_GPQA_REFERENCE_PATH",
            "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH",
            "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH",
            "MLXFAST_SEMANTIC_GPQA_MODEL",
            "MLXFAST_PRIVATE_DIR",
            "MLXFAST_PRIVATE_PROMPTS_R2_PRESENT",
            "MLXFAST_ANTHROPIC_PRESENT",
            "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE",
            "R2_ACCESS_KEY_ID",
            "R2_BUCKET_ENDPOINT",
            "R2_SECRET_ACCESS_KEY",
        ] {
            environment.removeValue(forKey: key)
        }
        process.environment = environment
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        self.process = process
        self.input = stdin.fileHandleForWriting
        self.output = stdout.fileHandleForReading
        self.errorOutput = stderr.fileHandleForReading
        let hello = try readResponseLine(validateNonce: false)
        guard hello.id == 0, hello.ok, let nonce = hello.nonce, !nonce.isEmpty else {
            throw MLXFastError.invalidInput("runtime worker did not return a valid protocol hello")
        }
        self.sessionNonce = nonce
    }

    deinit {
        close()
    }

    func close() {
        guard !closed else {
            return
        }
        closed = true
        try? input.close()
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness",
            promptTokens: promptTokens,
            steps: steps
        )
    }

    func beginTeacherForcedCorrectness(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_begin",
            promptTokens: promptTokens
        )
    }

    func teacherForcedCorrectnessBatch(
        promptTokens: [Int],
        expectedTokens: [Int],
        steps: Int
    ) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_teacher_forced_batch",
            promptTokens: promptTokens,
            expectedTokens: expectedTokens,
            steps: steps
        )
    }

    func teacherForcedCorrectnessStep(previousToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_step",
            token: previousToken
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "prefill",
            promptTokens: promptTokens
        )
    }

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_begin",
            seedTokens: seedTokens
        )
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_step",
            token: inputToken
        )
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        expectedTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        guard process.isRunning else {
            throw MLXFastError.invalidInput("runtime worker exited before request \(kind): \(workerExitDiagnostic())")
        }
        let id = nextID
        nextID += 1
        let request = RuntimeWorkerRequest(
            id: id,
            kind: kind,
            promptTokens: promptTokens,
            expectedTokens: expectedTokens,
            token: token,
            seedTokens: seedTokens,
            steps: steps
        )
        var data = try encoder.encode(request)
        data.append(0x0a)
        try input.write(contentsOf: data)

        let response = try readResponseLine(validateNonce: true)
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine(validateNonce: Bool) throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            let response = try decoder.decode(RuntimeWorkerResponse.self, from: data)
            if validateNonce, response.nonce != sessionNonce {
                throw MLXFastError.invalidInput("runtime worker returned a response with an invalid nonce")
            }
            return response
        }
    }

    private func readWorkerOutputLine() throws -> Data {
        var data = Data()
        while true {
            let byte = output.readData(ofLength: 1)
            if byte.isEmpty {
                throw MLXFastError.invalidInput(
                    "runtime worker closed stdout before returning a response: \(workerExitDiagnostic())"
                )
            }
            if byte[byte.startIndex] == 0x0a {
                return data
            }
            data.append(byte)
        }
    }

    private func workerExitDiagnostic() -> String {
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        let stderr = String(data: errorOutput.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = sanitizeWorkerDiagnostic(trimmed)
        if redacted.isEmpty {
            return "exit_status=\(process.terminationStatus)"
        }
        return "exit_status=\(process.terminationStatus) stderr=\(redacted)"
    }

    private func sanitizeWorkerDiagnostic(_ value: String) -> String {
        let singleLine = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        if singleLine.range(of: "expected", options: .caseInsensitive) != nil
            || singleLine.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return singleLine
    }
}

func generateRuntimeWorkerNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeMutableBytes { buffer in
        if let baseAddress = buffer.baseAddress {
            arc4random_buf(baseAddress, buffer.count)
        }
    }
    return bytes
        .map { String(format: "%02x", $0) }
        .joined()
}

func runtimeWorkerLineLooksLikeJSONResponse(_ data: Data) -> Bool {
    for byte in data where byte != 0x20 && byte != 0x09 && byte != 0x0d {
        return byte == 0x7b
    }
    return false
}
