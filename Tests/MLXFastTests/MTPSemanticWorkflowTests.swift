import Foundation
import Testing

private func mtpSemanticWorkflow() throws -> String {
    try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
}

private func mtpSemanticStepBody(
    _ workflow: String,
    from stepName: String,
    to nextStepName: String
) throws -> String {
    let start = try #require(
        workflow.range(of: stepName),
        "missing step \(stepName)"
    )
    let end = try #require(
        workflow.range(
            of: nextStepName,
            range: start.upperBound..<workflow.endIndex
        ),
        "missing step \(nextStepName)"
    )
    return String(workflow[start.lowerBound..<end.lowerBound])
}

@Test
func mtpSemanticGateIsSecondaryUntimedAndExactFailureDominates() throws {
    let workflow = try mtpSemanticWorkflow()
    #expect(workflow.contains(
        "MLXFAST_OFFICIAL_BENCHMARK_RUN: \"1\""
    ))
    let exact = try #require(
        workflow.range(
            of: "- name: MTP correctness and parity gate (untimed)"
        )
    )
    let exactReap = try #require(
        workflow.range(
            of: "- name: Reap exact MTP worker before semantic material"
        )
    )
    let exactScrub = try #require(
        workflow.range(
            of: "- name: Scrub exact MTP oracle before semantic capture"
        )
    )
    let capture = try #require(
        workflow.range(
            of: "- name: Capture MTP semantic GPQA answers (untimed)"
        )
    )
    let semantic = try #require(
        workflow.range(of: "- name: MTP semantic GPQA gate (untimed)")
    )
    let semanticScrub = try #require(
        workflow.range(of: "- name: Scrub MTP semantic GPQA material")
    )
    let correctnessStage = try #require(
        workflow.range(of: "- name: Stage MTP correctness artifacts")
    )
    let hiddenScrub = try #require(
        workflow.range(of: "- name: Scrub hidden material from bench workspace")
    )
    let timed = try #require(
        workflow.range(
            of: "- name: Timed paired MTP benchmark (measure-mtp-job)"
        )
    )
    #expect(exact.lowerBound < exactReap.lowerBound)
    #expect(exactReap.lowerBound < exactScrub.lowerBound)
    #expect(exactScrub.lowerBound < capture.lowerBound)
    #expect(capture.lowerBound < semantic.lowerBound)
    #expect(semantic.lowerBound < semanticScrub.lowerBound)
    #expect(semanticScrub.lowerBound < correctnessStage.lowerBound)
    #expect(correctnessStage.lowerBound < hiddenScrub.lowerBound)
    #expect(hiddenScrub.lowerBound < timed.lowerBound)

    let exactBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: MTP correctness and parity gate (untimed)",
        to: "- name: Reap exact MTP worker before semantic material"
    )
    #expect(exactBody.contains(".all_tokens_matched == true"))
    #expect(exactBody.contains("exit \"${status}\""))
    #expect(!exactBody.contains("mtp-generate-gpqa-answers"))
    #expect(!exactBody.contains("run-semantic-gpqa-gate.sh"))
    #expect(!exactBody.contains("mtp_semantic_gpqa_reference.json"))

    let exactReapBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: Reap exact MTP worker before semantic material",
        to: "- name: Scrub exact MTP oracle before semantic capture"
    )
    #expect(exactReapBody.contains("reap-bench-processes.sh"))
    #expect(exactReapBody.contains("bench_uid=\"$(id -u bench)\""))
    #expect(exactReapBody.contains("$2 !~ /^Z/"))
    #expect(exactReapBody.contains(
        "refusing semantic material while exact-gate bench processes remain alive"
    ))
    #expect(!exactReapBody.contains("mtp_semantic_gpqa_reference"))
    #expect(!exactReapBody.contains("ANTHROPIC_API_KEY"))

    let exactScrubBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: Scrub exact MTP oracle before semantic capture",
        to: "- name: Capture MTP semantic GPQA answers (untimed)"
    )
    #expect(exactScrubBody.contains(
        ".mtp-ranked-src/mtp_correctness_golden.json"
    ))
    #expect(exactScrubBody.contains("rm -f -- \"${exact_oracle}\""))
    #expect(exactScrubBody.contains(
        "[[ -e \"${exact_oracle}\" || -L \"${exact_oracle}\" ]]"
    ))
    #expect(!exactScrubBody.contains(
        "mtp_semantic_gpqa_reference.json"
    ))

    let captureBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: Capture MTP semantic GPQA answers (untimed)",
        to: "- name: MTP semantic GPQA gate (untimed)"
    )
    #expect(captureBody.contains(
        "exec .build/release/mlxfast-swift mtp-generate-gpqa-answers"
    ))
    #expect(captureBody.contains(
        "--gpqa .mtp-ranked-src/mtp_semantic_gpqa_reference.json"
    ))
    #expect(captureBody.contains(
        "--tokenizer \"${MLXFAST_MTP_TARGET_DIR}\""
    ))
    #expect(captureBody.contains("--require-trained-assistant"))
    #expect(captureBody.contains(
        "BENCH_GOLDEN_PATH: ${{ env.MLXFAST_JOB_WS }}/.mtp-ranked-src/mtp_semantic_gpqa_reference.json"
    ))
    #expect(captureBody.contains(
        "MLXFAST_PRIVATE_DIR=\"${MLXFAST_JOB_WS}/private\""
    ))
    #expect(captureBody.contains(
        "mkdir -m 0700 \"${MLXFAST_JOB_WS}/private\""
    ))
    #expect(captureBody.contains(
        "install -m 0600 \"${semantic_src}\" \"${semantic_dst}\""
    ))
    let exactAbsence = try #require(captureBody.range(
        of: "refusing semantic launch while exact MTP oracle remains staged"
    ))
    let semanticInstall = try #require(captureBody.range(
        of: "install -m 0600 \"${semantic_src}\" \"${semantic_dst}\""
    ))
    #expect(exactAbsence.lowerBound < semanticInstall.lowerBound)
    #expect(captureBody.contains(
        "\"user:bench allow list,search,readattr,readextattr,read,execute,"
    ))
    #expect(captureBody.contains(
        #"/bin/chmod +a "user:bench allow read" "${semantic_dst}""#
    ))
    #expect(captureBody.contains(
        "mtp-generate-gpqa-answers: completed"
    ))
    #expect(captureBody.contains(
        #"/bin/chmod +a "user:${trusted_reader} allow read" "${answers}""#
    ))
    let reapBeforeHandoff = try #require(
        captureBody.range(of: "reap-bench-processes.sh")
    )
    let rejectLiveBench = try #require(
        captureBody.range(
            of: "refusing MTP semantic answer handoff while bench processes remain alive"
        )
    )
    let answerACL = try #require(
        captureBody.range(
            of: #"/bin/chmod +a "user:${trusted_reader} allow read" "${answers}""#
        )
    )
    #expect(reapBeforeHandoff.lowerBound < rejectLiveBench.lowerBound)
    #expect(rejectLiveBench.lowerBound < answerACL.lowerBound)
    #expect(captureBody.contains("bench_uid=\"$(id -u bench)\""))
    #expect(captureBody.contains("$2 !~ /^Z/"))
    #expect(captureBody.contains(
        #"test "$(stat -f '%Lp' "${answers}")" = "600""#
    ))
    #expect(captureBody.contains("test -r \"${answers}\""))
    #expect(!captureBody.contains("ANTHROPIC_API_KEY"))
    #expect(!captureBody.contains(
        "exec .build/release/mlxfast-swift generate-gpqa-answers"
    ))
    #expect(!captureBody.contains("--golden"))
    #expect(!captureBody.contains("expected_token"))

    let semanticBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: MTP semantic GPQA gate (untimed)",
        to: "- name: Scrub MTP semantic GPQA material"
    )
    #expect(semanticBody.contains(
        "MLXFAST_SEMANTIC_GPQA_VERDICT_ONLY: \"1\""
    ))
    #expect(semanticBody.contains(
        "MLXFAST_SEMANTIC_GPQA_REQUIRED: \"1\""
    ))
    #expect(semanticBody.contains(
        "ANTHROPIC_API_KEY: ${{ secrets.ORG_ANTHROPIC_API_KEY }}"
    ))
    #expect(semanticBody.contains(
        ".github/scripts/run-semantic-gpqa-gate.sh"
    ))
    #expect(!semanticBody.contains("MLXFAST_SCORE_PATH"))
    #expect(!semanticBody.contains("score.json"))
}

@Test
func officialMTPBenchmarkDefaultsNoWriteWhileLocalRemainsOptIn() throws {
    let workflow = try mtpSemanticWorkflow()
    let exactBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: MTP correctness and parity gate (untimed)",
        to: "- name: Scrub exact MTP oracle before semantic capture"
    )
    #expect(exactBody.contains("--deny-worker-file-writes"))
    // Exactly two untimed gate invocations carry the no-write flag: the base
    // correctness gate and the extended-legs runner. The timed
    // measure-mtp-job invocation never does.
    #expect(
        workflow.components(
            separatedBy: "--deny-worker-file-writes"
        ).count - 1 == 2
    )

    let timedBody = try mtpSemanticStepBody(
        workflow,
        from: "- name: Timed paired MTP benchmark (measure-mtp-job)",
        to: "- name: Compute MTP score and enforce floor"
    )
    #expect(!timedBody.contains("--deny-worker-file-writes"))
    let localRunner = try String(
        contentsOfFile: "benchmark-mtp.sh",
        encoding: .utf8
    )
    #expect(!localRunner.contains("--deny-worker-file-writes"))

    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let benchmarkStart = try #require(
        cli.range(of: "private static func runExperimentalTrainedMTPBenchmark(")
    )
    let semanticStart = try #require(
        cli.range(
            of: "private static func runExperimentalTrainedMTPSemanticGPQA(",
            range: benchmarkStart.upperBound..<cli.endIndex
        )
    )
    let benchmarkBody = cli[
        benchmarkStart.lowerBound..<semanticStart.lowerBound
    ]
    #expect(benchmarkBody.contains(
        "deniesWorkerFileWrites: options.hasFlag("
    ))
    #expect(benchmarkBody.contains("\"--deny-worker-file-writes\""))
    #expect(benchmarkBody.contains(
        "deniesWorkerFileWritesInOfficialRuns: true"
    ))
    #expect(cli.contains(
        "let effectiveDeniesWorkerFileWrites ="
    ))
    #expect(cli.contains(
        "|| (officialRun && deniesWorkerFileWritesInOfficialRuns)"
    ))
    #expect(cli.contains(
        "deniesFileWrites: effectiveDeniesWorkerFileWrites"
    ))
    #expect(cli.contains(
        "deniesWorkerFileWritesInOfficialRuns: Bool = false"
    ))

    let semanticBody = cli[
        semanticStart.lowerBound..<cli.endIndex
    ]
    #expect(semanticBody.contains("deniesWorkerFileWrites: true"))
}

@Test
func mtpSemanticFixturePinsSandboxCleanupAndArtifactsStayPrivate() throws {
    let workflow = try mtpSemanticWorkflow()
    let stepsMarker = try #require(workflow.range(of: "\n    steps:"))
    let jobHeader = String(workflow[..<stepsMarker.lowerBound])
    #expect(!jobHeader.contains("MLXFAST_MTP_SEMANTIC_GPQA_R2_PATH"))
    #expect(jobHeader.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_REFERENCE_SHA256: "
            + "fc8bcdaff94aa89b2fc2a1a2adc28943ed026899ae805b3c52b3f81a235c20ff"
    ))
    #expect(jobHeader.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_REFERENCE_BYTES: \"9919\""
    ))
    #expect(jobHeader.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_MIN_PASS: \"1\""
    ))
    #expect(jobHeader.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_CASE_COUNT: \"5\""
    ))
    #expect(jobHeader.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_MAX_NEW_TOKENS: \"64\""
    ))
    #expect(!jobHeader.contains("vars.MLXFAST_MTP_SEMANTIC_GPQA"))
    #expect(workflow.contains("NOT an IT/Opus M5 calibration"))

    let enablement = try mtpSemanticStepBody(
        workflow,
        from: "- name: Enforce MTP track enablement",
        to: "- name: MTP host preflight"
    )
    #expect(enablement.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_REFERENCE_SHA256"
    ))
    #expect(enablement.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_REFERENCE_BYTES"
    ))
    #expect(enablement.contains(
        "MTP semantic GPQA non-inferiority threshold"
    ))
    #expect(enablement.contains("^[0-9a-f]{64}$"))

    let download = try mtpSemanticStepBody(
        workflow,
        from: "- name: Prepare hidden MTP goldens",
        to: "- name: Verify trusted harness before MTP correctness gate"
    )
    #expect(download.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_R2_PATH: "
            + "correctness_prompts/gpqa_reference_cases-gemma.json"
    ))
    #expect(download.contains(
        "\"${MLXFAST_PRIVATE_DIR}/mtp_semantic_gpqa_reference.json\""
    ))
    #expect(download.contains(
        "semantic_hash=\"$(shasum -a 256"
    ))
    #expect(download.contains(
        "semantic_bytes=\"$(wc -c"
    ))
    #expect(!download.contains(
        "install -m 0444 "
            + "\"${MLXFAST_PRIVATE_DIR}/mtp_semantic_gpqa_reference.json\""
    ))
    #expect(download.contains(
        ".mtp-ranked-src/mtp_correctness_golden.json"
    ))
    #expect(!download.contains(
        "semantic_dst=\"${MLXFAST_JOB_WS}/.mtp-ranked-src/"
            + "mtp_semantic_gpqa_reference.json\""
    ))

    let semanticScrub = try mtpSemanticStepBody(
        workflow,
        from: "- name: Scrub MTP semantic GPQA material",
        to: "- name: Stage MTP correctness artifacts"
    )
    #expect(semanticScrub.contains("if: ${{ always() }}"))
    #expect(semanticScrub.contains(
        "mtp_semantic_gpqa_reference.json"
    ))
    #expect(semanticScrub.contains(
        ".mtp-ranked-src/mtp_correctness_golden.json"
    ))
    #expect(semanticScrub.contains(
        "mtp_semantic_gpqa_answers.json"
    ))
    #expect(semanticScrub.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_RESULTS_PATH"
    ))
    #expect(semanticScrub.contains("mtp-semantic-capture.log"))

    let broadScrub = try mtpSemanticStepBody(
        workflow,
        from: "- name: Scrub hidden material from bench workspace",
        to: "- name: Reap lingering bench processes before timing"
    )
    #expect(broadScrub.contains(
        ".mtp-ranked-src/mtp_semantic_gpqa_reference.json"
    ))
    #expect(broadScrub.contains(
        "MLXFAST_MTP_SEMANTIC_GPQA_RESULTS_PATH"
    ))
    #expect(broadScrub.contains("mtp-semantic-capture.stdout"))
    #expect(broadScrub.contains("'*semantic*gpqa*'"))
    #expect(broadScrub.contains(
        "if: ${{ success() && inputs.run_benchmark }}"
    ))

    let correctnessArtifacts = try mtpSemanticStepBody(
        workflow,
        from: "- name: Stage MTP correctness artifacts",
        to: "- name: Upload MTP correctness artifacts"
    )
    #expect(correctnessArtifacts.contains(
        "if: ${{ success() && !inputs.run_benchmark }}"
    ))
    let benchmarkArtifacts = try mtpSemanticStepBody(
        workflow,
        from: "- name: Stage MTP benchmark artifacts",
        to: "- name: Upload MTP benchmark artifacts"
    )
    let score = try mtpSemanticStepBody(
        workflow,
        from: "- name: Compute MTP score and enforce floor",
        to: "- name: Check MTP benchmark artifact paths"
    )
    for artifacts in [correctnessArtifacts, benchmarkArtifacts] {
        #expect(!artifacts.contains("semantic_gpqa"))
        #expect(!artifacts.contains("mtp-semantic"))
        #expect(!artifacts.contains("gpqa_reference"))
    }
    #expect(!score.contains("semantic_gpqa"))
    #expect(!score.contains("mtp-semantic"))
    #expect(!score.contains("gpqa_reference"))

    let deny = try String(
        contentsOfFile: ".github/scripts/deny-private-artifacts.sh",
        encoding: .utf8
    )
    #expect(deny.contains("*gpqa_reference*.json"))
    #expect(deny.contains("*semantic_gpqa*.json"))
    #expect(deny.contains("mtp-semantic-*"))
}

@Test
func mtpSemanticFixedPinsAndPolicyFloorAreDocumentedAsActivated() throws {
    for path in [
        "README.md",
        "TASK.md",
        "AGENTS.md",
        "docs/experimental-mtp-track.md",
        "docs/mtp-track-golive-runbook.md",
    ] {
        let source = try String(contentsOfFile: path, encoding: .utf8)
        #expect(source.contains(
            "fc8bcdaff94aa89b2fc2a1a2adc28943ed026899ae805b3c52b3f81a235c20ff"
        ), Comment(rawValue: path))
        #expect(source.contains("9919"), Comment(rawValue: path))
        #expect(
            source.localizedCaseInsensitiveContains("policy floor"),
            Comment(rawValue: path)
        )
        #expect(
            !source.localizedCaseInsensitiveContains("deployment blocker")
                && !source.localizedCaseInsensitiveContains(
                    "not rollout-ready"
                ),
            Comment(rawValue: path)
        )
    }
}

@Test
func semanticJudgeVerdictOnlyModeRequiresThresholdAndNeverPatchesScore()
    throws
{
    let fileManager = FileManager.default
    let repositoryRoot = fileManager.currentDirectoryPath
    let script = URL(fileURLWithPath: repositoryRoot)
        .appendingPathComponent(
            ".github/scripts/run-semantic-gpqa-gate.sh"
        ).path
    let temporaryRoot = fileManager.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-verdict-only-\(UUID().uuidString)"
    )
    try fileManager.createDirectory(
        at: temporaryRoot,
        withIntermediateDirectories: true
    )
    let root = URL(
        fileURLWithPath: temporaryRoot.path.hasPrefix("/var/")
            ? "/private" + temporaryRoot.path
            : temporaryRoot.path
    )
    let privateRoot = root.appendingPathComponent("private")
    let shimRoot = root.appendingPathComponent("bin")
    try fileManager.createDirectory(
        at: privateRoot,
        withIntermediateDirectories: true
    )
    try fileManager.createDirectory(
        at: shimRoot,
        withIntermediateDirectories: true
    )
    defer { try? fileManager.removeItem(at: root) }

    let answers = root.appendingPathComponent("answers.json")
    try Data(
        """
        {
          "version": 1,
          "cases": [{
            "id": "private-case",
            "prompt": "private question",
            "answer_key": "A",
            "reference_answer": "A. reference",
            "candidate_answer": "A",
            "candidate_tokens": [1, 2],
            "max_new_tokens": 2
          }]
        }
        """.utf8
    ).write(to: answers)
    try fileManager.setAttributes(
        [.posixPermissions: 0o600],
        ofItemAtPath: answers.path
    )
    let truncatedAnswers = root.appendingPathComponent(
        "truncated-answers.json"
    )
    try Data(
        """
        {
          "version": 1,
          "cases": [{
            "id": "truncated",
            "prompt": "private question",
            "answer_key": "A",
            "reference_answer": "A. reference",
            "candidate_answer": "A",
            "candidate_tokens": [1],
            "max_new_tokens": 2
          }]
        }
        """.utf8
    ).write(to: truncatedAnswers)
    let contradictoryAnswers = root.appendingPathComponent(
        "contradictory-answers.json"
    )
    try Data(
        """
        {
          "version": 1,
          "cases": [{
            "id": "contradictory",
            "prompt": "contradictory verdict question",
            "answer_key": "A",
            "reference_answer": "A. reference",
            "candidate_answer": "A",
            "candidate_tokens": [1, 2],
            "max_new_tokens": 2
          }]
        }
        """.utf8
    ).write(to: contradictoryAnswers)
    func makeStrictResponseFixture(
        name: String,
        prompt: String
    ) throws -> URL {
        let url = root.appendingPathComponent("\(name)-answers.json")
        let object: [String: Any] = [
            "version": 1,
            "cases": [[
                "id": name,
                "prompt": prompt,
                "answer_key": "A",
                "reference_answer": "A. reference",
                "candidate_answer": "A",
                "candidate_tokens": [1, 2],
                "max_new_tokens": 2,
            ]],
        ]
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return url
    }
    let promptInjectionAnswers = try makeStrictResponseFixture(
        name: "prompt-injection",
        prompt: "prompt injection echo"
    )
    let truncatedStopAnswers = try makeStrictResponseFixture(
        name: "truncated-stop",
        prompt: "truncated stop"
    )
    let fencedAnswers = try makeStrictResponseFixture(
        name: "fenced",
        prompt: "fenced verdict"
    )
    let multipleTextAnswers = try makeStrictResponseFixture(
        name: "multiple-text",
        prompt: "multiple text blocks"
    )
    let trailingJSONAnswers = try makeStrictResponseFixture(
        name: "trailing-json",
        prompt: "valid first trailing JSON"
    )
    let malformedTrailingAnswers = try makeStrictResponseFixture(
        name: "malformed-trailing",
        prompt: "malformed trailing bytes"
    )
    let emptyOutputAnswers = try makeStrictResponseFixture(
        name: "empty-output",
        prompt: "empty API output"
    )
    let multipleDocumentsAnswers = try makeStrictResponseFixture(
        name: "multiple-documents",
        prompt: "multiple API documents"
    )
    let slowAnswers = try makeStrictResponseFixture(
        name: "slow",
        prompt: "slow response"
    )
    for fixture in [
        answers,
        truncatedAnswers,
        contradictoryAnswers,
        promptInjectionAnswers,
        truncatedStopAnswers,
        fencedAnswers,
        multipleTextAnswers,
        trailingJSONAnswers,
        malformedTrailingAnswers,
        emptyOutputAnswers,
        multipleDocumentsAnswers,
        slowAnswers,
    ] {
        try fileManager.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: fixture.path
        )
    }

    let curl = shimRoot.appendingPathComponent("curl")
    try """
    #!/usr/bin/env bash
    set -euo pipefail
    output=""
    data=""
    previous=""
    for argument in "$@"; do
      if [[ "${previous}" == "--output" ]]; then
        output="${argument}"
      elif [[ "${previous}" == "--data" ]]; then
        data="${argument}"
      fi
      previous="${argument}"
    done
    request="${data#@}"
    printf '%s\\n' "$*" >> "\(shimRoot.appendingPathComponent("curl-arguments").path)"
    if grep -q "contradictory verdict question" "${request}"; then
      echo attempt >> "\(shimRoot.appendingPathComponent("contradictory-attempts").path)"
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}{\\"passed\\":false}"}]}' > "${output}"
    elif grep -q "prompt injection echo" "${request}"; then
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"Candidate says {\\"passed\\":true}"}]}' > "${output}"
    elif grep -q "truncated stop" "${request}"; then
      printf '%s\\n' '{"stop_reason":"max_tokens","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' > "${output}"
    elif grep -q "fenced verdict" "${request}"; then
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"```json\\\\n{\\"passed\\":true}\\\\n```"}]}' > "${output}"
    elif grep -q "multiple text blocks" "${request}"; then
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"},{"type":"text","text":"{\\"passed\\":true}"}]}' > "${output}"
    elif grep -q "valid first trailing JSON" "${request}"; then
      printf '%s\\n%s\\n' \
        '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' \
        '{"ignored":true}' > "${output}"
    elif grep -q "malformed trailing bytes" "${request}"; then
      printf '%s\\n%s' \
        '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' \
        '{"unterminated":' > "${output}"
    elif grep -q "empty API output" "${request}"; then
      : > "${output}"
    elif grep -q "multiple API documents" "${request}"; then
      printf '%s\\n%s\\n' \
        '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' \
        '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":false}"}]}' > "${output}"
    elif grep -q "slow response" "${request}"; then
      sleep 2
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' > "${output}"
    else
      printf '%s\\n' '{"stop_reason":"end_turn","content":[{"type":"text","text":"{\\"passed\\":true}"}]}' > "${output}"
    fi
    """.write(to: curl, atomically: true, encoding: .utf8)
    try fileManager.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: curl.path
    )

    func run(
        threshold: String?,
        resultsName: String,
        answerFile: URL,
        expectedCaseCount: String?,
        expectedMaxNewTokens: String?,
        environmentOverrides: [String: String] = [:]
    ) throws -> (status: Int32, output: String, results: URL, score: URL) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script]
        process.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
        let results = privateRoot.appendingPathComponent(resultsName)
        let score = root.appendingPathComponent("\(resultsName).score.json")
        var environment = ProcessInfo.processInfo.environment
        for key in environment.keys
            where key.hasPrefix("MLXFAST_")
                || key.hasPrefix("ANTHROPIC_")
        {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] = shimRoot.path + ":"
            + (ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin")
        environment["ANTHROPIC_API_KEY"] = "test-key"
        environment["MLXFAST_PRIVATE_DIR"] = privateRoot.path
        environment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] = answerFile.path
        environment["MLXFAST_SEMANTIC_GPQA_RESULTS_PATH"] = results.path
        environment["MLXFAST_SCORE_PATH"] = score.path
        environment["MLXFAST_SEMANTIC_GPQA_REQUIRED"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_VERDICT_ONLY"] = "1"
        if let expectedCaseCount {
            environment[
                "MLXFAST_SEMANTIC_GPQA_EXPECTED_CASE_COUNT"
            ] = expectedCaseCount
        }
        if let expectedMaxNewTokens {
            environment[
                "MLXFAST_SEMANTIC_GPQA_EXPECTED_MAX_NEW_TOKENS"
            ] = expectedMaxNewTokens
        }
        if let threshold {
            environment["MLXFAST_SEMANTIC_GPQA_MIN_PASS"] = threshold
        }
        for (key, value) in environmentOverrides {
            environment[key] = value
        }
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(decoding: data, as: UTF8.self),
            results,
            score
        )
    }

    let passed = try run(
        threshold: "1",
        resultsName: "passed.json",
        answerFile: answers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2",
        environmentOverrides: [
            "MLXFAST_SEMANTIC_GPQA_MAX_ANSWER_BYTES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_MAX_JUDGE_REQUEST_BYTES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_MAX_JUDGE_RESPONSE_BYTES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_MAX_RESULTS_BYTES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_MAX_CASES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_CONNECT_TIMEOUT_SECONDS": "999999999",
            "MLXFAST_SEMANTIC_GPQA_CURL_TIMEOUT_SECONDS": "999999999",
            "MLXFAST_SEMANTIC_GPQA_CURL_RETRY_MAX_TIME_SECONDS": "999999999",
            "MLXFAST_SEMANTIC_GPQA_CURL_RETRIES": "999999999",
            "MLXFAST_SEMANTIC_GPQA_JUDGE_ATTEMPTS": "999999999",
            "MLXFAST_SEMANTIC_GPQA_GATE_DEADLINE_SECONDS": "999999999",
        ]
    )
    #expect(passed.status == 0)
    #expect(passed.output.contains("semantic-gpqa: verdict passed"))
    #expect(!passed.output.contains("case 1/1"))
    #expect(!passed.output.contains("pass_count="))
    #expect(fileManager.fileExists(atPath: passed.results.path))
    #expect(!fileManager.fileExists(atPath: passed.score.path))
    let permissions = try #require(
        try fileManager.attributesOfItem(atPath: passed.results.path)[
            .posixPermissions
        ] as? NSNumber
    ).intValue
    #expect(permissions & 0o777 == 0o600)
    let curlArguments = try String(
        contentsOf: shimRoot.appendingPathComponent("curl-arguments"),
        encoding: .utf8
    )
    #expect(curlArguments.contains("--connect-timeout 30"))
    #expect(curlArguments.contains("--retry 3"))
    #expect(curlArguments.contains("--retry-max-time 600"))
    #expect(curlArguments.contains("--max-time 600"))

    let missing = try run(
        threshold: nil,
        resultsName: "missing-threshold.json",
        answerFile: answers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2"
    )
    #expect(missing.status != 0)
    #expect(missing.output.contains(
        "requires an explicitly configured non-inferiority threshold"
    ))
    #expect(!fileManager.fileExists(atPath: missing.score.path))

    let zero = try run(
        threshold: "0",
        resultsName: "zero-threshold.json",
        answerFile: answers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2"
    )
    #expect(zero.status != 0)
    #expect(zero.output.contains(
        "requires a positive non-inferiority threshold"
    ))
    #expect(!fileManager.fileExists(atPath: zero.score.path))

    let missingCaseCount = try run(
        threshold: "1",
        resultsName: "missing-case-count.json",
        answerFile: answers,
        expectedCaseCount: nil,
        expectedMaxNewTokens: "2"
    )
    #expect(missingCaseCount.status != 0)
    #expect(missingCaseCount.output.contains(
        "requires MLXFAST_SEMANTIC_GPQA_EXPECTED_CASE_COUNT"
    ))

    let missingMaxNewTokens = try run(
        threshold: "1",
        resultsName: "missing-max-new-tokens.json",
        answerFile: answers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: nil
    )
    #expect(missingMaxNewTokens.status != 0)
    #expect(missingMaxNewTokens.output.contains(
        "requires MLXFAST_SEMANTIC_GPQA_EXPECTED_MAX_NEW_TOKENS"
    ))

    let truncated = try run(
        threshold: "1",
        resultsName: "truncated.json",
        answerFile: truncatedAnswers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2"
    )
    #expect(truncated.status != 0)
    #expect(truncated.output.contains(
        "semantic GPQA answer document failed strict schema validation"
    ))
    #expect(!fileManager.fileExists(atPath: truncated.score.path))

    let contradictory = try run(
        threshold: "1",
        resultsName: "contradictory.json",
        answerFile: contradictoryAnswers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2"
    )
    #expect(contradictory.status != 0)
    #expect(!contradictory.output.contains("verdict passed"))
    #expect(!fileManager.fileExists(atPath: contradictory.score.path))
    let contradictoryResults = try String(
        contentsOf: contradictory.results,
        encoding: .utf8
    )
    #expect(contradictoryResults.contains("invalid_judge_response"))
    let contradictoryAttempts = try String(
        contentsOf: shimRoot.appendingPathComponent(
            "contradictory-attempts"
        ),
        encoding: .utf8
    )
    #expect(
        contradictoryAttempts.split(separator: "\n").count == 3
    )

    for (name, fixture) in [
        ("prompt-injection", promptInjectionAnswers),
        ("truncated-stop", truncatedStopAnswers),
        ("fenced", fencedAnswers),
        ("multiple-text", multipleTextAnswers),
        ("trailing-json", trailingJSONAnswers),
        ("malformed-trailing", malformedTrailingAnswers),
        ("multiple-documents", multipleDocumentsAnswers),
    ] {
        let invalid = try run(
            threshold: "1",
            resultsName: "\(name).json",
            answerFile: fixture,
            expectedCaseCount: "1",
            expectedMaxNewTokens: "2"
        )
        #expect(invalid.status != 0, Comment(rawValue: name))
        #expect(!invalid.output.contains("verdict passed"))
        #expect(!fileManager.fileExists(atPath: invalid.score.path))
        let result = try String(
            contentsOf: invalid.results,
            encoding: .utf8
        )
        #expect(
            result.contains("invalid_judge_response"),
            Comment(rawValue: name)
        )
    }

    let emptyOutput = try run(
        threshold: "1",
        resultsName: "empty-output.json",
        answerFile: emptyOutputAnswers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2"
    )
    #expect(emptyOutput.status != 0)
    #expect(emptyOutput.output.contains(
        "semantic GPQA judge response is empty"
    ))
    #expect(!fileManager.fileExists(atPath: emptyOutput.results.path))

    let deadline = try run(
        threshold: "1",
        resultsName: "deadline.json",
        answerFile: slowAnswers,
        expectedCaseCount: "1",
        expectedMaxNewTokens: "2",
        environmentOverrides: [
            "MLXFAST_SEMANTIC_GPQA_GATE_DEADLINE_SECONDS": "1",
            "MLXFAST_SEMANTIC_GPQA_JUDGE_ATTEMPTS": "1",
        ]
    )
    #expect(deadline.status != 0)
    #expect(deadline.output.contains("hard overall deadline"))
    #expect(!fileManager.fileExists(atPath: deadline.results.path))
    let privateChildren = try fileManager.contentsOfDirectory(
        atPath: privateRoot.path
    )
    #expect(!privateChildren.contains { $0.hasPrefix("semantic-gpqa.") })
}

@Test
func semanticJudgeRejectsTraversalSymlinkAncestorsAndOutsideResultsBeforeCurl()
    throws
{
    let fileManager = FileManager.default
    let repositoryRoot = fileManager.currentDirectoryPath
    let script = URL(fileURLWithPath: repositoryRoot)
        .appendingPathComponent(
            ".github/scripts/run-semantic-gpqa-gate.sh"
        ).path
    let temporaryRoot = fileManager.temporaryDirectory.appendingPathComponent(
        "mtp-semantic-paths-\(UUID().uuidString)"
    )
    try fileManager.createDirectory(
        at: temporaryRoot,
        withIntermediateDirectories: true
    )
    let root = URL(
        fileURLWithPath: temporaryRoot.path.hasPrefix("/var/")
            ? "/private" + temporaryRoot.path
            : temporaryRoot.path
    )
    defer { try? fileManager.removeItem(at: root) }
    let privateRoot = root.appendingPathComponent("private")
    let outsideRoot = root.appendingPathComponent("outside")
    let shimRoot = root.appendingPathComponent("bin")
    for directory in [privateRoot, outsideRoot, shimRoot] {
        try fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
    }
    let answers = root.appendingPathComponent("answers.json")
    try Data(
        """
        {"version":1,"cases":[{
          "id":"case",
          "prompt":"question",
          "answer_key":"A",
          "reference_answer":"A",
          "candidate_answer":"A",
          "candidate_tokens":[1],
          "max_new_tokens":1
        }]}
        """.utf8
    ).write(to: answers)
    try fileManager.setAttributes(
        [.posixPermissions: 0o600],
        ofItemAtPath: answers.path
    )
    let curlMarker = root.appendingPathComponent("curl-ran")
    let curl = shimRoot.appendingPathComponent("curl")
    try """
    #!/bin/sh
    : > "\(curlMarker.path)"
    exit 99
    """.write(to: curl, atomically: true, encoding: .utf8)
    try fileManager.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: curl.path
    )

    func run(
        privateDirectory: String,
        resultPath: String
    ) throws -> (Int32, String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script]
        process.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
        var environment = ProcessInfo.processInfo.environment
        for key in environment.keys
            where key.hasPrefix("MLXFAST_")
                || key.hasPrefix("ANTHROPIC_")
        {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] = shimRoot.path + ":/usr/bin:/bin"
        environment["ANTHROPIC_API_KEY"] = "test-key"
        environment["MLXFAST_PRIVATE_DIR"] = privateDirectory
        environment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] = answers.path
        environment["MLXFAST_SEMANTIC_GPQA_RESULTS_PATH"] = resultPath
        environment["MLXFAST_SEMANTIC_GPQA_MIN_PASS"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_REQUIRED"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_VERDICT_ONLY"] = "1"
        environment["MLXFAST_SEMANTIC_GPQA_EXPECTED_CASE_COUNT"] = "1"
        environment[
            "MLXFAST_SEMANTIC_GPQA_EXPECTED_MAX_NEW_TOKENS"
        ] = "1"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(decoding: output, as: UTF8.self)
        )
    }

    let traversal = try run(
        privateDirectory: privateRoot.path,
        resultPath: privateRoot.path + "/nested/../result.json"
    )
    #expect(traversal.0 != 0)
    #expect(traversal.1.contains("must not contain '..'"))

    let outside = try run(
        privateDirectory: privateRoot.path,
        resultPath: outsideRoot.appendingPathComponent("result.json").path
    )
    #expect(outside.0 != 0)
    #expect(outside.1.contains("strict real descendant"))

    let linkedParent = privateRoot.appendingPathComponent("linked")
    try fileManager.createSymbolicLink(
        at: linkedParent,
        withDestinationURL: outsideRoot
    )
    let symlinkAncestor = try run(
        privateDirectory: privateRoot.path,
        resultPath: linkedParent.appendingPathComponent("result.json").path
    )
    #expect(symlinkAncestor.0 != 0)
    #expect(symlinkAncestor.1.contains("symlink ancestor"))

    let linkedRoot = root.appendingPathComponent("private-link")
    try fileManager.createSymbolicLink(
        at: linkedRoot,
        withDestinationURL: privateRoot
    )
    let symlinkRoot = try run(
        privateDirectory: linkedRoot.path,
        resultPath: linkedRoot.appendingPathComponent("result.json").path
    )
    #expect(symlinkRoot.0 != 0)
    #expect(symlinkRoot.1.contains("symlink ancestor"))
    #expect(!fileManager.fileExists(atPath: curlMarker.path))
}
