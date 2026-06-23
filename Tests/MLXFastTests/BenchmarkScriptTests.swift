import Foundation
import Testing

@Test
func setupScriptDefaultsToFastReferenceMirror() throws {
    let setup = try String(
        contentsOfFile: "setup.sh",
        encoding: .utf8
    )

    #expect(setup.contains("DEFAULT_REFERENCE_BASE_URL=\"https://ds4.darkbloom.ai/deepseek-v4-flash-4bit\""))
    #expect(setup.contains("REFERENCE_BASE_URL=\"${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}\""))
    #expect(setup.contains("DEFAULT_HF_HOME=\"${MLXFAST_HF_HOME:-${HF_HOME:-${PWD}/.cache/huggingface}}\""))
    #expect(setup.contains("REFERENCE_CACHE_DIR=\"${MLXFAST_REFERENCE_CACHE_DIR:-${DEFAULT_HF_HUB_CACHE}/${REFERENCE_CACHE_REPO_DIR}/snapshots/${REFERENCE_CACHE_REVISION_DIR}}\""))
    #expect(setup.contains("REFERENCE_CACHE_LOCK_PATH=\"${MLXFAST_REFERENCE_CACHE_LOCK_PATH:-${REFERENCE_DIR}/.mlxfast-reference-cache.lock}\""))
    #expect(setup.contains("REFERENCE_POST_DOWNLOAD_FULL_VERIFY=\"${MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY:-1}\""))
    #expect(setup.contains("Usage: ./setup.sh"))
    #expect(setup.contains("reference_file_is_current"))
    #expect(setup.contains("reference_cache_lock_is_current"))
    #expect(setup.contains("reference_post_download_full_verify_enabled"))
    #expect(setup.contains("verify_reference_weights_after_verified_download"))
    #expect(setup.contains("cannot skip post-download full verification unless MLXFAST_REFERENCE_HASH_VERIFY=1"))
    #expect(setup.contains("write_reference_cache_lock"))
    #expect(setup.contains("redownloading ${label} from scratch after hash verification failed"))
    #expect(setup.contains("If you only installed the Command Line Tools and this still fails, install full"))
    #expect(setup.contains("reference cache path ${reference_dir}"))
    #expect(setup.contains("compatibility reference path exists and is not a symlink"))
    #expect(setup.contains("if ! verify_reference_manifest \"${reference_dir}\"; then"))
    #expect(setup.contains("downloaded ${total}/${total} safetensors shard(s)"))
    #expect(setup.contains("setup.sh: setup complete elapsed="))
    #expect(setup.contains("MLXFAST_OFFLINE_WRITABLE_PATHS=\"${PWD}/weights\" .github/scripts/run-offline.sh ${SWIFT_BIN} transform --reference \"${REFERENCE_DIR}\" --output weights"))
    #expect(setup.contains("${SWIFT_BIN} correctness --weights weights"))
}

@Test
func benchmarkWorkflowRunsTransformOfflineAfterSetup() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
    let setupRange = try #require(workflow.range(of: "- name: Setup Swift harness and reference checkpoint"))
    let transformRange = try #require(workflow.range(of: "- name: Transform reference checkpoint"))

    #expect(setupRange.lowerBound < transformRange.lowerBound)
    #expect(workflow.contains("MLXFAST_REFERENCE_DIR: .cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main"))
    #expect(workflow.contains("MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY: \"0\""))
    #expect(workflow.contains("MLXFAST_OFFLINE_WRITABLE_PATHS=\"${PWD}/weights\""))
    #expect(workflow.contains(".github/scripts/run-offline.sh .build/release/mlxfast-swift transform"))
    #expect(workflow.contains("--reference \"${MLXFAST_REFERENCE_DIR}\""))
    #expect(workflow.contains("--output weights"))
}

@Test
func ciSubmissionSmokeTestUsesIsolatedWorkspace() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/ci.yml",
        encoding: .utf8
    )

    #expect(workflow.contains("WORKSPACE=\"${RUNNER_TEMP}/mlxfast-submit-smoke\""))
    #expect(workflow.contains("git -C \"${WORKSPACE}\" -c user.name=\"CI\" -c user.email=\"ci@example.test\" commit -m \"base\""))
    #expect(workflow.contains("\"${BIN}\" submit --contract \"${WORKSPACE}/benchmark.json\" --base-ref HEAD --output /tmp/mlxfast-submission.zip"))
    #expect(!workflow.contains(".build/release/mlxfast-swift submit --output /tmp/mlxfast-submission.zip"))
}

@Test
func referenceCacheProbeWorkflowIsManualAndExperimental() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/reference-cache-probe.yml",
        encoding: .utf8
    )
    let ci = try String(
        contentsOfFile: ".github/workflows/ci.yml",
        encoding: .utf8
    )

    #expect(workflow.contains("name: reference-cache-probe"))
    #expect(workflow.contains("workflow_dispatch:"))
    #expect(!workflow.contains("pull_request:"))
    #expect(!workflow.contains("push:"))
    #expect(workflow.contains("cache_scope:"))
    #expect(workflow.contains("actions/cache/restore@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(workflow.contains("actions/cache/save@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(workflow.contains(".github/scripts/download-reference-cache-scope.sh \"${CACHE_SCOPE}\""))
    #expect(workflow.contains("MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY: \"0\""))
    #expect(ci.contains("bash -n .github/scripts/download-reference-cache-scope.sh"))
    #expect(ci.contains("bash -n .github/scripts/download-r2-object.sh"))
}

@Test
func benchmarkWorkflowCanBeDispatchedFromCurrentRefForTesting() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
    let guardScript = try String(
        contentsOfFile: ".github/scripts/enforce-trusted-benchmark-workflow.sh",
        encoding: .utf8
    )

    #expect(workflow.contains("MLXFAST_TRUSTED_BENCHMARK_REF: ${{ github.ref }}"))
    #expect(!workflow.contains("MLXFAST_TRUSTED_BENCHMARK_REF: refs/heads/main"))
    #expect(guardScript.contains("TRUSTED_REF=\"${MLXFAST_TRUSTED_BENCHMARK_REF:-${GITHUB_REF}}\""))
}

@Test
func benchmarkWorkflowUsesDispatchParseablePrivatePaths() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
    let validator = try String(
        contentsOfFile: ".github/scripts/validate-benchmark-artifacts.sh",
        encoding: .utf8
    )

    #expect(!workflow.contains("${{ runner.temp }}"))
    #expect(workflow.contains("MLXFAST_PRIVATE_DIR: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}"))
    #expect(workflow.contains("MLXFAST_CORRECTNESS_GOLDEN_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/correctness_golden.json"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_PROMPT_PATH: correctness_prompts/public_longcopy_gate_english_512.txt"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH: correctness_prompts/public_longcopy_gate_english_512_256.json"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256: 2a747bf797e16d58f5ffedacc0d4bf5ce0d14be00f2421dc04289a2154cb011d"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_BYTES: \"10320\""))
    #expect(workflow.contains("MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json"))
    #expect(workflow.contains("MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256: 830670206859a1b221508ae44a031205a3eba6f5f13e05b40383bf781bdbf067"))
    #expect(workflow.contains("MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES: \"26110\""))
    #expect(workflow.contains("benchmark: using checked-in public correctness golden"))
    #expect(workflow.contains("[[ -z \"${MLXFAST_CORRECTNESS_GOLDEN_URL:-}\" && \"${MLXFAST_RUN_BENCHMARK}\" == \"1\" && \"${MLXFAST_PRIVATE_PROMPTS_R2_PRESENT}\" == \"1\" ]]"))
    #expect(!workflow.contains("generate_golden_only"))
    #expect(!workflow.contains("MLXFAST_GENERATE_GOLDEN_ONLY"))
    #expect(workflow.contains("inputs.run_benchmark && steps.validate_benchmark_artifacts.outcome == 'success'"))
    #expect(!workflow.contains("inputs.submission_ref == '' || steps.validate_benchmark_artifacts.outcome == 'success'"))
    #expect(workflow.contains("!inputs.run_benchmark && inputs.trace_correctness_step != ''"))
    #expect(!workflow.contains("results.tsv\n          if-no-files-found"))
    #expect(validator.contains("and (.metrics.first_failing_case == null)"))
    #expect(validator.contains("and (.metrics.expected_token == null)"))
    #expect(validator.contains("and (.metrics.actual_token == null)"))
}

@Test
func offlineRunnerProvesNetworkIsBlockedBeforeRunningCommand() throws {
    let runner = try String(
        contentsOfFile: ".github/scripts/run-offline.sh",
        encoding: .utf8
    )

    #expect(runner.contains("(deny network*)"))
    #expect(runner.contains("(deny process-fork)"))
    #expect(runner.contains("(deny process-exec*)"))
    #expect(runner.contains("(allow process-exec (literal"))
    #expect(runner.contains("(deny file-write*)"))
    #expect(runner.contains("MLXFAST_OFFLINE_WRITABLE_PATHS"))
    #expect(runner.contains("write_allowed_writes"))
    #expect(runner.contains("-fsS --max-time 10 https://example.com"))
    #expect(runner.contains("sandbox profile did not block network access; refusing to run"))
    #expect(runner.contains("network egress and child process execution are blocked"))
    #expect(runner.contains("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"))
    #expect(runner.contains("export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9"))
}

@Test
func benchmarkScriptHidesPrivateDirectoryFromRuntimeWorker() throws {
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    #expect(benchmark.contains("MLXFAST_PRIVATE_DIR"))
    #expect(benchmark.contains("(deny file-read* (subpath"))
    #expect(benchmark.contains("(deny file-write* (subpath"))
    #expect(benchmark.contains("(deny process-fork)"))
    #expect(benchmark.contains("(deny process-exec*)"))
    #expect(benchmark.contains("(deny file-write*)"))
    #expect(benchmark.contains("(allow process-exec (literal"))
    #expect(!benchmark.contains("(allow network* (remote ip \"localhost:*\"))"))
    #expect(!benchmark.contains("(allow network* (local unix-socket))"))
    #expect(runtime.contains("\"MLXFAST_PRIVATE_DIR\""))
    #expect(cli.contains("(deny process-fork)"))
    #expect(cli.contains("(deny process-exec*)"))
    #expect(cli.contains("(deny file-write*)"))
    #expect(!cli.contains("(allow network* (remote ip \\\"localhost:*\\\"))"))
}

@Test
func benchmarkScriptAvoidsNestedSandboxWithRuntimeWorker() throws {
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )

    #expect(benchmark.contains("Blacksmith rejects nested sandbox-exec"))
    #expect(benchmark.contains("if [[ \"${USE_RUNTIME_WORKER}\" != \"1\" && \"${MLXFAST_IN_SANDBOX:-0}\" != \"1\" && \"${MLXFAST_NO_SANDBOX:-0}\" != \"1\" ]]; then"))
    #expect(benchmark.contains("run_offline_writable_command \"$(absolute_path \"${WEIGHTS_PATH}\")\""))
    #expect(benchmark.contains("run_offline_writable_command \"$(absolute_path \"${VERIFY_TRANSFORM_TMP_PARENT}\")\""))
    #expect(benchmark.contains("--tmp-parent \"${VERIFY_TRANSFORM_TMP_PARENT}\""))
}

@Test
func overlayScriptRejectsDangerousArtifactsAfterCopy() throws {
    let overlay = try String(
        contentsOfFile: ".github/scripts/overlay-editable-paths.sh",
        encoding: .utf8
    )

    #expect(overlay.contains("validate_overlay_tree \"${target_path}\""))
    #expect(overlay.contains("overlaid editable paths must not contain symlinks"))
    #expect(overlay.contains("overlaid editable paths must contain only regular files and directories"))
    #expect(overlay.contains("-links +1"))
    #expect(overlay.contains("setuid or setgid"))
}

@Test
func runtimeWorkerBenchmarkDecodeDoesNotReceiveBulkOracle() throws {
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )

    #expect(runtime.contains("kind: \"decode_begin\""))
    #expect(runtime.contains("kind: \"decode_step\""))
    #expect(runtime.contains("worker.decodeStep(inputToken: inputToken)"))
    #expect(!runtime.contains("expected_seed_token"))
    #expect(!runtime.contains("case decodeSteps = \"decode_steps\""))
    #expect(!runtime.contains("validation_delay_ms"))
    #expect(!runtime.contains("case secondsPerToken = \"seconds_per_token\""))
    #expect(!runtime.contains("case bandwidthGBPerToken = \"bandwidth_gb_per_token\""))
}

@Test
func runtimeWorkerProtocolUsesAuthenticatedPrivateIO() throws {
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )

    #expect(runtime.contains("nonce: sessionNonce"))
    #expect(runtime.contains("response.nonce == sessionNonce"))
    #expect(runtime.contains("RuntimeWorkerProtocolIO.isolatingStandardIO()"))
    #expect(runtime.contains("F_DUPFD_CLOEXEC"))
    #expect(runtime.contains("arc4random_buf(baseAddress, buffer.count)"))
    #expect(runtime.contains("redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY"))
    #expect(runtime.contains("redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY"))
    #expect(runtime.contains("dup2(devNullFD, descriptor)"))
    #expect(runtime.contains("try protocolIO.writeLine(data)"))
    #expect(!runtime.contains("FileHandle.standardOutput.write(data)"))
}

@Test
func benchmarkQuickModeUsesShortLocalPrefixAndPrintsScore() throws {
    let constants = try String(
        contentsOfFile: "Sources/MLXFastCore/Constants.swift",
        encoding: .utf8
    )
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )

    #expect(constants.contains("public static let quickCorrectnessSteps = 64"))
    #expect(constants.contains("public static let quickBenchmarkDecodeSteps = 64"))
    #expect(cli.contains("flagOptions: [\"--quick\"]"))
    #expect(cli.contains("printScorePayload(at: scorePath)"))
    #expect(runtime.contains("correctness_steps=\\(options.correctnessSteps)"))
    #expect(runtime.contains("benchmark_decode_steps=\\(options.benchmarkDecodeSteps)"))
    #expect(!runtime.contains("Array(expectedTokens.prefix(timingPlan.decodeSteps))"))
    #expect(runtime.contains("Array(expectedTokens.prefix(decodeSteps))"))
}

@Test
func benchmarkScriptForwardsQuickFlagToSwiftBenchmark() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let argLog = root.appendingPathComponent("args.txt")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    printf '%s\\n' "$@" > "\(argLog.path)"
    score_path="score.json"
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--score-path" ]; then
        shift
        score_path="$1"
      fi
      shift || exit 1
    done
    cat > "$score_path" <<'JSON'
    {
      "score": 1,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--quick"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ]) { _, new in new }

    try process.run()
    process.waitUntilExit()

    let args = try String(contentsOf: argLog, encoding: .utf8)
    #expect(process.terminationStatus == 0)
    #expect(args.contains("benchmark\n"))
    #expect(args.contains("--quick\n"))
}

@Test
func benchmarkScriptFailsWhenScorePayloadFails() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    score_path="score.json"
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--score-path" ]; then
        shift
        score_path="$1"
      fi
      shift || exit 1
    done
    cat > "$score_path" <<'JSON'
    {
      "score": null,
      "passed": false,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ]) { _, new in new }

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus != 0)
    #expect(FileManager.default.fileExists(atPath: score.path))
    #expect(FileManager.default.fileExists(atPath: integrity.path))
}

@Test
func privateArtifactGuardRejectsRenamedGoldenAndPromptFiles() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let golden = root.appendingPathComponent("correctness_golden_512_2048.json")
    let prompts = root.appendingPathComponent("my_private_prompts.json")
    let goldenPromptText = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512.txt")
    let goldenPromptJSON = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512_256.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    try "{}".write(to: prompts, atomically: true, encoding: .utf8)
    try "hidden prompt".write(to: goldenPromptText, atomically: true, encoding: .utf8)
    try "{}".write(to: goldenPromptJSON, atomically: true, encoding: .utf8)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        ".github/scripts/deny-private-artifacts.sh",
        golden.path,
        prompts.path,
        goldenPromptText.path,
        goldenPromptJSON.path,
    ]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_GITHUB_ANNOTATIONS": "0",
    ]) { _, new in new }

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus != 0)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-benchmark-script-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
