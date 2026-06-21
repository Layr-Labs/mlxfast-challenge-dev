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
    #expect(setup.contains("Usage: ./setup.sh"))
    #expect(setup.contains("reference_file_is_current"))
    #expect(setup.contains("redownloading ${label} from scratch after hash verification failed"))
    #expect(setup.contains("reference cache path ${reference_dir}"))
    #expect(setup.contains("if ! verify_reference_manifest \"${reference_dir}\"; then"))
    #expect(setup.contains("downloaded ${total}/${total} safetensors shard(s)"))
    #expect(setup.contains("setup.sh: setup complete elapsed="))
    #expect(setup.contains(".github/scripts/run-offline.sh ${SWIFT_BIN} transform"))
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
    #expect(workflow.contains("run: .github/scripts/run-offline.sh .build/release/mlxfast-swift transform"))
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

    #expect(!workflow.contains("${{ runner.temp }}"))
    #expect(workflow.contains("MLXFAST_PRIVATE_DIR: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}"))
    #expect(workflow.contains("MLXFAST_CORRECTNESS_GOLDEN_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/correctness_golden.json"))
}

@Test
func offlineRunnerProvesNetworkIsBlockedBeforeRunningCommand() throws {
    let runner = try String(
        contentsOfFile: ".github/scripts/run-offline.sh",
        encoding: .utf8
    )

    #expect(runner.contains("sandbox-exec -f \"${SANDBOX_PROFILE}\""))
    #expect(runner.contains("curl -fsS --max-time 10 https://example.com"))
    #expect(runner.contains("sandbox profile did not block network access; refusing to run"))
    #expect(runner.contains("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"))
    #expect(runner.contains("HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9"))
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

    #expect(benchmark.contains("MLXFAST_PRIVATE_DIR"))
    #expect(benchmark.contains("(deny file-read* (subpath"))
    #expect(benchmark.contains("(deny file-write* (subpath"))
    #expect(runtime.contains("\"MLXFAST_PRIVATE_DIR\""))
}

@Test
func benchmarkScriptAvoidsNestedSandboxWithRuntimeWorker() throws {
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )

    #expect(benchmark.contains("Blacksmith rejects nested sandbox-exec"))
    #expect(benchmark.contains("if [[ \"${USE_RUNTIME_WORKER}\" != \"1\" && \"${MLXFAST_IN_SANDBOX:-0}\" != \"1\" && \"${MLXFAST_NO_SANDBOX:-0}\" != \"1\" ]]; then"))
    #expect(benchmark.contains("run_offline_command \"${SWIFT_BIN}\" transform --reference"))
    #expect(benchmark.contains("run_offline_command \"${SWIFT_BIN}\" verify-transform --reference"))
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
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    try "{}".write(to: prompts, atomically: true, encoding: .utf8)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        ".github/scripts/deny-private-artifacts.sh",
        golden.path,
        prompts.path,
    ]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

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
