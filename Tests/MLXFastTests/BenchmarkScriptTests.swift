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
    #expect(setup.contains("SETUP_PARALLEL_METALLIB=\"${MLXFAST_SETUP_PARALLEL_METALLIB:-${MLXFAST_SETUP_PARALLEL_BUILD:-1}}\""))
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
    #expect(setup.contains("ensure_swift_harness_ready || return 1"))
    #expect(setup.contains("start_mlx_metallib_build"))
    #expect(setup.contains("MLXFAST_SETUP_PARALLEL_METALLIB must be 0 or 1"))
    #expect(setup.contains("setup.sh: mlx.metallib build running in background"))
    let mainRange = try #require(setup.range(of: "ensure_swift_toolchain\ntrap cleanup_background_builds EXIT"))
    let main = String(setup[mainRange.lowerBound...])
    #expect(main.contains("build_swift_harness\nstart_mlx_metallib_build\ndownload_reference_weights \"${REFERENCE_DIR}\""))
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
    let ci = try String(
        contentsOfFile: ".github/workflows/ci.yml",
        encoding: .utf8
    )
    let setupRange = try #require(workflow.range(of: "- name: Setup Swift harness and reference checkpoint"))
    let transformRange = try #require(workflow.range(of: "- name: Transform reference checkpoint"))
    let restoreCacheRange = try #require(workflow.range(of: "- name: Restore SwiftPM cache"))
    let saveCacheRange = try #require(workflow.range(of: "- name: Save SwiftPM cache"))

    #expect(restoreCacheRange.lowerBound < setupRange.lowerBound)
    #expect(setupRange.lowerBound < transformRange.lowerBound)
    #expect(setupRange.lowerBound < saveCacheRange.lowerBound)
    #expect(saveCacheRange.lowerBound < transformRange.lowerBound)
    #expect(workflow.contains("MLXFAST_REFERENCE_DIR: .cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/main"))
    #expect(workflow.contains("MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY: \"0\""))
    #expect(workflow.contains("default: \"12\""))
    #expect(workflow.contains("actions/cache/restore@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(workflow.contains("actions/cache/save@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(workflow.contains(".build/checkouts"))
    #expect(workflow.contains(".build/repositories"))
    #expect(workflow.contains(".build/artifacts"))
    #expect(!workflow.contains("path: .cache/huggingface"))
    #expect(workflow.contains("MLXFAST_OFFLINE_WRITABLE_PATHS=\"${PWD}/weights\""))
    #expect(workflow.contains(".github/scripts/run-offline.sh .build/release/mlxfast-swift transform"))
    #expect(workflow.contains("--reference \"${MLXFAST_REFERENCE_DIR}\""))
    #expect(workflow.contains("--output weights"))
    #expect(ci.contains("push:\n    branches:\n      - main"))
    #expect(ci.contains("- name: Restore SwiftPM cache"))
    #expect(ci.contains("- name: Save SwiftPM cache"))
    #expect(ci.contains("github.event.pull_request.head.repo.full_name == github.repository"))
    #expect(ci.contains("actions/cache/restore@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(ci.contains("actions/cache/save@0400d5f644dc74513175e3cd8d07132dd4860809"))
    #expect(ci.contains(".build/checkouts"))
    #expect(ci.contains(".build/repositories"))
    #expect(ci.contains(".build/artifacts"))
}

@Test
func benchmarkWorkflowProbesAndEnforcesRuntimeWorkerSandbox() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/benchmark.yml",
        encoding: .utf8
    )
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )
    let probe = try String(
        contentsOfFile: ".github/scripts/probe-runtime-worker-sandbox.sh",
        encoding: .utf8
    )
    let ci = try String(
        contentsOfFile: ".github/workflows/ci.yml",
        encoding: .utf8
    )

    let probeRange = try #require(workflow.range(of: "- name: Probe runtime worker sandbox"))
    let goldenSourceRange = try #require(workflow.range(of: "- name: Check correctness golden source"))
    let setupRange = try #require(workflow.range(of: "- name: Setup Swift harness and reference checkpoint"))
    #expect(probeRange.lowerBound < goldenSourceRange.lowerBound)
    #expect(probeRange.lowerBound < setupRange.lowerBound)
    #expect(workflow.contains("run: .github/scripts/probe-runtime-worker-sandbox.sh"))
    #expect(workflow.contains("MLXFAST_OFFICIAL_BENCHMARK_RUN: \"1\""))

    #expect(benchmark.contains("enforce_official_sandbox"))
    #expect(benchmark.contains("MLXFAST_OFFICIAL_BENCHMARK_RUN"))
    #expect(benchmark.contains("official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1"))
    #expect(benchmark.contains("official GitHub benchmark runs must use the runtime worker sandbox"))
    #expect(benchmark.contains("enforce_official_sandbox\n\nif [[ \"${MLXFAST_IN_SANDBOX:-0}\" != \"1\" && ! -x \"${SWIFT_BIN}\" ]]; then"))

    #expect(probe.contains("(deny network*)"))
    #expect(probe.contains("(deny process-fork)"))
    #expect(probe.contains("(deny process-exec*)"))
    #expect(probe.contains("(allow process-exec (literal"))
    #expect(probe.contains("(deny file-write*)"))
    #expect(probe.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(probe.contains("(deny file-read* (literal"))
    #expect(probe.contains("(deny file-read* (subpath"))
    #expect(probe.contains("expect_inet_network_denied()"))
    #expect(probe.contains("expect_unix_network_denied(argv[5])"))
    #expect(probe.contains("expect_fork_denied()"))
    #expect(probe.contains("expect_spawn_denied()"))
    #expect(ci.contains("bash -n .github/scripts/probe-runtime-worker-sandbox.sh"))
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
    #expect(ci.contains("bash -n .github/scripts/upload-r2-object.sh"))
    #expect(ci.contains("bash -n .github/scripts/stage-benchmark-artifacts.sh"))
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
    let stageArtifacts = try String(
        contentsOfFile: ".github/scripts/stage-benchmark-artifacts.sh",
        encoding: .utf8
    )
    let ci = try String(
        contentsOfFile: ".github/workflows/ci.yml",
        encoding: .utf8
    )
    let semanticGate = try String(
        contentsOfFile: ".github/scripts/run-semantic-gpqa-gate.sh",
        encoding: .utf8
    )

    #expect(!workflow.contains("${{ runner.temp }}"))
    #expect(workflow.contains("MLXFAST_PRIVATE_DIR: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}"))
    #expect(workflow.contains("MLXFAST_CORRECTNESS_GOLDEN_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/correctness_golden.json"))
    #expect(workflow.contains("MLXFAST_GPQA_REFERENCE_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/gpqa_reference_cases.json"))
    #expect(!workflow.contains("MLXFAST_GPQA_TTFT_RESULTS_PATH"))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/semantic_gpqa_answers.json"))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_RESULTS_PATH: /tmp/mlxfast-private-${{ github.run_id }}-${{ github.run_attempt }}/semantic_gpqa_results.json"))
    #expect(workflow.contains("MLXFAST_ARTIFACT_ROOT: /tmp/mlxfast-artifacts-${{ github.run_id }}-${{ github.run_attempt }}"))
    #expect(workflow.contains("MLXFAST_ANTHROPIC_PRESENT: ${{ secrets.ORG_ANTHROPIC_API_KEY != '' && '1' || '0' }}"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_PROMPT_PATH: correctness_prompts/public_longcopy_gate_english_512.txt"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH: correctness_prompts/public_longcopy_gate_english_512_256.json"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256: 2a747bf797e16d58f5ffedacc0d4bf5ce0d14be00f2421dc04289a2154cb011d"))
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_BYTES: \"10320\""))
    #expect(workflow.contains("MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/golden_prompt_benchmark_transcription_gate_english_512_256.json"))
    #expect(workflow.contains("MLXFAST_GPQA_R2_PATH: correctness_prompts/gpqa_reference_cases.json"))
    #expect(workflow.contains("MLXFAST_GPQA_CASE_COUNT: \"5\""))
    #expect(workflow.contains("MLXFAST_GPQA_MAX_NEW_TOKENS: \"10\""))
    #expect(workflow.contains("MLXFAST_GPQA_TTFT_CASE_COUNT: \"5\""))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_CASE_COUNT: \"5\""))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS: \"10\""))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_MIN_PASS: \"3\""))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_REQUIRED: \"1\""))
    #expect(workflow.contains("MLXFAST_SEMANTIC_GPQA_MODEL: claude-sonnet-4-5-20250929"))
    #expect(workflow.contains("calibrate_gpqa_reference:"))
    #expect(workflow.contains("MLXFAST_CALIBRATE_GPQA_REFERENCE: ${{ inputs.calibrate_gpqa_reference && '1' || '0' }}"))
    #expect(workflow.contains("calibrate_gpqa_reference cannot be combined with preserve_golden_only"))
    #expect(workflow.contains("mlxfast-swift calibrate-gpqa-gates"))
    #expect(workflow.contains("mlxfast-gpqa-calibration-private.log"))
    #expect(workflow.contains(".github/scripts/upload-r2-object.sh"))
    #expect(workflow.contains("uploaded calibrated GPQA reference cases to private R2"))
    #expect(workflow.contains("MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256: 830670206859a1b221508ae44a031205a3eba6f5f13e05b40383bf781bdbf067"))
    #expect(workflow.contains("MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_BYTES: \"26110\""))
    #expect(!workflow.contains("MLXFAST_EXPECTED_CORRECTNESS_CASES: \"10\""))
    #expect(workflow.contains("benchmark: using checked-in public correctness golden"))
    #expect(workflow.contains("hidden GPQA behavior gate requires private R2 secrets"))
    #expect(workflow.contains("semantic GPQA gate requires ORG_ANTHROPIC_API_KEY"))
    #expect(workflow.contains("mlxfast-swift attach-gpqa-gates"))
    #expect(workflow.contains("--case-count \"${MLXFAST_GPQA_CASE_COUNT}\""))
    #expect(workflow.contains("--max-new-tokens \"${MLXFAST_GPQA_MAX_NEW_TOKENS}\""))
    #expect(!workflow.contains("- name: Generate semantic GPQA answers"))
    #expect(!workflow.contains("- name: Measure GPQA TTFT gate"))
    #expect(workflow.contains("- name: Semantic GPQA gate"))
    #expect(!workflow.contains("mlxfast-swift measure-gpqa-ttft"))
    #expect(!workflow.contains(".github/scripts/patch-gpqa-ttft-metrics.sh"))
    #expect(workflow.contains("ANTHROPIC_API_KEY: ${{ secrets.ORG_ANTHROPIC_API_KEY }}"))
    #expect(!workflow.contains("mlxfast-swift generate-gpqa-answers"))
    #expect(!workflow.contains("--case-count \"${MLXFAST_SEMANTIC_GPQA_CASE_COUNT}\""))
    #expect(!workflow.contains("--max-new-tokens \"${MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS}\""))
    #expect(workflow.contains(".github/scripts/run-semantic-gpqa-gate.sh"))
    #expect(workflow.contains("using private GPQA-augmented correctness golden"))
    #expect(workflow.contains("[[ \"${MLXFAST_RUN_BENCHMARK}\" == \"1\" ]]"))
    #expect(!workflow.contains("generate_golden_only"))
    #expect(!workflow.contains("MLXFAST_GENERATE_GOLDEN_ONLY"))
    #expect(workflow.contains("hashFiles('score.json') != ''"))
    #expect(!workflow.contains("inputs.run_benchmark && steps.validate_benchmark_artifacts.outcome == 'success'"))
    #expect(workflow.contains(".github/scripts/stage-benchmark-artifacts.sh"))
    #expect(workflow.contains("inputs.run_benchmark && !inputs.calibrate_gpqa_reference"))
    #expect(workflow.contains("golden.sha256=\"${MLXFAST_CORRECTNESS_GOLDEN_PATH}.sha256\""))
    #expect(workflow.contains("path: ${{ env.MLXFAST_ARTIFACT_ROOT }}/benchmark-results"))
    #expect(workflow.contains("path: ${{ env.MLXFAST_ARTIFACT_ROOT }}/correctness-results"))
    #expect(!workflow.contains("inputs.submission_ref == '' || steps.validate_benchmark_artifacts.outcome == 'success'"))
    #expect(workflow.contains("!inputs.run_benchmark && !inputs.calibrate_gpqa_reference && inputs.trace_correctness_step != ''"))
    #expect(!workflow.contains("results.tsv\n          if-no-files-found"))
    #expect(validator.contains("and (.metrics.first_failing_case == null)"))
    #expect(validator.contains("and (.metrics.expected_token == null)"))
    #expect(validator.contains("and (.metrics.actual_token == null)"))
    #expect(validator.contains("MLXFAST_SEMANTIC_GPQA_CASE_COUNT is required"))
    #expect(validator.contains("MLXFAST_SEMANTIC_GPQA_MIN_PASS is required"))
    #expect(validator.contains("MLXFAST_SEMANTIC_GPQA_REQUIRED"))
    #expect(validator.contains("MLXFAST_SEMANTIC_GPQA_REQUIRED:-1"))
    #expect(validator.contains("MLXFAST_GPQA_TTFT_CASE_COUNT is required"))
    #expect(validator.contains("\"gpqa_ttft_passed\""))
    #expect(validator.contains("and (.metrics.gpqa_ttft_passed == true)"))
    #expect(validator.contains("and (.metrics.gpqa_ttft_case_count == $ttft_cases)"))
    #expect(validator.contains("\"semantic_gpqa_passed\""))
    #expect(validator.contains("and (.metrics.semantic_gpqa_passed | type == \"boolean\")"))
    #expect(validator.contains("if $semantic_required == 1 then"))
    #expect(validator.contains("and (.metrics.semantic_gpqa_pass_count >= $semantic_min_pass)"))
    #expect(validator.contains("\"decode_speedup_floor\""))
    #expect(validator.contains("\"prefill_speedup_floor\""))
    #expect(validator.contains("and (.metrics.passed_decode_speedup_floor == true)"))
    #expect(validator.contains("and (.metrics.passed_prefill_speedup_floor == true)"))
    #expect(validator.contains("and (.metrics.decode_speedup >= .metrics.decode_speedup_floor)"))
    #expect(validator.contains("and (.metrics.prefill_speedup >= .metrics.prefill_speedup_floor)"))
    let scoreArtifactCheck = try #require(validator.range(of: "require_file \"${SCORE_PATH}\""))
    let checkedStepsEnvCheck = try #require(validator.range(of: "MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS is required"))
    #expect(scoreArtifactCheck.lowerBound < checkedStepsEnvCheck.lowerBound)
    #expect(stageArtifacts.contains("/tmp/mlxfast-artifacts-*"))
    #expect(stageArtifacts.contains(".github/scripts/deny-private-artifacts.sh \"${dest}\""))
    #expect(!ci.contains("bash -n .github/scripts/patch-gpqa-ttft-metrics.sh"))
    #expect(ci.contains("bash -n .github/scripts/run-semantic-gpqa-gate.sh"))
    #expect(semanticGate.contains("ANTHROPIC_API_KEY is required"))
    #expect(semanticGate.contains("unset ANTHROPIC_API_KEY"))
    #expect(semanticGate.contains("env -u ANTHROPIC_API_KEY curl"))
    #expect(semanticGate.contains("header = \"x-api-key: %s\""))
    #expect(semanticGate.contains("anthropic-version: 2023-06-01"))
    #expect(semanticGate.contains("extract_judge_json()"))
    #expect(semanticGate.contains("```(?:json)?"))
    #expect(semanticGate.contains("judge response was not parseable JSON; retrying"))
    #expect(semanticGate.contains("MIN_PASS=\"${MLXFAST_SEMANTIC_GPQA_MIN_PASS:-3}\""))
    #expect(semanticGate.contains("REQUIRED=\"${MLXFAST_SEMANTIC_GPQA_REQUIRED:-1}\""))
    #expect(semanticGate.contains("MLXFAST_SEMANTIC_GPQA_REQUIRED"))
    #expect(semanticGate.contains("invalid_judge_response"))
    #expect(semanticGate.contains("diagnostic did not meet threshold"))
    #expect(semanticGate.contains(".metrics.semantic_gpqa_passed = $semantic_passed"))
    #expect(semanticGate.contains(".score_sha256 = $score_hash"))
    #expect(!semanticGate.contains("--header \"x-api-key: ${ANTHROPIC_API_KEY}\""))
    #expect(!semanticGate.contains("--arg question \"$(jq"))
    #expect(!semanticGate.contains("candidate_answer\" >&2"))
}

@Test
func cliSupportsHiddenGPQAGateAttachment() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let package = try String(
        contentsOfFile: "Package.swift",
        encoding: .utf8
    )
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )

    #expect(package.contains(".product(name: \"Tokenizers\", package: \"swift-transformers\")"))
    #expect(cli.contains("case \"attach-gpqa-gates\""))
    #expect(cli.contains("case \"calibrate-gpqa-gates\""))
    #expect(cli.contains("case \"generate-gpqa-answers\""))
    #expect(!cli.contains("case \"measure-gpqa-ttft\""))
    #expect(cli.contains("AutoTokenizer.from(modelFolder: modelFolder, strict: false)"))
    #expect(cli.contains("acceptedReferenceTokenSequences"))
    #expect(cli.contains("DeepSeekRuntime.generateGreedyTokens"))
    #expect(cli.contains("runtimeWorkerOptions(blockedGoldenPath: gpqaPath)"))
    #expect(cli.contains("calibrated_reference_outputs"))
    #expect(cli.contains("SemanticGPQAAnswerDocument"))
    #expect(cli.contains("referenceAnswer(for: testCase)"))
    #expect(cli.contains("generate-gpqa-answers requires --output or MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"))
    #expect(cli.contains("semantic GPQA answer output"))
    #expect(!cli.contains("measure-gpqa-ttft"))
    #expect(!cli.contains("FirstTokenTimingOptions(weightsPath: weightsPath, promptTokenSets: selectedPrompts)"))
    #expect(cli.contains("[\"\\(normalizedKey).\", \"\\(normalizedKey):\", \"\\(normalizedKey))\"]"))
    #expect(!cli.contains("[\"\\(normalizedKey).\", \"\\(normalizedKey):\", \"\\(normalizedKey))\", \"\\(normalizedKey)\"]"))
    #expect(cli.contains("existingSequences + [generated]"))
    #expect(!cli.contains("accepted_sequences="))
    #expect(cli.contains("accepted_token_sequences or accepted_responses generated from the reference model"))
    #expect(cli.contains("sequence.prefix(maxNewTokens)"))
    #expect(cli.contains("uniqueSortedTokenSequences"))
    #expect(cli.contains("buildGPQABehaviorCaseIfWithinPromptBudget"))
    #expect(cli.contains("skippedOverBudgetGPQACases"))
    #expect(cli.contains("GPQA reference produced"))
    #expect(!cli.contains("gpqa.cases.prefix(caseCount)"))
    #expect(!cli.contains("answerKey ??"))
    #expect(cli.contains("MLXFastConstants.correctnessGPQACaseCount"))
    #expect(cli.contains("MLXFastConstants.correctnessGPQAMaxNewTokens"))
    #expect(!cli.contains("print(testCase.prompt)"))

    #expect(runtime.contains("compareBehaviorFirstToken"))
    #expect(runtime.contains("testCase.maxNewTokens == 1"))
    #expect(runtime.contains("correctnessTokenAccepted("))
    #expect(runtime.contains("correctness_teacher_forced_batch"))
    #expect(runtime.contains("top_logit_rows"))
    #expect(runtime.contains("teacherForcedCorrectnessBatch"))
}

@Test
func offlineRunnerProvesNetworkIsBlockedBeforeRunningCommand() throws {
    let runner = try String(
        contentsOfFile: ".github/scripts/run-offline.sh",
        encoding: .utf8
    )

    #expect(runner.contains("(deny network*)"))
    #expect(runner.contains("pwd -P"))
    #expect(runner.contains("cd -P"))
    #expect(runner.contains("(deny process-fork)"))
    #expect(runner.contains("(deny process-exec*)"))
    #expect(runner.contains("(allow process-exec (literal"))
    #expect(runner.contains("if [[ \"${executable}\" == \"${workspace_root}/\"* ]]; then"))
    #expect(runner.contains("(allow process-exec (subpath"))
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
    #expect(benchmark.contains("pwd -P"))
    #expect(benchmark.contains("cd -P"))
    #expect(benchmark.contains("export MLXFAST_RUNTIME_WORKER_EXECUTABLE=\"$(absolute_path \"${SWIFT_BIN}\")\""))
    #expect(benchmark.contains("(deny file-read* (subpath"))
    #expect(benchmark.contains("(deny file-write* (subpath"))
    #expect(benchmark.contains("(deny process-fork)"))
    #expect(benchmark.contains("(deny process-exec*)"))
    #expect(benchmark.contains("(deny file-write*)"))
    #expect(benchmark.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(benchmark.contains("(allow process-exec (literal"))
    #expect(!benchmark.contains("(allow network* (remote ip \"localhost:*\"))"))
    #expect(!benchmark.contains("(allow network* (local unix-socket))"))
    #expect(runtime.contains("\"MLXFAST_PRIVATE_DIR\""))
    #expect(runtime.contains("\"ANTHROPIC_API_KEY\""))
    #expect(runtime.contains("\"MLXFAST_GPQA_REFERENCE_PATH\""))
    #expect(runtime.contains("\"MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH\""))
    #expect(runtime.contains("\"MLXFAST_SEMANTIC_GPQA_RESULTS_PATH\""))
    #expect(runtime.contains("gpqaTTFT: correctnessResult.gpqaTTFT"))
    #expect(runtime.contains("gpqaTTFTSource: gpqaTTFT.source"))
    #expect(cli.contains("resolvingSymlinksInPath()"))
    #expect(cli.contains("(deny process-fork)"))
    #expect(cli.contains("(deny process-exec*)"))
    #expect(cli.contains("(deny file-write*)"))
    #expect(cli.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(cli.contains("(deny file-read* (subpath"))
    #expect(cli.contains("absolutePath(privateDir)"))
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
func expertStreamingDiagnosticsUseTrustedCoreCounters() throws {
    let fileManager = FileManager.default
    #expect(fileManager.fileExists(atPath: "Sources/MLXFastCore/ExpertSlotBank.swift"))
    #expect(!fileManager.fileExists(atPath: "Sources/MLXFastModel/ExpertSlotBank.swift"))
    #expect(fileManager.fileExists(atPath: "Sources/MLXFastCore/ExpertStreaming.swift"))
    #expect(!fileManager.fileExists(atPath: "Sources/MLXFastModel/ExpertStreaming.swift"))

    let contract = try String(contentsOfFile: "benchmark.json", encoding: .utf8)
    #expect(!contract.contains("Sources/MLXFastCore"))

    let metrics = try String(
        contentsOfFile: "Sources/MLXFastCore/ExpertStreaming.swift",
        encoding: .utf8
    )
    #expect(metrics.contains("bandwidthSource = \"trusted_core_expert_slot_bank_reads\""))
    #expect(!metrics.contains("public func recordCache"))

    let slotBank = try String(
        contentsOfFile: "Sources/MLXFastCore/ExpertSlotBank.swift",
        encoding: .utf8
    )
    #expect(!slotBank.contains("public func tensorBytes"))

    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )
    #expect(runtime.contains("ExpertStreamingMetrics.bandwidthSource"))
}

@Test
func benchmarkTimingChargesDecodeSetupAndSeparatesWorkers() throws {
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )
    let workerStart = try #require(runtime.range(of: "private static func benchmarkWithWorker"))
    let workerRuntime = String(runtime[workerStart.lowerBound...])

    #expect(workerRuntime.contains("benchmark prefill worker start"))
    #expect(workerRuntime.contains("benchmark decode worker start"))
    #expect(workerRuntime.contains("worker-reported per-step timing"))
    #expect(workerRuntime.contains("let decodePhaseStart = DispatchTime.now().uptimeNanoseconds"))
    #expect(workerRuntime.contains("includes_seed_prefill=true"))
    #expect(workerRuntime.contains("let measuredSeconds = secondsSince(decodePhaseStart)"))

    let timedBenchmarkRange = try #require(workerRuntime.range(of: "progress(\"timed benchmark start\")"))
    let weightsDigestRange = try #require(workerRuntime.range(of: "progress(\"weights digest start\")"))
    let correctnessRange = try #require(workerRuntime.range(of: "progress(\"correctness start cases=\\(golden.totalCorrectnessCaseCount)\")"))
    #expect(timedBenchmarkRange.lowerBound < weightsDigestRange.lowerBound)
    #expect(weightsDigestRange.lowerBound < correctnessRange.lowerBound)
}

@Test
func runtimeWorkerProtocolUsesAuthenticatedPrivateIO() throws {
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/DeepSeekRuntime.swift",
        encoding: .utf8
    )

    #expect(runtime.contains("let hello = try readResponseLine(validateNonce: false)"))
    #expect(runtime.contains("self.sessionNonce = nonce"))
    #expect(!runtime.contains("RuntimeWorkerRequest(\n            id: id,\n            nonce"))
    #expect(runtime.contains("response.nonce != sessionNonce"))
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
    let gpqa = root.appendingPathComponent("gpqa_reference_cases.json")
    let goldenPromptText = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512.txt")
    let goldenPromptJSON = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512_256.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    try "{}".write(to: prompts, atomically: true, encoding: .utf8)
    try "{}".write(to: gpqa, atomically: true, encoding: .utf8)
    try "hidden prompt".write(to: goldenPromptText, atomically: true, encoding: .utf8)
    try "{}".write(to: goldenPromptJSON, atomically: true, encoding: .utf8)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        ".github/scripts/deny-private-artifacts.sh",
        golden.path,
        prompts.path,
        gpqa.path,
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
