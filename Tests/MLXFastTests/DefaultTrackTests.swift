import Foundation
import Testing

// Structural guards for the DEFAULT ranked track: the serial pipeline runs
// under the canonical workflow name benchmark.yml, benchmark.json registers
// the serial track as Yukon's default, and none of the retired MTP track's
// surface (workflow, manifests, contract fixture, weight manifests, local
// scripts) survives in the repository.
@Suite
struct DefaultTrackTests {
    // benchmark.yml IS the serial ranked pipeline: dispatch-only
    // (self-hosted safety), pinned by its own trusted-workflow script under
    // the canonical name, carrying the serial host contract and the serial
    // scoring floors — and none of the retired MTP pipeline's identity.
    @Test
    func benchmarkWorkflowIsTheSerialPipelineUnderTheCanonicalName() throws {
        let workflow = try String(
            contentsOfFile: ".github/workflows/benchmark.yml",
            encoding: .utf8
        )
        #expect(workflow.hasPrefix("name: benchmark\n"))
        #expect(workflow.contains("workflow_dispatch:"))
        #expect(!workflow.contains("pull_request"))
        #expect(!workflow.contains("\n  push:"))
        #expect(workflow.contains("run: .github/scripts/enforce-trusted-benchmark-workflow.sh"))

        let script = try String(
            contentsOfFile: ".github/scripts/enforce-trusted-benchmark-workflow.sh",
            encoding: .utf8
        )
        #expect(script.contains("WORKFLOW_PATH=\".github/workflows/benchmark.yml\""))
        #expect(script.contains("workflow_dispatch"))
        #expect(script.contains(
            "refs/heads/main|refs/heads/submissions/*|refs/heads/baseline/*|refs/heads/yukon/baseline/*"
        ))

        // Serial ranked identity: the serial measure wrapper, stable bench
        // v2 workspace, pinned baseline tree, calibration, and the frozen
        // prefill+decode floors.
        #expect(workflow.contains("MLXFAST_MEASURE_JOB: /opt/bench-runner/measure-job.sh"))
        #expect(workflow.contains(
            "MLXFAST_JOB_WS: /Users/Shared/bench-jobs/ranked-laguna-xs-2.1-serial-v2"
        ))
        #expect(workflow.contains(
            "MLXFAST_BASELINE_WS: /opt/bench-runner/baseline/laguna-xs-2.1-serial-v2/current"
        ))
        #expect(workflow.contains(
            "MLXFAST_BASELINE_CALIBRATION: /opt/bench-runner/state/laguna-xs-2.1-serial-v2/baseline-calibration.json"
        ))
        #expect(workflow.contains("MLXFAST_DECODE_SPEEDUP_FLOOR: \"0.95\""))
        #expect(workflow.contains("MLXFAST_PREFILL_SPEEDUP_FLOOR: \"0.95\""))
        #expect(workflow.contains("- name: Timed paired benchmark (measure-job)"))
        #expect(workflow.contains("- name: Overlay paired timing into final score"))

        // The retired MTP pipeline's runtime identity must not resurface.
        #expect(!workflow.contains("MLXFAST_MTP_"))
        #expect(!workflow.contains("mtp-ranked"))
        #expect(!workflow.contains("measure-mtp-job"))
        #expect(!workflow.contains("mtp-weights"))
        #expect(!workflow.contains("laguna-xs-2.1-mtp"))
        #expect(!workflow.contains("gemma4-31b-it"))
    }

    // benchmark.json is Yukon's authoritative default registration: the
    // serial track, dispatched through the canonical benchmark.yml, scored
    // with the serial prefill+decode weighted formula and 0.95 floors.
    @Test
    func defaultManifestRegistersTheSerialTrack() throws {
        let registration = try Data(contentsOf: URL(fileURLWithPath: "benchmark.json"))
        let track = try #require(
            try JSONSerialization.jsonObject(with: registration) as? [String: Any]
        )
        #expect(track["trackId"] as? String == "laguna-xs-2.1-serial-v2")
        #expect(track["name"] as? String == "mlxfast-challenge-dev-serial-v2")
        let runner = try #require(track["runner"] as? [String: Any])
        #expect(runner["provider"] as? String == "github-actions")
        #expect(runner["workflow"] as? String == "benchmark.yml")
        #expect(track["scorePath"] as? String == "score.json")
        #expect(track["setupCommand"] as? [String] == ["bash", "-c", "./setup.sh"])
        #expect(
            track["preSubmitCommand"] as? [String]
                == ["bash", "-c", "./benchmark.sh --local-submit"]
        )
        #expect(
            track["benchmarkCommand"] as? [String]
                == ["bash", "-c", "MLXFAST_SCORE_PATH=score.json ./benchmark.sh --local-iterate"]
        )
        let scoring = try #require(track["scoring"] as? [String: Any])
        #expect(scoring["mode"] as? String == "serial-paired-prefill-decode")
        #expect(
            scoring["formula"] as? String == "decode_speedup^0.75 * prefill_speedup^0.25"
        )
        #expect(scoring["decodeSpeedupFloor"] as? Double == 0.95)
        #expect(scoring["prefillSpeedupFloor"] as? Double == 0.95)
        let leaderboard = try #require(track["leaderboard"] as? [String: Any])
        #expect(leaderboard["namespace"] as? String == "laguna-xs-2.1-serial-v2")
        // The serial track has no separate contract fixture; the manifest is
        // the single source of truth.
        #expect(track["contractPath"] == nil)
        // No retired-track identity in the default registration.
        let manifestText = try #require(String(data: registration, encoding: .utf8))
        #expect(!manifestText.contains("laguna-xs-2.1-mtp"))
        #expect(!manifestText.contains("benchmark-mtp.sh"))
        #expect(!manifestText.contains("setup-mtp.sh"))
        #expect(!manifestText.contains("separateFromMTPTrack"))
        #expect(!manifestText.contains("separateFromSerialTrack"))
    }

    // Every editablePaths entry must exist in the repository. The ranked
    // overlay step (.github/scripts/overlay-editable-paths.sh) fails the
    // whole run when a listed path is missing from the submission worktree,
    // and a path that does not exist on trusted main cannot exist in any
    // honest submission (submission branches differ from main only inside
    // editablePaths). Stale entries therefore hard-fail every ranked
    // submission: the retired MTP track left LagunaMTP.swift /
    // LagunaMTPTarget.swift listed after their files were dropped.
    @Test
    func defaultManifestEditablePathsAllExist() throws {
        let registration = try Data(contentsOf: URL(fileURLWithPath: "benchmark.json"))
        let track = try #require(
            try JSONSerialization.jsonObject(with: registration) as? [String: Any]
        )
        let editablePaths = try #require(track["editablePaths"] as? [String])
        #expect(!editablePaths.isEmpty)
        #expect(editablePaths.count == 97)
        #expect(Set(editablePaths).count == editablePaths.count, "editablePaths must be unique")
        for guidePath in ["README.md", "AGENTS.md", "CLAUDE.md"] {
            let guide = try String(contentsOfFile: guidePath, encoding: .utf8)
            #expect(
                guide.contains("currently \(editablePaths.count) entries"),
                "\(guidePath) must report benchmark.json's editablePaths count"
            )
        }

        let requiredFullStackPaths = [
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/fp4.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/fp8.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sort.metal",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sort.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduce.metal",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduce.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduce_utils.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/atomic.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduction/reduce_col.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduction/reduce_init.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduction/reduce_row.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduction/ops.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/reduction/reduce_all.h",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/sort.cpp",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/reduce.cpp",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/reduce_utils.cpp",
        ]
        for path in requiredFullStackPaths {
            #expect(
                editablePaths.contains(path),
                "Laguna MoE/attention full-stack path \(path) must remain editable"
            )
        }
        #expect(
            !editablePaths.contains(
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal"
            ),
            "host-side MLX edits must stay limited to exact reviewed files"
        )
        for trustedRoot in [
            ".github",
            "correctness_prompts",
            "fixtures",
            "Sources/MLXFastCLI",
            "Sources/MLXFastCore",
            "Sources/MLXFastHarness",
            "Sources/MLXFastTrustedHarness",
        ] {
            #expect(
                !editablePaths.contains {
                    $0 == trustedRoot || $0.hasPrefix(trustedRoot + "/")
                },
                "distribution, golden, and trusted harness tooling must stay excluded: \(trustedRoot)"
            )
        }

        let fm = FileManager.default
        for path in editablePaths {
            #expect(
                fm.fileExists(atPath: path),
                "editablePaths entry \(path) does not exist; the ranked overlay step fails closed on missing editable paths"
            )
        }
    }

    // The MTP track is fully retired: no alternate manifests, no MTP
    // contract fixture or weight manifests, no separately named serial
    // workflow, and no MTP local scripts. benchmark.json + benchmark.yml are
    // the single source of truth for the ranked track.
    @Test
    func retiredMTPTrackSurfaceStaysRetired() throws {
        let fm = FileManager.default
        for retired in [
            ".github/workflows/serial-benchmark.yml",
            ".github/scripts/enforce-trusted-serial-benchmark-workflow.sh",
            "benchmark.mtp.json",
            "benchmark.serial.json",
            "fixtures/laguna_xs_2_1_mtp_track.json",
            "fixtures/mtp_laguna_xs_2_1_4bit.sha256",
            "fixtures/mtp_laguna_xs_2_1_dflash.sha256",
            "benchmark-mtp.sh",
            "setup-mtp.sh",
        ] {
            #expect(!fm.fileExists(atPath: retired), "\(retired) must stay retired")
        }
    }

    // Submission static review is serial-only: the retired MTP track id is
    // no longer reviewable (fails closed), while the serial controlling rule
    // remains the policy the judge receives.
    @Test
    func staticReviewAllowsOnlyTheSerialTrack() throws {
        let script = try String(
            contentsOfFile: ".github/scripts/run-submission-static-review.sh",
            encoding: .utf8
        )
        #expect(script.contains("TRACK_ID=\"${MLXFAST_SUBMISSION_TRACK_ID:-serial}\""))
        #expect(script.contains("  serial)\n"))
        #expect(script.contains("unsupported submission static-review track"))
        #expect(script.contains("Controlling serial-track rule"))
        #expect(!script.contains("gemma4-31b-it-mtp-v1"))
        #expect(!script.contains("laguna-xs-2.1-mtp-v1"))
        #expect(!script.contains("Controlling MTP-track rule"))
        #expect(!script.contains("mtp_decode_rule"))
    }
}

// MARK: - The DFlash contract's editable surface must actually be submittable

/// `benchmark.dflash.json` declares nine editable paths that `benchmark.json`
/// does not — the DFlash runtime a DFlash submitter exists to optimize. Both
/// surface gates (`overlay-editable-paths.sh` and `enforce-modifiable-surface.sh`)
/// default to `benchmark.json` and only read the DFlash contract when the job
/// exports `CONTRACT_PATH`. If the DFlash workflow does not set it, those nine
/// files are silently dropped by the overlay AND rejected by the surface check,
/// so the track's whole optimization target is unreachable.
///
/// This is the coverage test for that: for every DFlash editable path, either the
/// serial contract already covers it, or the DFlash contract is genuinely the one
/// in force.
@Suite
struct DFlashEditableSurfaceReachabilityTests {
    private func editablePaths(_ manifest: String) throws -> [String] {
        let data = try Data(contentsOf: URL(fileURLWithPath: manifest))
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        return try #require(json["editablePaths"] as? [String])
    }

    @Test
    func everyDFlashEditablePathIsReachableUnderTheContractTheJobEnforces() throws {
        let dflashPaths = try editablePaths("benchmark.dflash.json")
        let serialPaths = Set(try editablePaths("benchmark.json"))
        #expect(!dflashPaths.isEmpty)

        let fileManager = FileManager.default
        for path in dflashPaths {
            #expect(fileManager.fileExists(atPath: path), "editable path is missing: \(path)")
        }

        let dflashOnly = dflashPaths.filter { !serialPaths.contains($0) }.sorted()
        // If this ever becomes empty the track has no distinct optimization
        // surface, which is itself a regression worth failing on.
        #expect(
            !dflashOnly.isEmpty,
            "benchmark.dflash.json no longer declares any DFlash-specific editable path"
        )
        for path in dflashOnly {
            #expect(
                path.contains("DFlash") || path.contains("MLXSpeculative"),
                "unexpected DFlash-only editable path \(path); confirm it belongs to this track"
            )
        }

        // Reachability: the DFlash job must put its own contract in force for
        // BOTH surface gates, or these paths are dropped and rejected. The
        // wiring goes through a job-env indirection, so resolve it rather than
        // demanding a literal.
        let workflow = try String(
            contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8
        )
        let environment = try DFlashGateTextSupport.jobEnvironment(workflow)
        for stepName in [
            "Verify submitted commit and modifiable surface",
            "Overlay submitted editable paths",
        ] {
            let step = DFlashGateTextSupport.executable(
                try DFlashGateTextSupport.stepBody(workflow, stepName)
            )
            let resolved = DFlashGateTextSupport.resolvedContractPath(
                in: step, jobEnvironment: environment
            )
            #expect(
                resolved == "benchmark.dflash.json",
                """
                '\(stepName)' puts \(resolved ?? "<CONTRACT_PATH unset>") in \
                force, not benchmark.dflash.json. These \(dflashOnly.count) \
                DFlash-only editable paths are therefore dropped by the overlay \
                and rejected by the modifiable-surface check: \
                \(dflashOnly.joined(separator: ", "))
                """
            )
        }
    }
}
