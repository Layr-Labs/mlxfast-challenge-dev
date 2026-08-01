import Foundation
import Testing

// Structural guards for the DEFAULT ranked track. As of 2026-07-31 the default
// registered by benchmark.json is the DFlash speculative-decode track
// (laguna-xs-2.1-dflash-v1), dispatched through dflash-benchmark.yml. The serial
// track and its benchmark.yml pipeline were RETIRED in the same change: there is
// no second manifest (benchmark.dflash.json is gone — its content IS
// benchmark.json now) and no serial ranked workflow.
@Suite
struct DefaultTrackTests {
    // benchmark.json is Yukon's authoritative default registration. It must
    // positively identify the DFlash track — name and trackId are pinned here so
    // the default identity cannot silently revert to the retired serial track.
    @Test
    func defaultManifestRegistersTheDFlashTrack() throws {
        let registration = try Data(contentsOf: URL(fileURLWithPath: "benchmark.json"))
        let track = try #require(
            try JSONSerialization.jsonObject(with: registration) as? [String: Any]
        )
        // The pinned default identity (must NOT revert to serial).
        #expect(track["name"] as? String == "mlxfast-challenge-dev-dflash")
        #expect(track["trackId"] as? String == "laguna-xs-2.1-dflash-v1")
        #expect(track["name"] as? String != "mlxfast-challenge-dev-serial-v2")
        #expect(track["trackId"] as? String != "laguna-xs-2.1-serial-v2")

        let runner = try #require(track["runner"] as? [String: Any])
        #expect(runner["provider"] as? String == "github-actions")
        #expect(runner["workflow"] as? String == "dflash-benchmark.yml")
        #expect(track["scorePath"] as? String == "score.json")
        #expect(
            track["setupCommand"] as? [String] == ["bash", "-c", "./setup.sh && ./setup-dflash.sh"]
        )
        #expect(
            track["preSubmitCommand"] as? [String]
                == ["bash", "-c", "./benchmark-dflash.sh --local-submit"]
        )
        #expect(
            track["benchmarkCommand"] as? [String]
                == ["bash", "-c", "MLXFAST_SCORE_PATH=score.json ./benchmark-dflash.sh --local-iterate"]
        )
        let scoring = try #require(track["scoring"] as? [String: Any])
        #expect(scoring["mode"] as? String == "dflash-paired-decode-only")
        #expect(scoring["decodeSpeedupFloor"] as? Double == 0.95)
        let leaderboard = try #require(track["leaderboard"] as? [String: Any])
        #expect(leaderboard["namespace"] as? String == "laguna-xs-2.1-dflash-v1")
        // The DFlash track has its own contract fixture (the serial track never
        // did); the manifest points at it.
        #expect(track["contractPath"] as? String == "fixtures/laguna_xs_2_1_dflash_track.json")

        // No retired-track identity (MTP or serial) in the default registration.
        let manifestText = try #require(String(data: registration, encoding: .utf8))
        #expect(!manifestText.contains("laguna-xs-2.1-mtp"))
        #expect(!manifestText.contains("laguna-xs-2.1-serial-v2"))
        // (runner.workflow is dflash-benchmark.yml, pinned above; a bare
        // "benchmark.yml" substring would match that, so it is not asserted.)
    }

    // Every editablePaths entry must exist in the repository. The ranked overlay
    // step (.github/scripts/overlay-editable-paths.sh) fails the whole run when a
    // listed path is missing, and a path that does not exist on trusted main
    // cannot exist in any honest submission, so a stale entry hard-fails every
    // ranked submission.
    @Test
    func defaultManifestEditablePathsAllExist() throws {
        let registration = try Data(contentsOf: URL(fileURLWithPath: "benchmark.json"))
        let track = try #require(
            try JSONSerialization.jsonObject(with: registration) as? [String: Any]
        )
        let editablePaths = try #require(track["editablePaths"] as? [String])
        #expect(!editablePaths.isEmpty)
        #expect(editablePaths.count == 84)
        #expect(Set(editablePaths).count == editablePaths.count, "editablePaths must be unique")
        for guidePath in ["README.md", "AGENTS.md", "CLAUDE.md"] {
            let guide = try String(contentsOfFile: guidePath, encoding: .utf8)
            #expect(
                guide.contains("currently \(editablePaths.count) entries"),
                "\(guidePath) must report benchmark.json's editablePaths count"
            )
        }

        // The DFlash track's own optimization surface must remain editable: the
        // speculative runtime plus the target/verify shims.
        let requiredDFlashPaths = [
            "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashBatchedEngine.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/DFlashTarget.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/DFlashVerifyLinear.swift",
        ]
        for path in requiredDFlashPaths {
            #expect(
                editablePaths.contains(path),
                "DFlash speculative runtime path \(path) must remain editable"
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

    // The retired tracks stay retired: no serial ranked workflow, no second
    // manifest, no MTP contract fixture / weight manifests / local scripts.
    // benchmark.json + dflash-benchmark.yml are the single source of truth for
    // the ranked track.
    @Test
    func retiredTrackSurfaceStaysRetired() throws {
        let fm = FileManager.default
        for retired in [
            ".github/workflows/benchmark.yml",
            ".github/workflows/serial-benchmark.yml",
            "benchmark.dflash.json",
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

    // Submission static review still fails closed on the retired MTP track id and
    // carries the DFlash controlling rule (the default track). The serial rule
    // remains available for any serial submission branch still in flight, but MTP
    // is not reviewable.
    @Test
    func staticReviewCoversTheDFlashTrackAndRejectsMTP() throws {
        let script = try String(
            contentsOfFile: ".github/scripts/run-submission-static-review.sh",
            encoding: .utf8
        )
        #expect(script.contains("dflash)"))
        #expect(script.contains("unsupported submission static-review track"))
        #expect(!script.contains("gemma4-31b-it-mtp-v1"))
        #expect(!script.contains("laguna-xs-2.1-mtp-v1"))
        #expect(!script.contains("Controlling MTP-track rule"))
        #expect(!script.contains("mtp_decode_rule"))
    }
}
