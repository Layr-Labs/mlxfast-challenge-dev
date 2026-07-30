import CryptoKit
import Foundation
import Testing

// The validator types live in the harness twin the test target already depends
// on; the trusted twin is byte-identical for everything used here.
import MLXFastCore
import MLXFastHarness

// Structural guards for the EXPERIMENTAL DFlash speculative-decode track
// (laguna-xs-2.1-dflash-v1). The track is deliberately NOT the default and
// NOT enabled: it exists in the repository as reviewable scaffolding plus a
// written correctness contract, and it fails closed until Criterion E is
// implemented. These tests pin (a) that it stays fail-closed while its token
// fidelity gate is unspecified, (b) that it cannot bleed into the live serial
// ranked track, and (c) that its manifests, fixture and workflow agree.
//
// See docs/dflash-track-correctness-contract.md for why the retired MTP
// track's exact-vs-sequential token gate is unsatisfiable here and what
// replaces it.
@Suite
struct DFlashTrackTests {
    private static let manifestPath = "benchmark.dflash.json"
    private static let fixturePath = "fixtures/laguna_xs_2_1_dflash_track.json"
    private static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let guardPath =
        ".github/scripts/enforce-trusted-dflash-benchmark-workflow.sh"
    private static let drafterManifestPath =
        "fixtures/dflash_laguna_xs_2_1_drafter.sha256"
    // The DFlash track deliberately reuses the serial track's reference
    // checkpoint so its speedups stay comparable with serial-track timings.
    private static let referenceManifestPath =
        "fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256"

    private func json(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    private func text(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    /// Manifest entry lines are `<sha256> <byte_count> <relative_path>`;
    /// comment and blank lines carry provenance only.
    private func manifestEntries(_ path: String) throws -> [(String, Int, String)] {
        try text(path).split(separator: "\n").compactMap { line in
            let raw = String(line)
            guard !raw.hasPrefix("#"), !raw.trimmingCharacters(in: .whitespaces).isEmpty
            else { return nil }
            let parts = raw.split(separator: " ", maxSplits: 2).map(String.init)
            guard parts.count == 3, let bytes = Int(parts[1]) else { return nil }
            return (parts[0], bytes, parts[2])
        }
    }

    // THE CENTRAL SAFETY INVARIANT. The measured evidence (14/14 honest
    // divergences from the target's own block-vs-sequential NVFP4 numerics)
    // forced the retired bit-exact gate to be withdrawn. Withdrawing a gate
    // without landing its replacement must never be promotable into a scored
    // track, so official scoring and a specified fidelity gate are locked
    // together: the track may only be enabled once the gate says implemented.
    @Test
    func trackCannotBeEnabledWhileTheFidelityGateIsUnspecified() throws {
        let fixture = try json(Self.fixturePath)
        let manifest = try json(Self.manifestPath)
        let scoring = try #require(manifest["scoring"] as? [String: Any])

        let officialScoring = fixture["official_scoring_enabled"] as? Bool
        let gateStatus = scoring["tokenFidelityGateStatus"] as? String
        #expect(gateStatus != nil, "scoring must declare tokenFidelityGateStatus")

        if officialScoring == true {
            #expect(
                gateStatus == "implemented",
                """
                DFlash official scoring is enabled while tokenFidelityGateStatus \
                is \(gateStatus ?? "nil"). A scored speculative-decode track \
                requires the Criterion E fidelity gate to be implemented; see \
                docs/dflash-track-correctness-contract.md.
                """
            )
        }

        // Today the track ships inert. If this flips, the assertion above is
        // what keeps it honest.
        #expect(officialScoring == false)
        #expect(gateStatus == "pending-spec")
    }

    // The written contract must stay in the repository: it is the only record
    // of why the exact-vs-sequential gate cannot be reused and what the
    // replacement has to do.
    @Test
    func correctnessContractIsDocumented() throws {
        let doc = try text("docs/dflash-track-correctness-contract.md")
        #expect(doc.contains("Criterion E"))
        #expect(doc.contains("baseline_path_block_vs_sequential"))
        // The four layers that do the anti-cheat work must remain described.
        for marker in ["L1", "L2", "L3", "L4"] {
            #expect(doc.contains(marker), "contract must describe layer \(marker)")
        }
    }

    // Identity agreement across manifest, contract fixture and workflow. A
    // mismatch here means a run could score one track against another's
    // contract.
    @Test
    func trackIdentityAgreesAcrossManifestFixtureAndWorkflow() throws {
        let manifest = try json(Self.manifestPath)
        let fixture = try json(Self.fixturePath)
        let workflow = try text(Self.workflowPath)

        let trackID = "laguna-xs-2.1-dflash-v1"
        #expect(manifest["trackId"] as? String == trackID)
        #expect(fixture["track_id"] as? String == trackID)
        #expect(workflow.contains("MLXFAST_DFLASH_TRACK_ID: \(trackID)"))
        // The env surface is namespaced to this track. Leftover MLXFAST_MTP_*
        // names would mean the revived scaffolding still speaks the retired
        // track's contract to the box wrapper, which reads MLXFAST_DFLASH_*.
        #expect(!workflow.contains("MLXFAST_MTP_"))

        #expect(manifest["contractPath"] as? String == Self.fixturePath)
        let runner = try #require(manifest["runner"] as? [String: Any])
        #expect(runner["workflow"] as? String == "dflash-benchmark.yml")
    }

    // The DFlash workflow is dispatch-only (self-hosted safety), guarded by
    // its OWN trusted-workflow script, and keeps the fail-closed enablement
    // interlock.
    @Test
    func dflashWorkflowIsDispatchOnlyAndGuardedByItsOwnScript() throws {
        let workflow = try text(Self.workflowPath)
        #expect(workflow.contains("workflow_dispatch:"))
        #expect(!workflow.contains("pull_request"))
        #expect(!workflow.contains("\n  push:"))
        #expect(workflow.contains("enforce-trusted-dflash-benchmark-workflow.sh"))
        // It must NOT invoke the serial track's guard.
        #expect(!workflow.contains("run: .github/scripts/enforce-trusted-benchmark-workflow.sh"))
        // Fail-closed interlock on the contract's own enablement flags.
        #expect(workflow.contains("official_scoring_enabled"))

        let guardScript = try text(Self.guardPath)
        #expect(guardScript.contains("WORKFLOW_PATH=\"\(Self.workflowPath)\""))
        #expect(guardScript.contains("workflow_dispatch"))
    }

    // Track isolation: the DFlash pipeline must not read or write any of the
    // live serial track's measurement state. The serial state directory in
    // particular holds baseline-calibration.json, which is the denominator of
    // every ranked serial score.
    @Test
    func dflashPipelineDoesNotTouchSerialTrackState() throws {
        let workflow = try text(Self.workflowPath)
        #expect(workflow.contains("/opt/bench-runner/measure-dflash-job.sh"))
        #expect(!workflow.contains("/opt/bench-runner/measure-job.sh"))
        #expect(workflow.contains("/opt/bench-runner/state/laguna-xs-2.1-dflash-v1"))
        #expect(!workflow.contains("/opt/bench-runner/state/laguna-xs-2.1-serial-v2"))
        #expect(!workflow.contains("laguna-xs-2.1-serial-v2/current"))
        // Everything below inspects what the workflow DOES, not what it says:
        // comment lines are stripped first. The comments deliberately name the dead
        // MTP paths and flags so a future reader knows what was removed and why,
        // and asserting against them would make documenting the fix impossible.
        let executable = workflow
            .split(separator: "\n")
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")

        // The BARE state dir is the serial track's, and it is where
        // baseline-calibration.json lives for serial. measure-dflash-job.sh's own
        // header warns that pointing DFlash at it bands DFlash runs against serial
        // numbers and cross-writes the serial oracle cache. The earlier version of
        // this test only excluded the serial PER-TRACK dir, so it passed while the
        // workflow pointed MLXFAST_DFLASH_BASELINE_CALIBRATION at a
        // retired-MTP calibration file inside the bare serial state dir.
        for line in executable.split(separator: "\n") {
            let text = String(line)
            guard text.contains("/opt/bench-runner/state/") else { continue }
            #expect(
                text.contains("/opt/bench-runner/state/laguna-xs-2.1-dflash-v1"),
                "every /opt/bench-runner/state path must sit inside the DFlash per-track state dir"
            )
        }

        // Retired-MTP paths. The identity test only forbade the MLXFAST_MTP_
        // variable PREFIX, which let MLXFAST_DFLASH_* variables keep pointing at
        // mtp-baseline paths that exist on no box -- failing ranked runs at the
        // prerequisite check with exit 8.
        #expect(!executable.contains("/opt/bench-runner/mtp-baseline"))
        #expect(!executable.contains("mtp-baseline-calibration.json"))

        // Every mlxfast-swift invocation must use flags this CLI actually declares.
        // The restored scaffolding passed five that do not exist, each aborting on
        // argument validation -- the same drift that had left four dead flags in
        // the box wrapper. Invisible until a run is dispatched, cheap to pin.
        // NOTE: --contract is deliberately absent from this list. It does not
        // exist on the CLI but IS a real measure-dflash-job.sh flag, so banning
        // the bare string would forbid a correct wrapper invocation. --golden,
        // --tokens and --block-size are likewise shared by both surfaces.
        for dead in [
            "--target-source", "--assistant",
            "--target-verification", "--require-trained-assistant",
        ] {
            #expect(
                !executable.contains(dead),
                "the DFlash workflow passes a CLI flag that does not exist"
            )
        }
        // The wrapper reads DFLASH_*, not MTP_*: exporting MTP_TARGET_DIR and
        // MTP_ASSISTANT_DIR was silently ignored, and only worked because the
        // wrapper's own defaults happen to be correct.
        #expect(!executable.contains("MTP_TARGET_DIR"))
        #expect(!executable.contains("MTP_ASSISTANT_DIR"))

        // The transform output must be the directory the box wrapper reads
        // (`--weights weights`), not the retired track's mtp-weights.
        #expect(!executable.contains("mtp-weights"))

        // The baseline must be RESOLVED before the wrapper clones it. The wrapper
        // uses `cp -c -R`, which on a symlink source copies the symlink, so the
        // clone under the job root becomes a link back into /opt/bench-runner and
        // bench-exec refuses it as outside the allowed job root. Measured on M5-C:
        // passing `.../current` fails every pair, passing the resolved sha path
        // succeeds. Keep `current` as the env indirection, resolve at use.
        #expect(executable.contains("MLXFAST_DFLASH_BASELINE_RESOLVED"))
        #expect(executable.contains("readlink -f \"${MLXFAST_DFLASH_BASELINE_WS}\""))
        #expect(
            !executable.contains("--baseline \"${MLXFAST_DFLASH_BASELINE_WS}\""),
            "the wrapper must receive the resolved baseline, not the symlink"
        )
        // Distinct concurrency domain so a DFlash run cannot cancel or queue
        // behind a serial ranked run under the same key.
        #expect(workflow.contains("mlxfast-dflash-ranked-"))

        // ...and the serial pipeline stays ignorant of DFlash.
        let serial = try text(".github/workflows/benchmark.yml")
        #expect(!serial.lowercased().contains("dflash"))

        // RUNNER POOL ISOLATION. Sharing the serial track's `m5-bench` label let a
        // DFlash dispatch occupy a runner in the LIVE ranked pool for the length of
        // a DFlash job. A dedicated label makes an unserved dispatch queue rather
        // than run on the wrong box -- the safe failure for an inert track.
        #expect(workflow.contains("runs-on: [self-hosted, m5-laguna-dflash]"))
        #expect(!workflow.contains("runs-on: [self-hosted, m5-bench]"))
        // The serial workflow keeps its own label, untouched.
        #expect(serial.contains("runs-on: [self-hosted, m5-bench]"))
    }

    // Submitted code runs as the bench uid inside the job workspace; every
    // trusted surface it must not rewrite is ACL-denied. The DFlash manifest
    // and its two local scripts are part of that surface.
    @Test
    func benchUidCannotWriteTheDFlashTrustedSurface() throws {
        let workflow = try text(Self.workflowPath)
        for locked in [
            "benchmark.dflash.json", "benchmark-dflash.sh", "setup-dflash.sh",
        ] {
            #expect(
                workflow.contains("${MLXFAST_JOB_WS}/\(locked)"),
                "\(locked) must be ACL-denied against bench writes"
            )
        }
    }

    // The drafter is an ORGANIZER-provisioned artifact: participants cannot
    // upload weights, so its manifest must carry real pinned hashes (the
    // retired placeholder was deliberately entry-less) and the fixture must
    // pin the manifest's own digest.
    @Test
    func drafterArtifactIsPinnedWithRealHashes() throws {
        let entries = try manifestEntries(Self.drafterManifestPath)
        #expect(entries.count >= 2, "drafter manifest must pin real entries")
        #expect(entries.contains { $0.2 == "model.safetensors" })
        #expect(entries.contains { $0.2 == "config.json" })
        for (digest, bytes, path) in entries {
            #expect(digest.count == 64, "\(path) digest must be sha256 hex")
            #expect(digest.allSatisfy { $0.isHexDigit })
            #expect(bytes > 0)
        }

        let fixture = try json(Self.fixturePath)
        let assistant = try #require(fixture["assistant"] as? [String: Any])
        #expect(assistant["manifest_path"] as? String == Self.drafterManifestPath)
        let pinned = try #require(assistant["manifest_sha256"] as? String)
        let actual = SHA256
            .hash(data: try Data(contentsOf: URL(fileURLWithPath: Self.drafterManifestPath)))
            .map { String(format: "%02x", $0) }
            .joined()
        #expect(
            pinned == actual,
            "assistant.manifest_sha256 is stale: fixture pins \(pinned), file hashes to \(actual)"
        )

        // Architecture parameters the runtime depends on.
        #expect(assistant["block_size"] as? Int == 16)
        #expect(assistant["mask_token_id"] as? Int == 12)
        #expect(assistant["target_layer_ids"] as? [Int] == [1, 13, 25, 33, 39])
    }

    // The DFlash target must be the SAME NVFP4 reference checkpoint the serial
    // track measures, otherwise a DFlash speedup is not comparable with serial
    // timings and the two tracks silently measure different models.
    @Test
    func dflashTargetIsTheSharedNVFP4ReferenceCheckpoint() throws {
        let fixture = try json(Self.fixturePath)
        let target = try #require(fixture["target"] as? [String: Any])
        #expect(target["runtime_model_id"] as? String == "poolside/Laguna-XS-2.1-NVFP4-mlx")
        #expect(
            target["runtime_revision"] as? String
                == "841778bda563a36104dd521e37d99218e46f4f25"
        )
        #expect(target["manifest_path"] as? String == Self.referenceManifestPath)

        let quantization = try #require(target["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == "nvfp4")
        #expect(quantization["group_size"] as? Int == 16)
        #expect(quantization["bits"] as? Int == 4)

        // The shared reference manifest is populated (not a placeholder) and
        // the fixture's byte expectation matches it.
        let entries = try manifestEntries(Self.referenceManifestPath)
        #expect(entries.count >= 13)
        let total = entries.reduce(0) { $0 + $1.1 }
        #expect(target["expected_source_bytes"] as? Int == total)
    }

    // The optimization target must be reachable: every declared editable path
    // has to exist, and the DFlash runtime the track exists to optimize has to
    // be inside the editable surface.
    @Test
    func editableSurfaceExistsAndCoversTheDFlashRuntime() throws {
        let manifest = try json(Self.manifestPath)
        let paths = try #require(manifest["editablePaths"] as? [String])
        #expect(!paths.isEmpty)

        let fm = FileManager.default
        for path in paths {
            #expect(fm.fileExists(atPath: path), "editable path is missing: \(path)")
        }

        for required in [
            "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashGreedyRound.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXSpeculative/DFlashDraftModel.swift",
            "Vendor/mlx-swift-lm/Libraries/MLXLLM/DFlashTarget.swift",
        ] {
            #expect(
                paths.contains(required),
                "\(required) must be editable: it is what the track optimizes"
            )
        }
    }

    // The scored window and the block size are declared in three places that
    // must not drift: the contract fixture's protocol block, the fixture's
    // proposed_scoring block, and the manifest's scoring block. They disagreed
    // on the revived scaffolding (protocol said 128 while both scoring blocks
    // said 512), which would have scored a different window than the contract
    // documented.
    @Test
    func scoredWindowAndBlockSizeAgreeEverywhereTheyAreDeclared() throws {
        let fixture = try json(Self.fixturePath)
        let manifest = try json(Self.manifestPath)
        let workflow = try text(Self.workflowPath)

        let protocolBlock = try #require(fixture["protocol"] as? [String: Any])
        let proposedScoring = try #require(fixture["proposed_scoring"] as? [String: Any])
        let scoring = try #require(manifest["scoring"] as? [String: Any])

        let window = try #require(protocolBlock["decode_tokens"] as? Int)
        #expect(proposedScoring["ranked_decode_window_tokens"] as? Int == window)
        #expect(scoring["decodeTokens"] as? Int == window)
        #expect(workflow.contains("MLXFAST_DFLASH_DECODE_TOKENS: \"\(window)\""))

        // The protocol ceiling must not be below the scored window.
        let ceiling = try #require(protocolBlock["maximum_decode_tokens"] as? Int)
        #expect(ceiling >= window)

        // Block size: the drafter is trained for 16 and the measured peak on M5
        // is K=8, so the ranked default must be within the protocol maximum and
        // must not silently revert to the retired MTP track's 4.
        let maxBlock = try #require(protocolBlock["maximum_block_size"] as? Int)
        #expect(maxBlock == 16)
        #expect(workflow.contains("MLXFAST_DFLASH_BLOCK_SIZE: \"3\""))
    }

    // CONTRACT LAYER L6 (anti-lottery). A frozen timed prompt makes a failed
    // ranked run free to retry, which turns every output-side gate into
    // submit-until-green and lets a drafter-confidence threshold be tuned across
    // resubmissions. The timed target must therefore be sampled per run from a
    // pool, using entropy the participant cannot see or influence -- NOT the run
    // id, which is visible and stable within a run.
    @Test
    func timedTargetIsSampledPerRunFromAPoolAndFailsClosedWhenEmpty() throws {
        let workflow = try text(Self.workflowPath)
        let fixture = try json(Self.fixturePath)

        // The pool lives in the contract fixture and each entry is verifiable.
        let pool = try #require(fixture["timed_prompt_pool"] as? [Any])
        for case let entry as [String: Any] in pool {
            #expect(entry["r2_path"] as? String != nil)
            #expect(entry["sha256"] as? String != nil)
            #expect(entry["bytes"] as? Int != nil)
        }

        // Selection must draw from kernel entropy, not from the run id.
        #expect(workflow.contains("/dev/urandom"))
        #expect(!workflow.contains("github.run_id }} % "))
        // Rejection sampling rather than a biased plain modulo over the range.
        #expect(workflow.contains("(65536 / pool_size) * pool_size"))
        // Fail closed on an empty pool.
        #expect(workflow.contains("DFlash timed prompt pool is empty"))
        // The selection is auditable privately and must not be echoed to the
        // participant-visible job log.
        #expect(workflow.contains("dflash_timed_target_selection.json"))
        #expect(workflow.contains("${MLXFAST_PRIVATE_DIR}/dflash_timed_target_selection.json"))
        // The downloaded object is checked against the SAMPLED entry's own pin,
        // so a swapped or truncated object still fails closed even though the
        // key changes per run.
        #expect(workflow.contains("MLXFAST_DFLASH_BENCH_GOLDEN_SHA256_SELECTED"))
        #expect(workflow.contains("MLXFAST_DFLASH_BENCH_GOLDEN_BYTES_SELECTED"))

        // Today the pool ships empty on purpose, which keeps the track
        // unrunnable; that is consistent with official_scoring_enabled == false.
        if pool.isEmpty {
            #expect(fixture["official_scoring_enabled"] as? Bool == false)
        }
    }

    // CONTRACT LAYER L6, second half. The pool makes a tuned threshold a 1/N
    // gamble; the rate limit stops the gamble being repeated for free. The
    // critical property is that only PARTICIPANT-attributable failures are
    // charged: if an overloaded judge API or a thermal reject consumed budget,
    // the rate limit would become a denial of service against honest submitters.
    @Test
    func attributableRankedFailuresAreRateLimited() throws {
        let workflow = try text(Self.workflowPath)

        #expect(workflow.contains("Enforce DFlash ranked-failure rate limit"))
        #expect(workflow.contains("Record attributable DFlash ranked failure"))
        #expect(workflow.contains("MLXFAST_DFLASH_MAX_RANKED_FAILURES"))
        // Counter state lives in this track's own state dir, never the serial
        // track's.
        #expect(workflow.contains("${MLXFAST_MEASURE_STATE_DIR}/failure-counts"))
        // Branch names are attacker-influenced, so the counter filename is a
        // digest rather than the ref text.
        #expect(workflow.contains("shasum -a 256 | awk '{print $1}'"))
        // Operator faults must not be charged: the recorder allowlists
        // participant categories and exits 0 for anything else.
        #expect(workflow.contains("is not participant-attributable"))
        #expect(workflow.contains("dflash_*|floor_failed|correctness_failed"))
        // Both halves must be present for either to be meaningful.
        #expect(workflow.contains("timed_prompt_pool"))
    }

    // The retired GEMMA-era MTP surface stays retired under its own names.
    // DFlash carries its own filenames precisely so reviving a speculative
    // track cannot be confused with un-retiring the old one.
    // The work-binding tolerance can be widened from the command line, because
    // calibrating it means measuring the honest gap distribution with the check
    // effectively off. A flag that widens a gate is a bypass surface, so the
    // refusal on the official path is pinned here rather than left to review.
    @Test
    func wideningTheWorkBindingToleranceIsRefusedOnTheOfficialPath() throws {
        let cli = try text("Sources/MLXFastCLI/main.swift")
        let flag = "--work-binding-tolerance-absolute"
        #expect(cli.contains(flag), "the calibration flag should exist")

        // The guard must sit between the flag being read and the tolerance being
        // constructed, so locate the override block and require the official-run
        // refusal inside it.
        let block = try #require(
            cli.range(of: flag).flatMap { start in
                cli.range(
                    of: "tolerance = DFlashWorkBindingTolerance(",
                    range: start.upperBound ..< cli.endIndex
                ).map { end in String(cli[start.upperBound ..< end.lowerBound]) }
            },
            "could not find the tolerance override block"
        )
        #expect(
            block.contains("MLXFAST_OFFICIAL_BENCHMARK_RUN"),
            """
            The DFlash work-binding tolerance override must be refused when \
            MLXFAST_OFFICIAL_BENCHMARK_RUN=1; otherwise a ranked run could be \
            scored against a widened contract check.
            """
        )

        // The per-comparison gap trace is calibration output only: on a ranked
        // run it is a per-row proximity signal against a hidden prompt, which
        // contract layer L6 keeps out of every published artifact.
        #expect(
            cli.contains("payload[\"work_binding_logit_deltas\"]")
                && cli.contains("tolerance.absolute != toleranceDefaults.absolute"),
            """
            work_binding_logit_deltas must only be published when a calibration \
            tolerance is in use.
            """
        )
    }

    @Test
    func retiredMTPNamesStayRetired() throws {
        let fm = FileManager.default
        for retired in [
            "benchmark.mtp.json",
            "benchmark-mtp.sh",
            "setup-mtp.sh",
            "fixtures/laguna_xs_2_1_mtp_track.json",
            "fixtures/mtp_laguna_xs_2_1_4bit.sha256",
            "fixtures/mtp_laguna_xs_2_1_dflash.sha256",
        ] {
            #expect(!fm.fileExists(atPath: retired), "\(retired) must stay retired")
        }
    }
}

// MARK: - Near-tie admission (contract Amendment 14)

/// A row where the reference's own top-2 ordering is inside the measured
/// build-to-build drift envelope is not evidence about the candidate. These pin
/// that such rows are admitted WITHOUT spending the residual budget, while a
/// divergence at a CONFIDENT row still spends it.
@Suite
struct DFlashNearTieAdmissionTests {
    /// Oracle returning one fixed row, so the admission branch under test is the
    /// only thing that varies.
    private struct FixedRowOracle: DFlashReferenceOracle {
        let row: DFlashReferenceRow
        func referenceRows(
            emittedPrefix: [Int],
            startOffset: Int,
            count: Int,
            declaredBlockWidth: Int
        ) throws -> [DFlashReferenceRow] {
            Array(repeating: row, count: count)
        }
    }

    private func round(emitting token: Int, offset: Int) -> DFlashObservedRound {
        DFlashObservedRound(
            requestedBlockSize: 1,
            tokens: [token],
            declaredRows: 1,
            perRowTop2Tokens: [[token, 0]],
            perRowTop2Logits: [[20.0, 10.0]],
            acceptedDraftCount: 0,
            rejectedDraftCount: 0,
            targetCacheOffset: offset,
            latencySeconds: 0.01
        )
    }

    private func validator(row: DFlashReferenceRow, tokens: Int)
        -> LagunaDFlashBlockValidator
    {
        LagunaDFlashBlockValidator(
            oracle: FixedRowOracle(row: row),
            seedTokenCount: 8,
            totalTokenCount: tokens,
            // Wide work-binding tolerance: these tests are about TOKEN
            // admissibility, not about the L2 logit comparison.
            tolerance: DFlashWorkBindingTolerance(absolute: 1e6, relative: 1e6)
        )
    }

    /// The reference cannot break a gap inside the envelope, so emitting its
    /// top-2 token is admissible and costs no residual budget.
    @Test
    func nearTieRowsAreAdmittedWithoutSpendingTheResidualBudget() throws {
        let gap = MLXFastConstants.experimentalDFlashNearTieLogitEnvelope / 2
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [20.0, 20.0 - gap]
        )
        // The FROZEN ranked window, deliberately: the per-thousand rate rounds up
        // to a single slot for any window under 25 tokens, so a toy window would
        // test the rounding rather than the admission rule. At 128 tokens the cap
        // is 6, against a measured need of 3.
        let v = validator(row: row, tokens: 128)
        // Four divergences in a row: with the old single-slot residual budget the
        // second would already have thrown.
        for i in 0 ..< 4 {
            try v.acceptStructural(round: round(emitting: 200, offset: 8 + i + 1))
        }
        // The structural half must have scored NOTHING. Admissibility needs
        // reference rows teacher-forced on the candidate's own emitted prefix,
        // which is not knowable until the run has finished; a validator that
        // reached its oracle inline is the defect this split removes.
        #expect(v.outcomes.isEmpty)
        #expect(v.referenceValidated == false)

        try v.validateJournalAgainstReference()
        #expect(v.referenceValidated)
        #expect(v.admissibleNearTieCount == 4)
        #expect(v.residualDivergenceCount == 0)
    }

    /// A confident row is a different matter: the reference has a defensible
    /// answer there, so a top-2 divergence is real drift and still consumes the
    /// small residual budget until it is exhausted.
    @Test
    func confidentRowsStillSpendTheResidualBudgetAndCanExhaustIt() throws {
        let gap = MLXFastConstants.experimentalDFlashNearTieLogitEnvelope + 5.0
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [40.0, 40.0 - gap]
        )
        let v = validator(row: row, tokens: 8)
        for i in 0 ..< 8 {
            try v.acceptStructural(round: round(emitting: 200, offset: 8 + i + 1))
        }
        var thrown: DFlashContractViolation?
        do {
            try v.validateJournalAgainstReference()
        } catch let violation as DFlashContractViolation {
            thrown = violation
        }
        #expect(v.admissibleNearTieCount == 0)
        #expect(thrown?.kind == .residualBudgetExhausted)
    }

    /// LEGACY FORM (Amendment 14). A reference row that carries no emitted-token
    /// logit -- a golden written before the field existed -- still falls back to
    /// the top-2 membership test, so a token outside the reference's top 2 is
    /// rejected there exactly as it always was.
    @Test
    func nearTieRowsWithoutAnEmittedTokenLogitFallBackToTheTopTwoTest() throws {
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [20.0, 20.0]
        )
        #expect(row.emittedTokenLogit == nil)
        let v = validator(row: row, tokens: 4)
        // Structurally the round is legal -- in range, one row, ledger consistent.
        // Fabricated output is caught by the REFERENCE pass, not by arithmetic.
        try v.acceptStructural(round: round(emitting: 999, offset: 9))
        var thrown: DFlashContractViolation?
        do {
            try v.validateJournalAgainstReference()
        } catch let violation as DFlashContractViolation {
            thrown = violation
        }
        #expect(thrown?.kind == .tokenNotAdmissible)
    }

    /// BLOCKER 4c, the row this rule exists for. Three tokens sit inside a
    /// fraction of a logit and the candidate emitted the reference's #3, so a
    /// two-member admissible set cannot express the tie. Judged on the emitted
    /// token's OWN reference logit, the row is admitted -- and costs no residual
    /// budget, because the reference cannot rank those three either.
    @Test
    func aThirdRankedTokenIsAdmittedWhenTheReferenceCannotSeparateIt() throws {
        // The measured shape at row 109 of the varied golden: reference top-2
        // [268, 85] at [14.625, 13.875], candidate emitted 1972.
        let row = DFlashReferenceRow(
            sequentialArgmax: 268,
            declaredFrameArgmax: 268,
            top2Tokens: [268, 85],
            top2Logits: [14.625, 13.875],
            top1Logit: 14.625,
            emittedToken: 1972,
            emittedTokenLogit: 13.8125
        )
        let v = validator(row: row, tokens: 128)
        try v.acceptStructural(round: round(emitting: 1972, offset: 9))
        try v.validateJournalAgainstReference()
        #expect(v.admissibleNearTieCount == 1)
        #expect(v.residualDivergenceCount == 0)
    }

    /// The envelope is still the whole control. A token the reference prices
    /// further than the envelope below its own top-1 is rejected even though the
    /// reference's top-2 pair is dead flat: the rule widened WHICH tokens a flat
    /// row can admit, not the distance any row admits.
    @Test
    func aTokenPricedBelowTheEnvelopeIsRejectedEvenAtAFlatTopTwo() throws {
        let envelope = MLXFastConstants.experimentalDFlashNearTieLogitEnvelope
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [20.0, 20.0],
            top1Logit: 20.0,
            emittedToken: 999,
            emittedTokenLogit: 20.0 - envelope - 0.5
        )
        let v = validator(row: row, tokens: 128)
        try v.acceptStructural(round: round(emitting: 999, offset: 9))
        var thrown: DFlashContractViolation?
        do {
            try v.validateJournalAgainstReference()
        } catch let violation as DFlashContractViolation {
            thrown = violation
        }
        #expect(thrown?.kind == .tokenNotAdmissible)
        #expect(v.admissibleNearTieCount == 0)
    }

    /// The emitted-token logit is only a statement about the token it was
    /// measured for. A stored golden's row was teacher-forced on the GOLDEN's
    /// chain, so its zero-distance readout must not admit a candidate token the
    /// row never described -- otherwise every divergence would pass for free the
    /// moment a golden carried the field.
    @Test
    func anEmittedTokenLogitForADifferentTokenAdmitsNothing() throws {
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [20.0, 5.0],
            top1Logit: 20.0,
            // The golden's own chain token, at zero distance from top-1.
            emittedToken: 100,
            emittedTokenLogit: 20.0
        )
        let v = validator(row: row, tokens: 128)
        try v.acceptStructural(round: round(emitting: 999, offset: 9))
        var thrown: DFlashContractViolation?
        do {
            try v.validateJournalAgainstReference()
        } catch let violation as DFlashContractViolation {
            thrown = violation
        }
        #expect(thrown?.kind == .tokenNotAdmissible)
        #expect(v.admissibleNearTieCount == 0)
    }

    /// The new predicate must SUBSUME Amendment 14's: emitting the reference's
    /// #2 token at a flat row is exactly the case where `top1 - emitted` IS the
    /// top-2 gap, so it stays admitted, and it stays capped by the same budget.
    @Test
    func theEmittedLogitRuleSubsumesTheTopTwoRule() throws {
        let gap = MLXFastConstants.experimentalDFlashNearTieLogitEnvelope / 2
        let row = DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 200],
            top2Logits: [20.0, 20.0 - gap],
            top1Logit: 20.0,
            emittedToken: 200,
            emittedTokenLogit: 20.0 - gap
        )
        let v = validator(row: row, tokens: 128)
        for i in 0 ..< 4 {
            try v.acceptStructural(round: round(emitting: 200, offset: 8 + i + 1))
        }
        try v.validateJournalAgainstReference()
        #expect(v.admissibleNearTieCount == 4)
        #expect(v.residualDivergenceCount == 0)
    }

    /// The envelope is derived, not guessed, and its basis was re-taken once the
    /// contaminated one was found (Amendment 16).
    ///
    /// Amendment 6 measured a maximum candidate-vs-reference top-2 logit delta of
    /// 3.375 against a PRE-GENERATED golden and set the envelope to 2 x 3.375 =
    /// 6.75. That statistic was the sum of candidate-vs-reference drift and the
    /// golden's own pre-generation drift. Under the live post-run replay of the
    /// candidate's own chain the same statistic is 0 at K=1 -- the serial control
    /// is bit-identical to the reference's width-1 walk -- and at most 2.4375 at
    /// K=4 across both fixtures and several schedule seeds. Reordering top-1 and
    /// top-2 needs a gap below the DIFFERENCE of two per-logit drifts, so the
    /// envelope is 2 x 2.4375 = 4.875.
    ///
    /// The L2 absolute tolerance shares the number but no longer shares the
    /// derivation: see `workBindingAbsoluteArmIsNotTightenedBecauseCrossBuildDriftDominates`.
    @Test
    func nearTieEnvelopeIsTwiceTheMeasuredWorkBindingDrift() {
        let liveMaximum = 2.4375
        #expect(
            MLXFastConstants.experimentalDFlashNearTieLogitEnvelope
                == 2 * liveMaximum
        )
        #expect(MLXFastConstants.experimentalDFlashNearTieLogitEnvelope == 4.875)
        // Neither constant may come out of a recalibration looser than what it
        // replaced: the measured basis shrank, so the gates must not widen.
        #expect(MLXFastConstants.experimentalDFlashNearTieLogitEnvelope < 6.75)
        let tolerance = DFlashWorkBindingTolerance()
        #expect(tolerance.absolute == 2 * liveMaximum)
        #expect(tolerance.absolute < 5.0)
        #expect(tolerance.relative == 0.25)
        // And the cap leaves headroom over the measured density of 2-3 near-tie
        // rows per 128 positions on varied prose.
        let budget =
            (128 * MLXFastConstants
                .experimentalDFlashNearTieAdmissionBudgetPerThousand + 999) / 1_000
        #expect(budget >= 6)
    }
}

// MARK: - What the surviving cheat exploited (contract Amendment 18)

/// A red-team run measured a verifier that ran ONE lm_head instead of K,
/// blind-accepted every draft, and copied the one computed row's work-binding
/// readouts into every other row. It passed the frozen window at the ranked block
/// width with a `max_top2_logit_delta` of 5.125 -- over the 4.875 absolute arm --
/// because the tolerance was an OR and the relative arm rescued it. These pin both
/// halves of the fix.
@Suite
struct DFlashWorkBindingHardeningTests {
    private struct FixedRowOracle: DFlashReferenceOracle {
        let row: DFlashReferenceRow
        func referenceRows(
            emittedPrefix: [Int], startOffset: Int, count: Int,
            declaredBlockWidth: Int
        ) throws -> [DFlashReferenceRow] {
            Array(repeating: row, count: count)
        }
    }

    private func validator() -> LagunaDFlashBlockValidator {
        LagunaDFlashBlockValidator(
            oracle: FixedRowOracle(
                row: DFlashReferenceRow(
                    sequentialArgmax: 100,
                    declaredFrameArgmax: 100,
                    top2Tokens: [100, 200],
                    top2Logits: [23.75, 20.0]
                )
            ),
            seedTokenCount: 8,
            totalTokenCount: 128
        )
    }

    /// The exact numbers the surviving cheat produced: 5.125 absolute against a
    /// 4.875 arm, but 5.125 / 23.75 = 0.216 against a 0.25 relative arm. Under an
    /// OR this matched. It must not.
    @Test
    func theExactDeltaThatDefeatedTheOrIsNowRejected() {
        let tolerance = DFlashWorkBindingTolerance()
        let reference = 23.75
        let candidate = reference - 5.125
        #expect(abs(candidate - reference) > tolerance.absolute)
        #expect(
            abs(candidate - reference) / reference < tolerance.relative,
            "the relative arm alone would still admit this, which is the point"
        )
        #expect(
            tolerance.matches(candidate: candidate, reference: reference) == false,
            "both arms must hold; an OR is what made L2 decorative"
        )
    }

    /// The absolute arm is NOT tightened, and this pins WHY -- so a future
    /// per-width "improvement" has to restate the measurement rather than
    /// rediscover it.
    ///
    /// Amendment 19 proposed calibrating the absolute arm per block width, on the
    /// stated basis that honest frame divergence is 0 at width 1 and grows
    /// monotonically with the verify width, which would have put the ranked width
    /// near 3.5-4.1. A 50-run, 12,800-comparison sweep with every gap attributed
    /// to its round's width measured that basis false beyond the first step, and a
    /// first-ever cross-build measurement measured the proposed arm unsafe.
    @Test
    func workBindingAbsoluteArmIsNotTightenedBecauseCrossBuildDriftDominates() {
        let tolerance = DFlashWorkBindingTolerance()

        // Honest SAME-BUILD maxima, per verify width, from the width-attributed
        // sweep. Width 1 is exactly 0; from width 2 on there is no monotone growth
        // -- 4.5 at width 6 sits above 2.5 at width 7 -- so width is not a
        // predictor and a per-width schedule has nothing to key on.
        let sameBuildMaxByWidth: [Int: Double] = [
            1: 0.0, 2: 3.125, 3: 3.375, 4: 2.4375,
            5: 3.375, 6: 4.5, 7: 2.5, 8: 3.25,
        ]
        #expect(sameBuildMaxByWidth[1] == 0.0)
        #expect(sameBuildMaxByWidth[6]! > sameBuildMaxByWidth[7]!)
        for (width, drift) in sameBuildMaxByWidth {
            #expect(
                tolerance.matches(
                    candidate: 26.0 - drift,
                    reference: 26.0
                ),
                "honest same-build drift at width \(width) must pass"
            )
        }

        // Honest CROSS-BUILD drift, from one semantics-preserving reassociation of
        // the MoE expert reduction against an otherwise identical reference build.
        // Width 1 is where same-build drift is exactly 0, so this is the
        // cross-build term isolated -- and it is the LARGEST honest number
        // measured anywhere.
        let crossBuildMaxAtWidthOne = 4.625
        #expect(crossBuildMaxAtWidthOne > sameBuildMaxByWidth[6]!)
        #expect(
            tolerance.matches(
                candidate: 26.0 - crossBuildMaxAtWidthOne,
                reference: 26.0
            ),
            "an honest cross-build submission must not be rejected"
        )

        // The arm is only two BF16 ULPs (0.125 each) above that, so the 1.5x
        // headroom discipline of Amendments 6 and 15 no longer holds against the
        // term a real submission actually produces. Tightening to Amendment 19's
        // proposed ranked-width arm would have rejected honest cross-build work.
        #expect(tolerance.absolute - crossBuildMaxAtWidthOne == 0.25)
        let amendment19ProposedRankedArm = 4.1
        #expect(amendment19ProposedRankedArm < crossBuildMaxAtWidthOne)
        #expect(amendment19ProposedRankedArm < sameBuildMaxByWidth[6]!)
    }

    /// Honest drift stays admissible.
    @Test
    func measuredHonestDriftStillPassesBothArms() {
        let tolerance = DFlashWorkBindingTolerance()
        #expect(tolerance.matches(candidate: 23.75 - 2.75, reference: 23.75))
        // Small magnitudes: the relative arm is the binding one there.
        #expect(tolerance.matches(candidate: 4.0, reference: 4.4))
        #expect(tolerance.matches(candidate: 1.0, reference: 4.0) == false)
    }

    /// Row j's logits produced emitted token j, so the candidate's own reported
    /// top-1 for row j must be that token. The surviving cheat's copied rows
    /// reported the anchor row's ids while emitting the drafter's tokens.
    @Test
    func aRowWhoseReportedTopOneIsNotItsEmittedTokenIsRejected() {
        let round = DFlashObservedRound(
            requestedBlockSize: 3,
            tokens: [100, 200],
            declaredRows: 3,
            // Row 1 reports the row-0 anchor's ids -- the fabrication signature.
            perRowTop2Tokens: [[100, 7], [100, 7], [100, 7]],
            perRowTop2Logits: [[23.75, 20.0], [23.75, 20.0], [23.75, 20.0]],
            acceptedDraftCount: 1,
            rejectedDraftCount: 1,
            targetCacheOffset: 10,
            latencySeconds: 0.02
        )
        var thrown: DFlashContractViolation?
        do { try validator().acceptStructural(round: round) }
        catch let v as DFlashContractViolation { thrown = v }
        catch {}
        #expect(thrown?.kind == .workBindingLogitMismatch)
        #expect(thrown?.step == 1, "must fire on the first fabricated row")
    }

    /// The same round with honest per-row ids passes the structural half, so the
    /// check above is specific to the fabrication and not to the shape.
    @Test
    func honestPerRowTopOnesPassTheStructuralHalf() throws {
        let round = DFlashObservedRound(
            requestedBlockSize: 3,
            tokens: [100, 200],
            declaredRows: 3,
            perRowTop2Tokens: [[100, 7], [200, 9], [301, 11]],
            perRowTop2Logits: [[23.75, 20.0], [22.0, 19.0], [21.5, 18.0]],
            // Draft 0 was accepted, so it IS the token emitted at row 0; draft 1
            // is the one the target overruled (Amendment 21's draft binding).
            draftTokens: [100, 777],
            acceptedDraftCount: 1,
            rejectedDraftCount: 1,
            targetCacheOffset: 10,
            latencySeconds: 0.02
        )
        try validator().acceptStructural(round: round)
    }
}

// MARK: - The rejected tail (contract Amendment 21)

/// L2 used to price only the EMITTED rows. Everything after the first rejected
/// draft carried per-row readouts that were length-checked and then compared to
/// nothing -- and those are exactly the rows an honest verifier must COMPUTE
/// (computing them is how it discovers the rejection) and a cheating one can
/// fabricate for free. At width 4 and 69% acceptance the ratio is ~1.25 declared
/// rows per emitted token, so the fabrication recovers the entire ~16%
/// speculation tax while emitting bit-identical tokens.
///
/// These pin the three binds that close it: the reference-free draft binding, the
/// reference's replay of the candidate's OWN verify block over the tail rows, and
/// the rejection claim itself.
@Suite
struct DFlashRejectedTailBindingTests {
    /// Oracle that answers the emitted rows AND replays a verify block.
    private struct ScriptedOracle: DFlashReferenceOracle {
        let emittedRows: [DFlashReferenceRow]
        let verifyTop2Tokens: [[Int]]?
        let verifyTop2Logits: [[Double]]?
        /// Records what the validator asked for, so a test can prove the verify
        /// block was reconstructed from the journal rather than invented.
        let observedDrafts = DraftRecorder()

        final class DraftRecorder: @unchecked Sendable {
            private(set) var drafts = [[Int]]()
            func record(_ value: [Int]) { drafts.append(value) }
        }

        func referenceRows(
            emittedPrefix: [Int],
            startOffset: Int,
            count: Int,
            declaredBlockWidth: Int
        ) throws -> [DFlashReferenceRow] {
            Array(emittedRows[startOffset ..< (startOffset + count)])
        }

        func referenceBatch(
            emittedPrefix: [Int],
            startOffset: Int,
            count: Int,
            declaredBlockWidth: Int,
            declaredRows: Int,
            draftTokens: [Int]
        ) throws -> DFlashReferenceBatch {
            observedDrafts.record(draftTokens)
            return DFlashReferenceBatch(
                rows: try referenceRows(
                    emittedPrefix: emittedPrefix,
                    startOffset: startOffset,
                    count: count,
                    declaredBlockWidth: declaredBlockWidth
                ),
                verifyBlockTop2Tokens: verifyTop2Tokens,
                verifyBlockTop2Logits: verifyTop2Logits
            )
        }
    }

    /// A pre-Amendment-21 oracle: emitted rows only, no verify-block replay. It
    /// deliberately does NOT implement `referenceBatch`, so it takes the protocol
    /// default -- which is the legacy path a stored golden falls back to.
    private struct LegacyRowsOnlyOracle: DFlashReferenceOracle {
        let emittedRows: [DFlashReferenceRow]
        func referenceRows(
            emittedPrefix: [Int],
            startOffset: Int,
            count: Int,
            declaredBlockWidth: Int
        ) throws -> [DFlashReferenceRow] {
            Array(emittedRows[startOffset ..< (startOffset + count)])
        }
    }

    // Emitted rows the candidate's tokens 100 and 200 are the argmax of. Wide
    // top-1/top-2 gaps: this model answers confidently almost everywhere (3
    // near-tie rows per 128 positions measured on varied prose).
    private static let emittedRows = [
        DFlashReferenceRow(
            sequentialArgmax: 100,
            declaredFrameArgmax: 100,
            top2Tokens: [100, 7],
            top2Logits: [23.75, 12.0]
        ),
        DFlashReferenceRow(
            sequentialArgmax: 200,
            declaredFrameArgmax: 200,
            top2Tokens: [200, 9],
            top2Logits: [22.0, 11.0]
        ),
    ]

    /// The reference's replay of the candidate's own width-4 verify block. Rows 0
    /// and 1 are the emitted ones; rows 2 and 3 are the tail rollback discarded.
    private static let referenceVerifyTokens = [
        [100, 7], [200, 9], [301, 11], [302, 12],
    ]
    private static let referenceVerifyLogits = [
        [23.75, 12.0], [22.0, 11.0], [21.5, 10.0], [20.5, 9.0],
    ]

    private func validator(
        oracle: any DFlashReferenceOracle
    ) -> LagunaDFlashBlockValidator {
        LagunaDFlashBlockValidator(
            oracle: oracle,
            seedTokenCount: 8,
            totalTokenCount: 128
        )
    }

    /// Width 4, one accepted draft, two rejected rows. `draftTokens[0] == 100`
    /// because an accepted draft IS the token emitted at its own index.
    private func round(
        perRowTop2Tokens: [[Int]] = [[100, 7], [200, 9], [301, 11], [302, 12]],
        perRowTop2Logits: [[Double]] = [
            [23.75, 12.0], [22.0, 11.0], [21.5, 10.0], [20.5, 9.0],
        ],
        draftTokens: [Int] = [100, 777, 888],
        acceptedDraftCount: Int = 1
    ) -> DFlashObservedRound {
        DFlashObservedRound(
            requestedBlockSize: 4,
            tokens: [100, 200],
            declaredRows: 4,
            perRowTop2Tokens: perRowTop2Tokens,
            perRowTop2Logits: perRowTop2Logits,
            draftTokens: draftTokens,
            acceptedDraftCount: acceptedDraftCount,
            rejectedDraftCount: 3 - acceptedDraftCount,
            targetCacheOffset: 10,
            latencySeconds: 0.02
        )
    }

    private func violation(_ body: () throws -> Void) -> DFlashContractViolation? {
        do {
            try body()
            return nil
        } catch let violation as DFlashContractViolation {
            return violation
        } catch {
            return nil
        }
    }

    // --- 1. the honest round is unaffected, and the tail is now priced -----

    @Test
    func anHonestRoundPassesAndItsRejectedTailIsActuallyPriced() throws {
        let oracle = ScriptedOracle(
            emittedRows: Self.emittedRows,
            verifyTop2Tokens: Self.referenceVerifyTokens,
            verifyTop2Logits: Self.referenceVerifyLogits
        )
        let validator = validator(oracle: oracle)
        try validator.acceptStructural(round: round())
        try validator.validateJournalAgainstReference()

        #expect(validator.rejectedRowsReferenceChecked == 2)
        #expect(validator.verifyBlockReplayedRoundCount == 1)
        // Two emitted rows plus two tail rows: the whole declared block is now
        // covered, which is what the ledger could never claim before.
        #expect(validator.referenceCheckedRowTotal == 4)
        #expect(validator.referenceCheckedRowTotal == validator.declaredRowTotal)
        #expect(validator.rejectedTailComparisonCount == 4)
        #expect(validator.maxRejectedTailLogitDelta == 0)
        // The verify block the reference replayed was built from the journalled
        // drafts, not from anything the reference chose.
        #expect(oracle.observedDrafts.drafts == [[100, 777, 888]])
    }

    /// Honest cross-build drift on the tail rows still passes -- the tail is
    /// judged by the SAME tolerance as the emitted rows, and 4.625 is the largest
    /// honest cross-build gap Amendment 20 measured.
    @Test
    func honestCrossBuildDriftOnTheTailIsAdmissible() throws {
        let drifted: [[Double]] = [
            [23.75, 12.0], [22.0, 11.0], [21.5 - 4.625, 10.0], [20.5, 9.0],
        ]
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: Self.referenceVerifyTokens,
                verifyTop2Logits: Self.referenceVerifyLogits
            )
        )
        try validator.acceptStructural(round: round(perRowTop2Logits: drifted))
        try validator.validateJournalAgainstReference()
        #expect(validator.rejectedRowsReferenceChecked == 2)
        #expect(validator.maxRejectedTailLogitDelta == 4.625)
    }

    // --- 2. the reference-free draft binding ------------------------------

    /// A worker that does not report its drafts cannot have its tail priced, so
    /// the round is refused rather than admitted unpriced.
    @Test
    func aRoundThatReportsNoDraftTokensIsRejected() {
        let thrown = violation {
            try validator(
                oracle: LegacyRowsOnlyOracle(emittedRows: Self.emittedRows)
            ).acceptStructural(round: round(draftTokens: []))
        }
        #expect(thrown?.kind == .draftTokenBindingMismatch)
        #expect(thrown?.step == 0)
    }

    /// `emitted[i] == draftTokens[i]` for every accepted draft, checkable with no
    /// reference at all: row `i + 1`'s input is `d_i`, and the accept walk emits
    /// row `i`'s own argmax, so an accepted draft equals the emitted token at that
    /// index by construction. This is what stops a worker choosing a convenient
    /// draft list for the prefix and thereby choosing the block the reference
    /// replays.
    @Test
    func anAcceptedDraftThatIsNotTheEmittedTokenIsRejected() {
        let thrown = violation {
            try validator(
                oracle: LegacyRowsOnlyOracle(emittedRows: Self.emittedRows)
            ).acceptStructural(round: round(draftTokens: [999, 777, 888]))
        }
        #expect(thrown?.kind == .draftTokenBindingMismatch)
        #expect(thrown?.step == 0, "fires at the offending row")
    }

    /// Claiming more accepted drafts than the block proposed is arithmetic.
    @Test
    func claimingMoreAcceptedDraftsThanTheBlockProposedIsRejected() {
        let thrown = violation {
            try validator(
                oracle: LegacyRowsOnlyOracle(emittedRows: Self.emittedRows)
            ).acceptStructural(
                round: round(draftTokens: [100, 777, 888], acceptedDraftCount: 4)
            )
        }
        #expect(thrown?.kind == .rowAccountingMismatch)
    }

    // --- 3. THE CHEAT: fabricated rejected-tail readouts ------------------

    /// The speed-profitable fabrication: run the per-row lm_head only until the
    /// accept walk breaks, then copy an accepted row's readouts into every
    /// rejected row. Emitted tokens are bit-identical and the emitted rows'
    /// readouts are genuine, so nothing before Amendment 21 looked at the tail.
    @Test
    func aTailFabricatedByCopyingAnAcceptedRowIsRejected() throws {
        let copiedRowZero: [[Int]] = [[100, 7], [200, 9], [100, 7], [100, 7]]
        let copiedLogits: [[Double]] = [
            [23.75, 12.0], [22.0, 11.0], [23.75, 12.0], [23.75, 12.0],
        ]
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: Self.referenceVerifyTokens,
                verifyTop2Logits: Self.referenceVerifyLogits
            )
        )
        // The structural half cannot see it: the emitted rows are honest and the
        // drafts bind. That is why this needed the reference.
        try validator.acceptStructural(
            round: round(
                perRowTop2Tokens: copiedRowZero,
                perRowTop2Logits: copiedLogits
            )
        )
        let thrown = violation {
            try validator.validateJournalAgainstReference()
        }
        #expect(thrown?.kind == .rejectedRowReadoutMismatch)
        #expect(thrown?.step == 2, "the first rejected row")
    }

    /// And the same round is admitted when the tail is NOT priced. This is the
    /// gap as it stood, and it is also the legacy fallback: an oracle that cannot
    /// replay a verify block leaves the tail unpriced rather than failing the run,
    /// and `rejectedRowsReferenceChecked` is how an audit tells the two apart.
    @Test
    func theSameFabricationPassesWhenTheTailIsNotPriced() throws {
        let validator = validator(
            oracle: LegacyRowsOnlyOracle(emittedRows: Self.emittedRows)
        )
        try validator.acceptStructural(
            round: round(
                perRowTop2Tokens: [[100, 7], [200, 9], [100, 7], [100, 7]],
                perRowTop2Logits: [
                    [23.75, 12.0], [22.0, 11.0], [23.75, 12.0], [23.75, 12.0],
                ]
            )
        )
        try validator.validateJournalAgainstReference()
        #expect(validator.rejectedRowsReferenceChecked == 0)
        #expect(validator.verifyBlockReplayedRoundCount == 0)
        #expect(validator.referenceCheckedRowTotal == 2)
        #expect(
            validator.referenceCheckedRowTotal < validator.declaredRowTotal,
            "the shortfall IS the tail that went unpriced"
        )
    }

    /// A fabrication whose tail ids happen to be right is still bound by the
    /// VALUES, under the shared tolerance -- the exact pair that defeated the old
    /// OR (5.125 against 23.75, relative 0.216).
    @Test
    func aTailWithHonestIdsIsStillBoundByTheSharedTolerance() throws {
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: Self.referenceVerifyTokens,
                verifyTop2Logits: [
                    [23.75, 12.0], [22.0, 11.0], [23.75, 10.0], [20.5, 9.0],
                ]
            )
        )
        try validator.acceptStructural(
            round: round(
                perRowTop2Logits: [
                    [23.75, 12.0], [22.0, 11.0],
                    [23.75 - 5.125, 10.0], [20.5, 9.0],
                ]
            )
        )
        let thrown = violation {
            try validator.validateJournalAgainstReference()
        }
        #expect(thrown?.kind == .rejectedRowReadoutMismatch)
        #expect(thrown?.step == 2)
    }

    /// The tail's exact-id bind is suppressed exactly where the emitted rows'
    /// near-tie admission is: at a row the REFERENCE cannot rank. No new constant,
    /// and the values still have to hold.
    @Test
    func aTailIdDisagreementAtAFlatReferenceRowIsNotAViolation() throws {
        let flat: [[Double]] = [
            [23.75, 12.0], [22.0, 11.0], [21.5, 21.0], [20.5, 9.0],
        ]
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: Self.referenceVerifyTokens,
                verifyTop2Logits: flat
            )
        )
        try validator.acceptStructural(
            round: round(
                // Row 2 reports the reference's #2 token as its top-1 -- which is
                // what a differently-ordered accumulation produces at a flat row.
                perRowTop2Tokens: [[100, 7], [200, 9], [11, 301], [302, 12]],
                perRowTop2Logits: flat
            )
        )
        try validator.validateJournalAgainstReference()
        #expect(validator.rejectedRowsReferenceChecked == 2)
    }

    // --- 4. the rejection claim itself ------------------------------------

    /// A round that claims draft `i` was overruled, at a row where the reference's
    /// own argmax in the candidate's verify block IS that draft, has asserted a
    /// rejection the reference denies.
    @Test
    func aRejectionTheReferenceContradictsIsRejected() throws {
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                // Row 1 is the first rejected draft's row, and the reference says
                // its argmax is 777 -- the very draft the candidate overruled.
                verifyTop2Tokens: [[100, 7], [777, 9], [301, 11], [302, 12]],
                verifyTop2Logits: Self.referenceVerifyLogits
            )
        )
        try validator.acceptStructural(round: round())
        let thrown = violation {
            try validator.validateJournalAgainstReference()
        }
        #expect(thrown?.kind == .fabricatedRejection)
        #expect(thrown?.step == 1, "the first rejected draft's own row")
    }

    /// ...but not at a row the reference cannot rank, where "the reference says
    /// 777" is not a fact about the candidate.
    @Test
    func aContradictedRejectionAtAFlatReferenceRowIsAdmitted() throws {
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: [[100, 7], [777, 200], [301, 11], [302, 12]],
                verifyTop2Logits: [
                    [23.75, 12.0], [22.0, 21.5], [21.5, 10.0], [20.5, 9.0],
                ]
            )
        )
        try validator.acceptStructural(round: round())
        try validator.validateJournalAgainstReference()
        #expect(validator.rejectedRowsReferenceChecked == 2)
    }

    /// A fully-accepted round has no rejected tail, so nothing is priced and
    /// nothing is claimed -- the checks must not invent a violation there.
    @Test
    func aFullyAcceptedRoundHasNoTailToPrice() throws {
        let validator = validator(
            oracle: ScriptedOracle(
                emittedRows: Self.emittedRows,
                verifyTop2Tokens: [[100, 7], [200, 9]],
                verifyTop2Logits: [[23.75, 12.0], [22.0, 11.0]]
            )
        )
        try validator.acceptStructural(
            round: DFlashObservedRound(
                requestedBlockSize: 2,
                tokens: [100, 200],
                declaredRows: 2,
                perRowTop2Tokens: [[100, 7], [200, 9]],
                perRowTop2Logits: [[23.75, 12.0], [22.0, 11.0]],
                draftTokens: [100],
                acceptedDraftCount: 1,
                rejectedDraftCount: 0,
                targetCacheOffset: 10,
                latencySeconds: 0.02
            )
        )
        try validator.validateJournalAgainstReference()
        #expect(validator.rejectedRowsReferenceChecked == 0)
        #expect(validator.verifyBlockReplayedRoundCount == 1)
        #expect(validator.referenceCheckedRowTotal == 2)
    }
}

// MARK: - The decode floor must not drift between its declarations

/// The floor lives in four places and only one of them rejects (the box wrapper's
/// `MIN_ACCEPTED_SPEEDUP`, which no test can reach). These pin the two the repo
/// owns, so a change to one cannot silently disagree with the other, and pin the
/// VALUE so a change is a deliberate edit to this test rather than a drift.
@Suite
struct DFlashDecodeFloorTests {
    private func json(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    /// 0.80, set by operator decision 2026-07-30. NOT 1.0: block decode costs ~16%
    /// on realistic prose, so an unmodified candidate measures 0.840x, and a 1.0
    /// floor would reject every honest sub-19% kernel win instead of rejecting
    /// regressions. Raising this back to 1.0 makes the track reject its own purpose.
    @Test
    func decodeFloorIsEightyHundredthsInBothManifests() throws {
        let manifest = try json("benchmark.dflash.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let floor = try #require(scoring["decodeSpeedupFloor"] as? Double)
        #expect(floor == 0.80)

        let fixture = try json("fixtures/laguna_xs_2_1_dflash_track.json")
        let proposed = try #require(fixture["proposed_scoring"] as? [String: Any])
        let text = try #require(proposed["component_floor"] as? String)
        #expect(
            text.contains(">= 0.80"),
            "the fixture's component_floor must state the same number the manifest enforces"
        )
        // The derivation has to travel with the number, or the next reader "fixes"
        // it back to 1.0.
        #expect(text.contains("0.840x"))
        #expect(text.contains("MAX_PLAUSIBLE_SPEEDUP stays 5.0"))
    }

    /// The workflow ALSO carries the floor as an env value and recomputes the score
    /// against it in a trusted shell -- a fifth site, found only when a real
    /// dispatch printed the whole env block. A 1.0 here would silently override the
    /// 0.80 landed everywhere else at go-live.
    @Test
    func workflowEnvFloorMatchesTheManifests() throws {
        let workflow = try String(
            contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8
        )
        #expect(workflow.contains("MLXFAST_DFLASH_DECODE_SPEEDUP_FLOOR: \"0.80\""))
        #expect(!workflow.contains("MLXFAST_DFLASH_DECODE_SPEEDUP_FLOOR: \"1.0\""))
    }

    /// The workflow's own header comments are what a reviewer reads first, so they
    /// must not still advertise the old floor.
    @Test
    func workflowCommentsDoNotStillAdvertiseAFloorOfOne() throws {
        let workflow = try String(
            contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8
        )
        #expect(workflow.contains("floor: aggregate >= 0.80"))
        #expect(!workflow.contains("floor: aggregate >= 1.0"))
        #expect(!workflow.contains("floor >= 1.0 on that aggregate"))
    }
}

// MARK: - The go-live runbook the workflow points at

/// `.github/workflows/dflash-benchmark.yml` refers operators to a "DFlash go-live
/// runbook" from four places, including a specific "step B" cited in an error
/// message an operator only sees when a dispatch fails closed. That reference
/// dangled at nothing until 2026-07-30. This pins its existence and the sections
/// the workflow names, so the pointer cannot rot again.
@Suite
struct DFlashGoLiveRunbookTests {
    private static let path = "docs/dflash-go-live-runbook.md"

    @Test
    func runbookExistsAndCoversTheStepsTheWorkflowCites() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // Steps named in the workflow's own error text.
        #expect(runbook.contains("Step A"))
        #expect(runbook.contains("Step B"))
        // The two trusted-contract fields the enablement guard requires.
        #expect(runbook.contains("official_scoring_enabled"))
        #expect(runbook.contains("publication_allowed"))
        // The blocking prerequisite the pool-selection step fails closed on.
        #expect(runbook.contains("timed_prompt_pool"))
    }

    /// The runbook must carry the measured limits an operator is accepting by
    /// flipping the switch, not just the mechanical steps. These are the ones that
    /// took a full day of measurement to establish and would otherwise be
    /// rediscovered the hard way.
    @Test
    func runbookRecordsTheLimitsBeingAcceptedAtGoLive() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // An unmodified candidate scores ~0.87, so the floor is not parity.
        #expect(runbook.contains("0.87"))
        // Draft acceptance collapses on realistic prose, which is why the pool
        // must be varied.
        #expect(runbook.contains("69%"))
        // The frequency floor false-reject, with the real throttle figures.
        #expect(runbook.contains("1447-1455"))
        // The guard is expected to fail closed before step D.
        #expect(runbook.lowercased().contains("fail-closed")
            || runbook.lowercased().contains("fails closed"))
    }
}
