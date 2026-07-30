import CryptoKit
import Foundation
import Testing

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
        // Distinct concurrency domain so a DFlash run cannot cancel or queue
        // behind a serial ranked run under the same key.
        #expect(workflow.contains("mlxfast-dflash-ranked-"))

        // ...and the serial pipeline stays ignorant of DFlash.
        let serial = try text(".github/workflows/benchmark.yml")
        #expect(!serial.lowercased().contains("dflash"))
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

    // The retired GEMMA-era MTP surface stays retired under its own names.
    // DFlash carries its own filenames precisely so reviving a speculative
    // track cannot be confused with un-retiring the old one.
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
