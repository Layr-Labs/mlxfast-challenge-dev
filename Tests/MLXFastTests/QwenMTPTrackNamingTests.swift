import Foundation
import Testing

// NAMING LAW FOR THE QWEN 3.6 NATIVE-MTP TRACK (qwen3.6-27b-mtp-v1).
//
// The organizer's rule is that no retired Gemma/Laguna MTP surface name may
// survive anywhere in the repository, as a substring, in any new name. The
// existing guard is DFlashTrackTests' `retiredMTPNamesStayRetired` (the retired
// FILES stay deleted) plus BenchmarkScriptTests'
// `localDFlashScriptsNameNoRetiredMTPSurface` (the local DFlash scripts do not
// NAME them). Both were written for the DFlash track and neither knows about
// this one.
//
// Renaming must be TESTED, not just done, and the test has to run in BOTH
// directions:
//
//   NEGATIVE  every entry of `retiredMTPNames` must still trip the guard's own
//             substring check. A track that "passes the guard" because the
//             guard was quietly narrowed has proven nothing.
//   POSITIVE  every name this track introduces must be proven free of all of
//             them -- enumerated here by name, so a future rename is caught by
//             this file rather than by a reviewer reading a diff.
//
// THE TRAP THIS FILE EXISTS FOR. The retired names are matched as SUBSTRINGS,
// so a "qwen-" prefix does NOT make a retired name new: the natural spellings
// for this track's workflow, its probe verb and its reference-row artifacts all
// reproduce a retired string with a prefix in front of it. Only changing what
// follows "mtp-" clears the rule. `retiredNamesAreNotClearedByAQwenPrefix`
// below pins exactly that, so the reasoning survives even if everyone who
// remembers it has moved on.
//
// Nothing here loads a model, runs a benchmark or needs a network: every
// assertion reads checked-in text.
@Suite
struct QwenMTPTrackNamingTests {
    private typealias S = DFlashGateTextSupport

    // MARK: - The guard's own check

    /// The substring check `localDFlashScriptsNameNoRetiredMTPSurface` applies,
    /// factored out so both directions below run the SAME predicate. If that
    /// guard's matching rule ever changes, this mirror must change with it --
    /// which is the point: two different notions of "names a retired surface"
    /// would let a name pass one and fail the other.
    private static func retiredNames(in text: String) -> [String] {
        S.retiredMTPNames.filter { text.contains($0) }
    }

    // MARK: - The new surface

    /// Every name this track introduces, labelled. The label is what a failure
    /// message says, so it names the thing to rename rather than the string.
    ///
    /// ADD TO THIS LIST WHEN THE TRACK GROWS A SURFACE. It is the enumeration
    /// the operator asked for: the value of the positive case comes entirely
    /// from its completeness, and a name that is not listed here is a name
    /// nobody is checking.
    static let newSurfaceNames: [(label: String, name: String)] = [
        // --- the four checked-in files -----------------------------------
        ("ranked workflow filename", "qwen-mtp-ranked-benchmark.yml"),
        ("ranked workflow path", ".github/workflows/qwen-mtp-ranked-benchmark.yml"),
        ("ranked workflow name", "qwen-mtp-ranked-benchmark"),
        // The provisioning + R2-key-probe workflows. The probe file dodges the
        // retired names deliberately: putting the probe verb directly after
        // "mtp-" would reproduce a retired substring (this file is scanned WHOLE,
        // so the retired string is described, never spelled), so the verb is
        // spelled "r2-key-probe" -- probe LAST -- and the file is
        // qwen-mtp-r2-key-probe.yml.
        ("provisioning workflow filename", "qwen-mtp-provision-goldens.yml"),
        ("provisioning workflow path", ".github/workflows/qwen-mtp-provision-goldens.yml"),
        ("provisioning workflow name", "qwen-mtp-provision-goldens"),
        ("r2-key-probe workflow filename", "qwen-mtp-r2-key-probe.yml"),
        ("r2-key-probe workflow path", ".github/workflows/qwen-mtp-r2-key-probe.yml"),
        ("r2-key-probe workflow name", "qwen-mtp-r2-key-probe"),
        // The trusted-context guards the two credentialled workflows invoke BY
        // PATH. A script path is as hard to rename as a workflow filename --
        // the `run:` line names it literally and a rename that misses one end
        // fails the job with exit 127 -- so both are enumerated here, and the
        // probe guard's own verb placement matters for the same reason the
        // probe workflow's does: the verb is spelled "r2-probe", never directly
        // after "mtp-".
        (
            "provisioning trusted-context guard",
            ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh"
        ),
        (
            "r2-key-probe trusted-context guard",
            ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh"
        ),
        ("local benchmark runner", "benchmark-qwen-mtp.sh"),
        ("local setup runner", "setup-qwen-mtp.sh"),
        ("track manifest filename", "benchmark.qwen-mtp.json"),
        // --- track identity ------------------------------------------------
        ("track id", "qwen3.6-27b-mtp-v1"),
        ("track name", "mlxfast-challenge-dev-qwen-mtp"),
        ("leaderboard namespace", "qwen3.6-27b-mtp-v1"),
        ("timed decode target id", "lowsim-prose-qwen-v1"),
        ("calibration-ready interlock", "MLXFAST_QWEN_MTP_CALIBRATION_READY"),
        ("workflow env prefix", "MLXFAST_QWEN_MTP_"),
        // --- CLI verbs -------------------------------------------------------
        // The natural spellings -- appending "probe", "benchmark" or
        // "reference" directly after "mtp-" -- are all retired entries.
        ("untimed verify verb", "mtp-verify"),
        ("timed measurement verb", "mtp-timed"),
        ("canonical depth flag", "--mtp-depth"),
        ("head tree flag", "--mtp-head"),
        // --- runtime-worker surface (phase 2) --------------------------------
        // The worker subcommand and the five protocol kinds. Wire strings are
        // surface names too: a retired substring in a request kind would survive
        // in every transcript and every sandbox profile that names it.
        ("runtime worker subcommand", "mtp-runtime-worker"),
        ("worker warm kind", "mtp_decode_warm"),
        ("worker begin kind", "mtp_decode_begin"),
        ("worker round kind", "mtp_decode_round"),
        ("worker reference prefill kind", "mtp_reference_prefill"),
        ("worker reference rows kind", "mtp_reference_rows"),
        ("head directory environment variable", "MLXFAST_QWEN_MTP_HEAD_DIR"),
        // --- Swift sources this track adds -----------------------------------
        ("worker hot-path source", "Qwen36MTPBlockSession.swift"),
        ("head attachment source", "Qwen36MTPHeadAttachment.swift"),
        ("serial reference source", "Qwen36MTPReferenceSession.swift"),
        ("worker harness source", "QwenRuntimeMTPWorker.swift"),
        ("model-surface protocol source", "Qwen36MTPTarget.swift"),
        ("model-surface protocol", "Qwen36MTPTarget"),
        ("backbone layout test suite source", "QwenMTPBackboneLayoutTests.swift"),
        ("metallib all-roots flag", "--all-build-roots"),
        ("pinned reference environment variable", "MLXFAST_QWEN_MTP_TARGET_DIR"),
        ("trusted contract source", "QwenRuntimeMTP.swift"),
        ("trusted driver source", "QwenRuntimeMTPDriver.swift"),
        ("verb test suite source", "QwenMTPVerbTests.swift"),
        ("rollback test suite source", "QwenMTPRollbackContractTests.swift"),
        ("trusted-context guard test suite source", "QwenMTPTrustedContextGuardTests.swift"),
        // --- payload fields ---------------------------------------------------
        // The evidence payload's own key names, enumerated for the same reason:
        // they are read by a box-owned wrapper and a workflow, so they are as
        // hard to rename as a filename.
        ("native head boolean", "uses_native_mtp_head"),
        ("pinned head boolean", "uses_pinned_mtp_head"),
        ("head attachment boolean", "mtp_head_attached"),
        ("depth field", "mtp_depth"),
        ("serial control depth field", "serial_control_depth"),
        ("row ledger field", "row_ledger"),
        ("parity gate field", "parity_all_ok"),
        // --- box-owned and per-track paths -----------------------------------
        ("measure wrapper filename", "measure-qwen-mtp-job.sh"),
        ("per-track state dir", "/opt/bench-runner/state/qwen3.6-27b-mtp-v1"),
        ("per-track state dir leaf", "qwen3.6-27b-mtp-v1"),
        ("per-track baseline dir", "/opt/bench-runner/baseline/qwen3.6-27b-mtp-v1/current"),
        ("per-track baseline calibration", "/opt/bench-runner/state/qwen3.6-27b-mtp-v1/baseline-calibration.json"),
        ("MTP head cache dir", "/opt/bench-runner/cache/qwen36/qwen3.6-27b-mtp-v1/mtp-head"),
        ("runner label", "m5-qwen3.6-27b-mtp"),
        ("bench job workspace", "/Users/Shared/bench-jobs/qwen-mtp-ranked-current"),
        // --- fixtures -------------------------------------------------------
        ("track contract fixture", "fixtures/qwen3_6_27b_mtp_track.json"),
        ("MTP head manifest fixture", "fixtures/qwen3_6_27b_mtp_head.sha256"),
        ("target manifest fixture", "fixtures/reference_qwen3_6_27b_4bit.sha256"),
        // --- workflow-internal names ----------------------------------------
        ("hidden-golden staging dir", ".qwen-mtp-ranked-src"),
        ("transformed weights hash artifact", "qwen-mtp-weights.sha256"),
        ("parity gate report artifact", "qwen-mtp-gates-report.json"),
        ("paired results artifact", "qwen-mtp-paired-results.json"),
        ("measure verdict artifact", "qwen-mtp-measure-verdict.txt"),
        ("correctness artifact bundle", "qwen-mtp-correctness-results"),
        ("audit artifact bundle", "qwen-mtp-ranked-audit"),
        ("redacted failure bundle", "qwen-mtp-ranked-failure"),
        ("bench job id prefix", "qwen-mtp-ranked-"),
        ("score mode", "qwen-mtp-paired-decode-only"),
        // --- local runner names ----------------------------------------------
        ("local scratch root", ".mlxfast-local-qwen-mtp"),
        ("local reference rows file", "qwen-mtp-golden-rows.json"),
        ("local head cache root", "~/.cache/mlxfast/qwen3.6-27b-mtp-v1"),
        ("local head cache leaf", "mtp-head"),
        ("local score oracle label", "candidate-local-mtp-golden-rows"),
    ]

    /// The checked-in files this track adds. Scanned WHOLE -- comments
    /// included, which is stricter than `DFlashGateTextSupport.executable`
    /// permits for the DFlash workflow. That is deliberate for a brand-new
    /// surface: there is no historical comment here that has to mention a dead
    /// name, so the strongest available check costs nothing, and it keeps a
    /// retired name from creeping in as "just a comment" and later being copied
    /// into a value.
    ///
    /// THIS FILE IS IN THE LIST. It is a new file too, and a naming test that
    /// exempted itself would be free to spell a retired name in a comment and
    /// then have that comment copied into a value. Every retired string this
    /// suite needs is built at run time from `retiredMTPNames`, never spelled
    /// here, so the self-check costs nothing and closes the loop.
    static let newSurfaceFiles = [
        ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        ".github/workflows/qwen-mtp-provision-goldens.yml",
        ".github/workflows/qwen-mtp-r2-key-probe.yml",
        // The two trusted-context guards. Scanned WHOLE like everything else
        // here: their headers explain themselves at length by comparison with
        // the DFlash twins, which is exactly the kind of prose that reaches for
        // a dead name.
        ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh",
        ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh",
        "benchmark-qwen-mtp.sh",
        "setup-qwen-mtp.sh",
        "benchmark.qwen-mtp.json",
        "fixtures/qwen3_6_27b_mtp_head.sha256",
        "Tests/MLXFastTests/QwenMTPTrackNamingTests.swift",
        // Phase 2: the runtime and verb sources. Scanned WHOLE, comments
        // included -- these files talk ABOUT the retired surface (the DFlash
        // rollback, the retired row equation) and must do so without ever
        // spelling a retired name.
        "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift",
        "Sources/MLXFastModel/Qwen36MTPReferenceSession.swift",
        "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift",
        "Sources/MLXFastModel/Qwen36MTPTarget.swift",
        "Tests/MLXFastTests/QwenMTPBackboneLayoutTests.swift",
        "Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift",
        "Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift",
        "Tests/MLXFastTests/QwenMTPVerbTests.swift",
        "Tests/MLXFastTests/QwenMTPRollbackContractTests.swift",
        "Tests/MLXFastTests/QwenMTPTrustedContextGuardTests.swift",
    ]

    // MARK: - NEGATIVE: the guard still catches every retired name

    /// Each retired name must still be caught by the guard's own check. This is
    /// the half that keeps the positive case honest: "no retired name found" is
    /// only evidence if the finder still finds them.
    @Test
    func everyRetiredMTPNameStillTripsTheGuard() throws {
        #expect(
            !S.retiredMTPNames.isEmpty,
            "the retired-name list is empty; the guard cannot catch anything"
        )
        for retired in S.retiredMTPNames {
            // A synthetic fixture in the shape the guard actually scans: a
            // retired name embedded in surrounding text, not the bare string.
            let fixture = """
                #!/usr/bin/env bash
                # a script that reaches for a dead surface
                exec mlxfast-swift \(retired) --weights weights
                """
            let caught = Self.retiredNames(in: fixture)
            #expect(
                caught.contains(retired),
                """
                the retired MTP name '\(retired)' no longer trips the guard's \
                substring check. Every entry must stay catchable: a guard that \
                has been narrowed makes every "no retired name found" result \
                below vacuous.
                """
            )
        }
    }

    /// The retired list must not SHRINK. The Qwen track was named to dodge the
    /// guard rather than to weaken it (the plan's phase 4 records that the
    /// tripwire may not need weakening at all), so a smaller list means someone
    /// took the other route -- deleting an entry instead of renaming a surface.
    @Test
    func theRetiredMTPNameListDoesNotShrink() throws {
        #expect(
            S.retiredMTPNames.count >= 12,
            """
            DFlashGateTextSupport.retiredMTPNames now holds \
            \(S.retiredMTPNames.count) entries, fewer than the 12 that were \
            retired with the Gemma/Laguna MTP track. The Qwen-MTP track is named \
            to PASS this guard unweakened; if a new surface collides, rename the \
            surface, do not delete the entry.
            """
        )
        // Duplicates would inflate the count above without widening coverage.
        #expect(
            Set(S.retiredMTPNames).count == S.retiredMTPNames.count,
            "retiredMTPNames contains duplicate entries"
        )
    }

    /// THE SUFFIX RULE, pinned. Prefixing a retired name with this track's
    /// prefix does NOT produce a new name: the retired string survives as a
    /// substring and the guard still fires. This is the single most repeatable
    /// mistake available here, and it is why this track's workflow filename does
    /// not put "-benchmark" directly after "mtp-" and its untimed verb does not
    /// end in "-probe": prefixing either of those with "qwen-" would have left a
    /// retired string intact inside the new name.
    @Test
    func retiredNamesAreNotClearedByAQwenPrefix() throws {
        for retired in S.retiredMTPNames {
            for prefix in ["qwen-", "qwen3.6-", "qwen36-"] {
                let prefixed = prefix + retired
                #expect(
                    !Self.retiredNames(in: prefixed).isEmpty,
                    """
                    '\(prefixed)' was expected to STILL trip the guard: prefixing \
                    a retired name does not clear it, because the check is a \
                    substring match. If this ever passes, the matching rule \
                    changed and every name in newSurfaceNames must be re-derived.
                    """
                )
            }
        }
    }

    // MARK: - POSITIVE: nothing this track introduces names a retired surface

    /// Every enumerated new surface name is free of every retired name.
    @Test
    func newQwenMTPSurfaceNamesCarryNoRetiredMTPName() throws {
        for (label, name) in Self.newSurfaceNames {
            let collisions = Self.retiredNames(in: name)
            #expect(
                collisions.isEmpty,
                """
                the \(label) '\(name)' contains the retired MTP name(s) \
                \(collisions.joined(separator: ", ")). Rename the surface -- \
                note that a "qwen-" prefix does not clear a retired name, only \
                changing what FOLLOWS "mtp-" does.
                """
            )
        }
    }

    /// The four checked-in files carry no retired name anywhere in their text.
    /// This is the file-level twin of `localDFlashScriptsNameNoRetiredMTPSurface`
    /// for this track, and it catches names the enumeration above forgot.
    @Test
    func newQwenMTPFilesCarryNoRetiredMTPName() throws {
        for path in Self.newSurfaceFiles {
            let body = try S.text(path)
            let collisions = Self.retiredNames(in: body)
            #expect(
                collisions.isEmpty,
                """
                \(path) names the retired MTP surface(s) \
                \(collisions.joined(separator: ", ")). Every one of these is a \
                subcommand that does not exist, a file that was deleted, or a \
                track id nothing will score.
                """
            )
        }
    }

    // MARK: - Anti-vacuity: the enumeration describes the real files

    /// An enumeration of names is worth nothing if the files use different
    /// ones. Pin each load-bearing name to the file that must actually contain
    /// it, so renaming a surface without updating this list fails HERE rather
    /// than passing a check of strings nobody uses.
    @Test
    func theEnumeratedNamesAreTheOnesTheFilesUse() throws {
        let workflow = try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        let provisionWorkflow = try S.text(
            ".github/workflows/qwen-mtp-provision-goldens.yml")
        let probeWorkflow = try S.text(".github/workflows/qwen-mtp-r2-key-probe.yml")
        let benchmarkRunner = try S.text("benchmark-qwen-mtp.sh")
        let setupRunner = try S.text("setup-qwen-mtp.sh")
        let manifest = try S.text("benchmark.qwen-mtp.json")
        let cli = try S.text("Sources/MLXFastCLI/main.swift")
        let workerCLI = try S.text("Sources/MLXFastRuntimeWorkerCLI/main.swift")
        let workerHarness = try S.text(
            "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift")

        let expectations: [(needle: String, haystack: String, file: String)] = [
            ("qwen3.6-27b-mtp-v1", workflow, "the workflow"),
            ("qwen3.6-27b-mtp-v1", manifest, "the manifest"),
            ("qwen3.6-27b-mtp-v1", benchmarkRunner, "the local benchmark runner"),
            ("mlxfast-challenge-dev-qwen-mtp", manifest, "the manifest"),
            ("MLXFAST_QWEN_MTP_CALIBRATION_READY", workflow, "the workflow"),
            ("measure-qwen-mtp-job.sh", workflow, "the workflow"),
            ("/opt/bench-runner/state/qwen3.6-27b-mtp-v1", workflow, "the workflow"),
            ("m5-qwen3.6-27b-mtp", workflow, "the workflow"),
            ("mtp-verify", workflow, "the workflow"),
            ("mtp-timed", workflow, "the workflow"),
            ("mtp-verify", benchmarkRunner, "the local benchmark runner"),
            ("mtp-timed", benchmarkRunner, "the local benchmark runner"),
            ("benchmark-qwen-mtp.sh", manifest, "the manifest"),
            ("setup-qwen-mtp.sh", manifest, "the manifest"),
            ("qwen-mtp-ranked-benchmark.yml", manifest, "the manifest"),
            ("qwen-mtp-ranked-benchmark.yml", benchmarkRunner, "the local benchmark runner"),
            ("setup-qwen-mtp.sh", benchmarkRunner, "the local benchmark runner"),
            ("fixtures/qwen3_6_27b_mtp_head.sha256", setupRunner, "the local setup runner"),
            // Phase 2: the verbs, the worker subcommand and the canonical depth
            // flag have to be the names the shipped code actually uses.
            ("--mtp-depth", workflow, "the workflow"),
            ("--mtp-depth", benchmarkRunner, "the local benchmark runner"),
            ("--mtp-head", workflow, "the workflow"),
            ("--mtp-head", benchmarkRunner, "the local benchmark runner"),
            ("mtp-verify", cli, "the trusted CLI"),
            ("mtp-timed", cli, "the trusted CLI"),
            ("mtp-runtime-worker", workerCLI, "the runtime-worker CLI"),
            ("mtp_decode_round", workerHarness, "the worker harness"),
            ("mtp_reference_rows", workerHarness, "the worker harness"),
            ("uses_native_mtp_head", cli, "the trusted CLI"),
            ("uses_pinned_mtp_head", cli, "the trusted CLI"),
            ("row_ledger", cli, "the trusted CLI"),
            ("parity_all_ok", cli, "the trusted CLI"),
            // The trusted-context guards: each credentialled workflow must name
            // its own guard by path. A rename that misses the `run:` line fails
            // the job with exit 127, which is how both of these shipped before
            // the scripts existed at all.
            (
                ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh",
                provisionWorkflow,
                "the provisioning workflow"
            ),
            (
                ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh",
                probeWorkflow,
                "the r2-key-probe workflow"
            ),
        ]
        for expectation in expectations {
            #expect(
                expectation.haystack.contains(expectation.needle),
                """
                \(expectation.file) no longer contains '\(expectation.needle)'. \
                Either the surface was renamed without updating \
                QwenMTPTrackNamingTests.newSurfaceNames, or this list is \
                checking a string nothing uses -- both make the naming guard \
                vacuous.
                """
            )
        }

        // Every needle above must itself be an enumerated surface name (or a
        // prefix of one), so the two lists cannot drift into checking disjoint
        // sets of strings.
        let enumerated = Set(Self.newSurfaceNames.map(\.name))
        for expectation in expectations {
            #expect(
                enumerated.contains(expectation.needle)
                    || enumerated.contains(where: { $0.hasSuffix(expectation.needle) })
                    || enumerated.contains(where: { $0.hasPrefix(expectation.needle) }),
                """
                '\(expectation.needle)' is checked against \(expectation.file) \
                but is not listed in newSurfaceNames, so it is never checked \
                against the retired-name list.
                """
            )
        }
    }

    /// The manifest and the workflow must agree on the track identity. A
    /// mismatch is not merely cosmetic: the workflow refuses a dispatch whose
    /// contract track_id differs from its own, and the leaderboard namespace is
    /// what keeps this track's scores out of another track's ranking.
    @Test
    func theManifestAndWorkflowAgreeOnTrackIdentity() throws {
        let manifest = try S.json("benchmark.qwen-mtp.json")
        #expect(manifest["trackId"] as? String == "qwen3.6-27b-mtp-v1")
        #expect(manifest["name"] as? String == "mlxfast-challenge-dev-qwen-mtp")
        #expect(manifest["staticReviewTrackId"] as? String == "qwen3.6-27b-mtp-v1")
        let leaderboard = try #require(manifest["leaderboard"] as? [String: Any])
        #expect(leaderboard["namespace"] as? String == "qwen3.6-27b-mtp-v1")
        let runner = try #require(manifest["runner"] as? [String: Any])
        #expect(runner["workflow"] as? String == "qwen-mtp-ranked-benchmark.yml")
        let benchmarkCommand = try #require(manifest["benchmarkCommand"] as? [String])
        #expect(
            benchmarkCommand.contains { $0.contains("./benchmark-qwen-mtp.sh") },
            "benchmarkCommand must drive ./benchmark-qwen-mtp.sh, got \(benchmarkCommand)"
        )

        let workflowEnvironment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        )
        #expect(workflowEnvironment["MLXFAST_QWEN_MTP_TRACK_ID"] == "qwen3.6-27b-mtp-v1")
        #expect(
            workflowEnvironment["MLXFAST_QWEN_MTP_TRACK_NAME"]
                == manifest["name"] as? String
        )
        // The editable-surface contract the workflow passes to BOTH the surface
        // enforcement and the overlay must be THIS manifest. Naming a different
        // file in one of the two silently drops files the other accepted.
        #expect(
            workflowEnvironment["MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT"]
                == "benchmark.qwen-mtp.json"
        )
    }

    /// The track is LIVE (operator go-live 2026-08-13, on the qwen36-mtp-track
    /// ref; the merge to main is deferred). This suite used to pin the INERT
    /// posture; it now pins the ARMED one, which is the same discipline pointed
    /// the other way. Each of these is still a one-character edit away from
    /// flipping, so taking the track back OFFLINE means editing this test in the
    /// same commit as the flags -- that friction is deliberate and runs in both
    /// directions (NEW-MODEL-BRINGUP 7.5, docs/qwen-mtp-go-live-runbook.md).
    @Test
    func theQwenMTPTrackIsLive() throws {
        let workflow = try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        let environment = try S.jobEnvironment(workflow)

        // 1. The calibration-ready interlock is a hard-pinned literal "1" --
        //    never an expression, never a dispatch input. It attests that the
        //    model-derived private fixtures and the final-SHA on-box baseline +
        //    calibration are installed on m5-max-128gb-3.
        #expect(environment["MLXFAST_QWEN_MTP_CALIBRATION_READY"] == "1")
        #expect(!workflow.contains("MLXFAST_QWEN_MTP_CALIBRATION_READY: ${{"))
        // ... and the runbook-shaped refusal message exists, scanned on the
        // comment-stripped view so a mention in prose cannot satisfy it.
        #expect(
            S.executable(workflow).contains(
                "Qwen 3.6 MTP model-derived fixtures and final-SHA baseline are not calibrated"
            )
        )

        // 2. THE TRUSTED CONTRACT IS ARMED.
        //
        //    "Enforce Qwen-MTP track enablement" refuses a RANKED dispatch while
        //    EITHER of these is false, and a gates-only dry run publishes no
        //    score. Both were flipped to true at go-live on 2026-08-13, once the
        //    R2 objects these pins name were uploaded and re-verified by digest
        //    on download. Flipping either back to false takes the track offline
        //    and MUST edit this test in the same commit.
        let contract = try S.json("fixtures/qwen3_6_27b_mtp_track.json")
        #expect(
            contract["official_scoring_enabled"] as? Bool == true,
            "official_scoring_enabled must be true while the track is live"
        )
        let baseline = contract["reference_baseline"] as? [String: Any]
        #expect(
            baseline?["publication_allowed"] as? Bool == true,
            "reference_baseline.publication_allowed must be true while the track is live"
        )
        #expect(
            contract["track_id"] as? String == "qwen3.6-27b-mtp-v1",
            "the contract track_id must match the workflow's track"
        )
        // The pool the normalised floor divides by: 8 entries, all distinct.
        let pool = contract["timed_prompt_pool"] as? [[String: Any]]
        #expect(pool?.count == 8, "the timed prompt pool must carry 8 entries")
        #expect(
            Set((pool ?? []).compactMap { $0["sha256"] as? String }).count == 8,
            "the timed prompt pool must name 8 DISTINCT targets"
        )
        #expect(
            (pool ?? []).allSatisfy {
                ($0["noop_decode_speedup"] as? Double).map { $0 > 0 } ?? false
            },
            "every pool entry needs a positive measured no-op reference"
        )

        //    The marker itself must remain spelled consistently in the workflow,
        //    because the pin-completeness gate still greps for it: resolving the
        //    values must not have deleted the guard that catches a REGRESSION
        //    back to a placeholder.
        let marker = "QWEN-MTP-PENDING-ORGANIZER"
        #expect(
            workflow.contains("PENDING_MARKER: \(marker)"),
            "the pin-completeness gate lost its marker definition"
        )
        // And no pin may quietly become a placeholder again.
        for pin in [
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR",
        ] {
            let value = try #require(
                environment[pin],
                "the workflow declares no \(pin)"
            )
            #expect(
                !value.contains(marker),
                "\(pin) regressed to a placeholder ('\(value)')"
            )
        }

        // 3. Every pin is a real value, not a placeholder. These are what the
        //    armed track actually ranks against, so a regression here is a
        //    scoring bug rather than a missing-work marker.
        //    The last four are the box-3-generated model-derived pins (the RAW
        //    hidden correctness golden and the GPQA reference); pinning them
        //    here forces the R2 object key and the job-env digest to move
        //    together, exactly as the DFlash goldens are mutation-tested.
        for (pin, expected) in [
            ("MLXFAST_EXPECTED_NUM_LAYERS", "64"),
            ("MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS", "16"),
            ("MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES", "16081490064"),
            ("MLXFAST_METAL_TOOLCHAIN_IDENTIFIER", "com.apple.dt.toolchain.Metal.32023.883"),
            ("MLXFAST_QWEN_MTP_CHECKPOINT_REPO", "mlx-community/Qwen3.6-27B-4bit"),
            (
                "MLXFAST_QWEN_MTP_CHECKPOINT_REVISION",
                "c000ac2c2057d94be3fa931000c31723aac53282"
            ),
            // faf1a679 = eef3b817 (generate-golden + attach-free-run-gate)
            // plus the derived `.benchmark` oracle the ranked gates phase
            // requires, attached by attach-benchmark-oracle. Additive only:
            // .cases, .correctness_gates, .model_provenance and .version are
            // byte-identical to eef3b817's.
            (
                "MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256",
                "faf1a679e993a3ea2ffb263213c66a526ba9cea591954898f8851bae42c9d5c5"
            ),
            ("MLXFAST_RAW_CORRECTNESS_GOLDEN_BYTES", "36183"),
            (
                "MLXFAST_GPQA_REFERENCE_SHA256",
                // c8bce79c = d05e93e9 with the previously empty
                // accepted_responses filled from the reference model's own
                // captures (its own note called for exactly that fill).
                // Prompts, answer_keys, domains and accepted_token_sequences
                // are byte-identical. Self-match restored: an unmodified
                // candidate scores 9/9.
                "c8bce79c0258f2b67882cb3a609937066a27eb7d8f6170e0fd6db8cbac8ce0d7"
            ),
            ("MLXFAST_GPQA_REFERENCE_BYTES", "11125"),
            // Phase 5, generated on box 3 against the Qwen tower with the
            // pinned MTP head. Pinned here for the same reason as the four
            // above: the digest, the byte count and the R2 object key that
            // embeds the digest must move in ONE commit or the download gate
            // fails on a key/digest mismatch.
            (
                "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_SHA256",
                "0c1dcdabd85e7655f5df03e5a1265a5fde1a43ef223283021463f7af80481add"
            ),
            ("MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_BYTES", "110168"),
            // NORMALISED floor: every pool entry's own measured
            // noop_decode_speedup normalises to 1.0, so 0.95 is the margin, not
            // a raw ratio. Measured raw ratios span 0.756-1.0915.
            ("MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR", "0.95"),
        ] {
            #expect(
                environment[pin] == expected,
                "\(pin) is \(environment[pin] ?? "unset"), expected \(expected)"
            )
        }
    }

    /// `benchmark.qwen-mtp.json`'s own `decodeSpeedupFloorNote` says the field
    /// "must be pinned equal to" the workflow env, and that the workflow is the
    /// enforcing site. Nothing checked it, and the manifest sat at `null` while
    /// the workflow had already resolved to 0.95. Modelled on the DFlash
    /// equivalent, which exists because the same gap opened there.
    @Test
    func theManifestDeclaresTheDecodeFloorTheWorkflowEnforces() throws {
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        let raw = try #require(
            environment["MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR"],
            "the Qwen-MTP workflow must declare the floor it enforces"
        )
        let enforced = try #require(
            Double(raw),
            "MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR is not a number: \(raw)"
        )

        let manifest = try S.json("benchmark.qwen-mtp.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let declared = try #require(
            scoring["decodeSpeedupFloor"] as? Double,
            "benchmark.qwen-mtp.json declares no numeric decodeSpeedupFloor"
        )
        #expect(
            enforced == declared,
            """
            MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR=\(raw) is the only value the \
            workflow actually rejects on, while benchmark.qwen-mtp.json \
            declares \(declared).
            """
        )
    }

    /// A track cannot be ranking submissions while its own manifest says the
    /// token-fidelity gate is unimplemented. The DFlash suite pins exactly this
    /// invariant for its manifest; the Qwen manifest shipped "pending" against a
    /// fixture that already said "implemented", which go-live had to reconcile.
    @Test
    func anEnabledTrackDeclaresAnImplementedTokenFidelityGate() throws {
        let manifest = try S.json("benchmark.qwen-mtp.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let gateStatus = try #require(scoring["tokenFidelityGateStatus"] as? String)

        let contract = try S.json("fixtures/qwen3_6_27b_mtp_track.json")
        let proposed = try #require(contract["proposed_scoring"] as? [String: Any])
        #expect(
            proposed["token_fidelity_gate_status"] as? String == gateStatus,
            "fixture token_fidelity_gate_status disagrees with the manifest"
        )

        if contract["official_scoring_enabled"] as? Bool == true {
            #expect(
                gateStatus == "implemented",
                """
                Qwen-MTP official scoring is enabled while \
                tokenFidelityGateStatus is '\(gateStatus)'. Ranked scoring \
                without a proven fidelity gate is exactly the state the \
                enablement interlock exists to prevent.
                """
            )
        }
    }
}

// MARK: - The go-live runbook the workflows point at

/// `.github/workflows/qwen-mtp-provision-goldens.yml` refers operators to a
/// "Qwen-MTP go-live runbook" from four places, including the "step A" and
/// "step B" cited in error messages an operator only ever sees when a dispatch
/// fails closed. That reference dangled at a file that did not exist until
/// go-live on 2026-08-13. This pins its existence and the sections the workflow
/// names, so the pointer cannot rot again -- the same guard the DFlash track
/// grew after its runbook reference dangled for a week.
@Suite
struct QwenMTPGoLiveRunbookTests {
    private static let path = "docs/qwen-mtp-go-live-runbook.md"

    @Test
    func runbookExistsAndCoversTheStepsTheWorkflowsCite() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // Steps named in the provisioning workflow's own error text.
        #expect(runbook.contains("Step A"))
        #expect(runbook.contains("Step B"))
        // The two trusted-contract fields the enablement guard requires.
        #expect(runbook.contains("official_scoring_enabled"))
        #expect(runbook.contains("publication_allowed"))
        // The blocking prerequisite the pool-selection step fails closed on.
        #expect(runbook.contains("timed_prompt_pool"))
        // The interlock this track's go-live actually turns.
        #expect(runbook.contains("MLXFAST_QWEN_MTP_CALIBRATION_READY"))
    }

    /// The runbook must carry the measured facts and the known limits an
    /// operator accepts by flipping the switch, not just the mechanical steps.
    @Test
    func runbookRecordsTheLimitsBeingAcceptedAtGoLive() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // The pool's raw no-op spread -- the reason the score is normalised.
        #expect(runbook.contains("0.756"))
        #expect(runbook.contains("1.0915"))
        // Prefill is measured but UNSCORED on this decode-only track.
        #expect(runbook.contains("prefill_component"))
        // The GPQA floor's honest limitation: a constant-"A" answerer scores
        // exactly the floor, so the floor does not reject it.
        #expect(runbook.lowercased().contains("constant-\"a\"")
            || runbook.lowercased().contains("constant-a"))
        // The guard fails closed before the flags are flipped.
        #expect(runbook.lowercased().contains("fail-closed")
            || runbook.lowercased().contains("fails closed"))
    }
}
