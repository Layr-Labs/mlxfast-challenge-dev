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
        // --- box-owned and per-track paths -----------------------------------
        ("measure wrapper filename", "measure-qwen-mtp-job.sh"),
        ("per-track state dir", "/opt/bench-runner/state/qwen3.6-27b-mtp-v1"),
        ("per-track state dir leaf", "qwen3.6-27b-mtp-v1"),
        ("per-track baseline dir", "/opt/bench-runner/baseline/qwen3.6-27b-mtp-v1/current"),
        ("per-track baseline calibration", "/opt/bench-runner/state/qwen3.6-27b-mtp-v1/baseline-calibration.json"),
        ("MTP head cache dir", "/opt/bench-runner/cache/qwen-mtp/qwen3.6-27b-mtp-v1/mtp-head"),
        ("runner label", "m5-qwen-mtp"),
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
        "benchmark-qwen-mtp.sh",
        "setup-qwen-mtp.sh",
        "benchmark.qwen-mtp.json",
        "Tests/MLXFastTests/QwenMTPTrackNamingTests.swift",
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
        let benchmarkRunner = try S.text("benchmark-qwen-mtp.sh")
        let setupRunner = try S.text("setup-qwen-mtp.sh")
        let manifest = try S.text("benchmark.qwen-mtp.json")

        let expectations: [(needle: String, haystack: String, file: String)] = [
            ("qwen3.6-27b-mtp-v1", workflow, "the workflow"),
            ("qwen3.6-27b-mtp-v1", manifest, "the manifest"),
            ("qwen3.6-27b-mtp-v1", benchmarkRunner, "the local benchmark runner"),
            ("mlxfast-challenge-dev-qwen-mtp", manifest, "the manifest"),
            ("MLXFAST_QWEN_MTP_CALIBRATION_READY", workflow, "the workflow"),
            ("measure-qwen-mtp-job.sh", workflow, "the workflow"),
            ("/opt/bench-runner/state/qwen3.6-27b-mtp-v1", workflow, "the workflow"),
            ("m5-qwen-mtp", workflow, "the workflow"),
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

    /// The track ships INERT. These three are the fail-closed posture the draft
    /// is required to hold, and each is a one-character edit away from being
    /// armed, so pin them rather than trusting a reviewer to notice.
    @Test
    func theQwenMTPTrackShipsInert() throws {
        let workflow = try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        let environment = try S.jobEnvironment(workflow)

        // 1. The calibration-ready interlock is a hard-pinned literal "0" --
        //    never an expression, never a dispatch input.
        #expect(environment["MLXFAST_QWEN_MTP_CALIBRATION_READY"] == "0")
        #expect(!workflow.contains("MLXFAST_QWEN_MTP_CALIBRATION_READY: ${{"))
        // ... and the runbook-shaped refusal message exists, scanned on the
        // comment-stripped view so a mention in prose cannot satisfy it.
        #expect(
            S.executable(workflow).contains(
                "Qwen 3.6 MTP model-derived fixtures and final-SHA baseline are not calibrated"
            )
        )

        // 2. The hidden-artifact pins are still placeholders, and the marker is
        //    spelled consistently so the pin-completeness gate can find them.
        let marker = "QWEN-MTP-PENDING-ORGANIZER"
        for pin in [
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_RAW_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_GPQA_REFERENCE_SHA256",
            "MLXFAST_GPQA_REFERENCE_BYTES",
            "MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR",
        ] {
            let value = try #require(
                environment[pin],
                "the workflow declares no \(pin)"
            )
            #expect(
                value.contains(marker),
                """
                \(pin) is '\(value)', not a placeholder. If the organizer \
                artifact has landed, this test is the reminder to move the \
                matching R2 object key, the manifest's scoring block and this \
                assertion in the SAME commit.
                """
            )
        }

        // 3. The pins that ARE computable from this branch are real, not
        //    placeholders -- the inert posture must not have swallowed them.
        for (pin, expected) in [
            ("MLXFAST_EXPECTED_NUM_LAYERS", "64"),
            ("MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS", "15"),
            ("MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES", "16081488494"),
            ("MLXFAST_METAL_TOOLCHAIN_IDENTIFIER", "com.apple.dt.toolchain.Metal.32023.883"),
            ("MLXFAST_QWEN_MTP_CHECKPOINT_REPO", "mlx-community/Qwen3.6-27B-4bit"),
            (
                "MLXFAST_QWEN_MTP_CHECKPOINT_REVISION",
                "c000ac2c2057d94be3fa931000c31723aac53282"
            ),
        ] {
            #expect(
                environment[pin] == expected,
                "\(pin) is \(environment[pin] ?? "unset"), expected \(expected)"
            )
        }
    }
}
