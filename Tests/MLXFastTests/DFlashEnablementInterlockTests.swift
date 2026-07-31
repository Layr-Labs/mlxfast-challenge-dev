import Foundation
import Testing

/// The DFlash enablement interlock answers two separate questions, and the
/// suite's job is to keep them separate:
///
///   may this job RUN?              `confirm_track_enabled`, matching
///                                  `track_id`, non-empty golden pins.
///   may it PUBLISH a ranked score? `official_scoring_enabled` AND
///                                  `reference_baseline.publication_allowed`.
///
/// Requiring the PUBLISH answer in order to RUN made the pipeline untestable
/// before go-live: the correctness gates and every fail-closed guard could only
/// be exercised by first throwing the switch they gate. A gates-only dispatch
/// (`run_benchmark=false`) produces no ranked score, so it is admitted against a
/// not-yet-enabled track; anything that can publish still requires both flags.
///
/// The risk this suite guards is the obvious one: that the split quietly became
/// a way to publish a score without the go-live flags. So it asserts the ranked
/// refusal survives, and asserts the check exists at the SCORING step too --
/// a guard 49 steps upstream of the thing it protects is a guard whose coverage
/// depends on step order.
@Suite("DFlash enablement interlock")
struct DFlashEnablementInterlockTests {
    private static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let fixturePath = "fixtures/laguna_xs_2_1_dflash_track.json"
    private static let enablementStep = "Enforce DFlash track enablement"
    private static let scoringStep = "Compute DFlash score and enforce floor"

    private static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    /// A step's `run:` body with comment-only lines removed, so no assertion
    /// here can be satisfied by the prose explaining it.
    private static func executableBody(_ workflow: String, _ step: String) throws -> String {
        let marker = "- name: \(step)\n"
        let start = try #require(
            workflow.range(of: marker),
            "step '\(step)' is missing from \(workflowPath)"
        )
        let rest = workflow[start.upperBound...]
        let end = rest.range(of: "\n      - name: ")
        let body = end.map { String(rest[..<$0.lowerBound]) } ?? String(rest)
        return body
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }

    // MARK: - the RUN question stays mandatory for both kinds of dispatch

    @Test
    func confirmTrackEnabledIsRequiredEvenForAGatesOnlyDryRun() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)

        // The interlock check must NOT be conditioned on run_benchmark: a dry
        // run is still a run on the operator's box.
        let check = try #require(
            step.range(of: #"if [[ "${CONFIRM_TRACK_ENABLED}" != "1" ]]; then"#),
            "the confirm_track_enabled interlock vanished"
        )
        let refusal = String(step[check.lowerBound...])
        #expect(
            refusal.contains("exit 1"),
            "the confirm_track_enabled interlock warns instead of refusing"
        )
        // The RUN gate must not have been made ranked-only by the split.
        let beforeCheck = String(step[..<check.lowerBound])
        let lastRunBenchmarkBranch = beforeCheck.range(
            of: #"if [[ "${RUN_BENCHMARK}" == "1" ]]; then"#,
            options: .backwards
        )
        if let branch = lastRunBenchmarkBranch {
            let between = String(beforeCheck[branch.upperBound...])
            #expect(
                between.contains("fi"),
                """
                the confirm_track_enabled interlock is inside an unclosed \
                RUN_BENCHMARK branch, so a gates-only dry run skips it.
                """
            )
        }
    }

    @Test
    func goldenPinsAreRequiredEvenForAGatesOnlyDryRun() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)
        for pin in [
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_DFLASH_BENCH_GOLDEN_SHA256",
            "MLXFAST_DFLASH_BENCH_GOLDEN_BYTES",
        ] {
            #expect(step.contains(pin), "golden pin \(pin) is no longer required")
        }
    }

    // MARK: - the PUBLISH question still refuses a ranked run

    @Test
    func aRankedRunAgainstAnUnenabledTrackIsStillRefused() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)

        #expect(step.contains("official_scoring_enabled"))
        #expect(step.contains("publication_allowed"))

        // The refusal must be reachable when RUN_BENCHMARK=1 and must exit.
        let unenabled = try #require(
            step.range(
                of: #"if [[ "${scoring_enabled}" != "true" || "${publication_allowed}" != "true" ]]; then"#
            ),
            "the enablement comparison vanished or changed shape"
        )
        let branch = String(step[unenabled.upperBound...])
        let rankedGuard = try #require(
            branch.range(of: #"if [[ "${RUN_BENCHMARK}" == "1" ]]; then"#),
            """
            an unenabled track no longer distinguishes a ranked dispatch from a \
            dry run, so either every run is refused (the old chicken-and-egg) or \
            every run is admitted (a ranked run against an unenabled track).
            """
        )
        let rankedBranch = String(branch[rankedGuard.upperBound...])
        let exitIndex = try #require(
            rankedBranch.range(of: "exit 1"),
            "a ranked dispatch against an unenabled track no longer exits"
        )
        // The exit must come before the branch closes -- i.e. it belongs to the
        // ranked arm, not to something after it.
        let fiIndex = try #require(rankedBranch.range(of: "\n            fi"))
        #expect(
            exitIndex.lowerBound < fiIndex.lowerBound,
            "the ranked refusal's exit 1 is outside the ranked branch"
        )
    }

    /// The dry-run arm must announce itself and must not be silent: an operator
    /// reading the log has to be able to tell a validated dry run from a ranked
    /// one that quietly published nothing.
    @Test
    func theDryRunArmAnnouncesThatNoScoreIsProduced() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)
        #expect(step.contains("::notice::"))
        #expect(
            step.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED=1"),
            """
            the dry-run arm does not record that it was admitted unenabled, so \
            the scoring step cannot refuse it independently.
            """
        )
        #expect(step.contains("GATES-ONLY"))
    }

    // MARK: - the second, order-independent check

    @Test
    func theScoringStepIndependentlyRefusesAnUnenabledTrack() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.scoringStep)

        #expect(
            step.contains("official_scoring_enabled"),
            """
            \(Self.scoringStep) does not check official_scoring_enabled itself. \
            The only enablement check would then be 49 steps upstream, so its \
            coverage depends on step order.
            """
        )
        #expect(step.contains("publication_allowed"))
        #expect(
            step.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED"),
            """
            \(Self.scoringStep) does not refuse a dispatch that was admitted as \
            unenabled, so a dry run taught to reach scoring would publish.
            """
        )
        // Both refusals must exit, not warn.
        let refusals = step
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { $0.contains("::error::") }
        #expect(
            refusals.count >= 2,
            "expected both the flag check and the dry-run check to error; found \(refusals.count)"
        )
    }

    // MARK: - behavioural: execute the real step under every input combination

    /// Everything above reads the step's text. This one RUNS it.
    ///
    /// The distinction is not academic: the redactor gap in Amendment 23 would
    /// have passed a text-matching test, because the missing arms were absent
    /// rather than misspelled. A split interlock is exactly the kind of change
    /// where the text can look right and the boolean can be inverted, so the
    /// truth table is asserted by execution.
    @Test
    func theInterlockAdmitsTheDryRunAndOnlyTheDryRun() throws {
        // (label, trackEnabled, rankedRun, confirmed, expectAdmit)
        let cases:
            [(String, Bool, Bool, Bool, Bool)] = [
                // The attack the split could have introduced.
                ("unenabled + ranked + confirmed", false, true, true, false),
                // The point of the split.
                ("unenabled + dry run + confirmed", false, false, true, true),
                // The RUN question must bind BOTH kinds of dispatch.
                ("unenabled + dry run + unconfirmed", false, false, false, false),
                ("enabled + ranked + unconfirmed", true, true, false, false),
                // A properly enabled track is unaffected.
                ("enabled + ranked + confirmed", true, true, true, true),
                ("enabled + dry run + confirmed", true, false, true, true),
            ]

        for (label, trackEnabled, rankedRun, confirmed, expectAdmit) in cases {
            let admitted = try runEnablementStep(
                trackEnabled: trackEnabled,
                rankedRun: rankedRun,
                confirmed: confirmed
            )
            #expect(
                admitted == expectAdmit,
                """
                enablement step \(admitted ? "ADMITTED" : "REFUSED") '\(label)', \
                expected \(expectAdmit ? "ADMIT" : "REFUSE"). The interlock's \
                truth table changed: a ranked dispatch against a track whose \
                official_scoring_enabled / publication_allowed are false must be \
                refused, a gates-only dispatch must be admitted, and neither may \
                run without confirm_track_enabled.
                """
            )
        }
    }

    /// Extracts the step's real `run:` body and executes it against a synthetic
    /// contract, returning whether it admitted the dispatch.
    private func runEnablementStep(
        trackEnabled: Bool,
        rankedRun: Bool,
        confirmed: Bool
    ) throws -> Bool {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-interlock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: workspace,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: workspace) }

        let contract = workspace.appendingPathComponent("contract.json")
        try """
        {"track_id":"laguna-xs-2.1-dflash-v1",
         "official_scoring_enabled":\(trackEnabled),
         "reference_baseline":{"publication_allowed":\(trackEnabled)}}
        """.write(to: contract, atomically: true, encoding: .utf8)

        let body = try Self.rawBody(try Self.workflow(), Self.enablementStep)
        let githubEnv = workspace.appendingPathComponent("github_env")
        FileManager.default.createFile(atPath: githubEnv.path, contents: nil)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", body]
        process.currentDirectoryURL = workspace
        process.environment = [
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "MLXFAST_DFLASH_CONTRACT_PATH": contract.path,
            "MLXFAST_DFLASH_TRACK_ID": "laguna-xs-2.1-dflash-v1",
            // Non-empty pins: this test is about the enablement booleans, and
            // goldenPinsAreRequiredEvenForAGatesOnlyDryRun covers the pins.
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256": "aa",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES": "1",
            "MLXFAST_DFLASH_BENCH_GOLDEN_SHA256": "bb",
            "MLXFAST_DFLASH_BENCH_GOLDEN_BYTES": "2",
            "CONFIRM_TRACK_ENABLED": confirmed ? "1" : "0",
            "RUN_BENCHMARK": rankedRun ? "1" : "0",
            "GITHUB_ENV": githubEnv.path,
        ]
        let sink = Pipe()
        process.standardOutput = sink
        process.standardError = sink
        try process.run()
        _ = sink.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        let admitted = process.terminationStatus == 0
        // An admitted dry run against an unenabled track must hand the marker to
        // the scoring step; without it the second guard cannot fire.
        if admitted && !trackEnabled {
            let recorded =
                (try? String(contentsOf: githubEnv, encoding: .utf8)) ?? ""
            #expect(
                recorded.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED=1"),
                """
                the step admitted an unenabled dispatch without recording \
                MLXFAST_DFLASH_DRY_RUN_UNENABLED, so the scoring step's \
                independent refusal has nothing to key on.
                """
            )
        }
        return admitted
    }

    /// The step's `run:` body verbatim, comments included -- this one is
    /// executed, so stripping comments would change what runs.
    private static func rawBody(_ workflow: String, _ step: String) throws -> String {
        let marker = "- name: \(step)\n"
        let start = try #require(workflow.range(of: marker))
        let rest = workflow[start.upperBound...]
        let end = rest.range(of: "\n      - name: ")
        let block = end.map { String(rest[..<$0.lowerBound]) } ?? String(rest)
        // Take the `run: |` scalar and strip its YAML block indentation.
        let runMarker = try #require(
            block.range(of: "run: |\n"),
            "step '\(step)' has no literal run: block"
        )
        let scalar = String(block[runMarker.upperBound...])
        let lines = scalar.split(separator: "\n", omittingEmptySubsequences: false)
        let indent =
            lines
            .first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }?
            .prefix { $0 == " " }
            .count ?? 0

        // A YAML block scalar ends at the first non-blank line indented LESS
        // than the block. Without this the extraction ran past the step into
        // the comment introducing the NEXT step, and a blind dedent turned
        // "      # DFlash-specific host contract..." into "lash-specific host
        // contract..." -- a line bash then tried to execute. That produced a
        // non-zero exit for every case and read exactly like the interlock
        // refusing everything.
        var body: [String] = []
        for line in lines {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank && line.prefix { $0 == " " }.count < indent { break }
            body.append(line.count >= indent ? String(line.dropFirst(indent)) : String(line))
        }
        return body.joined(separator: "\n")
    }

    /// A one-entry pool must not be enough to RANK, but must still be enough to
    /// validate the pipeline. That asymmetry is the anti-lottery floor.
    ///
    /// Why it matters: with a single timed target a failed ranked run is free to
    /// retry against a prompt it has already seen, which turns every output-side
    /// gate into submit-until-green and lets a drafter-confidence threshold be
    /// tuned across attempts. The fixture's timed_prompt_pool_note has always
    /// said 8; nothing enforced it, so a one-entry pool would have ranked.
    @Test
    func aRankedRunRequiresTheFullPoolButADryRunDoesNot() throws {
        let step = try Self.executableBody(
            try Self.workflow(), "Select hidden DFlash timed target from the pool"
        )

        let floor = try #require(
            step.range(of: #"pool_size}" -lt 8"#),
            """
            the pool-selection step has no 8-target floor, so a ranked run could \
            be scored against a pool small enough to memorise.
            """
        )
        let branch = String(step[floor.lowerBound...])
        #expect(
            branch.contains("exit 1"),
            "the anti-lottery floor warns instead of refusing"
        )

        // Scoped to ranked runs: gating a DRY run on 8 targets would recreate the
        // chicken-and-egg the enablement split removed.
        let before = String(step[..<floor.lowerBound])
        #expect(
            before.hasSuffix("MLXFAST_RUN_BENCHMARK:-0}\" == \"1\" && \"${")
                || step.contains(#"MLXFAST_RUN_BENCHMARK:-0}" == "1" && "${pool_size}" -lt 8"#),
            """
            the 8-target floor is not conditioned on MLXFAST_RUN_BENCHMARK, so it \
            either blocks gates-only validation or does not protect ranked runs.
            """
        )
        // The >= 1 check must survive: an EMPTY pool still fails both modes.
        #expect(step.contains(#"pool_size}" -lt 1"#))
    }

    /// The fixture must stay inert on main. This is the assertion that would
    /// catch the split being "completed" by simply flipping the flags.
    @Test
    func theTrackFixtureRemainsInertOnMain() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: Self.fixturePath))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(
            object["official_scoring_enabled"] as? Bool == false,
            """
            official_scoring_enabled is true on main. Splitting the interlock \
            was meant to make the track testable WITHOUT going live; if the \
            operator has completed the go-live runbook, update this test in the \
            same commit that flips the flag.
            """
        )
        let baseline = try #require(object["reference_baseline"] as? [String: Any])
        #expect(baseline["publication_allowed"] as? Bool == false)

        // The pool used to be asserted EMPTY here. That was the right property
        // expressed the wrong way: what makes the track inert is that a ranked
        // run cannot produce a score, and "empty" was merely how it happened to
        // be true. An empty pool also made the pipeline unvalidatable, which is
        // the same chicken-and-egg the enablement split above removed -- so the
        // pool now carries the generated timed golden, and the ANTI-LOTTERY
        // FLOOR in "Select hidden DFlash timed target from the pool" is what
        // keeps a ranked run from using it.
        //
        // Assert the property, not its old accident: fewer than 8 targets means
        // no ranked run can proceed, while a gates-only dispatch can.
        let pool = object["timed_prompt_pool"] as? [Any] ?? []
        #expect(
            pool.count < 8,
            """
            timed_prompt_pool has \(pool.count) targets, so a RANKED run is no \
            longer blocked by the anti-lottery floor. That is the go-live \
            threshold: if the operator has provisioned the full pool, this test \
            and the two flags above should change in one commit, deliberately.
            """
        )
        // Every entry must still be complete, or the selection step fails
        // mid-run rather than at dispatch.
        for (index, raw) in pool.enumerated() {
            let entry = try #require(raw as? [String: Any], "pool entry \(index) is not an object")
            let path = entry["r2_path"] as? String ?? ""
            let sha = entry["sha256"] as? String ?? ""
            let bytes = entry["bytes"] as? Int ?? 0
            #expect(!path.isEmpty, "pool entry \(index) has no r2_path")
            #expect(
                sha.count == 64 && sha.allSatisfy(\.isHexDigit),
                "pool entry \(index) sha256 is not a 64-char hex digest"
            )
            #expect(bytes > 0, "pool entry \(index) has no positive byte count")
        }
    }
}
