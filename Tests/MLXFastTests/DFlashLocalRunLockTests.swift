import Foundation
import Testing

/// `benchmark-dflash.sh` must take the SAME local run lock as `benchmark.sh`.
///
/// Why: both local modes hold the ~21.6 GB target plus the drafter, so two
/// overlapping local runs can out-of-memory the machine, and two runs sharing
/// one GPU invalidate both timings. benchmark.sh guarded only one direction of
/// this -- its resident-model scan lists the `dflash-*` subcommands, so a
/// serial run refuses to start against a live DFlash one -- while a DFlash run
/// started happily against a live serial run, because this script took no lock
/// at all.
///
/// These tests pin the reuse rather than the behaviour: the lock PATH is the
/// part that must not drift, since two implementations disagreeing about where
/// the lock lives would both "hold a lock" and exclude nothing.
@Suite("DFlash local run lock")
struct DFlashLocalRunLockTests {
    private static let dflashScriptPath = "benchmark-dflash.sh"
    private static let serialScriptPath = "benchmark.sh"

    private static func text(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    /// Lines with the leading `#` comment stripped, so an assertion cannot be
    /// satisfied by prose that merely mentions the thing.
    private static func executable(_ script: String) -> String {
        script
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }

    @Test
    func dflashAcquiresAndReleasesTheLocalRunLock() throws {
        let script = Self.executable(try Self.text(Self.dflashScriptPath))

        #expect(
            script.contains("acquire_local_run_lock"),
            """
            benchmark-dflash.sh does not acquire the local run lock. Both of its \
            modes hold the ~21.6 GB target plus the drafter; an overlapping local \
            run can out-of-memory the machine.
            """
        )
        #expect(
            script.contains("release_local_run_lock"),
            "benchmark-dflash.sh acquires the run lock but never releases it"
        )

        // Released from the EXIT trap, not just on the success path: a run that
        // aborts must not strand the lock and wedge the next one.
        let cleanup = try #require(
            script.range(of: "cleanup() {"),
            "benchmark-dflash.sh has no cleanup() function to release the lock from"
        )
        let trap = try #require(
            script.range(of: "trap cleanup EXIT"),
            "benchmark-dflash.sh never arms its cleanup trap"
        )
        let cleanupBody = String(script[cleanup.upperBound..<trap.lowerBound])
        #expect(
            cleanupBody.contains("release_local_run_lock"),
            """
            benchmark-dflash.sh releases the run lock outside cleanup(), so an \
            aborted run strands it and the next run refuses to start.
            """
        )

        // Acquired only once the trap can release it.
        let acquire = try #require(script.range(of: "acquire_local_run_lock"))
        #expect(
            acquire.lowerBound > trap.lowerBound
                || cleanupBody.contains("release_local_run_lock"),
            "the lock is acquired before its release trap is armed"
        )
    }

    @Test
    func dflashReusesTheSerialLockDefinitionsRatherThanCopyingThem() throws {
        let dflash = Self.executable(try Self.text(Self.dflashScriptPath))

        // Reuse, by the same awk-extract-and-eval idiom the source_hash() reuse
        // uses. A second implementation of the lock PATH is the failure this
        // asserts against: both scripts would hold "a lock" and exclude nothing.
        for definition in [
            "local_run_lock_path",
            "acquire_local_run_lock",
            "release_local_run_lock",
        ] {
            #expect(
                dflash.contains("awk '/^\(definition)\\(\\) \\{/,/^\\}/' benchmark.sh"),
                """
                benchmark-dflash.sh does not extract \(definition)() from \
                benchmark.sh. Reuse the single definition -- a copied lock path \
                that drifts excludes nothing.
                """
            )
        }
        // The lock filename may appear ONLY as the string the extraction is
        // sanity-checked against -- never in a path this script builds itself.
        // (That sanity check is the fail-closed guard, so it is the one
        // legitimate mention; excluding it is what keeps this assertion about
        // drift rather than about vocabulary.)
        let selfSpelled = dflash
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { $0.contains("mlxfast-local-benchmark-") }
            .filter { !$0.contains("run_lock_definitions") }
        #expect(
            selfSpelled.isEmpty,
            """
            benchmark-dflash.sh spells the lock filename itself instead of \
            reusing local_run_lock_path(): \
            \(selfSpelled.map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " | ")). \
            Two spellings of the path is exactly the drift this reuse prevents.
            """
        )
        #expect(
            dflash.contains("eval \"${run_lock_definitions}\""),
            "the extracted lock definitions are never evaluated"
        )

        // Fails CLOSED: if benchmark.sh is refactored and the extraction stops
        // finding the definitions, the script must refuse to run rather than
        // proceed unguarded.
        let guardStart = try #require(
            dflash.range(of: "run_lock_definitions=\"$("),
            "the lock extraction vanished"
        )
        let after = String(dflash[guardStart.lowerBound...])
        let sanityCheck = try #require(
            after.range(of: "refusing to run unguarded"),
            """
            benchmark-dflash.sh does not fail closed when the lock extraction \
            comes up empty -- a refactor of benchmark.sh would silently leave \
            DFlash local runs unguarded.
            """
        )
        let checkTail = String(after[sanityCheck.lowerBound...])
        #expect(
            checkTail.contains("exit 1"),
            "the failed-extraction path warns instead of exiting"
        )
    }

    @Test
    func serialStillDefinesTheFunctionsDFlashExtracts() throws {
        // The other half of the coupling: this is what turns a benchmark.sh
        // refactor into a red test here instead of an unguarded DFlash run
        // discovered at OOM time.
        let serial = try Self.text(Self.serialScriptPath)
        for definition in [
            "local_run_lock_path",
            "acquire_local_run_lock",
            "release_local_run_lock",
        ] {
            #expect(
                serial.contains("\n\(definition)() {"),
                """
                benchmark.sh no longer defines \(definition)() at the top level, \
                so benchmark-dflash.sh's extraction of it fails and DFlash local \
                runs refuse to start. Update both scripts together.
                """
            )
        }
        // The sanity strings benchmark-dflash.sh greps the extraction for.
        #expect(serial.contains("mlxfast-local-benchmark-"))
        #expect(serial.contains("LOCAL_RUN_LOCK_OWNED="))
    }

    /// The lock only helps if the two scripts agree it is the same lock. Drives
    /// the real extraction and asserts the resulting path matches the one
    /// benchmark.sh computes for the same environment.
    @Test
    func extractedLockPathMatchesTheSerialLockPath() throws {
        let lockRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-lock-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: lockRoot) }

        // Left: the path benchmark.sh's own definition produces. Right: the
        // path that same definition produces after DFlash's awk extraction and
        // eval. Any divergence means the two scripts lock different files.
        let program = """
        set -euo pipefail
        export MLXFAST_LOCAL_RUN_LOCK_DIR="\(lockRoot.path)"
        serial="$(bash -c 'eval "$(awk "/^local_run_lock_path\\(\\) \\{/,/^\\}/" benchmark.sh)"; local_run_lock_path')"
        defs="$(
          awk '/^local_run_lock_path\\(\\) \\{/,/^\\}/' benchmark.sh
          awk '/^acquire_local_run_lock\\(\\) \\{/,/^\\}/' benchmark.sh
          awk '/^release_local_run_lock\\(\\) \\{/,/^\\}/' benchmark.sh
        )"
        LOCAL_RUN_LOCK_OWNED=""
        local_run_guard_enabled() { [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]; }
        eval "${defs}"
        dflash="$(local_run_lock_path)"
        [[ "${serial}" == "${dflash}" ]] || { echo "MISMATCH ${serial} != ${dflash}"; exit 1; }
        acquire_local_run_lock >/dev/null 2>&1 || { echo "ACQUIRE-FAILED"; exit 1; }
        [[ -d "${dflash}" ]] || { echo "NO-LOCK-DIR"; exit 1; }
        ( LOCAL_RUN_LOCK_OWNED=""; acquire_local_run_lock ) >/dev/null 2>&1 \\
          && { echo "DOUBLE-ACQUIRED"; exit 1; }
        release_local_run_lock
        [[ -d "${dflash}" ]] && { echo "NOT-RELEASED"; exit 1; }
        echo "OK ${dflash}"
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", program]
        process.currentDirectoryURL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath
        )
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self
        )
        process.waitUntilExit()

        #expect(
            process.terminationStatus == 0,
            """
            the lock path benchmark-dflash.sh extracts diverges from \
            benchmark.sh's, or the extracted lock does not exclude a second \
            holder: \(output.trimmingCharacters(in: .whitespacesAndNewlines))
            """
        )
        #expect(output.contains("mlxfast-local-benchmark-"), "\(output)")
    }
}
