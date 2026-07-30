import Foundation
import Testing

/// The DFlash job and the serial job must bind the SAME GitHub environment.
///
/// Both read the same secret NAMES — `R2_ACCESS_KEY_ID`,
/// `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_ENDPOINT` — and those resolve per
/// environment. So a mismatched `environment:` hands one job different
/// credentials and a different endpoint for the same object names, and the
/// symptom is a bare HTTP 400 from curl with nothing naming the cause.
///
/// This is not hypothetical. The DFlash job inherited
/// `benchmark-private-prompts` from the retired MTP job while serial moved to
/// `benchmark-private-prompts-v2`. Run 30588892999 died in "Prepare hidden
/// correctness golden" — a copy of the serial step, fetching the same
/// `raw_golden.json` serial fetches successfully — on six retried 400s. Same
/// box, same object, same secret names; the binding was the only variable.
///
/// Pinning equality rather than a literal name is deliberate: when the
/// operator rotates to a `-v3`, this fails until BOTH jobs move, which is the
/// property that was missing.
@Suite("DFlash private material binding")
struct DFlashPrivateMaterialBindingTests {
    private static let dflashWorkflow = ".github/workflows/dflash-benchmark.yml"
    private static let serialWorkflow = ".github/workflows/benchmark.yml"

    /// The `environment:` of the workflow's single ranked job.
    private static func jobEnvironment(_ path: String) throws -> String {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let line = try #require(
            text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .first {
                    $0.hasPrefix("    environment:")
                        && !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#")
                },
            "\(path) declares no job-level environment:, so it gets no private-material secrets"
        )
        return line
            .replacingOccurrences(of: "    environment:", with: "")
            .trimmingCharacters(in: .whitespaces)
    }

    @Test
    func dflashBindsTheSamePrivateMaterialEnvironmentAsSerial() throws {
        let dflash = try Self.jobEnvironment(Self.dflashWorkflow)
        let serial = try Self.jobEnvironment(Self.serialWorkflow)

        #expect(
            dflash == serial,
            """
            DFlash binds environment '\(dflash)' but the serial ranked job binds \
            '\(serial)'. Both read the same R2 secret NAMES, which resolve per \
            environment, so the DFlash job would fetch the same hidden objects \
            with different credentials and a different endpoint -- curl HTTP 400 \
            with nothing naming the cause. Move BOTH jobs together.
            """
        )
        #expect(!dflash.isEmpty)
    }

    /// Neither job may read the R2 secrets outside the job that declares the
    /// environment — a secret reference in a job with no `environment:` silently
    /// resolves to empty and produces the same unexplained 400.
    @Test
    func everyR2SecretReferenceLivesInTheEnvironmentBoundJob() throws {
        for path in [Self.dflashWorkflow, Self.serialWorkflow] {
            let text = try String(contentsOfFile: path, encoding: .utf8)
            let jobCount = text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { $0.hasPrefix("  ") && $0.hasSuffix(":") && !$0.hasPrefix("   ") }
                .count
            // These workflows are single-job by construction; if that ever
            // changes, the environment/secret pairing needs re-checking rather
            // than this test quietly still passing.
            #expect(
                jobCount <= 2,
                """
                \(path) appears to declare more than one job. R2 secrets resolve \
                only inside the job that declares environment:, so a second job \
                referencing them would read empty strings. Re-check the pairing.
                """
            )
            #expect(
                text.contains("environment:"),
                "\(path) references private material but binds no environment"
            )
        }
    }
}
