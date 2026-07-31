import Foundation
import Testing

/// The R2 keys the DFlash goldens actually live under.
///
/// This exists because the prefix was dropped twice. The operator uploaded to
/// `gautham-experiments/correctness_prompts/laguna-xs-2.1-dflash/`, and I twice
/// pinned the key without the leading segment — first by assuming it was the
/// bucket name already carried by `R2_BUCKET_ENDPOINT` (the serial keys are
/// written with no bucket segment, which made the assumption feel safe), and
/// then again when the corrected commit missed the merge.
///
/// A wrong key costs a full 30–40 minute ranked dispatch to discover, so it is
/// worth pinning rather than re-deriving. If the objects genuinely move, change
/// this test in the same commit that moves them.
@Suite("DFlash golden R2 keys")
struct DFlashGoldenKeyTests {
    private static let prefix = "gautham-experiments/correctness_prompts/laguna-xs-2.1-dflash"

    @Test
    func theCorrectnessGoldenKeyMatchesWhereItWasUploaded() throws {
        let wf = try String(
            contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8
        )
        let expected =
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_R2_PATH: "
            + "\(Self.prefix)/dflash_correctness_golden_hidden.json"
        #expect(
            wf.contains(expected),
            """
            the correctness golden's R2 key is not the one the objects were \
            uploaded to. Expected the '\(Self.prefix)' prefix; a key missing it \
            fails as 404 NoSuchKey after a full ranked dispatch.
            """
        )
    }

    @Test
    func everyPoolEntryKeyMatchesWhereItWasUploaded() throws {
        let data = try Data(
            contentsOf: URL(fileURLWithPath: "fixtures/laguna_xs_2_1_dflash_track.json")
        )
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let pool = object["timed_prompt_pool"] as? [[String: Any]] ?? []
        #expect(!pool.isEmpty, "the pool is empty; nothing to key-check")
        for (index, entry) in pool.enumerated() {
            let path = entry["r2_path"] as? String ?? ""
            #expect(
                path.hasPrefix(Self.prefix + "/"),
                "pool entry \(index) key '\(path)' is not under \(Self.prefix)"
            )
            // The signer refuses anything outside this set BEFORE signing, so a
            // bad character is a dispatch-time failure, not a run-time one.
            #expect(
                path.allSatisfy { $0.isLetter || $0.isNumber || "._/-".contains($0) },
                "pool entry \(index) key would be refused by the signer's charset guard"
            )
        }
    }
}
