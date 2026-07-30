import Foundation
import Testing

/// The R2 helpers sign requests with AWS SigV4, and the signing must not depend
/// on which `openssl` happens to be first on PATH.
///
/// It did. `download-r2-object.sh` and `upload-r2-object.sh` derived the four
/// signing keys through a `hmac_hex` helper that used `-binary | xxd`, then
/// computed the FINAL signature with a separate pipeline that omitted `-binary`
/// and parsed the text form with `awk '{print $2}'`:
///
///     OpenSSL 3.x    "SHA2-256(stdin)= <hex>"   -> $2 is the hex
///     LibreSSL 3.3   "<hex>"                    -> $2 is EMPTY ($1 is the hex)
///
/// macOS ships LibreSSL as `/usr/bin/openssl`. On a box without Homebrew
/// OpenSSL ahead of it the signature came out empty, and R2 answered HTTP 400
/// `InvalidArgument: Signature element value should not be blank` — which reads
/// like a credentials fault and is not one. Measured on M5-C (LibreSSL 3.3.6);
/// the serial box only worked because OpenSSL 3 was first on its PATH.
///
/// Two guards, because either alone is weak: a structural one (no `openssl
/// dgst` may parse the text form) and a behavioural known-answer test (the
/// script's real signing chain must reproduce a pinned signature).
@Suite("R2 SigV4 signing")
struct R2SignatureTests {
    private static let scripts = [
        ".github/scripts/download-r2-object.sh",
        ".github/scripts/upload-r2-object.sh",
    ]

    /// Every `openssl dgst` must use `-binary`, so no output-format field index
    /// exists to get wrong.
    @Test
    func noOpensslDigestParsesTheTextOutputFormat() throws {
        for path in Self.scripts {
            let text = try String(contentsOfFile: path, encoding: .utf8)
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)

            var invocations = 0
            for (index, line) in lines.enumerated() {
                guard line.contains("openssl dgst"),
                    !line.trimmingCharacters(in: .whitespaces).hasPrefix("#")
                else { continue }
                invocations += 1

                // The invocation may continue onto the next lines via `\`.
                var statement = String(line)
                var cursor = index
                while statement.hasSuffix("\\"), cursor + 1 < lines.count {
                    cursor += 1
                    statement += "\n" + lines[cursor]
                }

                #expect(
                    statement.contains("-binary"),
                    """
                    \(path):\(index + 1) runs `openssl dgst` without -binary, so it \
                    depends on the text output format. LibreSSL prints the bare \
                    hex and OpenSSL 3 prefixes "SHA2-256(stdin)= ", so any field \
                    index is right on one implementation and wrong on the other.
                    """
                )
                #expect(
                    !statement.contains("awk '{print $2}'"),
                    """
                    \(path):\(index + 1) parses openssl's text output by field \
                    index. On LibreSSL the hex is $1 and $2 is empty, which \
                    yields an EMPTY signature and a 400 that looks like bad \
                    credentials.
                    """
                )
            }
            #expect(
                invocations > 0,
                "\(path) no longer invokes openssl dgst; if signing moved, retarget this test"
            )
        }
    }

    /// Runs the script's OWN `hmac_hex` definition over the full SigV4 key
    /// derivation and asserts the pinned signature. Independently computed with
    /// Python's `hmac` and verified identical under OpenSSL 3.6.3 and LibreSSL
    /// 3.3.6 before pinning.
    @Test
    func theScriptSigningChainReproducesThePinnedSignature() throws {
        let expected = "b8b5506ede410169fc30c5c95a42e4f9217b438b8aa8674553b8b5152e20deb3"

        for path in Self.scripts {
            let text = try String(contentsOfFile: path, encoding: .utf8)

            // Extract the real hmac_hex definition rather than restating it, so
            // this test tracks the script instead of a copy of it.
            let start = try #require(
                text.range(of: "hmac_hex() {"),
                "\(path) no longer defines hmac_hex()"
            )
            let tail = text[start.lowerBound...]
            let end = try #require(
                tail.range(of: "\n}"),
                "\(path) hmac_hex() definition is unterminated"
            )
            let definition = String(tail[..<end.upperBound])

            let program = """
                set -euo pipefail
                \(definition)
                SECRET='wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'
                k_date="$(hmac_hex "key:AWS4${SECRET}" "20260730")"
                k_region="$(hmac_hex "hexkey:${k_date}" "auto")"
                k_service="$(hmac_hex "hexkey:${k_region}" "s3")"
                k_signing="$(hmac_hex "hexkey:${k_service}" "aws4_request")"
                printf '%s' "$(hmac_hex "hexkey:${k_signing}" 'AWS4-HMAC-SHA256
                20260730T000000Z
                20260730/auto/s3/aws4_request
                deadbeef')"
                """

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", program]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            let output = String(
                decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            process.waitUntilExit()

            #expect(
                process.terminationStatus == 0,
                "\(path) hmac_hex failed to run: \(output)"
            )
            #expect(
                !output.isEmpty,
                """
                \(path) hmac_hex produced an EMPTY digest. That is the exact \
                failure mode that made the SigV4 signature blank and R2 answer \
                "Signature element value should not be blank".
                """
            )
            #expect(
                output == expected,
                """
                \(path) signing chain produced \(output), expected \(expected). \
                The SigV4 key derivation changed meaning; a wrong-but-nonempty \
                signature fails as SignatureDoesNotMatch instead of a blank \
                element, which is harder to recognise.
                """
            )
        }
    }
}
