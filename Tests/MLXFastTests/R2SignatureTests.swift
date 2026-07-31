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
/// like a credentials fault and is not one. Measured on M5-C (LibreSSL 3.3.6).
///
/// Not because the serial box had a better openssl — the serial box has the aws
/// CLI on its runner PATH, so `download_with_aws_cli()` short-circuits and the
/// signer never runs there at all. Every successful hidden-golden fetch in
/// either repo announced "using AWS CLI S3 path-style download". These tests
/// are the only coverage this signing path has.
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

    /// What the two tests above do NOT cover, and the gap that cost a second
    /// failed run.
    ///
    /// `theScriptSigningChainReproducesThePinnedSignature` signs a string-to-sign
    /// ending in the literal `deadbeef` — a stand-in for the canonical-request
    /// hash. So it proves the HMAC chain is right while saying nothing about the
    /// canonical request being hashed, and both scripts built a MALFORMED one:
    ///
    ///     canonical_headers="$(printf 'host:%s\n...\nx-amz-date:%s\n' ...)"
    ///     canonical_request="$(printf 'GET\n%s\n\n%s\n%s\n%s' ...)"
    ///
    /// SigV4 is `METHOD \n URI \n QUERY \n CanonicalHeaders \n SignedHeaders \n
    /// PayloadHash`, and CanonicalHeaders is `name:value\n` per header — so a
    /// BLANK LINE must separate the last header from SignedHeaders. The scripts
    /// spelled that terminating newline inside `canonical_headers`' own printf,
    /// where `$(...)` ATE it (command substitution strips every trailing
    /// newline). The canonical request went out one line short, hashed
    /// differently from the one R2 computed for the same bytes, and R2 answered
    /// HTTP 403 SignatureDoesNotMatch — with a perfectly well-formed signature,
    /// which reads like a bucket-permission fault and is not one.
    ///
    /// It survived because it was never executed: `download_with_aws_cli()`
    /// short-circuits the signer whenever `aws` is on PATH, and the serial
    /// ranked box has it. M5-C's runner PATH is `/usr/bin:/bin:/usr/sbin:/sbin`,
    /// so it became the first box to run this code at all. The aws CLI fallback
    /// must not be what makes R2 work.
    ///
    /// Reference hashes are botocore's own `CanonicalRequest` for the identical
    /// request (aws-cli 2.35.21, `--debug`, path-style endpoint), so this pins
    /// against an independent SigV4 implementation rather than against a
    /// restatement of the script.
    @Test
    func theCanonicalRequestMatchesAnIndependentSigV4Implementation() throws {
        // (script, HTTP method, request path, payload hash, x-amz-date,
        //  botocore's canonical-request hash for exactly those inputs)
        let cases:
            [(
                path: String, method: String, requestPath: String, payloadHash: String,
                amzDate: String, expected: String
            )] = [
                (
                    ".github/scripts/download-r2-object.sh",
                    "GET",
                    "/mybucket/correctness_prompts/x.json",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "20260731T000829Z",
                    "18ec091efbf67625090e1e4b4faa2909228244578b5802a1b8222ad60a7096ad"
                ),
                (
                    ".github/scripts/upload-r2-object.sh",
                    "PUT",
                    "/mybucket/up/x.json",
                    "e38af85860a1452206b018e69c01595e89ce0626bd0068d69ea1b270e993cd41",
                    "20260731T001036Z",
                    "48c8f7da8097d5493dcb5140f1f66b1fd9c609d67a70c68637127dc60ab6651a"
                ),
            ]

        for testCase in cases {
            let text = try String(contentsOfFile: testCase.path, encoding: .utf8)

            // Pull the REAL assignments out of the script so this test tracks
            // the shipped construction instead of a copy of it.
            let assignments = text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter {
                    $0.hasPrefix("canonical_headers=") || $0.hasPrefix("canonical_request=")
                }
                .joined(separator: "\n")
            #expect(
                assignments.contains("canonical_headers=")
                    && assignments.contains("canonical_request="),
                """
                \(testCase.path) no longer assigns canonical_headers/canonical_request \
                at top level; if the canonical request moved, retarget this test rather \
                than deleting it.
                """
            )

            let program = """
                set -euo pipefail
                host='acct.r2.cloudflarestorage.com'
                request_path='\(testCase.requestPath)'
                payload_hash='\(testCase.payloadHash)'
                amz_date='\(testCase.amzDate)'
                signed_headers='host;x-amz-content-sha256;x-amz-date'
                \(assignments)
                printf '%s' "${canonical_request}" | shasum -a 256 | awk '{print $1}'
                printf '%s' "${canonical_request}" >&2
                """

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", program]
            let out = Pipe()
            let err = Pipe()
            process.standardOutput = out
            process.standardError = err
            try process.run()
            let hash = String(
                decoding: out.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            let rendered = String(
                decoding: err.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self
            )
            process.waitUntilExit()

            #expect(
                process.terminationStatus == 0,
                "\(testCase.path) canonical request failed to build: \(rendered)"
            )

            // The blank separator is the byte that was missing, so assert it
            // directly too: a hash mismatch alone does not say which line moved.
            let lines = rendered.split(separator: "\n", omittingEmptySubsequences: false)
            #expect(
                lines.count == 9,
                """
                \(testCase.path) canonical request has \(lines.count) lines, expected 9 \
                (method, uri, empty query, 3 headers, BLANK separator, signed headers, \
                payload hash). Rendered:
                \(rendered)
                """
            )
            if lines.count == 9 {
                #expect(
                    lines[2].isEmpty,
                    "\(testCase.path) line 3 must be the empty canonical query string"
                )
                #expect(
                    lines[6].isEmpty,
                    """
                    \(testCase.path) line 7 must be the BLANK line that terminates \
                    CanonicalHeaders and separates it from SignedHeaders. Omitting it \
                    yields HTTP 403 SignatureDoesNotMatch with a well-formed signature. \
                    Do not spell that newline inside canonical_headers' printf -- \
                    command substitution strips trailing newlines.
                    """
                )
            }

            #expect(
                hash == testCase.expected,
                """
                \(testCase.path) canonical-request hash \(hash), expected \
                \(testCase.expected) (botocore's own CanonicalRequest for the same \
                \(testCase.method) request). Rendered:
                \(rendered)
                """
            )
        }
    }
}
