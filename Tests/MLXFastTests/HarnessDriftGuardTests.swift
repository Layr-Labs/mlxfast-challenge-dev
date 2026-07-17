import Foundation
import Testing

// Drift guard for the two harness source trees created by the #630 binary
// split:
//
//   Sources/MLXFastTrustedHarness/  builds module MLXFastHarness, linked into
//                                   the trusted `mlxfast-swift` binary
//                                   (timing, gates, scoring, weights digest).
//   Sources/MLXFastHarness/         builds module MLXFastRuntimeWorkerSupport,
//                                   linked into the sandboxed
//                                   `mlxfast-runtime-worker` binary.
//
// The trees are near-twins and most fixes must land in both. A branch race
// around the split silently dropped #636's digest memory fix
// (GemmaRuntimePreflight.swift, fileDigest) from the trusted copy and
// re-OOMed 36 GB machines, while CI stayed green because the regression test
// asserted against the other tree's file (restored in #652). These tests fail
// on any undocumented divergence so a one-sided fix cannot land silently
// again. The structural cleanup (naming inversion, dead halves, duplicated
// literals) is tracked in issue #655.

private let workerHarnessDirectory = "Sources/MLXFastHarness"
private let trustedHarnessDirectory = "Sources/MLXFastTrustedHarness"

/// Twin files allowed to differ beyond `#if MLXFAST_TRUSTED_HARNESS` plumbing
/// and whitespace, with the reason the divergence is intentional. The guard
/// also fails when an entry goes stale (the twins converged), so this list
/// always mirrors reality. Before adding an entry, prefer mirroring the
/// change into both trees: every file listed here is surface this guard
/// cannot protect from the next dropped fix.
private let allowedHarnessDivergences: [String: String] = [
    "GemmaRuntimeCorrectnessCompare.swift":
        "the trusted copy traces correctness through the sandboxed "
        + "runtime-worker client (optional worker options plus validation of "
        + "worker-reported top logits); the worker copy traces in-process "
        + "with direct model access",
    "GemmaRuntimeMTP.swift":
        "the trusted copy cannot link MLXFastModel, so it re-declares "
        + "Gemma4MTPVerificationMode and inlines the shippedCheckpoint "
        + "assistant-unavailable reason string; deduplication is tracked in "
        + "issue #655",
    "GemmaRuntimeWorker.swift":
        "the trusted copy's spawn sites carry #646's sandbox exec-allow "
        + "rebinding (runtimeWorkerSandboxProfile) and drop a doc comment the "
        + "worker copy keeps; the worker copy's spawn/orchestration half is "
        + "unreachable in the worker binary; pruning is tracked in issue #655",
]

@Test
func harnessTreesContainTheSameSourceFiles() throws {
    let workerFiles = try harnessSourceFileNames(in: workerHarnessDirectory)
    let trustedFiles = try harnessSourceFileNames(in: trustedHarnessDirectory)
    let workerOnly = workerFiles.subtracting(trustedFiles).sorted()
    let trustedOnly = trustedFiles.subtracting(workerFiles).sorted()
    #expect(
        workerFiles == trustedFiles,
        "the harness trees must stay file-for-file twins (see issue #655); only in \(workerHarnessDirectory): \(workerOnly); only in \(trustedHarnessDirectory): \(trustedOnly); add the missing twin instead of forking the trees"
    )
}

@Test
func harnessTwinsOnlyDivergeWhereDocumented() throws {
    let workerFiles = try harnessSourceFileNames(in: workerHarnessDirectory)
    let trustedFiles = try harnessSourceFileNames(in: trustedHarnessDirectory)

    for file in allowedHarnessDivergences.keys.sorted() {
        #expect(
            workerFiles.contains(file) && trustedFiles.contains(file),
            "allowedHarnessDivergences lists \(file), which no longer exists in both harness trees; update the allowlist"
        )
    }

    for file in workerFiles.intersection(trustedFiles).sorted() {
        let workerPath = "\(workerHarnessDirectory)/\(file)"
        let trustedPath = "\(trustedHarnessDirectory)/\(file)"
        let workerLines = try workerVisibleLines(
            in: try String(contentsOfFile: workerPath, encoding: .utf8),
            file: workerPath
        )
        let trustedLines = try workerVisibleLines(
            in: try String(contentsOfFile: trustedPath, encoding: .utf8),
            file: trustedPath
        )
        let twinsMatch = condensedText(workerLines) == condensedText(trustedLines)

        if let reason = allowedHarnessDivergences[file] {
            #expect(
                !twinsMatch,
                "stale allowlist entry: the \(file) twins are identical again; remove the file from allowedHarnessDivergences (entry reason was: \(reason))"
            )
            continue
        }

        let hint = twinsMatch ? "" : firstMismatchHint(worker: workerLines, trusted: trustedLines)
        #expect(
            twinsMatch,
            "\(file) drifted between \(workerHarnessDirectory) (worker binary) and \(trustedHarnessDirectory) (trusted binary). Fixes must land in both trees: dropping one side is how #636's GemmaRuntimePreflight.swift fileDigest fix vanished from the trusted binary and re-OOMed 36 GB machines (#652). Mirror the change into the other tree, or if the divergence is truly intentional, add an allowedHarnessDivergences entry explaining why. \(hint)"
        )
    }
}

@Test
func harnessFileDigestTwinsAreTextuallyIdentical() throws {
    // fileDigest is the proven hot spot: both binaries digest multi-gigabyte
    // artifacts on the memory-critical startup path (the trusted binary for
    // the weights integrity hash, the worker during preflight), and it is
    // exactly where the #636 fix was dropped from one tree (#652). The twins
    // must stay byte-identical, not merely equivalent.
    let workerPath = "\(workerHarnessDirectory)/GemmaRuntimePreflight.swift"
    let trustedPath = "\(trustedHarnessDirectory)/GemmaRuntimePreflight.swift"
    let workerDigest = try fileDigestRegion(inFileAt: workerPath)
    let trustedDigest = try fileDigestRegion(inFileAt: trustedPath)
    #expect(
        workerDigest == trustedDigest,
        "the fileDigest implementations drifted; mirror the change into both \(workerPath) and \(trustedPath) (see #636/#652)"
    )
    for (path, digestSource) in [(workerPath, workerDigest), (trustedPath, trustedDigest)] {
        #expect(
            digestSource.contains("F_NOCACHE"),
            "fileDigest in \(path) must keep #636's uncached read"
        )
        #expect(
            digestSource.contains("F_RDAHEAD"),
            "fileDigest in \(path) must keep #636's read-ahead opt-out"
        )
        #expect(
            digestSource.contains("autoreleasepool"),
            "fileDigest in \(path) must keep #636's bounded chunk buffers"
        )
    }
}

private func harnessSourceFileNames(in directory: String) throws -> Set<String> {
    let names = try FileManager.default.contentsOfDirectory(atPath: directory)
    return Set(names.filter { $0.hasSuffix(".swift") })
}

private struct HarnessDriftGuardFailure: Error, CustomStringConvertible {
    let description: String
}

private enum HarnessConditionalRegion {
    case keptForWorker
    case droppedForWorker
    case unrelated
}

/// Reduces harness source text to the lines the sandboxed worker build would
/// compile: `#if MLXFAST_TRUSTED_HARNESS` branches are dropped,
/// `#if !MLXFAST_TRUSTED_HARNESS` branches are kept (their `#else` dropped),
/// and any other conditional is preserved literally. The trusted tree may
/// carry trusted-only plumbing, but every shared line must stay identical to
/// its worker twin. Lines are trimmed and blank lines skipped so the
/// downstream comparison ignores re-indentation and re-wrapping.
private func workerVisibleLines(in source: String, file: String) throws -> [String] {
    var stack: [HarnessConditionalRegion] = []
    var kept: [String] = []
    func insideDroppedRegion() -> Bool {
        stack.contains(.droppedForWorker)
    }
    for rawLine in source.components(separatedBy: "\n") {
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        if line.hasPrefix("#elseif") {
            guard stack.last == .unrelated else {
                throw HarnessDriftGuardFailure(
                    description: "\(file) uses #elseif on a MLXFAST_TRUSTED_HARNESS conditional; teach workerVisibleLines(in:file:) how to evaluate it before landing"
                )
            }
            if !insideDroppedRegion() {
                kept.append(line)
            }
            continue
        }
        if line.hasPrefix("#if") {
            let condition = line.dropFirst("#if".count)
                .trimmingCharacters(in: .whitespaces)
            if condition == "!MLXFAST_TRUSTED_HARNESS" {
                stack.append(.keptForWorker)
            } else if condition == "MLXFAST_TRUSTED_HARNESS" {
                stack.append(.droppedForWorker)
            } else {
                stack.append(.unrelated)
                if !insideDroppedRegion() {
                    kept.append(line)
                }
            }
            continue
        }
        if line.hasPrefix("#else") {
            switch stack.last {
            case .keptForWorker:
                stack[stack.count - 1] = .droppedForWorker
            case .droppedForWorker:
                stack[stack.count - 1] = .keptForWorker
            default:
                if !insideDroppedRegion() {
                    kept.append(line)
                }
            }
            continue
        }
        if line.hasPrefix("#endif") {
            if let region = stack.popLast() {
                if region == .unrelated, !insideDroppedRegion() {
                    kept.append(line)
                }
            } else {
                kept.append(line)
            }
            continue
        }
        if !line.isEmpty, !insideDroppedRegion() {
            kept.append(line)
        }
    }
    guard stack.isEmpty else {
        throw HarnessDriftGuardFailure(
            description: "\(file) has an unterminated #if; the drift guard cannot parse it"
        )
    }
    return kept
}

/// Whitespace-insensitive fingerprint of the kept lines. Stripping all
/// whitespace means pure re-wrapping of an argument list never counts as
/// drift (the trees re-wrap many call sites differently); the trade-off is
/// that an edit changing nothing but spacing inside a string literal is
/// tolerated too. Any fix that adds, removes, or reorders non-whitespace
/// characters cannot hide behind this normalization.
private func condensedText(_ lines: [String]) -> String {
    lines.joined().filter { !$0.isWhitespace }
}

private func firstMismatchHint(worker: [String], trusted: [String]) -> String {
    let limit = min(worker.count, trusted.count)
    var index = 0
    while index < limit, worker[index] == trusted[index] {
        index += 1
    }
    let workerLine = index < worker.count ? truncatedForMessage(worker[index]) : "<end of file>"
    let trustedLine = index < trusted.count ? truncatedForMessage(trusted[index]) : "<end of file>"
    return "First mismatching worker-visible statement (index \(index); may be a re-wrap near the real drift): worker copy has [\(workerLine)], trusted copy has [\(trustedLine)]."
}

private func truncatedForMessage(_ line: String) -> String {
    line.count <= 96 ? line : String(line.prefix(96)) + "..."
}

private func fileDigestRegion(inFileAt path: String) throws -> String {
    let source = try String(contentsOfFile: path, encoding: .utf8)
    let start = try #require(
        source.range(of: "static func fileDigest("),
        "expected a fileDigest implementation in \(path)"
    )
    let end = try #require(
        source.range(of: "\n    static func ", range: start.upperBound..<source.endIndex),
        "expected a declaration after fileDigest in \(path)"
    )
    return String(source[start.lowerBound..<end.lowerBound])
}
