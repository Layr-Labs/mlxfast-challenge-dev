import Foundation
import Testing

@Test
func setupScriptDefaultsToFastReferenceMirror() throws {
    let setup = try String(
        contentsOfFile: "setup.sh",
        encoding: .utf8
    )

    #expect(setup.contains("DEFAULT_REFERENCE_BASE_URL=\"https://ds4.darkbloom.ai/deepseek-v4-flash-4bit\""))
    #expect(setup.contains("REFERENCE_BASE_URL=\"${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}\""))
    #expect(setup.contains("Usage: ./setup.sh"))
    #expect(setup.contains("downloaded ${total}/${total} safetensors shard(s)"))
    #expect(setup.contains("setup.sh: setup complete elapsed="))
    #expect(setup.contains("${SWIFT_BIN} transform"))
}

@Test
func benchmarkScriptFailsWhenScorePayloadFails() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    score_path="score.json"
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--score-path" ]; then
        shift
        score_path="$1"
      fi
      shift || exit 1
    done
    cat > "$score_path" <<'JSON'
    {
      "score": null,
      "passed": false,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ]) { _, new in new }

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus != 0)
    #expect(FileManager.default.fileExists(atPath: score.path))
    #expect(FileManager.default.fileExists(atPath: integrity.path))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-benchmark-script-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
