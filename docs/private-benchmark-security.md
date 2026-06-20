# Private Benchmark Security

The private correctness prompts and private golden data must be treated as
trusted harness material. They are not part of the contestant modifiable
surface.

## Required GitHub setup

Store private prompt/golden download credentials only as secrets on the
`benchmark-private-prompts` GitHub Environment. Do not store them as
repository-wide or organization-wide secrets.

Configure the `benchmark-private-prompts` Environment with:

- Deployment branches restricted to `main`.
- Required reviewers for private benchmark runs.
- No fork or submission branch access to the environment.

The benchmark workflow also verifies at runtime that it is executing from:

```text
Layr-Labs/mlxfast-challenge-dev/.github/workflows/benchmark.yml@refs/heads/main
```

This runtime check is defense in depth. The GitHub Environment branch
restriction is the real boundary that prevents a changed workflow file from
printing private secrets before the guard can run.

## Submission flow

Run private benchmarks by dispatching the trusted `benchmark.yml` workflow on
`main` and passing the contestant commit or branch as `submission_ref`.

The workflow checks out trusted `main`, then overlays only the `editablePaths`
from `benchmark.json` out of the submitted ref. Submitted workflow files,
scripts, tests, and harness code are not used.

## Output policy

For `submission_ref` runs:

- Correctness and benchmark process logs are redirected to private runner temp
  files and are not uploaded.
- The workflow prints only fixed heartbeat lines while submitted code is
  running.
- `score.json`, `benchmark-integrity.json`, and correctness artifacts are
  uploaded only after strict schema and hash validation succeeds.
- Correctness traces are disabled.

This prevents submitted code from using GitHub logs or uploaded artifacts as a
direct prompt-exfiltration channel.

## Residual channel

Submitted code still participates in inference on hidden prompt tokens. Any
public feedback from that run, including pass/fail, score, timing, or repeated
submission attempts, is a possible low-bandwidth covert channel. The workflow
hardening above blocks direct extraction paths, but competition policy should
still limit repeated private benchmark attempts and avoid exposing per-case
failure details for hidden cases.

The committed public fixture is temporary test data and is not a substitute for
the private hidden prompt/golden store.
