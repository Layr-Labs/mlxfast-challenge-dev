#!/usr/bin/env python3
"""Render score.json as markdown.

Prints the report to stdout and, when running inside GitHub Actions
(GITHUB_STEP_SUMMARY set), appends it to the job's step summary.
"""
import json
import os
import sys


def render(payload: dict) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "## quantizationfail benchmark",
        "",
        f"**Score: `{payload['score']}`** (lower is better)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    # The metrics schema evolves with the harness; report whatever it emits.
    lines += [f"| {k} | {v} |" for k, v in sorted(metrics.items())]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    score_path = sys.argv[1] if len(sys.argv) > 1 else "score.json"
    with open(score_path) as f:
        payload = json.load(f)
    summary = render(payload)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
