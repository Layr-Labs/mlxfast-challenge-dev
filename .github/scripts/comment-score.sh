#!/usr/bin/env bash
# Upserts the benchmark score as a marker-tagged PR comment via the gh CLI.
# Usage: GH_TOKEN=<token> PR_NUMBER=<n> comment-score.sh
# GITHUB_REPOSITORY is provided by the Actions runtime.
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${PR_NUMBER:?PR_NUMBER is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

marker='<!-- quantizationfail-benchmark -->'
# Suppress the step-summary side effect; this invocation only renders the body.
report="$(GITHUB_STEP_SUMMARY='' python3 .github/scripts/report-score.py)"
body="$(printf '%s\n%s' "${marker}" "${report}")"

existing_id="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" --paginate \
  --jq "[.[] | select(.body | startswith(\"${marker}\"))][0].id // empty")"

if [[ -n "${existing_id}" ]]; then
  gh api -X PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${existing_id}" -f body="${body}" >/dev/null
else
  gh api -X POST "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" -f body="${body}" >/dev/null
fi
echo "Score comment posted to PR #${PR_NUMBER}."
