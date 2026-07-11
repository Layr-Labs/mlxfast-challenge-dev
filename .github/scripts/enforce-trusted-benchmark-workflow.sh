#!/usr/bin/env bash
# Ensure private benchmark material is only used by this repository's benchmark
# workflow FILE on trusted main. For workflow_dispatch, GitHub loads the
# workflow definition and every `run:` body from the dispatched ref, so the
# only way to guarantee the privileged job (self-hosted runner, private
# environment secrets, hidden goldens) executes trusted code is to require the
# dispatched ref to be `refs/heads/main` AND the resolved workflow ref to be
# that same trusted-main file. Candidate commits are supplied as DATA (the
# `submission_ref` input) and are checked out/overlaid separately; they can
# never select the workflow ref, so a pushed submissions/baseline branch can
# no longer run its own copy of this workflow.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/benchmark.yml"
readonly TRUSTED_REF="refs/heads/main"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

# Anchored to trusted main, NOT to ${GITHUB_REF}. Anchoring to the dispatched
# ref would be a tautology for workflow_dispatch (the workflow ref IS the
# dispatched ref), so it would never bind privilege to the trusted main SHA.
readonly EXPECTED_WORKFLOW_REF="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${TRUSTED_REF}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::private benchmark workflow must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::private benchmark workflow only supports workflow_dispatch" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "${TRUSTED_REF}" ]]; then
  echo "::error::private benchmark workflow must be dispatched on ${TRUSTED_REF}; current ref is ${GITHUB_REF}" >&2
  echo "::error::submit candidates via the submission_ref input, not by dispatching a submissions/baseline branch" >&2
  exit 1
fi

if [[ "${GITHUB_WORKFLOW_REF}" != "${EXPECTED_WORKFLOW_REF}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${EXPECTED_WORKFLOW_REF}" >&2
  exit 1
fi

echo "benchmark: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
