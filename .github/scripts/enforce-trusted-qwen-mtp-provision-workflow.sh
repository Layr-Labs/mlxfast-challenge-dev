#!/usr/bin/env bash
# Trusted-context guard for the Qwen-MTP golden PROVISIONING workflow.
#
# The Qwen-MTP twin of enforce-trusted-dflash-provision-workflow.sh, checked
# against qwen-mtp-provision-goldens.yml instead of the DFlash provisioning
# workflow. Same four checks, same order, same fail-closed posture: this
# repository only, workflow_dispatch only, an allowlisted ref only, and a
# GITHUB_WORKFLOW_REF that names THIS workflow file at THAT ref.
#
# WHY THE BRANCH SET IS NARROW. The ranked job admits submissions/*, baseline/*
# and yukon/baseline/* because a submission has to be able to run against
# trusted main's harness. This job holds R2 WRITE credentials and its whole
# output is an object that later becomes ranked hidden material -- so a dispatch
# from any branch a participant can create must never reach it. Copying the
# ranked allowlist would hand submissions/* the ability to overwrite a hidden
# golden.
#
# WHAT THIS GUARD DOES NOT DO. `workflow_dispatch` runs the workflow file, and
# these scripts, from the ref the dispatcher selected, so this check is code the
# dispatched ref supplies about itself. It is decisive against a dispatch from a
# ref outside the allowlist, a fork or mirror of this repository, a non-dispatch
# trigger, and a job borrowing this guard under a different workflow file. It is
# NOT, by itself, a defence against an actor who can both push an allowlisted
# branch and dispatch it, because that actor can edit this file on that branch.
# The control that binds such an actor is the environment's deployment-branch
# policy on the credentialled environment, plus the branch protection on each
# allowlisted ref; this guard is the in-repo half of that pair, not a
# replacement for it.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/qwen-mtp-provision-goldens.yml"

##############################################################################
##                                                                          ##
##   TEMPORARY REF ALLOWLIST ENTRY -- READ THIS BEFORE GO-LIVE              ##
##                                                                          ##
##   This guard allows refs/heads/qwen36-mtp-track IN ADDITION to           ##
##   refs/heads/main. The DFlash twin is main-only; this extra ref is the   ##
##   one deliberate difference and it is scheduled for REMOVAL.             ##
##                                                                          ##
##   Grep this file for "TEMPORARY (2026-08-13)": three marked sites -- the ##
##   constant below, the refusal message, and the runtime ::warning:: this  ##
##   guard prints on every dispatch that actually uses the extra ref.       ##
##                                                                          ##
##############################################################################

# TEMPORARY (2026-08-13): pre-merge migration window -- qwen36-mtp-track is
# signature-protected (verified-committer branch rule); REMOVE this ref at
# go-live merge
readonly MIGRATION_REF="refs/heads/qwen36-mtp-track"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::Qwen-MTP golden provisioning must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::Qwen-MTP golden provisioning only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" && "${GITHUB_REF}" != "${MIGRATION_REF}" ]]; then
  echo "::error::Qwen-MTP golden provisioning ref is not allowed: ${GITHUB_REF}" >&2
  echo "::error::this job holds R2 WRITE credentials and freezes ranked hidden material; only refs/heads/main and ${MIGRATION_REF} may dispatch it" >&2
  # TEMPORARY (2026-08-13): pre-merge migration window -- qwen36-mtp-track is
  # signature-protected (verified-committer branch rule); REMOVE this ref at
  # go-live merge
  exit 1
fi

if [[ "${GITHUB_REF}" == "${MIGRATION_REF}" ]]; then
  # TEMPORARY (2026-08-13): pre-merge migration window -- qwen36-mtp-track is
  # signature-protected (verified-committer branch rule); REMOVE this ref at
  # go-live merge
  echo "::warning::TEMPORARY REF IN USE: this dispatch came from ${MIGRATION_REF}, not refs/heads/main. The extra ref exists only for the pre-merge migration window and must be removed from this guard at the go-live merge."
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "qwen-mtp-provision-goldens: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
