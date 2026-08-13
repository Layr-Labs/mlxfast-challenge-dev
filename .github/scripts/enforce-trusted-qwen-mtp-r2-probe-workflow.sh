#!/usr/bin/env bash
# Trusted-context guard for the Qwen-MTP R2 key probe workflow.
#
# The Qwen-MTP twin of enforce-trusted-dflash-probe-workflow.sh, checked against
# qwen-mtp-r2-key-probe.yml instead of the DFlash probe workflow. Same four
# checks, same order, same fail-closed posture: this repository only,
# workflow_dispatch only, an allowlisted ref only, and a GITHUB_WORKFLOW_REF
# that names THIS workflow file at THAT ref.
#
# WHY A READ-ONLY DIAGNOSTIC NEEDS THIS AT ALL. qwen-mtp-r2-key-probe.yml binds
# the real R2 credentials and executes `.github/scripts/download-r2-object.sh`
# FROM THE DISPATCHED REF against hidden competition material. "It only prints
# an HTTP status, a byte count and a sha256" is a property of the trusted copy
# of that script, not of the workflow: a dispatch from a branch whose
# download-r2-object.sh had been edited to print the object body would have
# copied hidden golden bytes into an Actions log.
#
# WHY THE BRANCH SET IS NARROW, and not the ranked allowlist. The ranked job
# admits submissions/*, baseline/* and yukon/baseline/* because a submission has
# to be able to run against the trusted harness on this track's base branch
# (qwen36-mtp-track -- this track is branch-targeted, so its base is not main);
# those namespaces carry
# participant-authored content by design. This job has no such need -- it
# resolves object keys and prints a status, a byte count and a sha256 -- and it
# reads raw hidden material while doing it. Copying the ranked allowlist here
# would hand every branch namespace a participant can influence the ability to
# choose the code that touches a hidden golden, which is the same reasoning that
# makes enforce-trusted-qwen-mtp-provision-workflow.sh narrow.
#
# WHAT THIS GUARD DOES NOT DO. `workflow_dispatch` runs the workflow file, and
# these scripts, from the ref the dispatcher selected, so this check is code the
# dispatched ref supplies about itself. It is decisive against a dispatch from a
# ref outside the allowlist, a fork or mirror of this repository, a non-dispatch
# trigger, and a job borrowing this guard under a different workflow file -- and
# it makes any of those a loud refusal in the log. It is NOT, by itself, a
# defence against an actor who can both push an allowlisted branch and dispatch
# it, because that actor can edit this file on that branch. The control that
# binds such an actor is the environment's deployment-branch policy on the
# credentialled environment, plus the branch protection on each allowlisted ref;
# this guard is the in-repo half of that pair, not a replacement for it.
set -euo pipefail

readonly TRUSTED_REPOSITORY="Layr-Labs/mlxfast-challenge-dev"
readonly WORKFLOW_PATH=".github/workflows/qwen-mtp-r2-key-probe.yml"

##############################################################################
##                                                                          ##
##   TRACK-BASE REF ALLOWLIST ENTRY -- PERMANENT BY DESIGN                  ##
##                                                                          ##
##   This guard allows refs/heads/qwen36-mtp-track IN ADDITION to           ##
##   refs/heads/main. The DFlash twin is main-only; this extra ref is the   ##
##   one deliberate difference, and it is NOT scheduled for removal.        ##
##   The Qwen-MTP track is branch-targeted permanently (operator design     ##
##   decision 2026-08-13): qwen36-mtp-track is the track's base branch,     ##
##   main is the DFlash track's, and no merge between them is planned.      ##
##   Operators dispatch this probe from the track base, so that ref is the  ##
##   NORMAL one here and main is the vestigial one.                         ##
##                                                                          ##
##   Grep this file for "PERMANENT BY DESIGN (2026-08-13)": two marked      ##
##   sites -- the constant below and the refusal message. A dispatch from   ##
##   the track base emits no annotation, exactly as a dispatch from main    ##
##   emits none: both are ordinary trusted operations.                      ##
##                                                                          ##
##############################################################################

# PERMANENT BY DESIGN (2026-08-13): qwen36-mtp-track is this track's base branch
# and is signature-protected (verified-committer branch rule), the same property
# that makes main trustworthy here. KEEP this ref; it has no removal date.
readonly TRACK_BASE_REF="refs/heads/qwen36-mtp-track"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKFLOW_REF:?GITHUB_WORKFLOW_REF is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"

if [[ "${GITHUB_REPOSITORY}" != "${TRUSTED_REPOSITORY}" ]]; then
  echo "::error::Qwen-MTP R2 key probe must run in ${TRUSTED_REPOSITORY}, not ${GITHUB_REPOSITORY}" >&2
  exit 1
fi

if [[ "${GITHUB_EVENT_NAME}" != "workflow_dispatch" ]]; then
  echo "::error::Qwen-MTP R2 key probe only supports workflow_dispatch (got ${GITHUB_EVENT_NAME}); it must never run on push" >&2
  exit 1
fi

if [[ "${GITHUB_REF}" != "refs/heads/main" && "${GITHUB_REF}" != "${TRACK_BASE_REF}" ]]; then
  echo "::error::Qwen-MTP R2 key probe ref is not allowed: ${GITHUB_REF}" >&2
  # PERMANENT BY DESIGN (2026-08-13): ${TRACK_BASE_REF} is this track's base
  # branch, not a migration ref. KEEP it in this allowlist.
  echo "::error::this job holds R2 credentials and downloads hidden competition material with scripts taken from the dispatched ref; only refs/heads/main and ${TRACK_BASE_REF} (this track's base branch) may dispatch it" >&2
  exit 1
fi

expected_workflow_ref="${TRUSTED_REPOSITORY}/${WORKFLOW_PATH}@${GITHUB_REF}"
if [[ "${GITHUB_WORKFLOW_REF}" != "${expected_workflow_ref}" ]]; then
  echo "::error::unexpected workflow ref ${GITHUB_WORKFLOW_REF}" >&2
  echo "::error::expected ${expected_workflow_ref}" >&2
  exit 1
fi

echo "qwen-mtp-r2-key-probe: trusted workflow verified ${GITHUB_WORKFLOW_REF}"
