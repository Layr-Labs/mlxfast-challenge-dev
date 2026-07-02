#!/usr/bin/env bash
# Write a reproducible content hash of a transformed weights/ tree to stdout (or a file
# if given a second argument). Run identically on every machine in a parallel-correctness
# fleet so combine-parallel-correctness.sh can confirm they all transformed to
# byte-identical output before trusting any of their results together.
set -euo pipefail

WEIGHTS_PATH="${1:?usage: hash-weights-directory.sh WEIGHTS_PATH [OUTPUT_PATH]}"
OUTPUT_PATH="${2:-}"

if [[ ! -d "${WEIGHTS_PATH}" ]]; then
  echo "hash-weights-directory: ${WEIGHTS_PATH} is not a directory" >&2
  exit 1
fi

hash="$(
  find "${WEIGHTS_PATH}" -type f ! -name '.benchmark-source.sha256' ! -name '.gitkeep' -print0 \
    | LC_ALL=C sort -z \
    | while IFS= read -r -d '' path; do
        # shasum's own output line embeds the exact path it was given
        # ("<hash>  <path>"), which includes WEIGHTS_PATH -- that varies between
        # machines even for byte-identical trees, so extract just the hash and
        # pair it with our own path made relative to WEIGHTS_PATH instead of
        # feeding shasum's raw output (path and all) into the outer hash.
        file_hash="$(shasum -a 256 "${path}" | awk '{print $1}')"
        printf '%s  %s\n' "${file_hash}" "${path#"${WEIGHTS_PATH}"/}"
      done \
    | shasum -a 256 \
    | awk '{print $1}'
)"

if [[ -n "${OUTPUT_PATH}" ]]; then
  printf '%s\n' "${hash}" > "${OUTPUT_PATH}"
else
  printf '%s\n' "${hash}"
fi
