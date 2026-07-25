#!/usr/bin/env bash
# Write a reproducible content hash of a transformed weights/ tree to stdout (or a file
# if given a second argument). The ranked benchmark.yml hashes the transform output
# with this immediately after the (untrusted) transform runs, and the gates run's
# reported weights hash is cross-checked against it before any score is trusted.
set -euo pipefail

WEIGHTS_PATH="${1:?usage: hash-weights-directory.sh WEIGHTS_PATH [OUTPUT_PATH]}"
OUTPUT_PATH="${2:-}"

if [[ ! -d "${WEIGHTS_PATH}" || -L "${WEIGHTS_PATH}" ]]; then
  echo "hash-weights-directory: ${WEIGHTS_PATH} is not a non-symlink directory" >&2
  exit 1
fi
WEIGHTS_PATH="$(cd -P "${WEIGHTS_PATH}" && pwd -P)"

XXD_BIN="$(command -v xxd 2>/dev/null || true)"
if [[ -z "${XXD_BIN}" ]]; then
  echo "hash-weights-directory: xxd is required to encode raw SHA256 digest bytes" >&2
  exit 1
fi

# Match LagunaRuntime.directoryDigest exactly: for every sorted relative path,
# hash path bytes + NUL + the raw 32-byte file SHA256 + NUL.
invalid_entry="$(find "${WEIGHTS_PATH}" -mindepth 1 ! -type d ! -type f -print -quit)"
if [[ -n "${invalid_entry}" ]]; then
  echo "hash-weights-directory: weights tree contains a symlink or non-regular entry: ${invalid_entry}" >&2
  exit 1
fi
# Reject hardlinked files (link count != 1): a second name aliasing these
# bytes -- e.g. planted by the sandboxed bench uid -- must not slip through the
# transform-output gate. Mirrors overlay-editable-paths.sh and
# LagunaRuntime.directoryDigest / requireSingleHardLink.
hardlinked_entry="$(find "${WEIGHTS_PATH}" -type f -links +1 -print -quit)"
if [[ -n "${hardlinked_entry}" ]]; then
  echo "hash-weights-directory: weights tree contains a hardlinked file: ${hardlinked_entry}" >&2
  exit 1
fi

# MLXFAST_HASH_JOBS>1 hashes the (few, very large) weight files concurrently.
# It DEFAULTS TO 1 -- byte-identical to the original sequential loop -- because
# one caller must stay sequential: the pre-timing scrub re-hashes this tree as
# the last action before the quiescence wait and the 40C thermal gate, and a
# parallel CPU burn there would push heat into the scored window. Only the
# ranked audit-hash step, which runs early, opts in.
#
# The digest is order-sensitive (it must match LagunaRuntime.directoryDigest),
# and these filenames come from the UNTRUSTED transform, so the parallel path
# never parses hasher stdout: each file's digest is written to a temp file
# named by its sorted index, then consumed in index order. Paths move through
# NUL-terminated files only, so any byte except NUL survives intact.
HASH_JOBS="${MLXFAST_HASH_JOBS:-1}"
if [[ ! "${HASH_JOBS}" =~ ^[0-9]+$ || "${HASH_JOBS}" -lt 1 ]]; then
  echo "hash-weights-directory: MLXFAST_HASH_JOBS must be a positive integer" >&2
  exit 1
fi

if [[ "${HASH_JOBS}" -eq 1 ]]; then
  hash="$(
    find "${WEIGHTS_PATH}" -type f -print0 \
      | LC_ALL=C sort -z \
      | while IFS= read -r -d '' path; do
          relative_path="${path#"${WEIGHTS_PATH}"/}"
          if [[ "${relative_path}" == '.benchmark-source.sha256' \
              || "${relative_path}" == '.gitkeep' ]]; then
            continue
          fi
          file_hash="$(shasum -a 256 "${path}" | awk '{print $1}')"
          printf '%s\0' "${relative_path}"
          printf '%s' "${file_hash}" | "${XXD_BIN}" -r -p
          printf '\0'
        done \
      | shasum -a 256 \
      | awk '{print $1}'
  )"
else
  work_dir="$(mktemp -d)"
  trap 'rm -rf "${work_dir}"' EXIT
  entries=0
  while IFS= read -r -d '' path; do
    relative_path="${path#"${WEIGHTS_PATH}"/}"
    if [[ "${relative_path}" == '.benchmark-source.sha256' \
        || "${relative_path}" == '.gitkeep' ]]; then
      continue
    fi
    printf '%s\0' "${path}" > "${work_dir}/${entries}.abs"
    printf '%s\0' "${relative_path}" > "${work_dir}/${entries}.rel"
    entries=$((entries + 1))
  done < <(find "${WEIGHTS_PATH}" -type f -print0 | LC_ALL=C sort -z)
  if [[ "${entries}" -eq 0 ]]; then
    echo "hash-weights-directory: no hashable files under ${WEIGHTS_PATH}" >&2
    exit 1
  fi
  seq 0 "$((entries - 1))" \
    | xargs -P "${HASH_JOBS}" -I '{}' /bin/bash -c '
        set -euo pipefail
        IFS= read -r -d "" p < "$1/{}.abs"
        shasum -a 256 "${p}" | awk "{print \$1}" > "$1/{}.hex"
      ' _ "${work_dir}"
  hash="$(
    for ((i = 0; i < entries; i++)); do
      IFS= read -r -d '' rel < "${work_dir}/${i}.rel"
      file_hash="$(tr -d '[:space:]' < "${work_dir}/${i}.hex")"
      if [[ ! "${file_hash}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "hash-weights-directory: parallel digest missing or malformed for entry ${i}" >&2
        exit 1
      fi
      printf '%s\0' "${rel}"
      printf '%s' "${file_hash}" | "${XXD_BIN}" -r -p
      printf '\0'
    done \
      | shasum -a 256 \
      | awk '{print $1}'
  )"
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
  printf '%s\n' "${hash}" > "${OUTPUT_PATH}"
else
  printf '%s\n' "${hash}"
fi
