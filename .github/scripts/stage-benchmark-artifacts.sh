#!/usr/bin/env bash
# Copy already-validated benchmark outputs into a single upload directory.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: stage-benchmark-artifacts.sh DEST NAME=PATH..." >&2
  exit 2
fi

dest="$1"
shift

case "${dest}" in
  /tmp/mlxfast-artifacts-*|"${RUNNER_TEMP:-/tmp}"/mlxfast-artifacts-*) ;;
  *)
    echo "artifact destination must be under /tmp/mlxfast-artifacts-* or RUNNER_TEMP/mlxfast-artifacts-*: ${dest}" >&2
    exit 2
    ;;
esac

rm -rf "${dest}"
mkdir -p "${dest}"

for mapping in "$@"; do
  case "${mapping}" in
    *=*) ;;
    *)
      echo "artifact mapping must be NAME=PATH, got ${mapping}" >&2
      exit 2
      ;;
  esac

  name="${mapping%%=*}"
  source_path="${mapping#*=}"

  if [[ "${name}" == */* || "${name}" == "." || "${name}" == ".." || -z "${name}" ]]; then
    echo "artifact name must be a simple filename, got ${name}" >&2
    exit 2
  fi
  if [[ ! -f "${source_path}" || -L "${source_path}" ]]; then
    echo "artifact source must be a regular non-symlink file: ${source_path}" >&2
    exit 1
  fi

  cp "${source_path}" "${dest}/${name}"
done

.github/scripts/deny-private-artifacts.sh "${dest}"
echo "benchmark: staged artifacts in ${dest}"
