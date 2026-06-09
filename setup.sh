#!/usr/bin/env bash
# Bootstrap Python dependencies and, by default, the reference Gemma weights.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

find_python() {
  local candidate

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
      return 0
    fi
    echo "setup.sh: PYTHON_BIN=${PYTHON_BIN} was not found" >&2
    return 1
  fi

  for candidate in python3.11 python3 python; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done

  return 1
}

host_python="$(find_python || true)"
if [[ -z "${host_python}" ]]; then
  echo "setup.sh: Python 3.11+ is required" >&2
  exit 1
fi

"${host_python}" -m venv "${VENV_DIR}"
python="${VENV_DIR}/bin/python"
if [[ ! -x "${python}" ]]; then
  echo "setup.sh: virtualenv Python not found at ${python}" >&2
  exit 1
fi

"${python}" -m pip install --upgrade pip setuptools wheel
"${python}" -m pip install -e . "huggingface_hub>=0.23"

if [[ "${QUANTIZATIONFAIL_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "setup.sh: skipping reference weight download"
  exit 0
fi

"${python}" -m quantizationfail.cli weights
