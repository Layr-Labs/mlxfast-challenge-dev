#!/usr/bin/env bash
# Bootstrap system tools, Python dependencies, and the reference weights.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

load_homebrew_shellenv() {
  local candidate
  local candidates=()

  if [[ -n "${HOMEBREW_PREFIX:-}" ]]; then
    candidates+=("${HOMEBREW_PREFIX}/bin/brew")
  fi
  candidates+=(
    "/opt/homebrew/bin/brew"
    "/usr/local/bin/brew"
    "${HOME}/.linuxbrew/bin/brew"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      eval "$("${candidate}" shellenv)"
      return 0
    fi
  done

  return 1
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if load_homebrew_shellenv && command -v brew >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_HOMEBREW_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: Homebrew is not installed and MLXFAST_SKIP_HOMEBREW_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic Homebrew installation is only supported on macOS" >&2
    return 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "setup.sh: curl is required to install Homebrew" >&2
    return 1
  fi

  echo "setup.sh: Homebrew not found; installing Homebrew"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if ! load_homebrew_shellenv || ! command -v brew >/dev/null 2>&1; then
    echo "setup.sh: Homebrew installation finished, but brew is still not on PATH" >&2
    echo "setup.sh: open a new shell or run Homebrew's shellenv command, then retry" >&2
    return 1
  fi
}

ensure_mactop() {
  if [[ "${MLXFAST_SKIP_MACTOP_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: skipping mactop install"
    return 0
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: skipping mactop install; mactop is only available on macOS"
    return 0
  fi

  if command -v mactop >/dev/null 2>&1 || [[ -x "/opt/homebrew/bin/mactop" || -x "/usr/local/bin/mactop" ]]; then
    return 0
  fi

  ensure_homebrew
  echo "setup.sh: installing mactop with Homebrew"
  brew install mactop

  if ! command -v mactop >/dev/null 2>&1 && [[ ! -x "/opt/homebrew/bin/mactop" && ! -x "/usr/local/bin/mactop" ]]; then
    echo "setup.sh: mactop installation finished, but the mactop binary was not found" >&2
    return 1
  fi
}

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

ensure_mactop

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

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "setup.sh: skipping reference weight download"
  exit 0
fi

"${python}" -m mlxfast.cli weights
