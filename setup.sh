#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only DeepSeek harness.
set -euo pipefail

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

  if [[ -n "${MLXFAST_MACTOP_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_MACTOP_BIN}" ]]; then
      echo "setup.sh: using mactop at ${MLXFAST_MACTOP_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_MACTOP_BIN is set but not executable: ${MLXFAST_MACTOP_BIN}" >&2
    return 1
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

ensure_swift_toolchain() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: this Swift harness targets macOS on Apple Silicon" >&2
    exit 1
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "setup.sh: swift was not found; install Xcode command line tools" >&2
    exit 1
  fi

  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "setup.sh: xcodebuild was not found; install Xcode" >&2
    exit 1
  fi
}

ensure_mactop

ensure_swift_toolchain

echo "setup.sh: building Swift harness"
mkdir -p .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
swift build -c release

if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
  echo "setup.sh: skipping mlx.metallib build"
else
  echo "setup.sh: building mlx.metallib for MLX Swift runtime"
  tools/build-mlx-metallib.sh
fi

REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "setup.sh: skipping reference weight download"
  exit 0
fi

if [[ -f "${REFERENCE_DIR}/config.json" ]]; then
  echo "setup.sh: reference weights already present at ${REFERENCE_DIR}"
  exit 0
fi

cat >&2 <<EOF
setup.sh: reference weights are not present at ${REFERENCE_DIR}.

The Swift-only harness no longer bootstraps Python just to download weights.
Place the DeepSeek V4 Flash checkpoint there, or rerun with:

  MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 ./setup.sh

EOF
exit 1
