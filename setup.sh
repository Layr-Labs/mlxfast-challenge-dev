#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only DeepSeek harness.
set -euo pipefail

REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-mlx-community/DeepSeek-V4-Flash-4bit}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-main}"
REFERENCE_BASE_URL="${MLXFAST_REFERENCE_BASE_URL:-https://huggingface.co/${REFERENCE_MODEL_REPO}/resolve/${REFERENCE_REVISION}}"
REFERENCE_MIN_FREE_GIB="${MLXFAST_REFERENCE_MIN_FREE_GIB:-170}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
REFERENCE_METADATA_FILES=(
  "README.md"
  "chat_template.jinja"
  "config.json"
  "generation_config.json"
  "model.safetensors.index.json"
  "tokenizer.json"
  "tokenizer_config.json"
)

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

ensure_reference_space() {
  local directory="$1"
  local available_kib
  local required_kib

  if ! [[ "${REFERENCE_MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_MIN_FREE_GIB must be an integer" >&2
    return 1
  fi

  available_kib="$(df -Pk "${directory}" | awk 'NR == 2 {print $4}')"
  required_kib=$((REFERENCE_MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kib}" || "${available_kib}" -lt "${required_kib}" ]]; then
    cat >&2 <<EOF
setup.sh: not enough free disk space for ${REFERENCE_MODEL_REPO}.

Need at least ${REFERENCE_MIN_FREE_GIB} GiB free under ${directory}; available is $((available_kib / 1024 / 1024)) GiB.
Set MLXFAST_REFERENCE_DIR to a larger SSD, or set MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
and place/mount the checkpoint manually.

EOF
    return 1
  fi
}

download_reference_file() {
  local file="$1"
  local output_path="$2"
  local marker_path="${output_path}.complete"
  local url="${REFERENCE_BASE_URL%/}/${file}"

  if [[ -f "${marker_path}" && -s "${output_path}" ]]; then
    echo "setup.sh: already downloaded ${file}"
    return 0
  fi

  if [[ "${url}" == http://* || "${url}" == https://* ]]; then
    url="${url}?download=true"
  fi

  mkdir -p "$(dirname "${output_path}")"
  echo "setup.sh: downloading ${file}"
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 2 \
    --continue-at - \
    --output "${output_path}" \
    "${url}"
  touch "${marker_path}"
}

list_reference_shards() {
  local index_path="$1"

  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: Swift binary missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi

  "${SWIFT_BIN}" checkpoint-shards --index "${index_path}"
}

download_reference_weights() {
  local reference_dir="$1"
  local parent_dir
  local partial_dir
  local file
  local index_path
  local shard_list
  local shard_files=()

  if [[ -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference weights already present at ${reference_dir}"
    return 0
  fi

  if [[ -e "${reference_dir}" ]]; then
    cat >&2 <<EOF
setup.sh: ${reference_dir} exists but does not contain config.json.

Move it aside or set MLXFAST_REFERENCE_DIR to a complete checkpoint directory.

EOF
    return 1
  fi

  parent_dir="$(dirname "${reference_dir}")"
  partial_dir="${reference_dir}.partial"
  mkdir -p "${parent_dir}"

  ensure_reference_space "${parent_dir}"
  if [[ -e "${partial_dir}" && ! -d "${partial_dir}" ]]; then
    echo "setup.sh: partial download path exists but is not a directory: ${partial_dir}" >&2
    return 1
  fi
  mkdir -p "${partial_dir}"

  echo "setup.sh: downloading ${REFERENCE_MODEL_REPO} from ${REFERENCE_BASE_URL}"
  for file in "${REFERENCE_METADATA_FILES[@]}"; do
    download_reference_file "${file}" "${partial_dir}/${file}"
  done

  if [[ ! -f "${partial_dir}/config.json" ]]; then
    echo "setup.sh: downloaded checkpoint is missing config.json" >&2
    return 1
  fi
  index_path="${partial_dir}/model.safetensors.index.json"
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: downloaded checkpoint is missing model.safetensors.index.json" >&2
    return 1
  fi

  if ! shard_list="$(list_reference_shards "${index_path}")"; then
    return 1
  fi
  while IFS= read -r file; do
    if [[ -n "${file}" ]]; then
      shard_files+=("${file}")
    fi
  done <<< "${shard_list}"
  if [[ "${#shard_files[@]}" -eq 0 ]]; then
    echo "setup.sh: checkpoint index did not list any safetensors shards" >&2
    return 1
  fi

  echo "setup.sh: checkpoint index lists ${#shard_files[@]} safetensors shard(s)"
  for file in "${shard_files[@]}"; do
    download_reference_file "${file}" "${partial_dir}/${file}"
  done

  find "${partial_dir}" -name "*.complete" -type f -delete
  mv "${partial_dir}" "${reference_dir}"
  echo "setup.sh: downloaded reference weights to ${reference_dir}"
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

download_reference_weights "${REFERENCE_DIR}"
