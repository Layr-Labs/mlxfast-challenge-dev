#!/usr/bin/env bash
# Bootstrap system tools and build the Swift-only DeepSeek harness.
set -euo pipefail

REFERENCE_MODEL_REPO="${MLXFAST_REFERENCE_MODEL_REPO:-mlx-community/DeepSeek-V4-Flash-4bit}"
REFERENCE_REVISION="${MLXFAST_REFERENCE_REVISION:-main}"
DEFAULT_REFERENCE_BASE_URL="https://ds4.darkbloom.ai/deepseek-v4-flash-4bit"
REFERENCE_BASE_URL="${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}"
REFERENCE_AUTH_HEADER="${MLXFAST_REFERENCE_AUTH_HEADER:-}"
REFERENCE_APPEND_DOWNLOAD_QUERY="${MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}"
REFERENCE_MANIFEST_PATH="${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_deepseek_v4_flash_4bit.sha256}"
REFERENCE_HASH_VERIFY="${MLXFAST_REFERENCE_HASH_VERIFY:-1}"
REFERENCE_MIN_FREE_GIB="${MLXFAST_REFERENCE_MIN_FREE_GIB:-170}"
REFERENCE_DOWNLOAD_JOBS="${MLXFAST_REFERENCE_DOWNLOAD_JOBS:-8}"
SWIFT_BIN="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
MLX_METALLIB="${MLXFAST_MLX_METALLIB:-$(dirname "${SWIFT_BIN}")/mlx.metallib}"
REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-reference_weights/DeepSeek-V4-Flash-4bit}"
SETUP_STARTED_SECONDS="${SECONDS}"
REFERENCE_REQUIRED_METADATA_FILES=(
  "config.json"
  "model.safetensors.index.json"
)
REFERENCE_OPTIONAL_METADATA_FILES=(
  "README.md"
  "chat_template.jinja"
  "generation_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
)

print_help() {
  cat <<EOF
Usage: ./setup.sh

Checks the local macOS/Apple Silicon toolchain, builds the Swift harness,
builds mlx.metallib, and downloads the DeepSeek V4 Flash 4-bit reference
checkpoint when it is not already present.

Important environment variables:
  MLXFAST_REFERENCE_DIR              Reference checkpoint directory.
                                     Default: ${REFERENCE_DIR}
  MLXFAST_REFERENCE_BASE_URL         HTTP prefix for checkpoint files.
                                     Default: ${DEFAULT_REFERENCE_BASE_URL}
  MLXFAST_REFERENCE_MANIFEST_PATH    SHA256 manifest for the reference files.
                                     Default: ${REFERENCE_MANIFEST_PATH}
  MLXFAST_REFERENCE_DOWNLOAD_JOBS    Parallel safetensors downloads.
                                     Default: ${REFERENCE_DOWNLOAD_JOBS}
  MLXFAST_REFERENCE_MIN_FREE_GIB     Required free space before download.
                                     Default: ${REFERENCE_MIN_FREE_GIB}
  MLXFAST_REFERENCE_HASH_VERIFY=0    Skip reference SHA256 verification.
  MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1    Build tools only; do not download weights.
  MLXFAST_SKIP_MLX_METALLIB=1        Skip mlx.metallib build.
  MLXFAST_SKIP_MACTOP_INSTALL=1      Skip mactop install/check.
  MLXFAST_MACTOP_BIN=/path/mactop    Use a specific mactop binary.

After setup:
  .github/scripts/run-offline.sh .build/release/mlxfast-swift transform
  ./benchmark.sh
EOF
}

if [[ "$#" -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_help
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument '$1'" >&2
      echo "Run ./setup.sh --help for usage." >&2
      exit 2
      ;;
  esac
fi

format_duration() {
  local total_seconds="${1:-0}"
  printf '%02d:%02d:%02d' \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

path_size_gib() {
  local path="$1"
  local size_kib

  if [[ ! -e "${path}" ]]; then
    printf '0.0'
    return 0
  fi

  size_kib="$(du -sk "${path}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${size_kib}" ]]; then
    printf 'unknown'
    return 0
  fi

  awk -v kib="${size_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }'
}

print_setup_summary() {
  local reference_status="${1:-ready}"
  local elapsed="$((SECONDS - SETUP_STARTED_SECONDS))"
  local reference_line
  local metallib_line

  if [[ "${reference_status}" == "skipped" ]]; then
    reference_line="skipped (${REFERENCE_DIR})"
  elif [[ -f "${REFERENCE_DIR}/config.json" ]]; then
    reference_line="${REFERENCE_DIR} ($(path_size_gib "${REFERENCE_DIR}") GiB)"
  else
    reference_line="missing (${REFERENCE_DIR})"
  fi

  if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
    metallib_line="skipped (${MLX_METALLIB})"
  else
    metallib_line="${MLX_METALLIB}"
  fi

  cat <<EOF
setup.sh: setup complete elapsed=$(format_duration "${elapsed}")
setup.sh: summary
  binary: ${SWIFT_BIN}
  mlx.metallib: ${metallib_line}
  reference checkpoint: ${reference_line}
  next:
    .github/scripts/run-offline.sh ${SWIFT_BIN} transform
    ./benchmark.sh
EOF
}

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

find_cmake() {
  local candidate
  if [[ -n "${MLXFAST_CMAKE_BIN:-}" ]]; then
    if [[ -x "${MLXFAST_CMAKE_BIN}" ]]; then
      printf '%s\n' "${MLXFAST_CMAKE_BIN}"
      return 0
    fi
    echo "setup.sh: MLXFAST_CMAKE_BIN is set but not executable: ${MLXFAST_CMAKE_BIN}" >&2
    return 1
  fi

  if candidate="$(command -v cmake 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  for candidate in /opt/homebrew/bin/cmake /usr/local/bin/cmake; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

ensure_cmake() {
  if find_cmake >/dev/null; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_CMAKE_INSTALL:-0}" == "1" ]]; then
    echo "setup.sh: cmake is not installed and MLXFAST_SKIP_CMAKE_INSTALL=1" >&2
    return 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: automatic cmake installation is only supported on macOS" >&2
    return 1
  fi

  ensure_homebrew
  echo "setup.sh: installing cmake with Homebrew"
  brew install cmake

  if ! find_cmake >/dev/null; then
    echo "setup.sh: cmake installation finished, but cmake was not found" >&2
    return 1
  fi
}

ensure_swift_toolchain() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh: this Swift harness targets macOS on Apple Silicon" >&2
    exit 1
  fi

  if [[ "$(uname -m)" != "arm64" ]]; then
    echo "setup.sh: this Swift harness requires Apple Silicon (arm64)" >&2
    exit 1
  fi

  if ! command -v swift >/dev/null 2>&1; then
    echo "setup.sh: swift was not found; install Xcode command line tools with xcode-select --install" >&2
    exit 1
  fi

  if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "setup.sh: xcodebuild was not found; install Xcode" >&2
    exit 1
  fi

  if ! xcodebuild -version >/dev/null 2>&1; then
    cat >&2 <<EOF
setup.sh: xcodebuild is installed but not usable.

Open Xcode once, select its command line tools, and accept the license, then retry:

  sudo xcodebuild -license accept

EOF
    exit 1
  fi
}

ensure_metal_toolchain() {
  if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL:-0}" == "1" ]]; then
    cat >&2 <<EOF
setup.sh: Xcode's Metal Toolchain is not installed and MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1.

Install it, then retry:

  xcodebuild -downloadComponent MetalToolchain

EOF
    return 1
  fi

  echo "setup.sh: installing Xcode Metal Toolchain"
  if ! xcodebuild -downloadComponent MetalToolchain; then
    cat >&2 <<EOF
setup.sh: failed to install Xcode's Metal Toolchain.

Install it manually, then retry:

  xcodebuild -downloadComponent MetalToolchain

EOF
    return 1
  fi

  if ! xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    echo "setup.sh: Metal Toolchain installation finished, but xcrun still cannot execute metal" >&2
    return 1
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

download_url_for_file() {
  local url="$1"
  local append_query=0
  local separator="?"

  case "${REFERENCE_APPEND_DOWNLOAD_QUERY}" in
    1|true|TRUE|yes|YES)
      append_query=1
      ;;
    0|false|FALSE|no|NO)
      append_query=0
      ;;
    auto|"")
      if [[ "${url}" == https://huggingface.co/* || "${url}" == http://huggingface.co/* ]]; then
        append_query=1
      fi
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
      return 1
      ;;
  esac

  if [[ "${append_query}" == "1" ]]; then
    if [[ "${url}" == *\?* ]]; then
      separator="&"
    fi
    url="${url}${separator}download=true"
  fi

  printf '%s\n' "${url}"
}

download_reference_file() {
  local file="$1"
  local output_path="$2"
  local label="${3:-${file}}"
  local marker_path="${output_path}.complete"
  local url="${REFERENCE_BASE_URL%/}/${file}"
  local started_seconds

  if [[ -f "${marker_path}" && -s "${output_path}" ]]; then
    echo "setup.sh: already downloaded ${label}"
    return 0
  fi

  if ! url="$(download_url_for_file "${url}")"; then
    return 1
  fi

  mkdir -p "$(dirname "${output_path}")"
  started_seconds="${SECONDS}"
  echo "setup.sh: downloading ${label}"
  if [[ -n "${REFERENCE_AUTH_HEADER}" ]]; then
    curl \
      --fail \
      --location \
      --retry 5 \
      --retry-all-errors \
      --retry-delay 2 \
      --continue-at - \
      -H "${REFERENCE_AUTH_HEADER}" \
      --output "${output_path}" \
      "${url}"
  else
    curl \
      --fail \
      --location \
      --retry 5 \
      --retry-all-errors \
      --retry-delay 2 \
      --continue-at - \
      --output "${output_path}" \
      "${url}"
  fi
  touch "${marker_path}"
  echo "setup.sh: downloaded ${label} elapsed=$(format_duration "$((SECONDS - started_seconds))")"
}

download_optional_reference_file() {
  local file="$1"
  local output_path="$2"

  if download_reference_file "${file}" "${output_path}"; then
    return 0
  fi

  rm -f "${output_path}" "${output_path}.complete"
  echo "setup.sh: optional metadata ${file} was not available; continuing"
}

download_reference_shards() {
  local output_dir="$1"
  shift
  local jobs="${REFERENCE_DOWNLOAD_JOBS}"
  local total="$#"
  local started_seconds="${SECONDS}"

  if ! [[ "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "setup.sh: MLXFAST_REFERENCE_DOWNLOAD_JOBS must be a positive integer" >&2
    return 1
  fi

  if [[ "${jobs}" == "1" || "$#" -le 1 ]]; then
    local file
    local ordinal=0
    echo "setup.sh: downloading ${total} safetensors shard(s) with 1 parallel job"
    for file in "$@"; do
      ordinal=$((ordinal + 1))
      download_reference_file "${file}" "${output_dir}/${file}" "shard ${ordinal}/${total}: ${file}"
    done
    echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
    return 0
  fi

  echo "setup.sh: downloading ${total} safetensors shard(s) with ${jobs} parallel job(s)"
  export REFERENCE_BASE_URL
  export REFERENCE_AUTH_HEADER
  export REFERENCE_APPEND_DOWNLOAD_QUERY
  local ordinal=0
  for file in "$@"; do
    ordinal=$((ordinal + 1))
    printf "%s|%s\0" "${ordinal}" "${file}"
  done | xargs -0 -I{} -P "${jobs}" bash -c '
    set -euo pipefail
    record="$1"
    output_dir="$2"
    total="$3"
    ordinal="${record%%|*}"
    file="${record#*|}"
    output_path="${output_dir}/${file}"
    marker_path="${output_path}.complete"
    url="${REFERENCE_BASE_URL%/}/${file}"
    started_seconds="${SECONDS}"

    download_url_for_file() {
      local url="$1"
      local append_query=0
      local separator="?"

      case "${REFERENCE_APPEND_DOWNLOAD_QUERY:-auto}" in
        1|true|TRUE|yes|YES)
          append_query=1
          ;;
        0|false|FALSE|no|NO)
          append_query=0
          ;;
        auto|"")
          if [[ "${url}" == https://huggingface.co/* || "${url}" == http://huggingface.co/* ]]; then
            append_query=1
          fi
          ;;
        *)
          echo "setup.sh: MLXFAST_REFERENCE_APPEND_DOWNLOAD_QUERY must be auto, true, or false" >&2
          return 1
          ;;
      esac

      if [[ "${append_query}" == "1" ]]; then
        if [[ "${url}" == *\?* ]]; then
          separator="&"
        fi
        url="${url}${separator}download=true"
      fi

      printf "%s\n" "${url}"
    }

    if [[ -f "${marker_path}" && -s "${output_path}" ]]; then
      echo "setup.sh: already downloaded shard ${ordinal}/${total}: ${file}"
      exit 0
    fi

    url="$(download_url_for_file "${url}")"

    mkdir -p "$(dirname "${output_path}")"
    echo "setup.sh: downloading shard ${ordinal}/${total}: ${file}"
    if [[ -n "${REFERENCE_AUTH_HEADER:-}" ]]; then
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --silent \
        --show-error \
        -H "${REFERENCE_AUTH_HEADER}" \
        --output "${output_path}" \
        "${url}"
    else
      curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --retry-delay 2 \
        --continue-at - \
        --silent \
        --show-error \
        --output "${output_path}" \
        "${url}"
    fi
    touch "${marker_path}"
    echo "setup.sh: downloaded shard ${ordinal}/${total}: ${file} elapsed=$((SECONDS - started_seconds))s"
  ' _ {} "${output_dir}" "${total}"
  echo "setup.sh: downloaded ${total}/${total} safetensors shard(s) elapsed=$(format_duration "$((SECONDS - started_seconds))")"
}

list_reference_shards() {
  local index_path="$1"

  if [[ ! -x "${SWIFT_BIN}" ]]; then
    echo "setup.sh: Swift binary missing at ${SWIFT_BIN}; build failed or MLXFAST_SWIFT_BIN is wrong" >&2
    return 1
  fi

  "${SWIFT_BIN}" checkpoint-shards --index "${index_path}"
}

verify_reference_weights() {
  local reference_dir="$1"
  local index_path="${reference_dir}/model.safetensors.index.json"
  local shard_list
  local file
  local shard_files=()
  local missing=0

  if [[ ! -f "${reference_dir}/config.json" ]]; then
    echo "setup.sh: reference checkpoint is missing config.json at ${reference_dir}" >&2
    return 1
  fi
  if [[ ! -f "${index_path}" ]]; then
    echo "setup.sh: reference checkpoint is missing model.safetensors.index.json at ${reference_dir}" >&2
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

  for file in "${shard_files[@]}"; do
    if [[ ! -s "${reference_dir}/${file}" ]]; then
      echo "setup.sh: reference checkpoint is missing shard ${file} at ${reference_dir}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 1
  fi

  verify_reference_manifest "${reference_dir}"
  echo "setup.sh: verified reference checkpoint at ${reference_dir} (${#shard_files[@]} safetensors shard(s))"
}

verify_reference_manifest() {
  local reference_dir="$1"
  local line
  local expected_hash
  local expected_size
  local relative_path
  local extra
  local file_path
  local actual_size
  local actual_hash
  local checked=0

  case "${REFERENCE_HASH_VERIFY}" in
    0|false|FALSE|no|NO)
      echo "setup.sh: skipping reference SHA256 verification"
      return 0
      ;;
    1|true|TRUE|yes|YES)
      ;;
    *)
      echo "setup.sh: MLXFAST_REFERENCE_HASH_VERIFY must be 0 or 1" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${REFERENCE_MANIFEST_PATH}" ]]; then
    echo "setup.sh: reference manifest missing at ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    read -r expected_hash expected_size relative_path extra <<< "${line}"
    if [[ -n "${extra:-}" || -z "${expected_hash:-}" || -z "${expected_size:-}" || -z "${relative_path:-}" ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ ! "${expected_hash}" =~ ^[0-9a-f]{64}$ || ! "${expected_size}" =~ ^[0-9]+$ ]]; then
      echo "setup.sh: malformed reference manifest line: ${line}" >&2
      return 1
    fi
    if [[ "${relative_path}" == /* || "${relative_path}" == *\\* ]]; then
      echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
      return 1
    fi
    case "/${relative_path}/" in
      *"/../"*|*"/./"*)
        echo "setup.sh: unsafe reference manifest path: ${relative_path}" >&2
        return 1
        ;;
    esac

    file_path="${reference_dir}/${relative_path}"
    if [[ ! -f "${file_path}" ]]; then
      echo "setup.sh: reference checkpoint is missing manifest file ${relative_path}" >&2
      return 1
    fi
    actual_size="$(wc -c < "${file_path}" | tr -d ' ')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "setup.sh: reference file ${relative_path} size mismatch: expected ${expected_size}, got ${actual_size}" >&2
      return 1
    fi
    actual_hash="$(shasum -a 256 "${file_path}" | awk '{print $1}')"
    if [[ "${actual_hash}" != "${expected_hash}" ]]; then
      echo "setup.sh: reference file ${relative_path} sha256 mismatch" >&2
      echo "setup.sh: expected ${expected_hash}" >&2
      echo "setup.sh: actual   ${actual_hash}" >&2
      return 1
    fi
    checked=$((checked + 1))
  done < "${REFERENCE_MANIFEST_PATH}"

  if [[ "${checked}" -eq 0 ]]; then
    echo "setup.sh: reference manifest contained no files: ${REFERENCE_MANIFEST_PATH}" >&2
    return 1
  fi
  echo "setup.sh: verified ${checked} reference file hash(es)"
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
    if verify_reference_weights "${reference_dir}"; then
      echo "setup.sh: reference weights already present at ${reference_dir}"
      return 0
    fi
    return 1
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
  for file in "${REFERENCE_REQUIRED_METADATA_FILES[@]}"; do
    download_reference_file "${file}" "${partial_dir}/${file}"
  done
  for file in "${REFERENCE_OPTIONAL_METADATA_FILES[@]}"; do
    download_optional_reference_file "${file}" "${partial_dir}/${file}"
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
  download_reference_shards "${partial_dir}" "${shard_files[@]}"
  verify_reference_weights "${partial_dir}"

  find "${partial_dir}" -name "*.complete" -type f -delete
  mv "${partial_dir}" "${reference_dir}"
  echo "setup.sh: downloaded reference weights to ${reference_dir}"
}

ensure_swift_toolchain

ensure_mactop

echo "setup.sh: building Swift harness"
mkdir -p .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}"
swift build -c release

if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
  echo "setup.sh: skipping mlx.metallib build"
else
  ensure_cmake
  ensure_metal_toolchain
  echo "setup.sh: building mlx.metallib for MLX Swift runtime"
  tools/build-mlx-metallib.sh
fi

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "setup.sh: skipping reference weight download"
  print_setup_summary "skipped"
  exit 0
fi

download_reference_weights "${REFERENCE_DIR}"
print_setup_summary "ready"
