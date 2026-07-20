#!/usr/bin/env bash
# Deterministic per-run selection from the pinned MTP correctness-gate
# rotation pool (fixtures/mtp_correctness_gate_rotation_pool.json).
#
# Usage: select-mtp-gate-rotation.sh POOL_JSON SEED
#
# Contract:
#   - Only ELIGIBLE members participate: non-empty r2_key, a well-formed
#     64-hex sha256 pin, a positive integer bytes pin, decode_tokens in
#     1...1536 (MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens;
#     keep in sync), and block_size in 2...4 (the trained-MTP CLI floor is
#     2 and MLXFastConstants.experimentalMTPMaxBlockSize is 4).
#   - Selection is index = SEED % eligible_count over eligible members in
#     manifest order, so a given run id/attempt always draws the same
#     member (auditable; the draw is logged to stderr).
#   - With no eligible members the script prints "{}" and exits 0 so the
#     workflow can skip the rotation leg gracefully until the operator
#     provisions the pool.
#   - Malformed manifests, malformed seeds, and member values outside the
#     trusted bounds fail closed (non-zero exit).
set -euo pipefail

readonly MAX_DECODE_TOKENS=1536
readonly MIN_BLOCK_SIZE=2
readonly MAX_BLOCK_SIZE=4

if [[ "$#" -ne 2 ]]; then
  echo "usage: select-mtp-gate-rotation.sh POOL_JSON SEED" >&2
  exit 2
fi
pool_path="$1"
seed="$2"

if [[ ! -f "${pool_path}" || -L "${pool_path}" ]]; then
  echo "::error::rotation pool manifest must be a regular file: ${pool_path}" >&2
  exit 1
fi
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
  echo "::error::rotation seed must be a non-negative integer" >&2
  exit 1
fi

jq -e '
  type == "object"
  and .schema_version == 1
  and (.track_id | type == "string" and length > 0)
  and (.members | type == "array")
  and all(.members[];
    type == "object"
    and (.id | type == "string" and length > 0)
    and (.r2_key | type == "string")
    and (.sha256 | type == "string")
    and (.bytes | type == "string")
    and (.decode_tokens | type == "number" and floor == . and . >= 1)
    and (.block_size | type == "number" and floor == .)
  )
  and ([.members[].id] | length == (unique | length))
' "${pool_path}" >/dev/null || {
  echo "::error::rotation pool manifest failed strict schema validation: ${pool_path}" >&2
  exit 1
}

eligible="$(jq -c \
  --argjson max_tokens "${MAX_DECODE_TOKENS}" \
  --argjson min_block "${MIN_BLOCK_SIZE}" \
  --argjson max_block "${MAX_BLOCK_SIZE}" \
  '[.members[]
    | select(
        (.r2_key | length > 0)
        and (.sha256 | test("^[0-9a-f]{64}$"))
        and (.bytes | test("^[1-9][0-9]*$"))
        and (.decode_tokens <= $max_tokens)
        and (.block_size >= $min_block and .block_size <= $max_block)
      )
  ]' "${pool_path}")"

eligible_count="$(jq 'length' <<< "${eligible}")"
if [[ "${eligible_count}" -eq 0 ]]; then
  echo "mtp-gate-rotation: no provisioned pool members (seed=${seed}); rotation leg will be skipped" >&2
  printf '{}\n'
  exit 0
fi

# Bash arithmetic is 64-bit; run ids are far below overflow.
index=$((seed % eligible_count))
selected="$(jq -c --argjson index "${index}" '.[$index]' <<< "${eligible}")"
selected_id="$(jq -r '.id' <<< "${selected}")"
echo "mtp-gate-rotation: drew member ${selected_id} (seed=${seed}, eligible=${eligible_count}, index=${index})" >&2
printf '%s\n' "${selected}"
