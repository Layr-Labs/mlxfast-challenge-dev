#!/bin/bash
# MTP decode benchmark matrix: 3 workloads x depth {1,2,3} + a warm-cache probe.
# Produces matrix.log in the current directory; parse with parse_matrix.py.
#
# Usage:
#   MTP_BIN=<path to Qwen36MTPDriver> MTP_WEIGHTS=<merged checkpoint dir> ./run-matrix.sh
# Optional:
#   MTP_LOCK_DIR=<path>   mkdir-based mutex honored around every model run
#                         (for shared machines; omit on a dedicated box)
#
# Requires DEVELOPER_DIR to point at Xcode (Metal). Run warm: the first process
# on a cold page cache measures ~23% slower on serial; discard a cold first run.
set -u
BIN=${MTP_BIN:?set MTP_BIN to the Qwen36MTPDriver binary}
W=${MTP_WEIGHTS:?set MTP_WEIGHTS to the merged checkpoint directory}
LOCK=${MTP_LOCK_DIR:-}

acquire() {
  [ -z "$LOCK" ] && return 0
  local t=0
  until mkdir "$LOCK" 2>/dev/null; do
    t=$((t+1)); [ $t -eq 1 ] && echo "[lock] busy, waiting"
    [ $t -gt 720 ] && { echo "[lock] TIMEOUT"; exit 9; }
    sleep 5
  done
  echo "mtp-verify" > "$LOCK/owner"
  trap 'rm -rf "$LOCK"; echo "[lock] released"' EXIT
}

W1P="Implement binary search in Python with tests."
W2P="Write a short poem about Apple Silicon."
W3P="Q: What is the capital of France?\nA:"

cell() {
  local id=$1; shift
  echo "########## BEGIN $id :: $* ##########"
  /usr/bin/time -p "$BIN" "$W" "$@" 2>&1
  echo "########## END $id ##########"
}

acquire
{
  for d in 1 2 3; do cell "W1-d$d" --chat --prompt "$W1P" --tokens 256 --depth $d; done
  for d in 1 2 3; do cell "W2-d$d" --prompt "$W2P" --tokens 256 --depth $d; done
  for d in 1 2 3; do cell "W3-d$d" --prompt "$W3P" --tokens 32 --depth $d; done
  # warm-cache probe: same cell twice back-to-back; timing spread should be ~1%
  cell "W1-d2-rep1" --chat --prompt "$W1P" --tokens 256 --depth 2
  cell "W1-d2-rep2" --chat --prompt "$W1P" --tokens 256 --depth 2
  echo MATRIX_COMPLETE
} | tee matrix.log
