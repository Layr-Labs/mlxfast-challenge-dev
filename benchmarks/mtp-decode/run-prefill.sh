#!/bin/bash
# Prefill throughput at ~480/960/1920/3840 prompt tokens (synthetic repeated
# sentence; matrix prompts are too short to resolve against 0.1s log markers).
# Usage: MTP_BIN=... MTP_WEIGHTS=... [MTP_LOCK_DIR=...] ./run-prefill.sh
set -u
BIN=${MTP_BIN:?set MTP_BIN}
W=${MTP_WEIGHTS:?set MTP_WEIGHTS}
LOCK=${MTP_LOCK_DIR:-}
if [ -n "$LOCK" ]; then
  t=0; until mkdir "$LOCK" 2>/dev/null; do t=$((t+1)); [ $t -gt 720 ] && exit 9; sleep 5; done
  echo "mtp-verify" > "$LOCK/owner"
  trap 'rm -rf "$LOCK"' EXIT
fi
U="The quick brown fox jumps over the lazy dog near the river bank at dawn. "
for reps in 30 60 120 240; do
  P=""; i=0; while [ $i -lt $reps ]; do P="$P$U"; i=$((i+1)); done
  echo "########## BEGIN PF-$reps ##########"
  "$BIN" "$W" --prompt "$P" --tokens 2 --depth 1 2>&1 \
    | grep -E "\[setup\] prompt tokens|\[mtp\] prefill done|\[serial\] 1/|\[serial\] 2/"
  echo "########## END PF-$reps ##########"
done
echo PREFILL_COMPLETE
