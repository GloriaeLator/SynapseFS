#!/usr/bin/env bash
# The crash harness. Starts a commit, kills it with -9 at a random point,
# restarts, and asserts the repository either verifies clean or explicitly
# refuses. Loops.
#
# Keep the counter. "We ran 40,000 iterations and every one recovered or
# refused" is a different claim from "we handle crashes", and it is a
# presentation slide.
#
#   ./scripts/run_crash_test.sh <build-dir> <fixture.safetensors> [iterations]
set -euo pipefail

BUILD="${1:?usage: run_crash_test.sh <build-dir> <fixture> [iterations]}"
FIXTURE="${2:?}"
ITERS="${3:-1000}"
SFS="$BUILD/apps/sfs/sfs"

pass=0; refused=0; broke=0
STATE="${CRASH_STATE_FILE:-crash-harness.log}"

for i in $(seq 1 "$ITERS"); do
    WORK="$(mktemp -d)"
    "$SFS" init "$WORK/repo" >/dev/null
    "$SFS" --repo "$WORK/repo" commit "$FIXTURE" -m base >/dev/null

    # Kill somewhere between 5 ms and 400 ms in — long enough to be inside the
    # write path, short enough to land at many different points.
    DELAY="0.$(printf '%03d' $((RANDOM % 400 + 5)))"
    "$SFS" --repo "$WORK/repo" commit "$FIXTURE" -m victim >/dev/null 2>&1 &
    VICTIM=$!
    sleep "$DELAY"
    kill -9 "$VICTIM" 2>/dev/null || true
    wait "$VICTIM" 2>/dev/null || true

    set +e
    OUT="$("$SFS" --repo "$WORK/repo" verify --full 2>&1)"
    RC=$?
    set -e

    case "$RC" in
        0) pass=$((pass+1)) ;;
        # Exit 4 is an integrity failure, which is NOT an acceptable outcome
        # here. Exit 6 (locked) and a clean refusal from --repair are.
        3|6) refused=$((refused+1)) ;;
        *) broke=$((broke+1))
           echo "ITERATION $i FAILED (delay=$DELAY, rc=$RC)" | tee -a "$STATE"
           echo "$OUT" | tee -a "$STATE"
           cp -a "$WORK/repo" "./crash-repro-$i"
           echo "repository preserved at ./crash-repro-$i" ;;
    esac
    rm -rf "$WORK"

    if (( i % 50 == 0 )); then
        printf '\r%6d iterations: %d clean, %d refused, %d BROKEN' \
               "$i" "$pass" "$refused" "$broke"
    fi
done

printf '\n\n%d iterations: %d recovered clean, %d refused safely, %d BROKEN\n' \
       "$ITERS" "$pass" "$refused" "$broke"
echo "$(date -Is) iterations=$ITERS clean=$pass refused=$refused broken=$broke" >> "$STATE"
[[ "$broke" -eq 0 ]]
