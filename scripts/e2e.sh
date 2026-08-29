#!/usr/bin/env bash
# End-to-end: init -> commit x2 -> checkout -> verify, asserting byte-exactness
# at each step. With --with-mount, additionally mounts and asserts that the
# mount's bytes are identical to the checkout's — the PS's consistency
# requirement, which belongs in the test suite and not in a manual check.
#
#   ./scripts/e2e.sh [--with-mount] <build-dir> <fixtures-dir>
set -euo pipefail

WITH_MOUNT=0
if [[ "${1:-}" == "--with-mount" ]]; then WITH_MOUNT=1; shift; fi

BUILD="${1:?usage: e2e.sh [--with-mount] <build-dir> <fixtures-dir>}"
FIXTURES="${2:?usage: e2e.sh [--with-mount] <build-dir> <fixtures-dir>}"
SFS="$BUILD/apps/sfs/sfs"
WORK="$(mktemp -d)"
trap 'fusermount3 -u "$WORK/mnt" 2>/dev/null || true; rm -rf "$WORK"' EXIT

A="$FIXTURES/mlp_step0.safetensors"
B="$FIXTURES/mlp_step1.safetensors"
[[ -f "$A" && -f "$B" ]] || { echo "missing fixtures; run: make fixtures-small" >&2; exit 1; }

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "init"
"$SFS" init "$WORK/repo"

step "commit x2"
"$SFS" --repo "$WORK/repo" commit "$A" -m "step 0"
"$SFS" --repo "$WORK/repo" commit "$B" -m "step 1"
"$SFS" --repo "$WORK/repo" log --graph

step "verify (standalone, no checkout, no mount)"
"$SFS" --repo "$WORK/repo" verify --full

step "checkout is byte-exact"
"$SFS" --repo "$WORK/repo" checkout HEAD --out "$WORK/restored.safetensors"
cmp "$WORK/restored.safetensors" "$B"
echo "checkout matches the original byte for byte"

step "history is byte-exact too"
PARENT="$("$SFS" --repo "$WORK/repo" log --max-count 2 --json | python3 -c \
  'import json,sys; print(json.load(sys.stdin)[1]["commit"])')"
"$SFS" --repo "$WORK/repo" checkout "$PARENT" --out "$WORK/restored0.safetensors"
cmp "$WORK/restored0.safetensors" "$A"
echo "the delta reconstructs its base exactly"

if [[ "$WITH_MOUNT" == "1" ]]; then
    step "mount bytes == checkout bytes"
    mkdir -p "$WORK/mnt"
    "$SFS" --repo "$WORK/repo" mount HEAD "$WORK/mnt" --foreground &
    MPID=$!
    for _ in $(seq 1 50); do [[ -e "$WORK/mnt/model.safetensors" ]] && break; sleep 0.1; done
    cmp "$WORK/mnt/model.safetensors" "$WORK/restored.safetensors"
    echo "mount matches checkout byte for byte"

    step "nothing was materialised"
    # The daemon must never write a checkpoint-sized file anywhere.
    if find "$WORK/repo/.synapsefs" -size +1M -newer "$WORK/restored.safetensors" | grep -q .; then
        echo "FAIL: the daemon wrote something large during the mount" >&2
        exit 1
    fi
    fusermount3 -u "$WORK/mnt"
    wait "$MPID" 2>/dev/null || true
fi

printf '\n\033[32mall end-to-end checks passed\033[0m\n'
