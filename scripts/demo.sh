#!/usr/bin/env bash
# The scripted demo, for the presentation. Pauses between steps so a presenter
# can talk over it; set DEMO_AUTO=1 to run straight through.
#
# Rehearse this. Do not generate fixtures live.
set -euo pipefail

BUILD="${BUILD:-build/release}"
FIXTURES="${FIXTURES:-fixtures/out}"
SFS="$BUILD/apps/sfs/sfs"
WORK="$(mktemp -d)"
trap 'fusermount3 -u "$WORK/mnt" 2>/dev/null || true; rm -rf "$WORK"' EXIT

say()  { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
pause() { [[ "${DEMO_AUTO:-0}" == "1" ]] || read -rp $'\n[enter]' _; }
run()  { printf '\033[2m$ %s\033[0m\n' "$*"; eval "$@"; }

say "1. Two checkpoints. Same function, different bytes."
run "ls -la $FIXTURES/mlp_step0.safetensors $FIXTURES/mlp_permuted.safetensors"
run "cmp -l $FIXTURES/mlp_step0.safetensors $FIXTURES/mlp_permuted.safetensors | wc -l"
echo "  ^ that many bytes differ, for a network that computes exactly the same thing"
pause

say "2. Store them both."
run "$SFS init $WORK/repo"
run "$SFS --repo $WORK/repo commit $FIXTURES/mlp_step0.safetensors -m 'original'"
run "$SFS --repo $WORK/repo commit $FIXTURES/mlp_permuted.safetensors -m 'permuted'"
run "du -sh $WORK/repo/.synapsefs/objects"
echo "  ^ the second checkpoint cost almost nothing"
pause

say "3. It comes back byte for byte."
run "$SFS --repo $WORK/repo checkout HEAD --out $WORK/out.safetensors"
run "cmp $WORK/out.safetensors $FIXTURES/mlp_permuted.safetensors && echo IDENTICAL"
pause

say "4. Integrity, standalone."
run "$SFS --repo $WORK/repo verify --full"
say "   Now corrupt one byte and try again."
run "python3 -c \"import glob,random;p=random.choice(glob.glob('$WORK/repo/.synapsefs/objects/*/*'));d=bytearray(open(p,'rb').read());i=len(d)//2;d[i]^=1;open(p,'wb').write(d);print('flipped one bit in',p)\""
set +e; run "$SFS --repo $WORK/repo verify --full"; echo "exit code: $?"; set -e
echo "  ^ it names the object AND the chunk"
pause

say "5. The mount. torch loads it, and nothing is written to disk."
run "$SFS init $WORK/repo2 && $SFS --repo $WORK/repo2 commit $FIXTURES/mlp_step0.safetensors -m x"
run "mkdir -p $WORK/mnt"
run "$SFS --repo $WORK/repo2 mount HEAD $WORK/mnt --foreground &"
sleep 1
run "ls -la $WORK/mnt"
run "python3 -c \"from safetensors.torch import load_file; t=load_file('$WORK/mnt/model.safetensors'); print(len(t),'tensors loaded from a file that does not exist')\""
say "   strace: no writes."
run "strace -f -e trace=write,openat -o $WORK/trace.txt python3 -c \"from safetensors.torch import load_file; load_file('$WORK/mnt/model.safetensors')\" 2>/dev/null || true"
run "grep -c 'openat.*O_WRONLY\\|openat.*O_CREAT' $WORK/trace.txt || echo 0"
pause
run "fusermount3 -u $WORK/mnt"
say "done"
