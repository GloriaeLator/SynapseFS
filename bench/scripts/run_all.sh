#!/usr/bin/env bash
# Run every benchmark and write JSON into bench-out/.
#
# Release preset only. docs/benchmarks.md says a RelWithDebInfo number is not a
# result, and this script enforces it rather than trusting anyone to remember.
set -euo pipefail

BUILD="${1:?usage: run_all.sh <build-dir>}"
OUT="${2:-bench-out}"
mkdir -p "$OUT"

if ! grep -q 'CMAKE_BUILD_TYPE:STRING=Release' "$BUILD/CMakeCache.txt" 2>/dev/null; then
    echo "refusing: $BUILD is not a Release build. Use: cmake --preset release" >&2
    exit 1
fi

echo "== machine =="
{
    echo "date:     $(date -Is)"
    echo "commit:   $(git rev-parse --short HEAD)"
    echo "kernel:   $(uname -r)"
    echo "cpu:      $(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | xargs)"
    echo "cores:    $(nproc)"
    echo "sha_ni:   $(grep -qm1 sha_ni /proc/cpuinfo && echo yes || echo no)"
    echo "avx512:   $(grep -qm1 avx512f /proc/cpuinfo && echo yes || echo no)"
    echo "ram:      $(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
    echo "fuse:     $(pkg-config --modversion fuse3 2>/dev/null || echo n/a)"
    echo "compiler: $($CXX --version 2>/dev/null | head -1 || g++ --version | head -1)"
} | tee "$OUT/machine.txt"

for b in align_time lap_bench sparse_bench residual_codec verify_time mmap_throughput; do
    if [[ -x "$BUILD/bench/$b" ]]; then
        echo; echo "== $b =="
        # A benchmark reporting a real finding (e.g. align_time's ground-truth
        # regressions from the known dependent-group bug) exits non-zero on
        # purpose -- that's data, not a script failure, and shouldn't stop
        # the remaining benchmarks under `set -e`.
        "$BUILD/bench/$b" --json | tee "$OUT/$b.json" || true
    else
        echo "skipping $b (not built)"
    fi
done

echo
echo "results in $OUT/. Copy the numbers into docs/benchmarks.md WITH the"
echo "machine block above — a throughput number without a machine is a rumour."
