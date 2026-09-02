#!/usr/bin/env bash
# Run a command and report its peak resident set from /proc/<pid>/status VmHWM.
#
# VmHWM, not RSS sampled in a loop: a sampler misses the peak, and the peak is
# what the 16 GB cap is about. An OOM at fixture size fails the metric even if
# it passes locally, so the scale tests assert against this rather than just
# printing it.
#
#   ./bench/scripts/peak_rss.sh <command> [args...]
set -euo pipefail

"$@" &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
    if HWM=$(awk '/VmHWM/ {print $2}' "/proc/$PID/status" 2>/dev/null); then
        [[ -n "$HWM" && "$HWM" -gt "$PEAK" ]] && PEAK="$HWM"
    fi
    sleep 0.05
done
wait "$PID" || RC=$?

printf '\npeak RSS (VmHWM): %d KiB = %.2f MiB = %.3f GiB\n' \
       "$PEAK" "$(awk -v k="$PEAK" 'BEGIN { printf "%f", k/1024 }')" \
       "$(awk -v k="$PEAK" 'BEGIN { printf "%f", k/1048576 }')"
exit "${RC:-0}"
