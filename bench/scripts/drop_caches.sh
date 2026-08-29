#!/usr/bin/env bash
# Drop the page cache so that a "cold cache" benchmark actually is one.
# Needs root. The PS benchmarks the mount from a cold page cache, so a warm
# number is not a result — it is a measurement of RAM.
set -euo pipefail
sync
if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches
else
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
fi
echo "page cache dropped"
