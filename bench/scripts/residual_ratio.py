#!/usr/bin/env python3
"""Summarise residual-codec results into the table docs/tradeoffs.md wants.

Reports RATIO AND DECOMPRESSION THROUGHPUT for every candidate, because ratio
is 7% of the grade and mmap throughput is 8%. Choosing on ratio alone picks the
smaller number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="bench-out/ directory or a .json file")
    args = ap.parse_args()

    path = args.results
    files = sorted(path.glob("residual_codec*.json")) if path.is_dir() else [path]
    if not files:
        print(f"no residual_codec results under {path}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        data = json.loads(f.read_text())
        for r in data.get("candidates", []):
            rows.append((
                r.get("residual", "?"),
                r.get("transform", "?"),
                float(r.get("ratio", 0.0)),
                float(r.get("decompress_mb_s", 0.0)),
                float(r.get("compress_mb_s", 0.0)),
            ))

    if not rows:
        print("no candidates in the results", file=sys.stderr)
        return 1

    print("| Residual | Transform | Ratio | Decompress MB/s | Compress MB/s |")
    print("|---|---|---|---|---|")
    for res, tr, ratio, dmb, cmb in sorted(rows, key=lambda r: r[2]):
        print(f"| `{res}` | {tr} | {ratio:.4f} | {dmb:.0f} | {cmb:.0f} |")

    best_ratio = min(rows, key=lambda r: r[2])
    best_tput = max(rows, key=lambda r: r[3])
    print()
    print(f"best ratio:      {best_ratio[0]} + {best_ratio[1]}  "
          f"(ratio {best_ratio[2]:.4f}, {best_ratio[3]:.0f} MB/s)")
    print(f"best throughput: {best_tput[0]} + {best_tput[1]}  "
          f"(ratio {best_tput[2]:.4f}, {best_tput[3]:.0f} MB/s)")
    if best_ratio[:2] != best_tput[:2]:
        print("\nThese disagree. Ratio is 7% of the grade and throughput is 8%.")
        print("Record BOTH rows in docs/tradeoffs.md and say which you picked and why.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
