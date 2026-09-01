#!/usr/bin/env python3
"""demo_mount.py — read a checkpoint straight out of a SynapseFS mount.

This does NOT mount anything itself. It expects `sfs mount` to already be
running against some mountpoint (see the usage steps below) and just proves
the thing docs/testing.md's mount test ladder cares about most: a totally
unmodified `safetensors.torch.load_file()` call succeeds against the mounted
file, with no bytes ever written to disk.

Usage:
    python3 demo_mount.py /path/to/mountpoint
"""

from __future__ import annotations

import argparse
import mmap
import os
import sys
import time
from pathlib import Path


def find_mounted_file(mountpoint: Path) -> Path:
    """The mount serves exactly one file, per fs.hpp: <mountpoint>/<name>."""
    entries = list(mountpoint.iterdir())
    if not entries:
        sys.exit(f"error: {mountpoint} is empty — is `sfs mount` actually running there?")
    if len(entries) > 1:
        sys.exit(f"error: expected exactly one file under {mountpoint}, found {len(entries)}")
    return entries[0]


def report_stat(path: Path) -> int:
    st = path.stat()
    print(f"[1/4] getattr: {path.name}  size={st.st_size:,} bytes  mode={oct(st.st_mode)}")
    return st.st_size


def report_sequential_read(path: Path, total_size: int) -> None:
    """A plain, boring streaming read — no library involved — so a failure
    here points at the mount itself rather than at safetensors' own parsing.
    """
    t0 = time.monotonic()
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            read += len(chunk)
    dt = time.monotonic() - t0
    assert read == total_size, f"read {read} bytes but stat() reported {total_size}"
    mb_s = (read / (1024 * 1024)) / dt if dt > 0 else float("inf")
    print(f"[2/4] sequential read: {read:,} bytes in {dt:.3f}s ({mb_s:.1f} MB/s)")


def report_mmap(path: Path) -> None:
    """mmap must work against the mount (FOPEN_DIRECT_IO is deliberately OFF
    — see daemon.hpp). This is the one property that a plain read() test can
    pass while still being broken for real ML loaders.
    """
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
            first = mm[0:8]
            last = mm[-8:]
    print(f"[3/4] mmap: first 8 bytes = {first.hex()}  last 8 bytes = {last.hex()}")


def report_safetensors_load(path: Path) -> None:
    """The actual point of the mount: an unmodified ML loader, unmodified."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        print("[4/4] safetensors.torch not installed — skipping "
              "(pip install safetensors torch to exercise this step)")
        return

    t0 = time.monotonic()
    tensors = load_file(str(path))
    dt = time.monotonic() - t0
    print(f"[4/4] safetensors.torch.load_file(): {len(tensors)} tensors in {dt:.3f}s")
    for i, (name, tensor) in enumerate(tensors.items()):
        print(f"       {name:<30} {tuple(tensor.shape)!s:<20} {tensor.dtype}")
        if i >= 4 and len(tensors) > 6:
            print(f"       ... and {len(tensors) - i - 1} more")
            break


def report_no_writes_hint() -> None:
    print(
        "\nTo confirm nothing was materialised to disk, rerun this script under:\n"
        "    strace -f -e trace=write,openat -o /tmp/trace.txt python3 demo_mount.py <mnt>\n"
        "    grep -c 'O_WRONLY\\|O_CREAT' /tmp/trace.txt   # expect 0"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mountpoint", type=Path, help="an active `sfs mount` mountpoint")
    args = ap.parse_args()

    if not args.mountpoint.is_dir():
        sys.exit(f"error: {args.mountpoint} is not a directory")

    path = find_mounted_file(args.mountpoint)
    size = report_stat(path)
    report_sequential_read(path, size)
    report_mmap(path)
    report_safetensors_load(path)
    report_no_writes_hint()


if __name__ == "__main__":
    main()
