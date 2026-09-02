#!/usr/bin/env python3
"""tests/e2e.py — one of the four properties docs/testing.md says must never
break: mmap works, load_file() works, unmodified.

Drives the real `sfs` binary through init -> commit -> mount -> read -> unmount
against a real, from-scratch .safetensors file (built with the standard
library only, deliberately not through `safetensors`/numpy, so this test
never depends on the same library it's trying to validate on the read side),
and asserts:
  - the mounted file's size and raw bytes match the original exactly
  - mmap against the mount works (daemon.hpp: FOPEN_DIRECT_IO is off)
  - safetensors.torch.load_file() -- UNMODIFIED, no flags, no wrapper --
    succeeds and returns the exact tensor values that were committed

Registered by tests/CMakeLists.txt as `ctest -R e2e.python`, invoked as
`python3 e2e.py --sfs <path-to-sfs-binary>`. Exits 0 on success, non-zero
(with a message naming the failing step) otherwise -- CTest reads the exit
code, not stdout content.
"""
from __future__ import annotations

import argparse
import json
import mmap
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def build_safetensors(path: Path) -> dict[str, list[float]]:
    """Writes a tiny, real .safetensors file using only struct/json -- no
    numpy, no safetensors package -- so committing it never depends on the
    library this test exists to validate on the read side.

    Returns {tensor_name: flat_values} for later comparison against what
    load_file() reports.
    """
    tensors = {
        "weight": {"shape": [2, 3], "values": [0.5, -1.25, 2.0, 3.75, -4.5, 5.125]},
        "bias": {"shape": [2], "values": [0.25, -0.75]},
    }

    header: dict = {}
    payload = bytearray()
    for name, t in tensors.items():
        data = b"".join(struct.pack("<f", v) for v in t["values"])
        header[name] = {
            "dtype": "F32",
            "shape": t["shape"],
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data

    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(payload)

    return {name: t["values"] for name, t in tensors.items()}


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(
            f"error: command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


def wait_for_mount(mounted_file: Path, expected_size: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if mounted_file.exists() and mounted_file.stat().st_size == expected_size:
            return
        time.sleep(0.05)
    sys.exit(f"error: {mounted_file} never appeared with size {expected_size}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sfs", required=True, help="path to the sfs binary")
    args = ap.parse_args()
    sfs = str(Path(args.sfs).resolve())

    workdir = Path(tempfile.mkdtemp(prefix="sfs_e2e_"))
    repo_dir = workdir / "repo"
    repo_dir.mkdir()
    checkpoint_path = workdir / "model.safetensors"
    mountpoint = workdir / "mnt"
    mountpoint.mkdir()

    mount_proc: subprocess.Popen | None = None
    try:
        expected = build_safetensors(checkpoint_path)
        original_bytes = checkpoint_path.read_bytes()

        run([sfs, "init", "."], cwd=repo_dir)
        run([sfs, "commit", str(checkpoint_path), "-m", "e2e checkpoint"], cwd=repo_dir)

        mount_proc = subprocess.Popen(
            [sfs, "mount", "main", str(mountpoint)],
            cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        mounted_file = mountpoint / checkpoint_path.name
        wait_for_mount(mounted_file, len(original_bytes))
        print(f"[1/4] mounted: {mounted_file} ({len(original_bytes)} bytes)")

        # Sequential read must match the original bytes exactly.
        read_back = mounted_file.read_bytes()
        assert read_back == original_bytes, "mounted file bytes differ from the committed checkpoint"
        print("[2/4] sequential read: byte-exact")

        # mmap must work (FOPEN_DIRECT_IO is deliberately off).
        with open(mounted_file, "rb") as f:
            with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
                assert bytes(mm[:]) == original_bytes, "mmap contents differ from the committed checkpoint"
        print("[3/4] mmap: byte-exact")

        # The actual point: an UNMODIFIED safetensors.torch.load_file() call.
        from safetensors.torch import load_file
        loaded = load_file(str(mounted_file))
        assert set(loaded.keys()) == set(expected.keys()), (
            f"tensor names differ: got {sorted(loaded.keys())}, want {sorted(expected.keys())}"
        )
        for name, want_values in expected.items():
            got = loaded[name].flatten().tolist()
            assert got == want_values, f"tensor '{name}' values differ: got {got}, want {want_values}"
        print(f"[4/4] safetensors.torch.load_file(): {len(loaded)} tensors, values match exactly")

    finally:
        try:
            run([sfs, "unmount", str(mountpoint)], cwd=repo_dir)
        except SystemExit:
            pass  # best-effort during cleanup; the real assertions already ran
        if mount_proc is not None:
            try:
                mount_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # `unmount` (fusermount3 -u) already returned above; the
                # daemon process failing to notice within 20s is a real
                # shutdown-latency issue worth knowing about, but not one
                # this test should hang forever over -- terminate it so the
                # suite still completes and reports what happened.
                print(f"warning: mount daemon did not exit within 20s of unmount; "
                     f"sending SIGTERM", file=sys.stderr)
                mount_proc.terminate()
                mount_proc.wait(timeout=10)
        shutil.rmtree(workdir, ignore_errors=True)

    print("PASS")


if __name__ == "__main__":
    main()
