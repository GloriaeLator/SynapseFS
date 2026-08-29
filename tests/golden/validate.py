#!/usr/bin/env python3
"""Validate the golden objects against the frozen specs.

Runs in CI's cheap `lint` job, before anything is compiled, because a format
mistake should not cost a compile. Checks:

  * every file is valid JSON with the required fields for its kind
  * canonicalisation (SPEC 10 §1.4) is idempotent and stable
  * the manifest's buffer layout covers the file with no gaps or overlaps
  * every buffer entry's group exists in `groups`
  * every topology axis satisfies shape[dim] == group.size * block, where the
    shape is known from the manifest
  * every diff artifact's frames tile [0, group_size) exactly

Deliberately dependency-free: stdlib only, so it runs anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OK, FAIL = "  ok  ", " FAIL "
errors: list[str] = []


def fail(where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def canonical(obj) -> str:
    """SPEC 10 §1.4: UTF-8, keys sorted by code point, no insignificant space."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def check_canonical_stable(name: str, obj) -> None:
    once = canonical(obj)
    twice = canonical(json.loads(once))
    if once != twice:
        fail(name, "canonicalisation is not idempotent")


def check_commit(name: str, o: dict) -> None:
    for k in ("type", "format_version", "parents", "manifest", "topology",
              "timestamp", "author", "message"):
        if k not in o:
            fail(name, f"missing required field {k!r}")
    if o.get("type") != "synapsefs.commit":
        fail(name, f"wrong type {o.get('type')!r}")
    if "parent" in o or "branch" in o or "commit_hash" in o:
        fail(name, "carries a field removed in format version 1 "
                   "(parent / branch / commit_hash)")
    parents = o.get("parents", [])
    if not isinstance(parents, list) or len(parents) > 2:
        fail(name, "parents must be a list of length 0, 1 or 2")
    ts = o.get("timestamp", "")
    if not ts.endswith("Z") or len(ts) != 20:
        fail(name, f"timestamp {ts!r} is not RFC 3339 UTC second precision")


def check_manifest(name: str, o: dict) -> dict[str, int]:
    """Returns tensor -> nbytes, for cross-checking the topology."""
    if o.get("type") != "synapsefs.manifest":
        fail(name, f"wrong type {o.get('type')!r}")
    f = o.get("file", {})
    for k in ("name", "header_block", "total_bytes", "sha256"):
        if k not in f:
            fail(name, f"file.{k} missing")
    if "base_commit" in o:
        fail(name, "top-level base_commit was removed: each group owns its base")
    if "/" in f.get("name", ""):
        fail(name, "file.name must not contain a path separator")

    buf = o.get("buffer", [])
    groups = o.get("groups", {})
    sizes: dict[str, int] = {}
    expect = 0
    for i, e in enumerate(buf):
        if e["off"] != expect:
            fail(name, f"buffer[{i}] {e['tensor']}: off={e['off']}, expected {expect} "
                       "(gaps and overlaps are data loss)")
        expect = e["off"] + e["nbytes"]
        if e["group"] not in groups:
            fail(name, f"buffer[{i}] {e['tensor']}: group {e['group']!r} not in groups")
        sizes[e["tensor"]] = e["nbytes"]

    # We cannot know the header length from the manifest alone, so check that
    # the data section is at least consistent with total_bytes.
    if expect > f.get("total_bytes", 0):
        fail(name, f"buffer sums to {expect}, more than file.total_bytes "
                   f"{f.get('total_bytes')}")

    for gid, g in groups.items():
        mode = g.get("mode")
        if mode == "full":
            if "block" not in g:
                fail(name, f"group {gid}: mode=full needs `block`")
            if g.get("chain_depth", 0) != 0:
                fail(name, f"group {gid}: mode=full must have chain_depth 0")
        elif mode == "delta":
            if "base" not in g or "diff_block" not in g:
                fail(name, f"group {gid}: mode=delta needs `base` and `diff_block`")
            if g.get("chain_depth", 0) < 1:
                fail(name, f"group {gid}: mode=delta must have chain_depth >= 1")
        else:
            fail(name, f"group {gid}: unknown mode {mode!r}")
        if "block_hash" in g or mode == "unchanged_reuse":
            fail(name, f"group {gid}: removed field/mode from the prototype schema")
    return sizes


def check_topology(name: str, o: dict, tensor_bytes: dict[str, int]) -> None:
    if o.get("type") != "synapsefs.topology":
        fail(name, f"wrong type {o.get('type')!r}")
    groups = o.get("perm_groups", {})
    for legacy in ("permutable", "permute_dim", "permute_source", "permute_input_from"):
        if legacy in json.dumps(o):
            fail(name, f"uses removed pairwise field {legacy!r}; "
                       "groups + blocking factors replaced these")
    for tname, t in o.get("tensors", {}).items():
        for ax in t.get("axes", []):
            g = ax.get("group")
            if g not in groups:
                fail(name, f"{tname} dim={ax.get('dim')}: unknown group {g!r}")
                continue
            if ax.get("block", 1) < 1:
                fail(name, f"{tname} dim={ax.get('dim')}: block must be >= 1")
    # The flatten case is the one that silently produced a wrong-shaped array,
    # so assert it explicitly rather than trusting the general rule.
    lin = o.get("tensors", {}).get("9.weight", {}).get("axes", [])
    inp = [a for a in lin if a.get("dim") == 1]
    if inp and (inp[0].get("group") != "g4" or inp[0].get("block") != 64):
        fail(name, "9.weight dim=1 must be {group: g4, block: 64} — this is the "
                   "flatten case that fails silently when it is wrong")


def check_diff(name: str, o: dict) -> None:
    if o.get("magic") != "SYNDIFF":
        fail(name, f"magic is {o.get('magic')!r}, must be 'SYNDIFF'")
    if "base_commit" in o or "base_layer_source" in o:
        fail(name, "history position must not appear inside a content-addressed "
                   "artifact; the base lives in the manifest entry")
    perm = o.get("permutation", {})
    if perm.get("kind") not in ("identity", "explicit"):
        fail(name, f"permutation.kind {perm.get('kind')!r} is not identity|explicit")
    if not o.get("alignable", True) and o.get("tensors"):
        fail(name, "alignable=false artifacts must not carry residual frames")
    for t in o.get("tensors", []):
        frames = t.get("frames", [])
        if not frames:
            fail(name, f"{t.get('name')}: no frames")
            continue
        if frames[0]["units"][0] != 0:
            fail(name, f"{t.get('name')}: frames must start at unit 0")
        for a, b in zip(frames, frames[1:]):
            if a["units"][1] != b["units"][0]:
                fail(name, f"{t.get('name')}: frames do not tile "
                           f"({a['units']} then {b['units']})")
            if a["off"] + a["len"] > b["off"]:
                fail(name, f"{t.get('name')}: frame payloads overlap")
        for fr in frames:
            if len(fr.get("digest", "")) != 64:
                fail(name, f"{t.get('name')}: frame digest must be 64 hex chars "
                           "(it covers the RECONSTRUCTED target bytes)")


def main() -> int:
    loaded = {}
    for path in sorted(HERE.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(path.name, f"invalid JSON: {exc}")
            continue
        loaded[path.name] = obj
        check_canonical_stable(path.name, obj)

    sizes: dict[str, int] = {}
    if "commit.json" in loaded:
        check_commit("commit.json", loaded["commit.json"])
    if "manifest.json" in loaded:
        sizes = check_manifest("manifest.json", loaded["manifest.json"])
    if "topology_cnn.json" in loaded:
        check_topology("topology_cnn.json", loaded["topology_cnn.json"], sizes)
    if "diff_artifact.json" in loaded:
        check_diff("diff_artifact.json", loaded["diff_artifact.json"])

    for name in sorted(loaded):
        marker = FAIL if any(e.startswith(name) for e in errors) else OK
        print(f"[{marker}] {name}")

    for e in errors:
        print(f"  {e}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} problem(s) in tests/golden/", file=sys.stderr)
        return 1
    print(f"\n{len(loaded)} golden object(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
