#!/usr/bin/env python3
"""fixtures/permute.py — the headline demo pair (fixtures/README.md):

"applies a valid permutation to a checkpoint, giving a file that computes
exactly the same function and shares almost no bytes."

Consumes the topology sidecar gen_mlp.py writes, generates one random
permutation per NON-PINNED group, applies it consistently to every tensor
axis bound to that group (so the network's function is unchanged — permuting
layer l's output units and layer l+1's matching input units together is
exactly what keeps it function-identical, spec 13 §1), and writes both the
permuted checkpoint and a small JSON recording the ground-truth permutations
used.

That JSON is deliberately in the same shape codec's C++ side wants
(group name -> permutation array): bench/residual_codec.cpp reads it directly
and builds a core::Topology + permutation by hand, rather than calling into
align/ (which has no implementation yet) — see the branch's planning notes.
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import load_file, save_file


def expand_permutation(perm: np.ndarray, block: int) -> np.ndarray:
    """core::expand_permutation, spec 13 §1: expand(p, block)[i*block+k] =
    p[i]*block + k. block == 1 (every axis in the MLP fixture) is the
    identity case; implemented in general because it costs nothing extra and
    is exactly what a conv-feeding-linear flatten would need later.
    """
    if block == 1:
        return perm
    n = perm.shape[0]
    out = np.empty(n * block, dtype=perm.dtype)
    for i in range(n):
        out[i * block:(i + 1) * block] = perm[i] * block + np.arange(block)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="path to a *_step0.safetensors file")
    ap.add_argument("--topology", required=True, help="path to the matching *_topology.json")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    tensors = load_file(args.source)
    with open(args.topology) as f:
        topo = json.load(f)

    rng = np.random.default_rng(args.seed)

    # One random permutation per non-pinned group. Pinned groups (spec 13
    # §2.1: input channels, classifier outputs) keep the identity — permuting
    # them would still reconstruct byte-for-byte, but the resulting model
    # would silently compute the wrong function, which is exactly the
    # failure mode pinning exists to rule out.
    group_perms: dict[str, list[int]] = {}
    for name, group in topo["perm_groups"].items():
        if group["pinned"]:
            continue
        perm = rng.permutation(group["size"])
        group_perms[name] = perm.tolist()

    permuted = {}
    for name, arr in tensors.items():
        axes = topo["tensors"].get(name, {}).get("axes", [])
        out = arr
        for axis in axes:
            group_name = axis["group"]
            if group_name not in group_perms:
                continue  # pinned: identity, nothing to do on this axis
            block = axis.get("block", 1)
            perm = expand_permutation(np.array(group_perms[group_name]), block)
            out = np.take(out, perm, axis=axis["dim"])
        permuted[name] = out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metadata = {"format": "synapsefs-fixture", "arch": topo.get("source", {}).get("arch", "")}
    # Insertion order intentionally mirrors `tensors` (safetensors' own load
    # order for the source file), not the permuted file re-sorting anything —
    # a permutation changes VALUES, not which keys exist or their order.
    save_file(permuted, args.out, metadata)

    perm_path = os.path.splitext(args.out)[0] + ".permutation.json"
    with open(perm_path, "w") as f:
        json.dump({"source": args.source, "groups": group_perms}, f, indent=2)

    changed = sum(
        int(not np.array_equal(tensors[n], permuted[n])) for n in tensors
    )
    print(f"wrote {args.out} and {perm_path} "
         f"({changed}/{len(tensors)} tensors byte-different, "
         f"{len(group_perms)} groups permuted)")


if __name__ == "__main__":
    main()
