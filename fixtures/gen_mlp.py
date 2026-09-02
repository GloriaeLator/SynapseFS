#!/usr/bin/env python3
"""fixtures/gen_mlp.py — the MLP fixture pair (docs/fixtures/manifest.toml).

Writes mlp_step0.safetensors and mlp_step1.safetensors: the same
Linear -> ReLU -> Linear -> ReLU -> Linear architecture, step1 being step0
nudged by a small amount of noise, standing in for one fine-tune step. No
permutation here — permute.py derives mlp_permuted.safetensors from step0
separately, since "reordered" and "fine-tuned" are two different, orthogonal
transformations of a checkpoint and the codec benchmark needs both kinds of
pair.

Tensor names follow the same numeric nn.Sequential-style convention used by
docs/spec/13-topology-config.md's own worked example ("0.weight", "0.bias",
...). Writes TWO different topology files, for two different consumers:
`{prefix}_config.json` is the real `align::topology_parser` "layers" schema
(see real_config()); `{prefix}_topology.json` is an older, pre-resolved
perm_groups/tensors shape that predates topology_parser.cpp and is kept only
because permute.py and bench/residual_codec.cpp still hand-parse it directly
rather than through the real parser (see topology_sidecar()'s own comment).

Deliberately minimal: this covers only the `mlp`/`tiny_mlp` rows of
manifest.toml, needed to unblock the codec benchmark. resnet/llm_7b are out
of scope.
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file


def build_mlp(rng: np.random.Generator, in_dim: int, h1: int, h2: int, out_dim: int):
    """One random-init MLP as fp16 numpy arrays, keyed nn.Sequential-style."""

    def linear(fan_in: int, fan_out: int):
        # Plain He-ish init; the exact distribution does not matter for a
        # codec fixture, only that step0 and step1 are two real, distinct
        # sets of fp16 weights.
        scale = np.sqrt(2.0 / fan_in)
        w = (rng.standard_normal((fan_out, fan_in)) * scale).astype(np.float16)
        # Small nonzero bias, not the conventional zero-init: a zero vector
        # stays a zero vector under any permutation, which would make the
        # permuted-pair benchmark's bias rows trivially "byte-identical" and
        # prove nothing.
        b = (rng.standard_normal((fan_out,)) * 0.01).astype(np.float16)
        return w, b

    w0, b0 = linear(in_dim, h1)
    w2, b2 = linear(h1, h2)
    w4, b4 = linear(h2, out_dim)

    # Deliberately awkward insertion order (biases before weights, and not
    # alphabetical) — fixtures/README.md calls this out explicitly: a
    # round-trip test against a file whose key order we chose ourselves
    # proves nothing, because the evaluator's checkpoints won't have been
    # written by our own tidy writer either. safetensors preserves dict
    # insertion order, so this is enough to get a non-trivial order without
    # needing the (also unimplemented) generate_synthetic_checkpoint.py.
    tensors = {
        "0.bias": b0, "2.bias": b2, "4.bias": b4,
        "0.weight": w0, "2.weight": w2, "4.weight": w4,
    }
    dims = {"in": in_dim, "g0": h1, "g2": h2, "out": out_dim}
    return tensors, dims


def real_config() -> dict:
    """The actual `align::topology_parser` schema (a flat "layers" list,
    modules/align/src/topology_parser.cpp's walk_layers) -- NOT the same
    shape as topology_sidecar() below. That function's own comment explains
    why the two differ: it predates topology_parser.cpp existing at all, and
    permute.py / bench/residual_codec.cpp still consume its pre-resolved
    perm_groups/tensors shape directly rather than through the real parser
    (see modules/align/tests/fixtures/generate.py for the fixture set that
    already exercises this exact real schema). Shapes aren't named here on
    purpose -- the parser derives them from the checkpoint's own tensors,
    not from this file.
    """
    return {"layers": [
        {"type": "linear"}, {"type": "relu"},
        {"type": "linear"}, {"type": "relu"},
        {"type": "linear"},
    ]}


def topology_sidecar(dims: dict) -> dict:
    """spec 13 §2 schema, by hand — align/'s real parser doesn't exist yet
    (see conversation notes); this is the same shape it would eventually
    produce for this exact architecture, hand-written so codec benchmarking
    isn't blocked on it.
    """
    return {
        "type": "synapsefs.topology",
        "format_version": 1,
        "source": {"kind": "synthetic", "arch": "mlp"},
        "perm_groups": {
            "in":  {"size": dims["in"],  "pinned": True},
            "g0":  {"size": dims["g0"],  "pinned": False},
            "g2":  {"size": dims["g2"],  "pinned": False},
            "out": {"size": dims["out"], "pinned": True},
        },
        "tensors": {
            "0.weight": {"axes": [{"dim": 0, "group": "g0", "block": 1},
                                  {"dim": 1, "group": "in", "block": 1}]},
            "0.bias":   {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            "2.weight": {"axes": [{"dim": 0, "group": "g2", "block": 1},
                                  {"dim": 1, "group": "g0", "block": 1}]},
            "2.bias":   {"axes": [{"dim": 0, "group": "g2", "block": 1}]},
            "4.weight": {"axes": [{"dim": 0, "group": "out", "block": 1},
                                  {"dim": 1, "group": "g2", "block": 1}]},
            "4.bias":   {"axes": [{"dim": 0, "group": "out", "block": 1}]},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory (e.g. fixtures/out)")
    ap.add_argument("--tiny", action="store_true", help="~65k params instead of ~1M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune-scale", type=float, default=0.02,
                    help="stddev of the noise added to produce step1 from step0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.tiny:
        in_dim, h1, h2, out_dim = 128, 192, 128, 64     # ~58k params
        prefix = "tiny_mlp"
    else:
        in_dim, h1, h2, out_dim = 512, 768, 512, 256    # ~919k params
        prefix = "mlp"

    step0, dims = build_mlp(rng, in_dim, h1, h2, out_dim)

    # step1: a fine-tune step, not a reordering — same key order, same
    # shapes, small per-element perturbation. This is what makes the
    # "fine-tune, 1 epoch" residual-ratio bench row meaningful: XOR/zigzag of
    # two real, close-but-not-identical fp16 tensors, not synthetic noise.
    step1 = {}
    for name, arr in step0.items():
        noise = (rng.standard_normal(arr.shape) * args.finetune_scale).astype(np.float16)
        step1[name] = (arr.astype(np.float32) + noise.astype(np.float32)).astype(np.float16)

    metadata = {"format": "synapsefs-fixture", "arch": "mlp"}
    save_file(step0, os.path.join(args.out, f"{prefix}_step0.safetensors"), metadata)
    save_file(step1, os.path.join(args.out, f"{prefix}_step1.safetensors"), metadata)

    topo_path = os.path.join(args.out, f"{prefix}_topology.json")
    with open(topo_path, "w") as f:
        json.dump(topology_sidecar(dims), f, indent=2)

    config_path = os.path.join(args.out, f"{prefix}_config.json")
    with open(config_path, "w") as f:
        json.dump(real_config(), f, indent=2)

    total_params = sum(a.size for a in step0.values())
    print(f"wrote {prefix}_step0.safetensors, {prefix}_step1.safetensors, "
         f"{prefix}_topology.json, {prefix}_config.json ({total_params} params, dims={dims})")


if __name__ == "__main__":
    main()
