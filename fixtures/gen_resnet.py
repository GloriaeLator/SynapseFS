#!/usr/bin/env python3
"""fixtures/gen_resnet.py — the `resnet` row of manifest.toml, minimally.

A small two-conv-layer CNN: conv2d -> batchnorm2d -> relu -> conv2d ->
batchnorm2d -> relu -> maxpool2d(2) -> flatten -> linear. Deliberately TWO
conv layers, not one: the second conv's input-channel axis (dim 1 of
[out_c, in_c, kh, kw]) depends on the FIRST conv's output group — the
multi-axis, rank-4, secondary-axis-is-not-the-last-dimension case
tests/byte_identity_cnn.cpp proved correct in isolation with hand-built
tensors. This is the same shape, but as a real .safetensors pair on disk, so
bench/residual_codec.cpp can measure it instead of just prove it works.

Same conventions as gen_mlp.py: nn.Sequential-style numeric tensor names,
awkward insertion order, a hand-written topology sidecar (spec 13 §2 shape)
since align/'s real parser output isn't what fixture generation drives —
permute.py and residual_codec.cpp both consume this sidecar directly.
"""
import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file


def conv2d(rng: np.random.Generator, out_c: int, in_c: int, k: int):
    fan_in = in_c * k * k
    scale = np.sqrt(2.0 / fan_in)
    w = (rng.standard_normal((out_c, in_c, k, k)) * scale).astype(np.float16)
    b = (rng.standard_normal((out_c,)) * 0.01).astype(np.float16)
    return w, b


def batchnorm2d(rng: np.random.Generator, c: int):
    weight = (np.ones(c) + rng.standard_normal(c) * 0.02).astype(np.float16)
    bias = (rng.standard_normal(c) * 0.01).astype(np.float16)
    running_mean = (rng.standard_normal(c) * 0.05).astype(np.float16)
    running_var = (np.abs(rng.standard_normal(c)) + 0.5).astype(np.float16)
    return weight, bias, running_mean, running_var


def linear(rng: np.random.Generator, out_dim: int, in_dim: int):
    scale = np.sqrt(2.0 / in_dim)
    w = (rng.standard_normal((out_dim, in_dim)) * scale).astype(np.float16)
    b = (rng.standard_normal((out_dim,)) * 0.01).astype(np.float16)
    return w, b


def build_resnet(rng: np.random.Generator, in_c: int, g0: int, g1: int, out_dim: int,
                 spatial: int, k: int = 3):
    """indices: 0=conv0 1=bn0 2=relu 3=conv1 4=bn1 5=relu 6=maxpool 7=flatten
    8=linear -- matches docs/spec/13's own nn.Sequential-index convention.
    """
    w0, b0 = conv2d(rng, g0, in_c, k)
    bn0_w, bn0_b, bn0_rm, bn0_rv = batchnorm2d(rng, g0)
    w3, b3 = conv2d(rng, g1, g0, k)
    bn1_w, bn1_b, bn1_rm, bn1_rv = batchnorm2d(rng, g1)

    pooled = spatial // 2  # maxpool2d(kernel_size=2, default stride=2)
    flat = g1 * pooled * pooled
    w8, b8 = linear(rng, out_dim, flat)

    # Awkward insertion order (fixtures/README.md): biases and running
    # stats before weights, not grouped by layer.
    tensors = {
        "0.bias": b0, "1.running_mean": bn0_rm, "1.running_var": bn0_rv,
        "3.bias": b3, "4.running_mean": bn1_rm, "4.running_var": bn1_rv,
        "8.bias": b8,
        "0.weight": w0, "1.weight": bn0_w, "1.bias": bn0_b,
        "3.weight": w3, "4.weight": bn1_w, "4.bias": bn1_b,
        "8.weight": w8,
    }
    dims = {"in": in_c, "g0": g0, "g1": g1, "out": out_dim, "flat": flat}
    return tensors, dims


def topology_sidecar(dims: dict) -> dict:
    """spec 13 §2 perm-groups/tensors schema -- same reasoning as
    gen_mlp.py's: NOT what align/topology_parser.cpp (`sfs commit
    --topology`) reads (see layers_config() below for that). This is for
    permute.py and bench/residual_codec.cpp, which want the axis->group
    mapping directly (confirmed independently by tests/byte_identity_cnn.cpp,
    which runs the REAL parser against an equivalent architecture and gets
    exactly this grouping).
    """
    return {
        "type": "synapsefs.topology",
        "format_version": 1,
        "source": {"kind": "synthetic", "arch": "resnet"},
        "perm_groups": {
            "in":  {"size": dims["in"], "pinned": True},
            "g0":  {"size": dims["g0"], "pinned": False},
            "g1":  {"size": dims["g1"], "pinned": False},
            "out": {"size": dims["out"], "pinned": True},
        },
        "tensors": {
            "0.weight": {"axes": [{"dim": 0, "group": "g0", "block": 1},
                                  {"dim": 1, "group": "in", "block": 1}]},
            "0.bias":   {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            "1.weight": {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            "1.bias":   {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            "1.running_mean": {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            "1.running_var":  {"axes": [{"dim": 0, "group": "g0", "block": 1}]},
            # The multi-axis case: dim 1 (in-channels) depends on g0, and is
            # NOT the last dimension -- kh, kw (block 1 each, folded into
            # "trailing" by the C++ side, not expressed as a block factor
            # here since block is only for a FLATTENED axis, spec 12 §2).
            "3.weight": {"axes": [{"dim": 0, "group": "g1", "block": 1},
                                  {"dim": 1, "group": "g0", "block": 1}]},
            "3.bias":   {"axes": [{"dim": 0, "group": "g1", "block": 1}]},
            "4.weight": {"axes": [{"dim": 0, "group": "g1", "block": 1}]},
            "4.bias":   {"axes": [{"dim": 0, "group": "g1", "block": 1}]},
            "4.running_mean": {"axes": [{"dim": 0, "group": "g1", "block": 1}]},
            "4.running_var":  {"axes": [{"dim": 0, "group": "g1", "block": 1}]},
            # flatten: g1 channels x pooled x pooled -> block = flat / g1.
            "8.weight": {"axes": [{"dim": 0, "group": "out", "block": 1},
                                  {"dim": 1, "group": "g1",
                                   "block": dims["flat"] // dims["g1"]}]},
            "8.bias":   {"axes": [{"dim": 0, "group": "out", "block": 1}]},
        },
    }


def layers_config(in_c: int, spatial: int, k: int = 3) -> dict:
    """SPEC 13 "layers" array -- the schema align/topology_parser.cpp's
    parse_topology() actually reads for `sfs commit --topology`. Mirrors
    build_resnet()'s conv0 -> bn0 -> relu -> conv1 -> bn1 -> relu ->
    maxpool2d(2) -> flatten -> linear chain index-for-index (same
    nn.Sequential numbering as the tensor names, e.g. "3.weight" is layer
    index 3). padding=(k-1)//2, stride=1 keeps each conv2d's spatial size
    unchanged, matching build_resnet()'s own assumption that only the
    maxpool2d halves H/W (dims["flat"] = g1 * pooled * pooled) -- get this
    out of sync with build_resnet() and the parser's tracked (C,H,W) will
    disagree with the real flatten width, though the axis union itself
    (block = len / group_size, derived from actual tensor shapes) still
    comes out correct regardless.
    """
    pad = (k - 1) // 2
    return {
        "input_shape": [in_c, spatial, spatial],
        "layers": [
            {"type": "conv2d", "kernel_size": k, "stride": 1, "padding": pad},
            {"type": "batchnorm2d"},
            {"type": "relu"},
            {"type": "conv2d", "kernel_size": k, "stride": 1, "padding": pad},
            {"type": "batchnorm2d"},
            {"type": "relu"},
            {"type": "maxpool2d", "kernel_size": 2},
            {"type": "flatten"},
            {"type": "linear"},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory (e.g. fixtures/out)")
    ap.add_argument("--tiny", action="store_true", help="smaller channel counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune-scale", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.tiny:
        in_c, g0, g1, out_dim, spatial = 3, 16, 24, 5, 8
        prefix = "tiny_resnet"
    else:
        in_c, g0, g1, out_dim, spatial = 3, 64, 96, 10, 32
        prefix = "resnet"

    step0, dims = build_resnet(rng, in_c, g0, g1, out_dim, spatial)

    step1 = {}
    for name, arr in step0.items():
        noise = (rng.standard_normal(arr.shape) * args.finetune_scale).astype(np.float16)
        step1[name] = (arr.astype(np.float32) + noise.astype(np.float32)).astype(np.float16)

    metadata = {"format": "synapsefs-fixture", "arch": "resnet"}
    save_file(step0, os.path.join(args.out, f"{prefix}_step0.safetensors"), metadata)
    save_file(step1, os.path.join(args.out, f"{prefix}_step1.safetensors"), metadata)

    topo_path = os.path.join(args.out, f"{prefix}_topology.json")
    with open(topo_path, "w") as f:
        json.dump(topology_sidecar(dims), f, indent=2)

    layers_path = os.path.join(args.out, f"{prefix}_layers_config.json")
    with open(layers_path, "w") as f:
        json.dump(layers_config(in_c, spatial), f, indent=2)

    total_params = sum(a.size for a in step0.values())
    print(f"wrote {prefix}_step0.safetensors, {prefix}_step1.safetensors, "
         f"{prefix}_topology.json, {prefix}_layers_config.json "
         f"({total_params} params, dims={dims})")
    print(f"note: pass {prefix}_layers_config.json to `sfs commit --topology` "
         f"-- {prefix}_topology.json is a different schema for permute.py / "
         f"bench/residual_codec.cpp only")


if __name__ == "__main__":
    main()
