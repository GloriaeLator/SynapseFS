#!/usr/bin/env python3
"""modules/align/tests/fixtures/generate.py — the fixture data
test_mlp_end_to_end.cpp and test_sparse_match.cpp need, generated on demand
rather than committed (deliverables note: "keep large synthetic test
fixtures out of the commit history where you can").

Writes, per fixture: base.safetensors, target.safetensors, config.json (the
"layers" schema align::topology_parser::parse_topology actually expects --
NOT the perm_groups/tensors shape fixtures/*_topology.json uses at the repo
root, which is a pre-resolved shape for tools that bypass the real parser),
and ground_truth.json recording the exact planted permutation each test
asserts recovery of.

Same nn.Sequential-index tensor naming and awkward-order-free-by-default
conventions as fixtures/gen_mlp.py at the repo root; kept separate because
these are align's own unit-test fixtures, not the codec-benchmark ones.

Run this once before `ctest -L align` (or any full test run) picks up
test_mlp_end_to_end / test_sparse_match:
    python modules/align/tests/fixtures/generate.py
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file


def linear(rng, out_dim, in_dim):
    scale = np.sqrt(2.0 / in_dim)
    w = (rng.standard_normal((out_dim, in_dim)) * scale).astype(np.float16)
    b = (rng.standard_normal((out_dim,)) * 0.01).astype(np.float16)
    return w, b


def build_and_permute(rng, in_dim, h1, h2, out_dim, seed_perm):
    w0, b0 = linear(rng, h1, in_dim)
    w2, b2 = linear(rng, h2, h1)
    w4, b4 = linear(rng, out_dim, h2)
    base = {"0.weight": w0, "0.bias": b0, "2.weight": w2, "2.bias": b2,
            "4.weight": w4, "4.bias": b4}

    prng = np.random.default_rng(seed_perm)
    perm0 = prng.permutation(h1)   # layer0's output group
    perm2 = prng.permutation(h2)   # layer2's output group

    # target[i] = base[perm[i]] convention (matches lap.cpp / commit_planner.cpp).
    target = {}
    target["0.weight"] = base["0.weight"][perm0, :]
    target["0.bias"] = base["0.bias"][perm0]
    # 2.weight: rows by its OWN group (perm2), columns by the PREVIOUS layer's
    # group (perm0) -- the two-axis case matcher.cpp's outgoing-evidence fix
    # (see test_mlp_end_to_end.cpp's own file comment) exists for.
    target["2.weight"] = base["2.weight"][perm2, :][:, perm0]
    target["2.bias"] = base["2.bias"][perm2]
    # 4.weight: dim0 (out) pinned -- unchanged; dim1 (h2) follows perm2.
    target["4.weight"] = base["4.weight"][:, perm2]
    target["4.bias"] = base["4.bias"]

    return base, target, perm0.tolist(), perm2.tolist()


CONFIG = {"layers": [
    {"type": "linear"}, {"type": "relu"},
    {"type": "linear"}, {"type": "relu"},
    {"type": "linear"},
]}


def write_fixture(out_dir, in_dim, h1, h2, out_dim, seed_data, seed_perm):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed_data)
    base, target, perm0, perm2 = build_and_permute(rng, in_dim, h1, h2, out_dim, seed_perm)

    save_file(base, os.path.join(out_dir, "base.safetensors"), {"format": "synapsefs-fixture"})
    save_file(target, os.path.join(out_dir, "target.safetensors"), {"format": "synapsefs-fixture"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump({"layer0_output_perm": perm0, "layer2_output_perm": perm2}, f, indent=2)
    print(f"wrote {out_dir}: in={in_dim} h1={h1} h2={h2} out={out_dim}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # test_mlp_end_to_end.cpp: modest two-hidden-layer MLP, dense/exact path only.
    write_fixture(os.path.join(base_dir, "mlp_end_to_end"),
                 in_dim=48, h1=64, h2=32, out_dim=10, seed_data=1, seed_perm=2)

    # test_sparse_match.cpp: layer0 >= 8192 (sparse path), layer2 = 64 (dense
    # path, same run) -- exactly what the test's own comments assert.
    write_fixture(os.path.join(base_dir, "mlp_large_sparse"),
                 in_dim=64, h1=8192, h2=64, out_dim=10, seed_data=3, seed_perm=4)
