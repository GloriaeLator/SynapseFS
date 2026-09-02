#!/usr/bin/env python3
"""bench/scripts/gen_sparse_scale.py — fixtures for bench/sparse_bench.cpp,
docs/benchmarks.md §1's sparse-path scaling table.

align::Matcher routes any group of size >= sparse_crossover (8192, default)
to match_group_sparse (fingerprint + Jacobi auction) instead of the dense
LAP path -- this is the one method in the whole alignment story with no
benchmark data at all before this file. Reuses
modules/align/tests/fixtures/generate.py's build_and_permute()/
write_fixture() (the same two-hidden-layer MLP shape mlp_large_sparse
already uses) rather than duplicating the fixture logic, since only the
sizes swept differ.

Writes fixtures/out/sparse_scale/n<h1>/{base,target}.safetensors,
config.json, ground_truth.json for each size in SIZES. h2 stays fixed and
small (8): it exists only so the fixture has a second, trivially-fast dense
group after the sparse one, matching the real shape align::Matcher would
see, not because sparse_bench.cpp times it.

    python bench/scripts/gen_sparse_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "modules", "align", "tests", "fixtures"))
from generate import write_fixture  # noqa: E402

SIZES = [8192, 16384, 32768, 65536]

if __name__ == "__main__":
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "fixtures", "out", "sparse_scale")
    for h1 in SIZES:
        write_fixture(os.path.join(out_root, f"n{h1}"),
                     in_dim=64, h1=h1, h2=8, out_dim=10, seed_data=5, seed_perm=6)
