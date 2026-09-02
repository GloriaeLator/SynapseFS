# ADR 0012 — libtorch for the large-group sparse alignment path

## Context

A dense `n x n` cost matrix and the greedy solver's `O(n^2)` enumerate-and-
sort are the whole story for a group up to a few thousand units — that's
`bench/lap_bench.cpp`'s territory, and `align::MatchOptions::lap_crossover`
picks between exact JV and greedy within it. Past `sparse_crossover` (8192
by default), the dense matrix itself stops fitting the PS's 16 GB RAM
ceiling, not just running slowly: an early version of this path read an
entire large checkpoint's tensor into a `std::vector<std::byte>` up front,
which was the direct cause of an OOM kill on a ~2B-parameter model. That
failure is why `align::tools::SimpleStSource` is mmap-backed and genuinely
lazy instead.

The fix for the matrix itself is algorithmic, not just "read lazily":
fingerprint each unit into a short vector, generate a handful of nearest-
neighbor candidates per unit instead of comparing against all `n` others,
and solve the resulting sparse assignment problem with a Jacobi auction
(`modules/align/src/sparse_match.cpp`, `fingerprint.cpp`, `auction.cpp`).
This is real, non-trivial numerical work — fingerprint construction,
whitening, candidate scoring, auction bidding rounds — and needed to run
fast enough to be worth doing lazily in the first place.

## Decision

Implement the sparse path's tensor math on top of libtorch (`torch::Tensor`),
not hand-rolled `std::vector<float>` loops, and let it run on CUDA when
available, falling back to CPU otherwise
(`modules/align/src/detail_device.hpp`). The PS's own stated 8 GB VRAM
grading ceiling implies a GPU is actually present in the grading
environment, not just a limit to respect defensively if one happens to
exist — so this path should actually use it.

This is the same module that already pulls in Torch as a build dependency
for the dense path's cost-matrix math (see the top-level `find_package(Torch
REQUIRED)` history), so the sparse path sharing that dependency doesn't add
a new one — it uses the one align already needs.

## Consequences

- Every `torch::Tensor::accessor<>()` call in this path requires a CPU
  tensor — it throws at runtime on a CUDA tensor, unlike `.item<>()` (a
  device-to-host scalar copy, safe on any device). Any function that reads
  a tensor element-by-element into plain C++ containers must call `.cpu()`
  on it first, regardless of which device it originated on — a real,
  easy-to-miss failure mode this path's own comments flag at each read site.
- `SparseMatchOptions::max_K` (512) bounds how large the sparse candidate
  matrix is allowed to grow: at `n = 57943` (this path's own measured large-
  group case), a `K = 512` cost matrix is ~120 MB against the dense path's
  ~13.4 GB for the same group — the memory saving this whole ADR exists for.
- This is a heuristic starting point (candidate count, whitening, auction
  epsilon-scaling), not an empirically calibrated one against real large
  checkpoints — the same caveat `ConfidenceOptions::distinct_match_floor`
  carries. Validating it against a real large model is future work, not
  done in this submission (see `docs/known-gaps.md` for what else is in
  that category).
- Pulling `align` into a target that only needs its plain-data types
  (`MatchReport`, `GroupMatch`, `Topology`) does NOT require linking this
  path or Torch at all — see `modules/store/CMakeLists.txt`'s and
  `bench/CMakeLists.txt`'s `target_include_directories`-only pattern. This
  ADR's dependency is real but scoped to the module that actually calls
  compiled `align::` code, not every consumer of its headers.
