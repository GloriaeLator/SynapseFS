# ADR 0004 — Weight matching by LAP, not activation matching or optimal transport

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

Module 1 asks us to recover the neuron correspondence between two checkpoints
of the same architecture. Alignment accuracy is 10% of the grade, wall-clock 8%,
residual ratio 7%.

The literature on permutation symmetry (Git Re-Basin and its neighbours) offers
three families.

## Options

**1. Weight matching.** Treat it as a linear assignment problem: the cost of
mapping base unit *j* to target unit *i* is a similarity between their
parameter vectors. Solve per layer, propagate, iterate to a fixed point.

- Needs only the two checkpoints. No data, no forward passes, no framework.
- Cost is `O(n²)` to build the matrix and `O(n³)` to solve it exactly.

**2. Activation matching.** Run both models on a batch of data, correlate unit
activations, match on that.

- Generally more accurate on genuinely different training runs.
- Requires a dataset, a working forward pass, and a framework. In a VCS this
  is fatal: `sfs commit` would need `torch` and representative data to store a
  checkpoint. It also makes the diff non-deterministic — two people committing
  the same pair get different artifacts.

**3. Optimal transport / soft matching (Sinkhorn).** Solve a relaxed problem,
then round to a permutation.

- Elegant, differentiable, handles near-degenerate symmetry gracefully.
- The rounding step is where the exactness goes, and we need an exact
  permutation because reconstruction is bit-exact. We would be adding a
  relaxation and then throwing it away.

## Decision

**Weight matching, solved as a LAP, iterated by coordinate descent across
layers.** Rejected the other two, and the reason is a VCS reason rather than an
ML one: a version control system must be able to store a checkpoint with no
inputs beyond the checkpoint. Requiring data to commit is a different product.

Cost function and solver:

- Cost between units is negative inner product of the concatenated incoming and
  outgoing parameter slices, normalised per unit. The ablation lives in
  `research/cost_ablation.py` and the chosen form is recorded in
  `docs/alignment_algorithm.md` with its numbers.
- Exact solver: Jonker–Volgenant (`modules/align/src/lap.cpp`). The prototype
  used `scipy`; in C++ we own it. `research/lap_bench.py` measures where the
  exact solver stops being acceptable.
- Above that crossover: greedy assignment plus local 2-swap refinement.
  **The crossover and the accuracy cost must be measured and stated**, not
  waved at — this is precisely the kind of thing the Q&A pushes on.
- Convergence: sweep layers in order, re-solving each against its current
  neighbours, until no permutation changes or a sweep cap is hit. Log the sweep
  count.

Groups, not layers, are the unit of matching — a residual block's branches
share a permutation (SPEC 13), so the LAP is solved once per group over the
concatenated evidence from every tensor in it.

## Consequences

- Alignment is deterministic and reproducible: the same pair of checkpoints
  always produces the same artifact. Two people committing the same thing get
  identical objects, which content addressing then deduplicates.
- No `torch` dependency in the commit path at all.
- We must detect **degenerate symmetry** (dead units, duplicated units) where
  many permutations are equally good. Any valid permutation is acceptable per
  the PS; we take the solver's answer and record `cost_normalized` so the
  ambiguity is visible.
- We must detect **not meaningfully alignable** — two checkpoints that do not
  correspond. Normalised cost above a threshold sets `alignable: false` with a
  reason, and the manifest entry falls back to `mode: full`. Having this at all
  is worth saying out loud in the presentation; most implementations do not.

## How we would know we were wrong

If planted-permutation recovery on the CNN fixture is below 100% on Day 3, the
cost function is wrong before the solver is. Check `research/cost_ablation.py`
first. If recovery is exact but the residual ratio is poor on real fine-tune
pairs, the problem is the encoding (ADR 0005), not the alignment.
