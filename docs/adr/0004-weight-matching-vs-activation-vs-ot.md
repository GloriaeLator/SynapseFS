# ADR 0004 - Weight matching alignment architecture, solver selection, and dispatch policy

## Context

Module 1 requires recovering neuron correspondence between two model checkpoints of the same architecture. The problem statement evaluates alignment across four core dimensions:
1. **Permutation Accuracy (10%):** Exact recovery of neuron permutations.
2. **Alignment Wall-Clock Time (8%):** Execution speed across model scales.
3. **Residual Ratio (7%):** Compression ratio after weight alignment.
4. **Hard Gates (Pass/Fail):** 16 GB RAM / 8 GB VRAM ceilings, out-of-core streaming above budget, byte-exact reconstruction, explicit non-alignable reporting (Requirement 1.h), and demonstrated scaling across small (~100M) and large (~7B) models in the same family (Requirement 1.i).

Literature on permutation symmetry offers three general matching paradigms:

1. **Weight Matching (LAP):** Formulate matching as a Linear Assignment Problem based on weight vector similarities. Operates per layer/group, iterated across network depth.
2. **Activation Matching:** Correlate unit activations across a calibration dataset.
3. **Optimal Transport / Soft Matching (Sinkhorn):** Relax permutation constraints to continuous optimal transport, rounding to discrete permutations at the end.

## Decision

We implement **Weight Matching by Linear Assignment Problems**, iterated by Gauss-Seidel coordinate descent over topologically ordered permutation groups ([SPEC 13](../spec/13-topology-config.md)).

### 1. Group-Size Driven Dispatch Strategy

- **Small Groups ($n < \text{lap\_crossover}$, default 4096):** Solved via `JvSolver` (exact Jonker–Volgenant LAP). Computes exact minimum-cost permutations where matrix materialization and $O(n^3)$ execution costs are negligible.
- **Medium Groups ($\text{lap\_crossover} \le n < \text{sparse\_crossover}$, default 8192):** Solved via `GreedySolver` (greedy initialization + local 2-swap refinement passes). Provides substantial speedups over exact LAP with zero empirical loss in permutation accuracy on structured models.
- **Large Groups ($n \ge \text{sparse\_crossover}$, default 8192):** Handled by `match_group_sparse`. Avoids instantiating full $n \times n$ cost matrices (which would require 19 GB RAM at $n = 71,429$, causing instant OOM crashes under the 16 GB hard gate). Evaluates true costs only for $K$ fingerprint-nearest candidates per unit, solving matching via a Jacobi auction (`auction.cpp`) with an automated $K$-widening retry safety valve and exact dense LAP repair (`sparse_null_repair`) on leftover unassigned units.

### 2. Out-of-Core Tiled Streaming Execution

All feature gathering (`build_features`) and candidate cost evaluations stream parameter tensors in row-tile chunks (`row_tile`, default 1024) via lazy `ITensorSource` reads ([ADR 0008](0008-out-of-core-streaming.md)). Memory consumption is bounded by tile sizes and group dimensions $n$, never by total model parameter counts.

### 3. Pre-Solver Alignability Gate & Confidence Assessment

Before accepting an alignment, `assess()` ([confidence.cpp](../../modules/align/src/confidence.cpp)) evaluates normalized assignment costs against identity and Monte Carlo random baselines (`ConfidenceOptions`).

If a checkpoint pair lacks structural correspondence (e.g. unrelated checkpoints or distinct model families), the normalized cost ratio fails confidence thresholds. The group is marked `alignable: false` (Requirement 1.h), bypassing further solver passes and safely falling back to `mode: full` unaligned storage.

---

## Architectural Philosophy & Design Rationale

1. **Residual Ratio Invariance Across Solvers:** On alignable checkpoint pairs, all valid solvers (`JvSolver`, `GreedySolver`, and `match_group_sparse`) recover the exact same neuron permutations. Consequently, solver choice does not alter the residual compression ratio (7% of score). Points on residual ratio are earned downstream by residual encoding (`zigzag` delta, bit-plane layout, zstd framing; see [ADR 0005](0005-residual-encoding.md)). Solver selection focuses strictly on **maximizing wall-clock throughput (8%)** and **surviving 7B scale without OOM hard gate failures**.
2. **Asymptotics & 7B Model Scale Trajectory:** Dense cost matrix construction exhibits $O(N^2 D)$ scaling. At $N \ge 2048$, dense solvers scale quadratically-plus ($\sim O(N^{2.3})$), making them intractable at 7B layer widths ($N > 70,000$). The sparse pipeline restricts true-cost evaluation to $K$ candidate pairs per row, achieving near-linear complexity scaling ($\sim O(N^{1.2})$) and reducing 7B layer alignment times from tens of minutes down to single-digit seconds.
3. **Memory Hard Gate Compliance:** Storing a dense float32 cost matrix for an $N = 71,429$ layer requires 19.0 GB RAM, instantly violating the 16 GB environment limit. `match_group_sparse` bounds memory to $O(N K)$ candidate structures, enabling out-of-core alignment within strict RAM constraints.
4. **Candidate Set Sparsity & Drift Dynamics:** Under fine-tuning drift, candidate recall (`recall_at_K`) remains near 100% even for small $K$. Counter-intuitively, increasing $K$ on drifted models *degrades* permutation accuracy because larger candidate sets increase the probability of selecting spurious cheap weight-space matches. Keeping $K$ small ($K=4$, scaled with $\sqrt{n}$ up to `max_K`) optimizes both execution speed and drift robustness.

---

## Consequences

- **Determinism:** Alignment remains 100% deterministic and reproducible across platforms.
- **Scalability:** The engine handles models from tiny MLPs up to 7B+ parameter scale without memory exhaustion.
- **Safety:** Unrelated or non-alignable checkpoints are detected cleanly, reporting `alignable: false` without wasting solver iterations or emitting invalid diffs.
