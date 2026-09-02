# The alignment algorithm

This describes how the alignment engine actually recovers the neuron correspondence between two
checkpoints.

## 1. Topology parsing - union-find over tensor axes

`align::parse_topology` (`src/topology_parser.cpp`) reads a `config.json`
describing a plain `nn.Sequential` - style layer chain (`linear`, `conv2d`,
`batchnorm2d`, `layernorm`, `relu`, `maxpool2d`, `flatten`, `dropout`,
`avgpool2d`/`adaptiveavgpool2d`; anything else is a hard parse error in
strict mode). It builds an `AxisUnionFind` (`graph.hpp`/`graph.cpp` - a
standard weighted union-find with path halving and union-by-rank) and, for
each parameterized layer, unions:

- a linear/conv layer's **input** axis with the previous layer's **output**
  axis,
- a bias vector with its own layer's output axis,
- BatchNorm/LayerNorm `weight`/`bias`/`running_mean`/`running_var` with the
  preceding layer's output axis,

and **pins** (marks "identity is the only legal permutation for this group")
the very first layer's input axis and the last parameterized layer's output
axis so that the raw input features and the classifier's output identity must not
move. Every axis the config doesn't mention gets its own fresh, pinned,
singleton group, so tensors like `num_batches_tracked` still round-trip.
`finalize()` sets each group's size to the **minimum** axis length among its
members and derives each member's block factor as `axis_len / group_size`;
a non-integer ratio is a hard `BlockFactorMismatch`, never silently
truncated. Config-less input is legal since everything just becomes a singleton
pinned group. There is no structural inference from tensor names or shapes
alone.

The parsed `Topology`'s wire schema uses `perm_groups` as the JSON key not `groups`.
## 2. Cost matrix and features

`CostMatrix` (`cost.hpp`/`cost.cpp`) is a dense, row-major `n × n` matrix of
`float`, where `n` is the **group size** (number of interchangeable units),
not the parameter count - this is what keeps a bounded memory budget
workable (8192 units ≈ 256 MB). Three metrics are implemented
(`CostMetric`): `NegInnerProduct` (default, additively decomposable across
tensor contributions), `L2`, `CosineDistance`.

Per-unit feature vectors (`matcher.cpp::build_features`) concatenate: the
unit's own outgoing rows across every tensor whose leading axis binds to the
group; incoming column slices from tensors whose *other* axis binds to the
group (when enabled); folded BatchNorm statistics as scalars; then
L2-normalize the concatenated row when normalization is enabled, so scale
doesn't dominate direction.

## 3. Assignment - three solvers, chosen by group size

This is genuinely three different algorithms, not one, dispatched by size:

- **Below `lap_crossover` (default 4096 units):** an exact solver
  (`lap.cpp`, wrapping `rectangular_lsap.cpp`, a port of SciPy's
  `linear_sum_assignment`). This implements the shortest-augmenting-path
  formulation from Crouse (2016), *"On implementing 2D rectangular
  assignment algorithms"* (IEEE TAES 52(4)) - commonly called the
  Jonker–Volgenant algorithm. It is exact, not a heuristic.
- **At or above `lap_crossover`, below `sparse_crossover` (default 8192):**
  a greedy heuristic (`make_greedy_solver`, `max_passes = 8` by default) -
  sort all `(i, j)` pairs ascending by cost, greedily assign non-conflicting
  pairs, then repeatedly scan for improving 2-swaps until none remain or the
  pass budget is exhausted.
- **At or above `sparse_crossover`:** the dense cost matrix is skipped
  entirely. `Matcher::match_group` switches to a fingerprint + candidate +
  auction pipeline operating on `torch::Tensor`s (CUDA if available):
  a sorted-quantile-position sketch (`fingerprint.hpp`, *not* a hash -
  "bit-invariant under permutation") generates a shortlist of `K` candidate
  partners per unit (`K` scales roughly as `2√n`, floor 4, cap 512,
  widened ×4 if the unassigned rate exceeds 2%), then a **Bertsekas
  parallel (Jacobi) auction algorithm** (`auction.hpp`/`auction.cpp`,
  integer-quantized costs, epsilon-scaling, a 400-round-per-phase guard)
  resolves the assignment among candidates, and any leftover unassigned
  units are cleaned up by an exact dense Jonker–Volgenant re-solve on just
  that leftover block.

`make_auto_solver(crossover)` is what most callers use; `MatchOptions`
exposes both crossover thresholds.

## 4. Coordinate descent across the network

`propagate.hpp`/`propagate.cpp` computes a group processing order via a
topological sort of "which axis-group's alignment depends on which other
axis-group" - Kahn's algorithm. ResNet-style residual-add cycles make a
true topological order impossible, so the code falls back to a deterministic
alphabetical order for those groups and lets the sweep loop resolve them
iteratively instead.

`Matcher::run()` performs **Gauss–Seidel** coordinate descent, not Jacobi:
each group in a sweep sees the *already-updated* permutations of groups
earlier in the same sweep, not just the previous sweep's results. It repeats
up to `max_sweeps` (default 8) times, stopping early once no group's
permutation changes between two consecutive full sweeps. Though convergence is typically 1–2 sweeps on real fine-tuned pairs.

## 5. Confidence scoring and the `alignable` gate

`confidence::assess(achieved_cost, identity_cost, random_cost, n,
distinct_matches, opts)` normalizes cost against the identity permutation's
cost (sign-aware - `NegInnerProduct` costs are negative), then compares it
against `effective_threshold = random_cost_normalized × random_baseline_margin`
(margin default 0.9 - the achieved cost must close at least 90% of the gap
between "random" and "perfect"). The random baseline itself is the exact
mean over all cost-matrix entries on the dense path, or a 4-sample Monte
Carlo average over random permutations on the sparse path.

Result is one of three verdicts (`Alignability`):

- **`NotAlignable`** - cost doesn't clear the threshold. The manifest must
  then store this group `Full`, not `Delta`.
- **`Degenerate`** - the threshold is cleared, but fewer than half the units
  (`distinct_match_floor`, default 0.5) have a uniquely-best match, i.e. the
  group has dead or duplicated units and the "correspondence" is ambiguous.
- **`Aligned`** - a real, usable correspondence was found.

## 6. Out-of-core streaming

`ooc_plan.hpp`/`ooc_plan.cpp` models peak memory as `2 × tile_bytes +
n² × sizeof(float)` (the tile plus the dense cost matrix) against a memory
budget that defaults to 8 GiB. **If the `n × n` cost matrix alone exceeds
the budget, `plan_tiles` returns a hard error**. Streaming reads of tensor rows (`detail_read_rows.hpp`) go through `ITensorSource::read_units` and widen fp16/bf16 to float only for cost computation - this is the only place in the whole codebase floating-point conversion happens; reconstruction is always an exact bijective integer operation (XOR or zigzag over raw bits),
never a float roundtrip. The sparse auction path bounds its own peak memory
independent of `n` via row-tiled reads (`row_tile`, default 1024).

## Constants

| Constant | Default | Meaning |
|---|---|---|
| `lap_crossover` | 4096 | exact JV below this group size |
| `sparse_crossover` | 8192 | switch to fingerprint+auction at/above this |
| `max_sweeps` | 8 | coordinate-descent sweep cap |
| greedy `max_passes` | 8 | 2-swap improvement passes |
| `MemoryBudget::bytes` | 8 GiB | out-of-core tiling budget |
| `random_baseline_margin` | 0.9 | confidence threshold vs. random baseline |
| `distinct_match_floor` | 0.5 | below this fraction, verdict is `Degenerate` |
| sparse `K` floor / cap | 4 / 512 | auction candidate shortlist size |
| sparse `row_tile` | 1024 | rows read per streaming tile |
| Monte Carlo samples (sparse random baseline) | 4 | |

The three assignment algorithms, the Gauss–Seidel sweep, the confidence gate, and out-of-core row streaming - is implemented, exercised by `modules/align/tests/` (including a test that
plants a known permutation and asserts `Matcher::run()` recovers it
exactly), and used unmodified from `apps/sfs/cmd/commit.cpp`.
