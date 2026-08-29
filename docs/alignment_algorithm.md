# The alignment algorithm

How SynapseFS recovers the neuron correspondence between two checkpoints of the
same architecture, and why it is done this way rather than the other two ways.

Normative details: [SPEC 13](spec/13-topology-config.md) (permutation groups)
and [SPEC 12](spec/12-residual-format.md) (what the alignment is used for).
The decision is [ADR 0004](adr/0004-weight-matching-vs-activation-vs-ot.md).

---

## 1. The symmetry we are exploiting

For a two-layer MLP with weights `W1, W2` and a permutation matrix `P`:

```
f(x) = W2 · σ(W1 · x)
     = (W2 Pᵀ) · σ((P W1) · x)
```

The permuted network computes *exactly the same function*. Its file shares
almost no bytes with the original. Any checkpoint store that does not
understand this is storing the same model many times over.

The job: given base checkpoint *A* and target checkpoint *B*, find the
permutation of each group of units that makes *B* look as much like *A* as
possible, so that what is left to store is small.

---

## 2. Permutation groups, not layers

The unit of alignment is a **group**, not a layer.

A ResNet block adds a skip connection onto the main path. The block's output
channels, the shortcut's output channels and every consumer downstream must all
carry the *same* permutation — that is a set, and pairwise links
(`permute_input_from`, and friends) cannot express it.

Groups come from a union-find over tensor axes: union an axis with another
whenever the architecture forces them to share a permutation (consecutive
layers, norm parameters with the conv or linear they follow, both branches of a
residual add, a flatten's grouped input axis with the producing conv's output
axis). The result is `{group_id, block}` per tensor axis, which is exactly what
the topology object stores.

### Blocking factors, and a failure with no exception

After the last pool, a conv with 16 output channels feeds a linear layer whose
input axis is 1024 = 16 × 8 × 8. Applying the 16-element channel permutation
directly to that axis is *legal numpy*:

```
topology says: linear_9.permute_input_from = 'conv2d_4'
conv2d_4 output channels : 16
linear_9 weight shape    : (10, 1024)   <- input axis is 1024, not 16

naive W[:, p]  -> shape (10, 16)  (silently keeps only 16 of 1024 columns)
               -> max output error 113.667   WRONG

blocking factor derived  : 1024 / 16 = 64   (= H*W after the last pool)
expanded column perm     : length 1024
max output error         : 4.96e-05   CORRECT
```

No exception, no warning, a model that computes garbage. The blocking factor is
**derived** (`block = axis_len / group_size`), never hardcoded, and a
non-integer result is a parse error naming both tensors.

One function covers every case:

```cpp
std::vector<uint32_t> expand(std::span<const uint32_t> p, uint32_t block) {
    if (block == 1) return {p.begin(), p.end()};
    std::vector<uint32_t> out(p.size() * block);
    for (size_t i = 0; i < p.size(); ++i)
        for (uint32_t k = 0; k < block; ++k)
            out[i * block + k] = p[i] * block + k;
    return out;
}
```

### The 4.96e-05, and what it forbids

Note that the *correct* result is not zero error. Permuting changes the order
in which a sum is accumulated, and floating-point addition is not associative.

The consequence is worth stating loudly because it shapes the entire test
strategy: **you cannot verify an alignment by comparing model outputs for exact
equality.** Alignment is verified by reconstructing bytes.

---

## 3. Why weight matching

Three families exist. We use the first.

| | Needs data? | Deterministic? | Exact permutation? |
|---|---|---|---|
| **Weight matching (LAP)** | no | yes | yes |
| Activation matching | **yes** | no | yes |
| Optimal transport / Sinkhorn | no | yes | only after rounding |

Activation matching is generally more accurate on genuinely independent
training runs, and it is disqualified here for a version-control reason rather
than an ML one: `sfs commit` would need a dataset and a working forward pass to
store a checkpoint. It would also make diffs non-deterministic, so two people
committing the same pair get different artifacts and content addressing
deduplicates nothing.

Optimal transport is elegant and ends in a rounding step, which is where the
exactness goes. We need an exact permutation, because reconstruction is
bit-exact.

---

## 4. The algorithm

```
for each permutation group g (in topology order):
    if g.pinned: p[g] = identity; continue
    C[g] = cost matrix over (base unit j, target unit i)        # §4.1
    p[g] = lap_solve(C[g])                                      # §4.2

repeat until no permutation changes, or sweep_cap:              # §4.3
    for each group g:
        rebuild C[g] using the CURRENT permutations of g's neighbours
        p[g] = lap_solve(C[g])

for each group g:                                               # §4.4
    if normalized_cost(g) > threshold: mark g not alignable
```

### 4.1 Cost

The cost of mapping base unit *j* onto target unit *i* is the negative inner
product of their parameter slices, gathered across **every tensor in the group**
and both sides of each unit:

- outgoing: the unit's row/filter in each tensor where the group indexes `dim 0`;
- incoming: the unit's column slice in each tensor where the group indexes an
  input axis, **permuted by that axis's group's current permutation**;
- norm parameters (`weight`, `bias`, `running_mean`, `running_var`) folded in as
  scalars per unit.

Everything is L2-normalised per unit before the inner product, so a group whose
units differ wildly in magnitude does not have its assignment decided by scale.
The ablation over cost variants lives in `research/cost_ablation.py`, and the
chosen form plus its numbers belong in [benchmarks.md](benchmarks.md).

Folding norms in matters more than it sounds: a BatchNorm following a conv
carries per-channel statistics that are highly discriminative between channels,
and they are cheap — one scalar per unit per tensor.

### 4.2 Solving

Linear assignment, minimising total cost.

- **Exact:** Jonker–Volgenant, `modules/align/src/lap.cpp`. O(n³) worst case,
  much better in practice on these matrices.
- **Above the measured crossover:** greedy assignment (sort candidate pairs by
  cost, take greedily) followed by local 2-swap refinement until no swap
  improves.

The crossover is **measured, not guessed** (`research/lap_bench.py`,
`bench/align_time.cpp`), and both the crossover and the accuracy cost of the
fallback are recorded in [benchmarks.md](benchmarks.md). "We used greedy above
n = 4096, and it costs us 0.3% permutation accuracy for a 40× speedup" is an
answer; "we used greedy because it's faster" is not.

### 4.3 Propagation

A group's cost depends on its neighbours' permutations, so one pass is not
enough. We sweep groups in topological order, re-solving each against the
current state, until nothing changes or a cap is hit. Sweep counts are logged;
on fine-tune pairs convergence is typically 1–2 sweeps because the answer is
close to identity, and that is a nice number to show.

### 4.4 Confidence, and saying "no"

The PS requires detecting and reporting "not meaningfully alignable". Most
implementations do not have this, and it is worth a slide.

We compute a normalised cost — the achieved assignment cost divided by the cost
of the identity assignment, or of a random one, whichever framing survives the
Day-3 measurement — and compare against a threshold. Above it, the artifact
carries `alignable: false` with a reason, and the manifest entry falls back to
`mode: full`.

Two distinct situations, both handled:

- **Degenerate symmetry** — dead units, duplicated units, many equally good
  permutations. The PS says any valid permutation is acceptable. We take the
  solver's answer and record `cost_normalized` so the ambiguity is visible.
- **Genuinely unrelated checkpoints** — different training runs, or a
  different model. High normalised cost, `alignable: false`, store full.

---

## 5. Out-of-core

Fixtures reach ~7B parameters in fp16 with a 16 GB RAM cap. Two checkpoints
resident is an instant OOM, and the PS says an OOM at fixture size fails the
metric even if it passes locally.

**The streaming path is the only path**, used even on the 1M-parameter MLP
fixture ([ADR 0008](adr/0008-out-of-core-streaming.md)). It is slower than
necessary on small inputs, and that is the price of never retrofitting it on
Day 4.

```
for each tile of target units Ti:
    read Ti's slices                       (bounded)
    for each tile of base units Bj:
        read Bj's slices                   (bounded)
        C[Ti, Bj] += contribution
        release Bj
    release Ti
```

Peak RSS is `tile_bytes × 2 + |C|`, where `|C|` is `n²` floats for the group —
256 MB at n = 8192. The **group size**, not the parameter count, is what bounds
the cost matrix, which is why this works at 7B at all: a 7B model has enormous
tensors and moderate channel counts.

Tile size is a tuning parameter, not a detail: alignment wall-clock is 8% of the
grade and I/O is inside it, so `bench/align_time.cpp` sweeps it.

---

## 6. What we do not claim

- **Transformers.** Attention head permutations and RoPE interleaving need
  their own group semantics. The PS puts them explicitly out of scope and out
  of the graded fixtures; we do not pretend otherwise.
- **Cross-architecture alignment.** Different shapes, no correspondence.
  `alignable: false`.
- **Alignment across dtypes.** fp16 to bf16 is a different bit layout; we store
  full.
- **Optimality.** Coordinate descent over groups converges to a local optimum.
  On planted permutations it recovers them exactly; on real fine-tune pairs
  there is no ground truth to be optimal against, and the metric that matters
  is the residual ratio.

---

## 7. How we know it works

| Claim | Test |
|---|---|
| A planted permutation is recovered exactly (MLP) | `modules/align/tests/test_known_permutation.cpp` |
| …and on a CNN with conv + BN + residual groups | same |
| …at two scales in the same family | `tests/test_end_to_end.cpp` |
| The flatten blocking factor comes out as 64 | `modules/align/tests/test_topology_parser.cpp` |
| Unrelated checkpoints report `alignable: false` | `modules/align/tests/test_unalignable.cpp` |
| The LAP solver finds the planted optimum | `modules/align/tests/test_lap.cpp` |
| Alignment never loads a whole checkpoint | `bench/scripts/peak_rss.sh`, asserted at fixture scale |
| Diff + base reconstructs the target byte-for-byte | `tests/byte_identity.cpp` |

The last one is the only one that actually matters. The rest exist to tell you
*where* it broke.
