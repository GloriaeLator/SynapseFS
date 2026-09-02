# Trade-offs

Every choice that could reasonably have gone another way, what we picked, and
the number or argument behind it.

---

## 1. Decided by measurement

### 1.1 Hash function — BLAKE3 over SHA-256

Prototype-machine hashing, 1 GB:

| Function | Throughput |
|---|---|
| SHA-256 | 0.38 GB/s |
| BLAKE3, 1 thread | 4.30 GB/s |
| BLAKE3, all cores | 8.37 GB/s |

Re-measured in-container, real vendored BLAKE3, with SHA-NI present (this
box's original caveat): SHA-256 0.355 GB/s, BLAKE3 4.697 GB/s — the ~13×
gap holds even with hardware SHA acceleration available. See
`docs/benchmarks.md` §3.

[ADR 0002](adr/0002-blake3-over-sha256.md)

### 1.2 Read granularity — chunked verification

32 MiB block, 64 KiB chunks, 2000 random 4 KiB reads:

| Strategy | Hashed | Time | Throughput | Diagnostic |
|---|---|---|---|---|
| Verify whole block | 33.554 MB | 430.4 ms | 0.6 MB/s | "corrupt" |
| Verify touched chunk | 0.066 MB | 44.5 ms | **183.9 MB/s** | "corrupt chunk 137" |
| No verification | 0 | 1.5 ms | 5375.3 MB/s | — |

Tamper detection (10%) and mmap throughput (8%) look like competing goals
but aren't — they only conflicted because verification granularity and read
granularity differed. 300× the throughput and a better diagnostic.

### 1.3 Residual granularity — frames, not layers

32 MiB layer, chain depth 5, one 4 KiB read:

| Strategy | Decompressed | Time | Peak RAM |
|---|---|---|---|
| Whole layer | 201.3 MB | 1347.8 ms | 134.2 MB |
| 128 KiB frames | 0.8 MB | **0.8 ms** | **0.4 MB** |

[ADR 0005](adr/0005-residual-encoding.md)

### 1.4 Residual encoding — measured on `tiny_mlp` (~58k params, fp16)

Re-measured under the real CMake `release` preset, in-container — matches
the dev-box numbers below almost exactly (`docs/benchmarks.md` §2).

**Fine-tune pair** (`tiny_mlp_step0` → `tiny_mlp_step1`). Baseline — plain
zstd, no alignment — is 0.9173; every row below beats it:

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a ^ b` | none | 0.8408 | 574 |
| `a ^ b` | byte-plane | 0.8412 | 786 |
| `a ^ b` | bitshuffle | 0.8605 | 1083 |
| `zigzag(b-a)` | none | **0.8196** | **1291** |
| `zigzag(b-a)` | byte-plane | 0.8196 | 966 |
| `zigzag(b-a)` | bitshuffle | 0.8156 | 991 |

`zigzag(b-a)` + none wins both axes at once — best ratio (tied with
byte-plane) and best throughput, no trade-off to argue. This refutes this
ADR's own starting intuition (byte-plane/bitshuffle should help zstd): here
the extra transform pass costs throughput without improving ratio.

**Permuted-only pair** (pure permutation, no fine-tune noise) — the
headline case: ratio **0.00019** across all six candidates (a correctly
aligned pure permutation has a near-zero residual, so transform/codec choice
can't be distinguished at that floor). Baseline plain zstd: 0.9187. Alignment
turns a 92%-of-original file into 0.02%-of-original.

**Unrelated checkpoints** (two independently-seeded inits, no relation):
ratios 0.94–1.0001 — confirms XOR/zigzag of unrelated fp16 tensors is noise
that can compress *larger* than the input. This is why `snapshot_alpha`
(§2.3) exists and must force `full` storage here.

**Decision: `zigzag_after_permute` + `Transform::None`, the default in
`EncodeOptions`.** [ADR 0005](adr/0005-residual-encoding.md), Accepted.

### 1.4.1 The conv case — measured on `tiny_cnn` (~6k params, fp16, two conv layers)

`tiny_mlp` only exercises rank-2 tensors — every permuted axis is the
tensor's last dimension. `tiny_cnn` (conv2d → batchnorm2d → relu → conv2d →
batchnorm2d → relu → maxpool2d → flatten → linear) has a real rank-4 case:
`3.weight`, shape `[24, 16, 3, 3]`, whose in-channel axis (dim 1) depends on
the first conv's output group and is *not* the last dimension. This exact
shape surfaced a real bug in three places (`store::commit_planner`'s
`ColumnPermutingSource`, `codec::reconstruct`'s secondary-axis gather, this
bench's own `gather_axis`) — each needed to move the trailing `kh×kw` block
with each in-channel index instead of treating it as a scalar. All three
fixed.

Topology now comes from the real `align::parse_topology_file()` against
`tiny_cnn_layers_config.json`, not a hand-parsed sidecar — confirmed
byte-identical ratio (0.00158) to before the switch, including through the
rank-4 case, which is stronger evidence of correctness than "didn't crash."

**Permuted-only pair**: ratio **0.00158**, same near-zero floor as
`tiny_mlp`, now with a real rank-4 tensor in the mix. Baseline: 0.9328.
`3.weight` alone is over half this fixture's bytes — had the fix been wrong
rather than absent, its residual would be high-entropy noise pulling the
ratio back toward ~1.0, not the ~0.0016 measured.

**Fine-tune pair**: best is `zigzag(b-a)` + bitshuffle at 0.8276, 444 MB/s —
`zigzag` still beats `a^b`, though the winning transform differs from the
MLP case (bitshuffle here, none there). Worth another look on a larger conv
fixture; one small fixture can't settle whether transform choice is
shape-dependent.

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a ^ b` | none | 0.8505 | 234 |
| `a ^ b` | byte-plane | 0.8524 | 449 |
| `a ^ b` | bitshuffle | 0.8773 | 409 |
| `zigzag(b-a)` | none | 0.8303 | 464 |
| `zigzag(b-a)` | byte-plane | 0.8290 | 356 |
| `zigzag(b-a)` | bitshuffle | **0.8276** | 444 |

Default (`zigzag` + none) isn't the CNN-optimal row, but the gap (0.8303 →
0.8276) is small relative to how much smaller this fixture is than a real
checkpoint — not enough to change the project-wide default from §1.4
without a larger conv fixture confirming the pattern holds.

### 1.5 LAP fallback crossover

Real numbers now in `docs/benchmarks.md` §1 (`bench/lap_bench.cpp`,
synthetic random cost matrices):

| n | Exact JV | Greedy + 2-swap | Accuracy cost |
|---|---|---|---|
| 512 | 18.4 ms | 50.2 ms | +127.4% |
| 2048 | 323.7 ms | 1065.3 ms | +167.2% |
| 4096 | 2108.9 ms | 4942.7 ms | +187.5% |

On random input, greedy is slower than JV and 127–188% more expensive —
the opposite of the intended trade-off. Real alignment cost matrices are
far more diagonal-dominant than uniform-random ones, so this likely
understates greedy's real-world quality, but the crossover's speed/quality
trade-off is not validated by this benchmark as constructed. See
`docs/benchmarks.md` §1 for the full breakdown, including which method
`align::Matcher` actually uses at each size, and the separate sparse-path
(`match_group_sparse`, n ≥ 8192) scaling measurement.

### 1.6 Frame size and chain depth — not swept

128 KiB and 5 are defaults chosen together, not separately: small frames are
what make a deep chain tolerable. Given more time, we would have swept both
against read latency and repository size on a multi-commit lineage to
confirm that pairing actually holds rather than just reasoning about it —
not done here.

---

## 2. Decided by argument

### 2.1 Weight matching, not activation matching

Activation matching is generally more accurate but needs a dataset and a
forward pass. A version control system that can't store a checkpoint
without representative data is a different product — and it makes diffs
non-deterministic, defeating content-addressed deduplication.

[ADR 0004](adr/0004-weight-matching-vs-activation-vs-ot.md)

### 2.2 Merge conflicts refuse; weights are never averaged

Averaging two conflicting tensor groups produces an artifact neither author
wrote. Defensible as model-soup research; indefensible as version control —
a VCS that silently invents content is broken. `merge` refuses and requires
`--ours` or `--theirs`.

### 2.3 Snapshot policy has two bounds, not one

`max_chain_depth` bounds **time**; `snapshot_alpha` bounds **space**.
Neither implies the other: a hundred 0.1% deltas cost a hundred hops on
every read, and one badly aligned group can produce a delta at 110% of
full (XOR of unrelated fp16 is high-entropy noise). Git makes the same pair
of choices.

### 2.4 The header is stored, not regenerated

4.6% of a 24 KB fixture, ~0.0005% of a 7B checkpoint, and it deduplicates
for free. The alternative — a reconstruction four bytes wrong with every
tensor bit-identical — isn't close.

### 2.5 No transfer journal

Content addressing already provides what a journal would record, and
derived state can't disagree with reality the way recorded state can.
Resume recomputes the want set from the store. Removing the prototype's
`resume.py` bookkeeping eliminated a whole class of "the journal says we
have it but we don't" bugs.

[SPEC 14 §5](spec/14-wire-protocol.md)

### 2.6 FUSE low-level over high-level

Needed direct control of `FOPEN_DIRECT_IO` (must be off, or `mmap` doesn't
work at all) and `FOPEN_KEEP_CACHE` (must be on, since commits are
immutable), plus zero-copy replies on the fault path. ~200 extra lines for
a module worth 25%.

[ADR 0003](adr/0003-fuse-lowlevel-vs-highlevel.md)

### 2.7 Loose objects before packfiles

Atomic-rename writes are trivially correct. Append-to-pack writes need
their own torn-write reasoning. This module is graded on integrity, not
inode efficiency — buy the hard property first.

[ADR 0006](adr/0006-packfiles-vs-loose-objects.md)

### 2.8 Streaming is the only path

Two code paths where only one is exercised at fixture scale means the one
that matters in production is the one we can't debug. Slower on the tiny
fixture — that's the price.

[ADR 0008](adr/0008-out-of-core-streaming.md)

### 2.9 Compile every ISA, dispatch at runtime

`-march=native` on a submitted binary is a SIGILL on the evaluator's
machine. Per-source-file ISA flags plus a CPUID-selected function pointer,
dispatched once per frame.

[ADR 0011](adr/0011-simd-dispatch-strategy.md)

### 2.10 C++23 rewrite of a working Python prototype

Genuinely expensive, defensible only because the on-disk formats were
already frozen and carried over unchanged — the rewrite is of code, not
design. Reasoning and revert-point in [ADR 0001](adr/0001-cpp23-and-toolchain.md).

---

## 3. Considered and rejected

| Idea | Why not |
|---|---|
| Trained zstd dictionary across frames | Real ratio win, but an optimization on top of a working codec — first on the cut list. |
| `rsync`-style rolling-hash chunking | The PS requires *us* to implement content-addressed block diffing; delegating it is disallowed. Also: rolling hashes find shifted content, ours is permuted, not shifted. |
| Storing float deltas instead of bit residuals | Reintroduces floating-point arithmetic into reconstruction, which must be bit-exact. XOR/zigzag are bijective; float subtraction isn't. |
| Verifying alignment by comparing model outputs | Permuting reorders summation, and float addition isn't associative — the correct answer differs at ~5e-05. Alignment is verified by reconstructing bytes instead. |
| Deriving the manifest from the topology | Silently loses every tensor the topology doesn't model (`num_batches_tracked`) — invisible to a "does it still load" test. |
| Averaging weights on merge | See 2.2. |
| GPG signing / authenticated refs | Out of scope per the PS; see [threat_model.md](threat_model.md). |
| Supporting `.pt` / `.bin` input | Explicit PS bonus. First thing cut if it starts. |
| Multi-file checkpoints (sharded safetensors) | Not in the graded fixtures. The `tree` object that carries them is specified (SPEC 10 §6a) and implemented (`format::Tree`), but no commit points at one; see §4. |

---

## 4. Known limitations

Stated here rather than discovered in Q&A.

- Transformers are not supported — out of scope in the PS.
- A single `.safetensors` file per commit. The `tree` object for sharded
  checkpoints is implemented and tested (`format::Tree`, SPEC 10 §6a), but
  `Commit.manifest` still always addresses a `manifest`, so nothing in this
  build writes or reads a tree end to end. Sharded input is still rejected.
- No authentication or transport encryption.
- `gc` refuses while a mount is attached, rather than coordinating with it.
- Alignment is a local optimum — no guarantee of the globally best
  permutation, and no ground truth to measure against on real pairs.
- Cross-dtype lineages (fp16 → bf16) fall back to full storage.
