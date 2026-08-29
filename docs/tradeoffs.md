# Trade-offs

Every choice that could reasonably have gone the other way, what we picked, and
the number or argument behind it. Rows marked **PENDING** have a measurement
scheduled and a placeholder — filling them in is a task, not a formality.

The rubric rewards original thinking and well-reasoned trade-offs over the
first-suggested approach, and a feature nobody can defend counts as not
implemented. This file is where the defence lives.

---

## 1. Decided by measurement

### 1.1 Hash function — BLAKE3 over SHA-256

Hashing 1 GB, prototype machine:

| Function | Throughput |
|---|---|
| SHA-256 | 0.38 GB/s |
| BLAKE3, 1 thread | 4.30 GB/s |
| BLAKE3, all cores | 8.37 GB/s |

Verification time is 10% of the grade. **Caveat, stated honestly:** this box may
not have SHA-NI. On hardware that does, SHA-256 runs 2–2.5 GB/s and the gap
narrows a lot. **PENDING:** rerun `bench/verify_time.cpp` on the demo machine
and record the result here. If SHA-256 wins there, switch — "we measured and
the difference did not justify a non-stdlib dependency" is the better answer.

[ADR 0002](adr/0002-blake3-over-sha256.md)

### 1.2 Read granularity — chunked verification

32 MiB block, 64 KiB chunks, 2000 random 4 KiB reads:

| Strategy | Hashed | Time | Throughput | Diagnostic |
|---|---|---|---|---|
| Verify whole block | 33.554 MB | 430.4 ms | 0.6 MB/s | "corrupt" |
| Verify touched chunk | 0.066 MB | 44.5 ms | **183.9 MB/s** | "corrupt chunk 137" |
| No verification | 0 | 1.5 ms | 5375.3 MB/s | — |

Tamper detection is 10% and mmap throughput is 8%. They look like a choice and
are not: they conflict only because verification granularity and read
granularity differed. 300× the throughput *and* a better diagnostic.

### 1.3 Residual granularity — frames, not layers

32 MiB layer, chain depth 5, one 4 KiB read:

| Strategy | Decompressed | Time | Peak RAM |
|---|---|---|---|
| Whole layer | 201.3 MB | 1347.8 ms | 134.2 MB |
| 128 KiB frames | 0.8 MB | **0.8 ms** | **0.4 MB** |

[ADR 0005](adr/0005-residual-encoding.md)

### 1.4 Residual encoding — **PENDING**, six numbers required

Timeboxed to 90 minutes, on a real fine-tune pair. Measure **ratio and
decompression throughput**, both, for all six:

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a ^ b` | none | _pending_ | _pending_ |
| `a ^ b` | byte-plane | _pending_ | _pending_ |
| `a ^ b` | bitshuffle | _pending_ | _pending_ |
| `zigzag(b-a)` | none | _pending_ | _pending_ |
| `zigzag(b-a)` | byte-plane | _pending_ | _pending_ |
| `zigzag(b-a)` | bitshuffle | _pending_ | _pending_ |

Ratio is 7% of the grade and throughput is 8%. **Do not choose on ratio alone.**
Record all six here even after picking one; the rejected rows are the answer to
"did you consider…".

### 1.5 LAP fallback crossover — **PENDING**

| n | Exact (JV) ms | Greedy + 2-swap ms | Permutation accuracy, greedy |
|---|---|---|---|
| 512 | _pending_ | _pending_ | _pending_ |
| 2048 | _pending_ | _pending_ | _pending_ |
| 8192 | _pending_ | _pending_ | _pending_ |

State the crossover and what it costs. `research/lap_bench.py`.

### 1.6 Frame size and chain depth — **PENDING sweep**

128 KiB and 5 are defaults chosen together, not separately: small frames are
what make a deep chain tolerable. Sweep both against read latency and repository
size on a 10-commit lineage and record the surface here.

---

## 2. Decided by argument

### 2.1 Weight matching, not activation matching

Activation matching is generally more accurate, and it needs a dataset and a
forward pass. A version control system that cannot store a checkpoint without
also being given representative data is a different product. It also makes
diffs non-deterministic, which defeats content-addressed deduplication.

[ADR 0004](adr/0004-weight-matching-vs-activation-vs-ot.md)

### 2.2 Merge conflicts refuse; weights are never averaged

Averaging two conflicting versions of a tensor group produces an artifact
neither author wrote. As model-soup research, defensible. As version control,
indefensible — a VCS that silently invents content is broken. `merge` refuses
and requires `--ours` or `--theirs`.

We say this explicitly because it is a good Q&A question and the tempting
answer is wrong.

### 2.3 Snapshot policy has two bounds, not one

`max_chain_depth` bounds **time**; `snapshot_alpha` bounds **space**. Neither
implies the other. A hundred 0.1% deltas cost a hundred hops on every read; one
badly aligned group can produce a delta at 110% of full, because the XOR of
unrelated fp16 is high-entropy noise. Git makes the same pair of choices.

### 2.4 The header is stored, not regenerated

4.6% of a 24 KB fixture, ~0.0005% of a 7B checkpoint, and it deduplicates for
free. Against: a reconstruction that was four bytes wrong with every tensor
bit-identical. Not close.

### 2.5 No transfer journal

Content addressing already provides what a journal would record, and derived
state cannot disagree with reality the way recorded state can. Resume
recomputes the want set from the store. Deleting the prototype's `resume.py`
bookkeeping removed a whole class of "the journal says we have it but we don't"
bugs.

[SPEC 14 §5](spec/14-wire-protocol.md)

### 2.6 FUSE low-level over high-level

Control of `FOPEN_DIRECT_IO` (must be **off**, or `mmap` does not work at all)
and `FOPEN_KEEP_CACHE` (must be **on**, since commits are immutable), plus
zero-copy replies on the fault path. ~200 extra lines for a module worth 25%.

[ADR 0003](adr/0003-fuse-lowlevel-vs-highlevel.md)

### 2.7 Loose objects before packfiles

Atomic-rename writes are trivially correct and easy to defend. Append-to-pack
writes need their own torn-write reasoning. Module 2 is graded on integrity, not
on inode efficiency; buy the hard property first.

[ADR 0006](adr/0006-packfiles-vs-loose-objects.md)

### 2.8 Streaming is the only path

Two code paths where one is exercised only at fixture scale means the one that
matters is the one we cannot debug. Slower on the tiny fixture, and that is the
price.

[ADR 0008](adr/0008-out-of-core-streaming.md)

### 2.9 Compile every ISA, dispatch at runtime

`-march=native` on a submitted binary is a SIGILL on the evaluator's machine.
Per-source-file ISA flags plus a CPUID-selected function pointer, dispatched
once per frame.

[ADR 0011](adr/0011-simd-dispatch-strategy.md)

### 2.10 C++23 rewrite of a working Python prototype

Genuinely expensive, and only defensible because the on-disk formats were
already frozen and carried over unchanged. The rewrite is of code, not of
design. The reasoning, and the checkpoint at which we would have reverted, is
in [ADR 0001](adr/0001-cpp23-and-toolchain.md).

---

## 3. Considered and rejected

| Idea | Why not |
|---|---|
| Trained zstd dictionary across frames | Real ratio win, and first on the cut list — it is an optimisation on top of a working codec. |
| `rsync`-style rolling-hash chunking | The PS requires *us* to implement content-addressed block diffing; delegating it is explicitly disallowed. Also: rolling hashes find shifted content, and our content is not shifted, it is permuted. |
| Storing float deltas instead of bit residuals | Reintroduces floating-point arithmetic into reconstruction, which must be bit-exact. XOR and zigzag are bijective; subtraction of floats is not. |
| Verifying alignment by comparing model outputs | Permuting reorders summation and float addition is not associative — the *correct* answer differs at ~5e-05. Alignment is verified by reconstructing bytes. |
| Deriving the manifest from the topology | Loses every tensor the topology does not model (`num_batches_tracked`), silently, invisibly to any "does it still load" test. |
| Averaging weights on merge | See 2.2. |
| GPG signing / authenticated refs | Out of scope per the PS; see [threat_model.md](threat_model.md). |
| Supporting `.pt` / `.bin` input | Explicit bonus in the PS. First thing cut if it ever starts. |
| Multi-file checkpoints (sharded safetensors) | Not in the graded fixtures. The manifest's `file` object would become an array; the design admits it and we have not built it. |

---

## 4. Known limitations

Stated here rather than discovered in the Q&A.

- Transformers are not supported and are out of scope in the PS.
- A single `.safetensors` file per commit; sharded checkpoints are not handled.
- No authentication or transport encryption.
- `gc` refuses while a mount is attached rather than coordinating with it.
- Alignment is a local optimum; there is no guarantee of the globally best
  permutation, and no ground truth to measure it against on real pairs.
- Cross-dtype lineages (fp16 → bf16) fall back to full storage.
