# Benchmarks

## Machine tested on

| | |
|---|---|
| CPU | AMD Ryzen 7 8840HS, 16 threads visible to the container, **SHA-NI and AVX-512 both present** |
| RAM | ~7.4 GiB visible inside the container (WSL2/Docker Desktop VM allocation, not the host's full RAM) |
| Storage | Host NVMe, via WSL2's virtualized filesystem — not a bare-metal number |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| libfuse | 3.14.0 |
| Compiler | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |

Grading caps: **16 GB RAM, 8 GB VRAM.** Every scale result below states the
fixture size it was measured against.

---

## 1. Alignment — wall-clock and permutation accuracy

*Alignment & Compression, 25% total: accuracy 10, wall-clock 8, ratio 7.*

```bash
./build/release/bench/align_time --json
```

(No arguments needed — runs a fixed fixture set. Or pass `--base`/`--target`/
`--topology`/`--ground-truth` for one ad-hoc fixture.)

| Fixture | Params | Groups | Sweeps | Wall-clock | Accuracy |
|---|---|---|---|---|---|
| MLP, dense path (planted perm) | 5546 | 4 | 2 | 0.76 ms | 66.7% (64/96) |
| MLP, sparse+dense mix (planted perm, layer0 size 8192) | 1,057,482 | 4 | 2 | 2.70 s | 99.2% (8192/8256) |
| MLP fine-tune, mid (~919k params) | 919,040 | 4 | 2 | 1.39 s | n/a |

Accuracy is exact-match recovery of a planted permutation — meaningless
without one, so n/a on the fine-tune row. §2's residual ratio is the
relevant metric there instead.

**Both planted-perm rows fail the same way, and it's a known bug, not new.**
Each has one independent group (100% recovered) and one dependent group
(0% recovered): 64/96 = layer0 (64/64, independent) right, layer2 (0/32,
dependent) wrong; 8192/8256 = layer0 (8192/8192, independent, sparse path)
right, layer2 (0/64, dependent, dense path) wrong. Same failure across both
solver paths, so it's `align::Matcher`'s dependent-group evidence
construction (`matcher.cpp`), not either LAP path. Root cause not yet
isolated — this is what `test_mlp_end_to_end.cpp`/`test_sparse_match.cpp`
already fail on.

### LAP solver crossover

```bash
./build/release/bench/lap_bench --sizes 512,1024,2048,4096,8192 --json
```

Synthetic random n × n cost matrices — the solvers never see a checkpoint,
so this benchmarks `align::make_jv_solver()`/`make_greedy_solver()` directly.

**`align::Matcher`'s real routing is three-way** (`matcher.cpp:238-274`):
`n >= sparse_crossover` (8192) skips dense matrices entirely for
`match_group_sparse` (fingerprint + Jacobi auction) — measured separately
below. Below that, `n < lap_crossover (4096) ? JV : Greedy` — note the
strict `<`, so n = 4096 itself already routes to Greedy. "Accuracy cost" is
how much more expensive greedy's assignment is than JV's true optimum:
`(greedy_cost − jv_cost) / jv_cost × 100`.

| n | Exact JV | Greedy + 2-swap | Accuracy cost | Method actually used at this n |
|---|---|---|---|---|
| 512 | 18.4 ms | 50.2 ms | +127.4% | Exact JV |
| 1024 | 72.7 ms | 242.2 ms | +171.6% | Exact JV |
| 2048 | 323.7 ms | 1065.3 ms | +167.2% | Exact JV |
| 4096 | 2108.9 ms | 4942.7 ms | +187.5% | Greedy (n < 4096 is false here) |
| 8192 | skipped (past crossover) | 23277.4 ms | n/a (no exact reference) | **Neither** — `match_group_sparse` (see below) |

Only the 512/1024/2048 rows measure the method actually used at that size.
4096's JV figure is a comparison point, not real behavior. 8192's Greedy
figure describes a path production never takes at all.

**Where it is a fair comparison (n < 4096), the result still contradicts
the crossover story.** Greedy is slower than JV at every size, and 127–172%
more expensive. `lap.hpp` sets the bar itself — *"greedy costs 0.Y%
accuracy for a Z× speedup" is an answer, "greedy is faster" is not* — and
here it's neither faster nor cheaper.

Likely cause: uniform-random costs give a greedy ascending-cost heuristic
nothing to exploit. Real alignment cost matrices (weight-similarity
features) are far more diagonal-dominant, which is probably the regime
`lap_crossover`'s default was tuned for. `GreedySolver::solve` also sorts
all n² pairs up front — not asymptotically cheaper than JV at these sizes
regardless of input structure.

**Bottom line:** this benchmark doesn't validate the dense crossover's
trade-off against realistic input. Future work, not done here:
`bench/align_time.cpp`'s `match_group` already builds a real cost matrix
per group internally but discards it instead of exposing it for reuse.

### Sparse path scaling

```bash
python3 bench/scripts/gen_sparse_scale.py    # once, writes fixtures/out/sparse_scale/
./build/release/bench/sparse_bench --json
```

The method actually used at n ≥ 8192 — untested by `lap_bench` since it
takes real tensors, not a cost matrix. Each size is a synthetic two-layer
MLP whose first hidden layer is the size under test (always past the
crossover) and a fixed size-8 second layer, present only to match a real
fixture's shape. Times `Matcher::match_group()` directly on just the first
layer, so the trivial second layer's cost doesn't dilute the number.

| n | Wall-clock | Accuracy |
|---|---|---|
| 8192 | 1.50 s | 100% (8192/8192) |
| 16384 | 3.86 s | 100% (16384/16384) |
| 32768 | 12.81 s | 100% (32768/32768) |
| 65536 | 51.94 s | 100% (65536/65536) |

Perfect recovery at every size — consistent with §1's own finding that
independent groups (this one has no dependency) always recover exactly,
regardless of dense or sparse path. `exact_solver` is `false` throughout,
as expected: the auction is an approximation, not a guarantee.

**Scaling is clearly super-linear, and gets worse as n grows** — each
doubling of n costs roughly 2.57×, 3.32×, then 4.05× the time (n^1.4 →
n^2.0 across this range), not the near-linear behavior you'd want from an
algorithm specifically built to avoid the dense path's O(n²) cost.
`SparseMatchOptions::K`'s own comment says starting K scales with
`sqrt(n)`, which alone would predict ~n^1.5 — consistent with the two
smaller jumps, but not the n^2.0 jump from 32768 to 65536. Not
root-caused here (candidate-widening retries on a low match rate are one
plausible cause, per `SparseMatchOptions::widen_on_null_rate`), but worth
flagging: this is the path a genuinely large model's biggest groups would
take, and its scaling isn't yet as good as the algorithm was designed for.

---

## 2. Residual ratio

```bash
./build/release/bench/residual_codec --pair <a>,<b> --json
python3 bench/scripts/residual_ratio.py build/release/bench-out
```

Every ratio below is post-`zstd`: `codec::compress_frame` (`modules/codec/src/compress.cpp`)
is `ZSTD_compress2` directly, so "Ratio" already means compressed/original,
not pre-compression residual size.

| Pair | Plain zstd | `zigzag` alone | `zigzag` + `zstd` | Notes |
|---|---|---|---|---|
| `tiny_mlp`, permuted-only (function-identical) | 0.9187 | 1.0000 | **0.000191** | Near-zero, as claimed |
| `tiny_cnn`, permuted-only, **conv** | 0.9318 | 1.0000 | **0.00158** | Same claim, real rank-4 tensor — `tradeoffs.md` §1.4.1 |
| `mlp`, fine-tune | 0.9157 | 1.0000 | 0.8680 | Full-size MLP (~919k params) |
| `tiny_mlp`, fine-tune | 0.9173 | 1.0000 | 0.8196 | |
| `tiny_cnn`, fine-tune | 0.9293 | 1.0000 | 0.8303 | |
| Unrelated checkpoints | — | 1.0000 | 1.0001 | Falls back to worse-than-raw, as ADR 0005 warns — confirms `full` storage is a necessary fallback |

"Plain zstd" is zstd on the raw checkpoint, no zigzag. "`zigzag` alone" is
exactly 1.0000 on every row, always — not a bug: zigzag is a bit-remapping,
not a compressor, and `six_candidates()`'s residual buffer is allocated at
exactly the target's byte count and filled one element at a time
(`bench/residual_codec.cpp:299-301`), so it can't change size by itself.
Its job is to reshape small deltas into small non-negative values; `zstd` is
what actually turns that into a smaller file — the real compression only
shows up once both steps run together, all the way to 0.0002–0.0016 on the
permuted pairs and ~0.82–0.87 on fine-tune pairs.

Measured via the real `release` preset in the project's own Docker
container. `./build/release/bench/residual_codec` with no arguments runs
every fixture pair under `fixtures/out/` and prints the full table.

The permuted-only rows are the core demonstration: a valid permutation
changes nearly every byte of a naive diff, but the residual ratio stays
near zero because alignment undoes the permutation first. Measured on
`tiny_mlp` (~58k params) then `tiny_cnn` (~6k params, two conv layers) —
the conv fixture exercises an in-channel axis that isn't the tensor's last
dimension, a case Linear-only fixtures can't reach, and surfaced (then
fixed) a real bug in the multi-axis permutation path.

### Codec experiment — six numbers

Ratio (7%) and throughput (8%) both matter; ratio alone would pick the
wrong candidate. Full comparison: `tradeoffs.md` §1.4.

Measured on the fine-tune pair (permuted-only floors all six at the same
near-zero ratio and doesn't discriminate). "Decompress MB/s" is zstd's own
throughput on each transformed candidate — the six numbers differ only in
what's fed to zstd, not in the compressor itself:

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a^b` | none | 0.8408 | 574 |
| `a^b` | byte-plane | 0.8412 | 786 |
| `a^b` | bitshuffle | 0.8605 | 1083 |
| `zigzag(b-a)` | none | **0.8196** | **1291** |
| `zigzag(b-a)` | byte-plane | 0.8196 | 966 |
| `zigzag(b-a)` | bitshuffle | 0.8156 | 991 |

**Winner: `zigzag(b-a)` + none** — best on both axes, no trade-off. Full
writeup: `tradeoffs.md` §1.4, ADR 0005 (Accepted).

### Kernel throughput by ISA

```bash
SFS_FORCE_ISA=scalar ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx2   ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx512 ./build/release/bench/residual_codec --kernel-only
```

64 MiB random buffers, 10 reps. Development machine (g++ 14.2, MSYS2
UCRT64, AVX-512F/BW/VL present), not in-container — re-measure on the
grading machine, which may lack AVX-512:

| ISA | XOR apply GB/s | Zigzag apply GB/s |
|---|---|---|
| scalar | 3.53 | 4.46 |
| avx2 | 9.37 | 10.94 |
| avx512 | 9.50 | 10.36 |

AVX-512 isn't meaningfully faster than AVX2 here — within noise on both
kernels, marginally slower on zigzag. Both kernels are memory-bandwidth-
bound, so wider registers stop helping once bandwidth saturates; AVX2's
~2.7× over scalar is the real gain. `util::best_isa()` still auto-selects
AVX-512 when present, but `SFS_FORCE_ISA=avx2` is a reasonable override
pending grading-hardware numbers.

---

## 3. Verification time

*Cryptographic Integrity, 20%: tamper detection 10, verification time 10.*

```bash
sudo ./bench/scripts/drop_caches.sh
./build/release/bench/verify_time --repo <repo> --json
```

| Repository | Objects | Bytes (`--full`) | `verify` | `verify --full` |
|---|---|---|---|---|
| 5 commits, `mlp` (in-container) | 7 | 1838584 | 0.203 ms | 1.308 ms |
| 5 commits, `tiny_mlp` (development machine) | 7 | 115944 | 0.422 ms | 0.687 ms |

Both `ok=1`, 0 findings. `Bytes` is `VerifyReport::bytes_hashed` from the
`--full` run — the quick `verify` never re-hashes chunks by design, so it's
correctly 0 there and not shown separately.

`10 commits, mid` and `3 commits, 7B` not measured — generating the larger
fixtures was out of scope for the time available.

### Hash function, on this machine

```bash
./build/release/bench/verify_time --hash-only --bytes 1G
```

| Function | GB/s (in-container) | GB/s (development machine) |
|---|---|---|
| SHA-256 | 0.355 | 0.296 |
| BLAKE3, 1 thread | 4.697 | 4.313 |
| BLAKE3, all cores | not measured — `core::Hasher` only exposes single-threaded hashing | |

BLAKE3 is ~13.2× faster than SHA-256 in-container, with SHA-NI available —
the gap doesn't close even with hardware SHA acceleration, answering ADR
0002's stated caveat. [ADR 0002](adr/0002-blake3-over-sha256.md) stands.

Development-machine numbers (0.296 / 4.313 GB/s) land within a few percent
of an earlier prototype (0.38 / 4.30 GB/s) — a useful cross-check, though
neither is the grading machine.

---

## 4. mmap throughput (cold cache)

*Filesystem Access & Memory, 25%: POSIX 10, mmap throughput 8, peak RSS 7.*

```bash
sudo ./bench/scripts/drop_caches.sh
./build/release/bench/mmap_throughput --mount /tmp/mnt --json
```

| Access pattern | Throughput | p50 | p99 |
|---|---|---|---|
| Sequential read, whole file | 236.1 MB/s | — | — |
| `mmap` sequential | 13132.4 MB/s | — | — |
| Random 4 KiB, depth 0 | 3806.3 MB/s | 0.9 us | 1.8 us |
| Baseline: same file on ext4 | 174.6 MB/s | — | — |

ext4 baseline included because a FUSE number alone isn't interpretable.

Not measured: concurrent-depth reads (`Random 4 KiB, depth 5` — no
concurrency dimension in this benchmark) and end-to-end `load_file()`
throughput. `load_file()` *correctness* (byte-identical tensors via a real
`safetensors.torch.load_file()` call) is covered separately by
`tests/e2e.py`.

First real mount-and-read under measurement for this project's FUSE daemon,
via a standalone driver (bypasses `apps/sfs`'s CLI/Torch dependency, calls
`store::`/`mount::` directly). Surfaced a real bug: `daemon.cpp` never
passed `max_read` to `fuse_session_new()`, so libfuse 3.14 rejected the
mount ("requested different maximum read size (131072 vs 0)"). Fixed by
passing `-omax_read=<N>` up front — see `daemon.cpp` at the fix site.

**Caveats:**
- **Not cold cache** — dropping caches needs interactive `sudo`, unavailable
  here. `FOPEN_KEEP_CACHE` is on by design (a commit is immutable), so
  these are warm-cache numbers throughout.
- **Fixture is small** (`mlp`, ~1.84 MB) — throughput here is dominated by
  per-request overhead and page-cache effects, not sustained streaming; the
  13 GB/s `mmap sequential` figure shows the path works, not a believable
  sustained number. Re-measure on a multi-GB fixture, cold cache, on the
  grading machine.
- Development machine, not the grading machine. `bench_mmap_throughput`
  isn't wired in as a CMake target yet (`bench/CMakeLists.txt`) — built and
  linked standalone instead.

---

## 5. Daemon peak RSS

```bash
./bench/scripts/peak_rss.sh ./build/release/apps/sfs/sfs mount <ref> /tmp/mnt
```

| Scenario | Cache budget | Peak RSS (`VmHWM`) |
|---|---|---|
| CNN checkpoint (~608 KB, 14 tensors), 1 reader | 1 GiB | 373.6 MiB (0.365 GiB) |

Measured in-container (`--cap-add SYS_ADMIN --device /dev/fuse`): init →
commit → `peak_rss.sh sfs mount main /tmp/mnt` → real read of the mounted
file → unmount.

Surfaced two tooling bugs, both fixed: `peak_rss.sh` used `bc` (not
installed in the build image) for its MiB/GiB conversion — the KiB figure
was already correct, only the derived columns were wrong — switched to
`awk`. Separately, the build stage had `libfuse3-dev` (headers) but not
`fuse3` (the `fusermount3` binary `sfs unmount` calls) — added in the
[Containerfile](../Containerfile).

The claim this section supports — peak RSS bounded by `cache_bytes +
frame_bytes × depth × readers`, independent of checkpoint size — is only
partly shown: this row proves the daemon mounts, serves reads, and stays
under budget, not that the bound holds independent of size (needs a second,
larger data point, out of scope here). Treat as measured, not proven at
scale.

---

## Reproducing all of it

```bash
cmake --preset release && cmake --build --preset release -j"$(nproc)"
make fixtures
./bench/scripts/run_all.sh build/release        # writes bench-out/*.json
```
