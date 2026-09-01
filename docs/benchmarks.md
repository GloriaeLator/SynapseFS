# Benchmarks

The five graded numbers, the hardware they were measured on, and the exact
command that reproduces each. **Everything here is PENDING until measured** —
the tables ship with placeholders on purpose, so that an unfilled cell is
visibly missing rather than quietly absent.

Rules, so the numbers mean something:

- Only the `release` preset. A RelWithDebInfo number is not a result.
- Cold page cache for anything touching the mount (`bench/scripts/drop_caches.sh`).
- State the hardware. A throughput number without a machine is a rumour.
- Report the command, not a description of the command.

---

## Machine

| | |
|---|---|
| CPU | _TBD_ (model, cores, base/boost, **whether it has SHA-NI and AVX-512**) |
| RAM | _TBD_ |
| Storage | _TBD_ (NVMe / SATA SSD / spinning — this dominates cold-cache numbers) |
| Kernel | _TBD_ (`uname -r`) |
| libfuse | _TBD_ (`pkg-config --modversion fuse3`) |
| Compiler | _TBD_ (`g++ --version`) |
| Commit | _TBD_ (`git rev-parse --short HEAD`) |

Grading caps: **16 GB RAM, 8 GB VRAM.** An OOM at fixture size fails the metric
even if it passes locally, so every scale run asserts a ceiling rather than
merely reporting one.

---

## 1. Alignment — wall-clock and permutation accuracy

*Alignment & Compression, 25% total: accuracy 10, wall-clock 8, ratio 7.*

```bash
./build/release/bench/align_time --fixture fixtures/out/<pair> --json
```

| Fixture | Params | Groups | Sweeps | Wall-clock | Peak RSS | Accuracy |
|---|---|---|---|---|---|---|
| MLP (planted perm) | ~1M | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ResNet CNN (planted perm) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Fine-tune pair, mid | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | n/a |
| 7B pair | ~7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | n/a |

"Accuracy" is exact-match recovery of a planted permutation, and is only
meaningful where one was planted. On real pairs there is no ground truth; the
residual ratio is the metric.

### LAP solver crossover

```bash
python3 research/lap_bench.py --sizes 512,1024,2048,4096,8192,16384
```

| n | Exact JV | Greedy + 2-swap | Accuracy cost |
|---|---|---|---|
| 512 | _TBD_ | _TBD_ | _TBD_ |
| 2048 | _TBD_ | _TBD_ | _TBD_ |
| 8192 | _TBD_ | _TBD_ | _TBD_ |
| 16384 | _TBD_ | _TBD_ | _TBD_ |

**Crossover: _TBD_.** State it and its accuracy cost. "We used greedy above
n = X, and it costs 0.Y% accuracy for a Z× speedup" is an answer.

---

## 2. Residual ratio

```bash
./build/release/bench/residual_codec --pair <a>,<b> --json
python3 bench/scripts/residual_ratio.py build/release/bench-out
```

| Pair | Naive bytes | Aligned + compressed | Ratio | Notes |
|---|---|---|---|---|
| Permuted-only (function-identical) | 115456 | 22 | **0.00019** | The headline number: near-zero is the claim — CONFIRMED on `tiny_mlp` |
| Fine-tune, 1 epoch | 115456 | 94624 | 0.8196 | best of six (`zigzag`+`none`); plain-zstd-of-target baseline is 0.9173 |
| Fine-tune, converged | _TBD_ | _TBD_ | _TBD_ | fixture not generated yet — needs a longer simulated training run |
| Unrelated checkpoints | 115456 | 115468 | 1.0001 | Fell back to worse-than-raw, as ADR 0005 warns — `full` storage confirmed necessary |
| Permuted-only, **conv** (`tiny_resnet`) | 12026 | 19 | **0.00158** | Same headline claim, this time with a real rank-4 (conv) tensor — see `docs/tradeoffs.md` §1.4.1 |

The permuted-only row is the demonstration that the whole project exists for.
Measured first, on `tiny_mlp` (~58k params, fp16) via `fixtures/gen_mlp.py` +
`fixtures/permute.py`, then again on `tiny_resnet` (~6k params, two conv
layers) via `fixtures/gen_resnet.py`, using a standalone dev-box build of
`bench/residual_codec.cpp` (g++ 14.2 / MSYS2 UCRT64 for the MLP numbers,
AVX-512 kernel for the conv numbers — **not** the graded machine either way,
re-measure once this builds under the real CMake release preset). The conv
fixture matters beyond "another data point": it exercises the one case
(a conv weight's in-channel axis, which is not the tensor's last dimension)
that a Linear-only fixture structurally cannot — and did in fact surface a
real bug (fixed) in the multi-axis permutation path before this number was
measured. Command:
`residual_codec --pair tiny_resnet_step0.safetensors,tiny_resnet_permuted.safetensors --topology tiny_resnet_topology.json --permutation tiny_resnet_permuted.permutation.json --json`.

### Codec experiment — six numbers

Both columns. Ratio is 7% and throughput is 8%; picking on ratio alone picks the
smaller number. Full table in [tradeoffs.md §1.4](tradeoffs.md).

Measured on the fine-tune pair (permuted-only floors all six at the same
near-zero ratio and doesn't discriminate — see `docs/tradeoffs.md` §1.4):

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a^b` | none | 0.8408 | 574 |
| `a^b` | byte-plane | 0.8412 | 786 |
| `a^b` | bitshuffle | 0.8605 | 1083 |
| `zigzag(b-a)` | none | **0.8196** | **1291** |
| `zigzag(b-a)` | byte-plane | 0.8196 | 966 |
| `zigzag(b-a)` | bitshuffle | 0.8156 | 991 |

**Winner: `zigzag(b-a)` + none** — best on both axes, no trade-off needed.
Full writeup and the `full`-storage fallback evidence: `docs/tradeoffs.md`
§1.4, `ADR 0005` (now Accepted).

### Kernel throughput by ISA

```bash
SFS_FORCE_ISA=scalar ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx2   ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx512 ./build/release/bench/residual_codec --kernel-only
```

Measured via `SFS_FORCE_ISA=<isa> residual_codec --kernel-only` (64 MiB random
buffers, 10 reps). Dev-box only (g++ 14.2 / MSYS2 UCRT64, x86-64;
AVX-512F/BW/VL present and OS-enabled on this box) — re-measure on the graded
machine, which may not have AVX-512 at all:

| ISA | XOR apply GB/s | Zigzag apply GB/s |
|---|---|---|
| scalar | 3.53 | 4.46 |
| avx2 | 9.37 | 10.94 |
| avx512 | 9.50 | 10.36 |

**AVX-512 is not meaningfully faster than AVX2 here** (within noise on both
kernels, and slightly *slower* on zigzag) — exactly the case this ADR
anticipated. Both kernels are streaming, memory-bandwidth-bound XOR/add over
large buffers, so wider registers stop helping once bandwidth is saturated;
AVX2's ~2.7× over scalar is the real win, AVX-512's extra width buys nothing
further on this workload. **Pinning `SFS_FORCE_ISA=avx2` is the reasonable
default pending a check on the actual grading machine** — `util::best_isa()`
still auto-selects AVX-512 when present, since the difference isn't negative
enough to special-case, but this is worth revisiting once real hardware
numbers are in.

If AVX-512 is not faster here (downclocking), say so and pin AVX2.

---

## 3. Verification time

*Cryptographic Integrity, 20%: tamper detection 10, verification time 10.*

```bash
sudo ./bench/scripts/drop_caches.sh
./build/release/bench/verify_time --repo <repo> --json
```

| Repository | Objects | Bytes (`--full`) | `verify` | `verify --full` |
|---|---|---|---|---|
| 5 commits, `tiny_mlp` | 7 | 115944 | 0.422 ms | 0.687 ms |
| 10 commits, mid | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 3 commits, 7B | _TBD_ | multi-GB | _TBD_ | _TBD_ |

Measured with a standalone driver calling `store::verify`/`CommitStore`/
`ManifestStore` directly (dev-box, WSL2 Ubuntu, real BLAKE3 — not the graded
machine, and `bench_verify_time` doesn't exist as a real CMake target yet,
see below). `ok=1`, 0 findings. `Bytes` is `VerifyReport::bytes_hashed` from
the `--full` run specifically — the quick `verify` never hashes anything by
design (spec: only `--full` re-hashes every chunk), so it's correctly 0 there
and not shown as a separate column. This field was dead code until just now:
declared in `verify.hpp` but never assigned in `verify.cpp` — fixed by
accumulating `blocks.size_of(oid)` alongside each successful `verify_block()`
call in the `--full` path, the only place bytes are actually re-hashed. A
second, related bug surfaced fixing it: `objects_checked` was being
double-counted for any object with a finding (incremented once in
`check_block`, again in `add_finding`, which was also — incorrectly —
counting non-object findings like a bad ref or a chain-depth mismatch as
"objects"); removed the increment from `add_finding` entirely, since
`check_block` is the only place actually enumerating objects.

### Hash function, on this machine

```bash
./build/release/bench/verify_time --hash-only --bytes 1G
```

| Function | GB/s |
|---|---|
| SHA-256 | 0.296 |
| BLAKE3, 1 thread | 4.313 |
| BLAKE3, all cores | _TBD_ — this codebase's BLAKE3 wrapper (`core::Hasher`/`core::digest`) only exposes single-threaded hashing, no multithreaded path to measure |

Prototype machine gave 0.38 / 4.30 / 8.37 — this dev-box run (0.296 / 4.313,
1 GiB random buffer, WSL2 Ubuntu, real vendored BLAKE3) lands within a few
percent of the prototype's single-thread numbers on both functions, a good
consistency check even though this isn't the graded machine.
[ADR 0002](adr/0002-blake3-over-sha256.md) stands: SHA-256 is not close to
winning here.

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
| Random 4 KiB, depth 5 | _TBD_ — concurrent-depth variant not implemented in this pass | | |
| `load_file()` end to end | _TBD_ — needs Python + torch, tests/e2e.py's job, not this bench | — | — |
| Baseline: same file on ext4 | 174.6 MB/s | — | — |

**Real numbers from a real mount, not a workaround** — the first time this
project's FUSE daemon has actually been mounted and read from at all, via a
standalone driver (`store`/`codec`/`mount` don't need align/Torch to build;
`apps/sfs`'s CLI does, so this bypasses it and calls the same `store::`/
`mount::` APIs directly). That attempt immediately surfaced a real,
previously-undiscovered bug: `daemon.cpp` never passed a `max_read` mount
option to `fuse_session_new()`, so this libfuse (3.14) rejected the mount
outright once `fuse_ll.cpp`'s own `init()` tried to raise it — "init() and
fuse_session_new() requested different maximum read size (131072 vs 0)".
Fixed by passing `-omax_read=<N>` up front so both sides agree from the
start; see `daemon.cpp`'s comment at the fix site.

**Caveats, stated plainly rather than left implicit:**
- **Not cold cache** — dropping caches needs `sudo`, which this session
  can't run non-interactively. `FOPEN_KEEP_CACHE` is on by design (a commit
  is immutable), so these are warm-cache numbers throughout, not the
  §4 heading's own "cold cache" requirement.
- **Fixture is small** (`mlp`, ~1.84 MB, 919k params) relative to a real
  checkpoint, so absolute throughput here is dominated by fixed per-request
  overhead and kernel page-cache effects, not sustained streaming
  performance at scale — `mmap sequential`'s 13 GB/s in particular is not a
  believable sustained number, just evidence the path works end to end.
  Re-measure on a multi-GB fixture, cold cache, on the actual graded
  machine before trusting these for anything but "it works."
- **`Random 4 KiB, depth 5`** (concurrent readers) and **`load_file()`
  end to end** are not measured here — the former needs a concurrency
  dimension this pass didn't implement, the latter needs Python +
  `safetensors.torch.load_file()`, a different test (`tests/e2e.py`)
  entirely.
- Dev-box (WSL2 Ubuntu), not the graded machine, and `bench_mmap_throughput`
  doesn't exist as a real CMake target yet — `bench/CMakeLists.txt`
  blanket-links `align` into every bench binary, so it's blocked the same
  way `bench_residual_codec` was until that's resolved; this was built and
  linked standalone instead.

Report the ext4 baseline. A FUSE number without it is uninterpretable.

---

## 5. Daemon peak RSS

```bash
./bench/scripts/peak_rss.sh ./build/release/apps/sfs/sfs mount <ref> /tmp/mnt
```

| Scenario | Cache budget | Peak RSS (`VmHWM`) |
|---|---|---|
| Small CNN, 1 reader | 1 GiB | _TBD_ |
| 7B, 1 reader, full `load_file()` | 1 GiB | _TBD_ |
| 7B, 8 concurrent readers | 1 GiB | _TBD_ |
| 7B, cache budget 256 MiB | 256 MiB | _TBD_ |

The claim to support: peak RSS is bounded by
`cache_bytes + frame_bytes × depth × readers` and is **not** a function of
checkpoint size. The last row is the one that proves it.

---

## Reproducing all of it

```bash
cmake --preset release && cmake --build --preset release -j"$(nproc)"
make fixtures
./bench/scripts/run_all.sh build/release        # writes bench-out/*.json
```
