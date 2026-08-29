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
| Permuted-only (function-identical) | _TBD_ | _TBD_ | _TBD_ | The headline number: near-zero is the claim |
| Fine-tune, 1 epoch | _TBD_ | _TBD_ | _TBD_ | |
| Fine-tune, converged | _TBD_ | _TBD_ | _TBD_ | |
| Unrelated checkpoints | _TBD_ | _TBD_ | ~1.0 | Should fall back to `full` |

The permuted-only row is the demonstration that the whole project exists for.
Measure it first.

### Codec experiment — six numbers

Both columns. Ratio is 7% and throughput is 8%; picking on ratio alone picks the
smaller number. Full table in [tradeoffs.md §1.4](tradeoffs.md).

| Residual | Transform | Ratio | Decompress MB/s |
|---|---|---|---|
| `a^b` | none | _TBD_ | _TBD_ |
| `a^b` | byte-plane | _TBD_ | _TBD_ |
| `a^b` | bitshuffle | _TBD_ | _TBD_ |
| `zigzag(b-a)` | none | _TBD_ | _TBD_ |
| `zigzag(b-a)` | byte-plane | _TBD_ | _TBD_ |
| `zigzag(b-a)` | bitshuffle | _TBD_ | _TBD_ |

### Kernel throughput by ISA

```bash
SFS_FORCE_ISA=scalar ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx2   ./build/release/bench/residual_codec --kernel-only
SFS_FORCE_ISA=avx512 ./build/release/bench/residual_codec --kernel-only
```

| ISA | GB/s |
|---|---|
| scalar | _TBD_ |
| avx2 | _TBD_ |
| avx512 | _TBD_ |

If AVX-512 is not faster here (downclocking), say so and pin AVX2.

---

## 3. Verification time

*Cryptographic Integrity, 20%: tamper detection 10, verification time 10.*

```bash
sudo ./bench/scripts/drop_caches.sh
./build/release/bench/verify_time --repo <repo> --json
```

| Repository | Objects | Bytes | `verify` | `verify --full` |
|---|---|---|---|---|
| 5 commits, small CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 commits, mid | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 3 commits, 7B | _TBD_ | multi-GB | _TBD_ | _TBD_ |

### Hash function, on this machine

```bash
./build/release/bench/verify_time --hash-only --bytes 1G
```

| Function | GB/s |
|---|---|
| SHA-256 | _TBD_ |
| BLAKE3, 1 thread | _TBD_ |
| BLAKE3, all cores | _TBD_ |

Prototype machine gave 0.38 / 4.30 / 8.37. **If this machine has SHA-NI the gap
will be much smaller** — record what it actually is, and revisit
[ADR 0002](adr/0002-blake3-over-sha256.md) if SHA-256 wins.

---

## 4. mmap throughput (cold cache)

*Filesystem Access & Memory, 25%: POSIX 10, mmap throughput 8, peak RSS 7.*

```bash
sudo ./bench/scripts/drop_caches.sh
./build/release/bench/mmap_throughput --mount /tmp/mnt --json
```

| Access pattern | Throughput | p50 | p99 |
|---|---|---|---|
| Sequential read, whole file | _TBD_ | _TBD_ | _TBD_ |
| `mmap` sequential | _TBD_ | _TBD_ | _TBD_ |
| Random 4 KiB, depth 0 | _TBD_ | _TBD_ | _TBD_ |
| Random 4 KiB, depth 5 | _TBD_ | _TBD_ | _TBD_ |
| `load_file()` end to end | _TBD_ | — | — |
| Baseline: same file on ext4 | _TBD_ | _TBD_ | _TBD_ |

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
