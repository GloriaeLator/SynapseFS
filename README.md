# SynapseFS

Permutation-aware, cryptographically verifiable version control and virtual
filesystem for neural network checkpoints.

Git stores a fine-tuned checkpoint as a whole new blob. SynapseFS stores it as
a delta — after recovering the neuron correspondence between the two
checkpoints, so that a model which was merely *permuted* costs almost nothing.
Reconstruction is **byte-for-byte**, headers and metadata included, and the
result is readable through a read-only POSIX mount that
`safetensors.torch.load_file()` opens without modification.

> Built for Takneek 2026 (Programming Club, IIT Kanpur). Status: **in
> development**. See [docs/](docs/) for the design, and
> [docs/adr/](docs/adr/) for why each decision was made.

---

## Why this is hard

Two reasons plain diffing does not work on checkpoints:

1. **Floats.** A tiny numeric update flips bits throughout every IEEE-754
   value, so a byte-level diff of two adjacent training steps is nearly the
   size of the file.
2. **Permutation invariance.** Swapping neuron indices in layer *l* and
   applying the matching permutation to layer *l+1* leaves the network's
   function *identical* and changes ~100% of the file's bytes.

SynapseFS recovers the permutation, applies it, and stores what is left.

---

## Quick start

```bash
# Build (see docs/build.md for prerequisites)
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset dev && cmake --build --preset dev -j"$(nproc)"
./build/dev/apps/sfs/sfs --version

# Or, with zero host setup:
docker build -t synapsefs . && docker run --rm synapsefs --version
```

```bash
# Generate the small fixtures (checkpoints are never committed to git)
make fixtures-small

# A repository, two commits, and a delta
sfs init myrepo && cd myrepo
sfs commit ../fixtures/out/mlp_step0.safetensors -m "initial"
sfs commit ../fixtures/out/mlp_step1.safetensors -m "after fine-tune"
sfs log --graph

# Prove it round-trips
sfs checkout HEAD --out /tmp/restored.safetensors
cmp /tmp/restored.safetensors ../fixtures/out/mlp_step1.safetensors && echo "byte-identical"

# Integrity, standalone — no checkout, no mount
sfs verify --full

# Mount it and load it with torch, with nothing materialised on disk
mkdir -p /tmp/mnt && sfs mount HEAD /tmp/mnt &
python3 -c "from safetensors.torch import load_file; print(len(load_file('/tmp/mnt/model.safetensors')))"
sfs unmount /tmp/mnt
```

---

## Commands

| Command | What it does |
|---|---|
| `sfs init [<path>]` | Create a repository. |
| `sfs commit <ckpt.safetensors> -m <msg>` | Align against the parent, store the delta, write a commit. `--no-delta` forces full storage. |
| `sfs checkout <ref> [--out <path>]` | Switch branch, or materialise a checkpoint file. |
| `sfs branch [<name>] [-d <name>]` | Create or list. Switching is `checkout <branch>` (pre-2.23 git semantics). |
| `sfs log [<ref>] [--graph]` | Walk history. |
| `sfs verify [<ref>] [--full] [--repair]` | Integrity check. **Works standalone.** Exits 4 on corruption, naming the object and chunk. |
| `sfs merge <branch> [--ours\|--theirs]` | Fast-forward, or three-way per tensor group. Refuses on conflict. |
| `sfs serve [--listen host:port]` | Start the sync listener. Default `127.0.0.1:9418`. |
| `sfs push <url> [<branch>]` | Send only the blocks the peer lacks. |
| `sfs pull <url> [<branch>]` | Fetch only the blocks we lack. Resumable. |
| `sfs mount <ref> <mountpoint>` | Read-only FUSE mount. Nothing is written to disk. |
| `sfs unmount <mountpoint>` | Unmount. |
| `sfs gc [--pack] [--prune]` | Repack and reclaim. |

Full flag reference and exit codes: **[docs/spec/15-cli-contract.md](docs/spec/15-cli-contract.md)**.

### Networking

`sfs serve` listens on `127.0.0.1:9418` by default. Override with
`--listen <host>:<port>` or by setting `listen` in `.synapsefs/config`.
The peer connects with `sfs push http://host:port` / `sfs pull …`.
Transport is plain TCP with length-prefixed frames; there is no authentication
or encryption, which is a deliberate scope decision — see
[docs/threat_model.md](docs/threat_model.md).

---

## How it works, in one page

```
 checkpoint A ──┐
                ├─► topology parser ──► permutation groups (union-find over tensor axes)
 checkpoint B ──┘                              │
                                               ▼
                                    alignment engine (LAP per group,
                                    coordinate descent, out-of-core)
                                               │
                                               ▼
                            diff artifact:  permutation + framed residuals
                                               │
                                               ▼
   content-addressable store  ─── BLAKE3 ───►  Merkle DAG (commit → manifest → blocks)
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
              sfs checkout (write a file)                  sfs mount (FUSE, lazy)
                        └────────────► ONE read_range() ◄─────────────┘
```

Four properties worth knowing before reading the code:

**The manifest describes a *file*, not a model.** It stores the verbatim
safetensors header as its own block plus a buffer layout covering every byte,
so reconstruction is concatenation rather than serialisation. Regenerating the
header instead produced a file that was four bytes wrong with every tensor
bit-identical. ([SPEC 10](docs/spec/10-object-model.md))

**Residuals are stored in frames.** Independently decompressible ranges of
output units, so serving one 4 KiB page fault touches ~0.8 MB instead of
201 MB on a depth-5 chain. ([SPEC 12](docs/spec/12-residual-format.md))

**Verification granularity equals read granularity.** Per-chunk digests inside
every block, plus a digest of the reconstructed bytes on every residual frame.
Tamper detection survives the reconstruction path and costs proportional to
what was actually read — 183.9 MB/s instead of 0.6 MB/s.
([SPEC 11](docs/spec/11-repo-layout.md))

**Checkout and mount call the same function.** `read_range` is the only
reconstructor, so "checkout bytes == mount bytes" is a property of the call
graph rather than a test that happens to pass.
([SPEC 16](docs/spec/16-consistency.md))

---

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Module boundaries and the object graph |
| [alignment_algorithm.md](docs/alignment_algorithm.md) | Permutation groups, LAP, propagation, out-of-core streaming |
| [storage_format.md](docs/storage_format.md) | Objects, framing, chunking, crash safety |
| [tradeoffs.md](docs/tradeoffs.md) | Every choice, with the measurement behind it |
| [threat_model.md](docs/threat_model.md) | What we detect, what is out of scope, and why |
| [benchmarks.md](docs/benchmarks.md) | The five graded numbers, the hardware, the commands |
| [build.md](docs/build.md) · [testing.md](docs/testing.md) | Building and testing |
| [spec/](docs/spec/) | Normative format specifications |
| [adr/](docs/adr/) | Architecture decision records |
| [ownership.md](docs/ownership.md) | Who owns what, and who to ask |

---

## Repository layout

```
modules/          nine libraries, in dependency order
  util/           atomic I/O, mmap, thread pool, CPUID, logging
  core/           object ids, errors, dtypes, interfaces  ← depends on nothing
  format/         on-disk object encode/decode
  stio/           lazy safetensors reader and byte-exact writer
  store/          block store, DAG, refs, journal, verify, merge, gc
  align/          topology parser, cost, LAP, propagation, confidence
  codec/          permute, residual encode/decode, SIMD kernels, chunking
  mount/          FUSE low-level daemon, inode table, block cache, prefetch
  net/            framing, have/want negotiation, server, client, resume
apps/sfs/         the CLI
tests/            unit, end-to-end, crash matrix, tamper, concurrency
bench/            the five graded measurements
fixtures/         checkpoint generators (Python) — output is never committed
research/         algorithm experiments that informed the ADRs
```

---

## Contributing

Read [docs/ownership.md](docs/ownership.md) first — it says who to ask about
what.

- The formats in `docs/spec/` are **frozen**. Changing one needs sign-off from
  all three teams and a same-commit update to the spec, the golden objects and
  the tests.
- `make format-check` and `make test-unit` before pushing. CI runs both.
- Commit small and often.
- If you build it, you write its three sentences of documentation the same day,
  in your own words.

## License

MIT. See [LICENSE](LICENSE).
