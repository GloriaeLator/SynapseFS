# SynapseFS

Permutation-aware, cryptographically verifiable version control and virtual
filesystem for neural network checkpoints.

Git stores a fine-tuned checkpoint as a whole new blob. SynapseFS stores it as
a delta - after recovering the neuron correspondence between the two
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
# Build container with Docker
docker build -t synapsefs -f Containerfile . && docker run --rm synapsefs --version
# To build with podman see [build.md](docs/build.md).
```

Everything below except `sfs mount` works in a plain `docker run --rm synapsefs
...`. `mount` needs the container started with `--cap-add SYS_ADMIN --device
/dev/fuse` (e.g. `docker run --rm -it --cap-add SYS_ADMIN --device /dev/fuse
synapsefs bash`) — without it, the mount fails outright rather than
succeeding partially, and looks like a hang if a script is polling for the
mounted file to appear.

```bash
# Generate the small fixtures (checkpoints are never committed to git)
make fixtures-small

# A repository, two commits, and a delta
sfs init myrepo && cd myrepo
sfs commit ../fixtures/out/mlp_step0.safetensors -m "initial"
sfs commit ../fixtures/out/mlp_step1.safetensors -m "after fine-tune"
sfs log

# Prove it round-trips
sfs checkout HEAD -o /tmp/restored.safetensors
cmp /tmp/restored.safetensors ../fixtures/out/mlp_step1.safetensors && echo "byte-identical"

# Integrity, standalone - no checkout, no mount
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
| `sfs commit <ckpt.safetensors> -m <msg>` | Align against the parent, store the delta, write a commit. Any alignment failure falls back silently to full storage - see [architecture.md](docs/architecture.md). |
| `sfs checkout <ref> [-o/--output <path>]` | Switch branch, or materialise a checkpoint file. |
| `sfs branch [<name>] [<start-point>] [-d/--delete <name>] [-f/--force]` | Create or list. Switching is `checkout <branch>` (pre-2.23 git semantics). |
| `sfs log [<ref>] [-n/--max-count <N>]` | Walk history. |
| `sfs verify [<ref>] [--full] [--repair]` | Integrity check. **Works standalone.** Exits 4 on corruption, naming the object and, for a `--full` chunk-level failure, the chunk index too (see [threat_model.md](docs/threat_model.md)). |
| `sfs merge <branch> [--ours\|--theirs]` | Fast-forward, or three-way per tensor group. Refuses on conflict. |
| `sfs serve [-p/--port <port>]` | Start the sync listener. Defaults `-p` to `9418`, matching `RepoConfig`; see [spec/14](docs/spec/14-wire-protocol.md). |
| `sfs push <ip:port>` | Sync the local repo to a listening peer. Exits `Network`(8) on a connection/transfer failure - see [spec/15](docs/spec/15-cli-contract.md). |
| `sfs pull <ip:port>` | Sync from a listening peer. Resumable by byte offset; received objects are hash-verified against their own address before being kept, refs are not - run `sfs verify --full` afterward regardless. Exits `Network`(8) on any transfer or integrity failure. |
| `sfs mount <ref> <mountpoint>` | Read-only FUSE mount. Nothing is written to disk. |
| `sfs unmount <mountpoint>` | Unmount. |
| `sfs gc [--prune] [--pack] [-n/--dry-run]` | Reclaim unreachable objects. `--pack` is not implemented yet (returns `NotImplemented`). |

Full flag reference and exit codes: **[docs/spec/15-cli-contract.md](docs/spec/15-cli-contract.md)**.

### Networking

`sfs serve -p <port>` starts a single-connection-at-a-time TCP listener
(defaults to `9418` if `-p` is omitted); the peer connects with `sfs push
<ip:port>` / `sfs pull <ip:port>` - a bare `ip:port` string, not a URL, with
no authentication or encryption (a deliberate scope decision). This is a
custom plaintext, newline-delimited protocol that decides *what* to
transfer by comparing file **size** (not a content hash) - that part is
unchanged. What it does with the bytes once received differs by kind:
a received loose object is re-hashed against its own claimed BLAKE3 address
before being kept, and the transfer is rejected if it doesn't match; refs,
`HEAD`, and the journal are still accepted on raw size/content equality
alone with no hash check at all. Treat a pulled repository as unverified
overall until `sfs verify --full` passes - object payloads are checked in
transit, but the ancestor-invariant/chain-depth walk `sfs verify` performs
is not. A failed `push`/`pull` exits `ExitCode::Network` (8) instead of
looking identical to success. See [docs/spec/14-wire-protocol.md](docs/spec/14-wire-protocol.md)
and [docs/threat_model.md](docs/threat_model.md).

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
   content-addressable store  ─── BLAKE3 ───►  Merkle DAG (commit -> manifest -> blocks)
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
what was actually read - 183.9 MB/s instead of 0.6 MB/s.
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
| [benchmarks.md](docs/benchmarks.md) | What the bench binaries measure, the hardware, the commands |
| [build.md](docs/build.md) · [testing.md](docs/testing.md) | Building and testing |
| [spec/](docs/spec/) | Format specifications, as implemented |
| [adr/](docs/adr/) | Architecture decision records |

---

## Repository layout

```
modules/          
  util/           atomic I/O, mmap, thread pool, CPUID, logging
  core/           object ids, errors, dtypes, interfaces  ← depends on nothing
  format/         on-disk object encode/decode
  stio/           lazy safetensors reader and byte-exact writer
  store/          block store, DAG, refs, journal, verify, merge, gc
  align/          topology parser, cost, LAP, propagation, confidence
  codec/          permute, residual encode/decode, SIMD kernels, chunking
  mount/          FUSE low-level daemon, inode table, block cache, prefetch
  net/            push/pull/serve - a plaintext, size-diff TCP sync
                  protocol
apps/sfs/         the CLI
tests/            2 integration tests (byte-identity) + per-module unit
                  tests
bench/            residual_codec, verify_time, mmap_throughput, align_time, lap_bench, sparse_bench
fixtures/         checkpoint generators (written in Python)
```

---

## License

MIT. See [LICENSE](LICENSE).
