# Architecture

How SynapseFS is put together: the nine modules, what depends on what, the
object graph, and the one primitive that three of the four graded modules are
built on.

The normative details live in `docs/spec/`. This document is the map.

---

## 1. Module graph

Arrows point from a module to what it depends on. No cycles; the build enforces
the order.

```
                         apps/sfs  (CLI)
                              │
      ┌────────┬──────────┬───┴────┬──────────┬──────────┐
      ▼        ▼          ▼        ▼          ▼          ▼
   mount     net       store     align      codec      stio
      │        │          │        │          │          │
      └────────┴────┬─────┴────────┴────┬─────┴──────────┘
                    ▼                   ▼
                 format               core
                    │                   │
                    └─────────┬─────────┘
                              ▼
                             util
```

| Module | Owns | Depends on |
|---|---|---|
| `util` | atomic write, mmap, file handles, CPUID, bit tricks, logging | — |
| `core` | `Oid`, `Error`/`ErrKind`, `DType`, `Tensor` view, topology types, repo config, endian helpers, **all cross-module interfaces** | util |
| `format` | encode/decode of every on-disk object: safetensors header, commit, manifest, tree, diff header | core |
| `stio` | lazy safetensors reading, byte-exact writing, axis-0 unit reads | format |
| `store` | block store (loose + packfiles), commit/manifest stores, refs, journal, DAG walk, verify, merge, gc, lockfile | format, codec |
| `align` | topology parser, alignment graph, cost, LAP, matcher, propagation, norm folding, confidence, out-of-core plan | stio, core |
| `codec` | permutation application, diff encode/decode, **`reconstruct` / `read_range`**, residual kernels, chunking, compression, snapshot policy | format, util |
| `mount` | FUSE low-level daemon, inode/interval table, frame cache, prefetch, stats | store, codec |
| `net` | wire framing, have/want negotiation, server, client, session, resume | store |

Two rules that keep this from rotting:

- **`core` depends on nothing but `util`, and defines every interface.** That is
  what lets a module be compiled and tested against stubs before its
  collaborators exist. On Day 1, it is the difference between eight people
  working and three people waiting.
- **`codec` does not know about repositories** and `store` does not know about
  permutations. The seam between them is `IBlockStore`.

---

## 2. The object graph

```
                 refs/heads/main
                        │
                        ▼
                    ┌────────┐   parents[]   ┌────────┐
                    │ commit │ ────────────► │ commit │ ──► …
                    └───┬────┘               └────────┘
              manifest  │  topology
                 ┌──────┴──────┐
                 ▼             ▼
           ┌──────────┐   ┌──────────┐
           │ manifest │   │ topology │
           └────┬─────┘   └──────────┘
                │
      header_block │ groups[g].block  │ groups[g].diff_block
        ┌──────────┼──────────────────┴─────┐
        ▼          ▼                        ▼
   ┌────────┐  ┌────────┐              ┌──────────┐
   │ header │  │  raw   │◄─────────────│   diff   │
   └────────┘  └────────┘  resolves    └──────────┘
                            against          │ base named by the
                                             │ MANIFEST entry, not
                                             │ by the artifact
```

Every node is content-addressed by BLAKE3 over its *framed* bytes, with the
object kind inside the hash (SPEC 10 §1.3). That last detail is what stops a
peer handing us a block that validates as both a tensor and a diff artifact.

The base of a delta is named by the manifest entry, never by the diff artifact.
Putting a position in history inside a content-addressed object would mean two
identical diffs never deduplicate.

---

## 3. The one primitive

```cpp
// modules/codec/include/synapsefs/codec/reconstruct.hpp
std::expected<std::size_t, Error>
read_range(const ReadCtx& ctx, std::string_view group,
           std::uint64_t offset, std::span<std::byte> out);
```

Everything that produces checkpoint bytes goes through it:

| Caller | Shape |
|---|---|
| `sfs checkout` | loop over the manifest's buffer entries, write to a file descriptor |
| mount, `read()` | interval lookup → one or more `read_range` → `fuse_reply_buf` |
| mount, `mmap` fault | same path, via the kernel's read of the backing pages |
| `sfs verify` | `read_range` over every reachable object |
| `sfs push` | reads whole blocks, so uses the store directly, not this |

`checkout` is about thirty lines. It is **not** a second reconstructor, and a
design where it is one has already failed the PS's consistency requirement — it
just does not know it yet.

Internally `read_range` recurses down the delta chain **per frame**, not per
layer (SPEC 12 §6). Peak memory is `frame_bytes × chain_depth`, and the digest
on each frame is checked at every hop, so corruption is caught and named at the
level it was introduced.

---

## 4. Data flow: `sfs commit`

```
 checkpoint.safetensors
        │
        ├─► stio::StSource ──────► header bytes (verbatim)  ──► store as kind=header
        │                     └──► buffer layout: every tensor, in buffer order
        │
        ├─► align::TopologyParser (checkpoint + config.json)
        │        └──► union-find over tensor axes ──► perm_groups + per-axis {group, block}
        │                                             └──► store as kind=topology
        │
        └─► for each permutation group g:
                parent manifest has g?
                  no  ──────────────────────────────────► mode=full
                  yes ─► align::Matcher (streamed, tiled)
                            │ alignable=false ──────────► mode=full
                            └─► permutation p
                                  └─► codec::DiffEncoder(base, target, p)
                                        │ len(delta) > α·len(full) ────► mode=full
                                        │ chain_depth+1 > max_depth ──► mode=full
                                        └─────────────────────────────► mode=delta
        │
        ├─► write every object atomically, fsync
        ├─► write manifest, write commit
        └─► compare-and-swap the branch ref          ← objects before refs, always
```

The four "fall back to full" conditions are all cheap to evaluate at the point
they are needed, and the size test in particular is free because the writer is
already holding both byte strings.

## 5. Data flow: a page fault on the mount

```
  torch → mmap → page fault → kernel → FUSE → daemon
                                                │
                          interval table (built once at open, binary search)
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                         header block range            (group, offset, len)
                                  │                           │
                                  └────────► codec::read_range ◄──────┐
                                                    │                 │
                                     frame cache hit? ──yes──► copy   │
                                                    │                 │
                                                   no                 │
                                                    ▼                 │
                                    single-flight fill:               │
                                      resolve base units p[a:b] ──────┘  (recursion,
                                      decompress frame                    per frame)
                                      apply residual
                                      check frame digest
                                      publish to LRU
                                                    │
                                                    ▼
                                            fuse_reply_buf
```

Nothing is materialised. The daemon holds a bounded LRU of decompressed
**frames** — small things — so peak RSS is
`cache_bytes + frame_bytes × depth × readers`, not a function of checkpoint
size.

---

## 6. Threading

| Component | Model |
|---|---|
| CLI commands | single-threaded, including the aligner |
| Aligner | single-threaded: tile accumulation, group matching, and the LAP solve all run serially. A `util::ThreadPool` was planned for this (one tile per task, groups run in parallel) but was never implemented — declared, uncalled, and removed as dead code rather than left as a misleading claim |
| LAP solver | single-threaded per group; groups are matched one at a time, not in parallel |
| Residual encode | single-threaded: one frame at a time |
| Mount daemon | libfuse's own thread pool; our state is the frame cache + interval table |
| Frame cache | sharded mutex + per-frame single-flight; entries immutable once published — genuinely real, since the mount daemon is the one place actual concurrency exists |
| `serve` | single-threaded today, not one-thread-per-connection as previously stated here |

The only genuinely subtle concurrency in the project is the frame cache, which
is why it has a dedicated TSan test (`test_blockcache_race.cpp`) and a
dedicated section in SPEC 16 §5.

---

## 7. Error handling

No exceptions across module boundaries. Every fallible API returns
`std::expected<T, sfs::core::Error>`; see `docs/interfaces/errors.md` for the
`ErrKind` taxonomy and how it maps to CLI exit codes and wire error codes.

The reason is not taste. An exception escaping into a libfuse callback is a
kernel-visible hang, not a stack trace, and the mount is 25% of the grade.

---

## 8. What lives outside C++

- `fixtures/` — checkpoint generation. Needs `torch`/`safetensors`, and the
  point of a fixture is that it was produced by *someone else's* writer.
- `tests/e2e.py` — the PS requires `safetensors.torch.load_file()` to work
  unmodified against our mount. Only Python can assert that.
- `research/` — the LAP benchmark and cost ablation that produced the numbers
  in the ADRs.
- `bench/scripts/` — cold-cache orchestration (`drop_caches`), `VmHWM`
  sampling, ratio reporting.

None of these are in the build's critical path, and none of them are needed to
build or run `sfs`.
