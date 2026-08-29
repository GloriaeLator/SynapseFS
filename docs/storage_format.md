# Storage format

The readable version. The normative definitions are
[SPEC 10](spec/10-object-model.md) (objects),
[SPEC 11](spec/11-repo-layout.md) (on-disk layout and crash safety) and
[SPEC 12](spec/12-residual-format.md) (residuals). Where this document and a
spec disagree, the spec wins and this document is a bug.

---

## 1. The idea in one paragraph

Every object — a commit, a manifest, a topology, a tensor group, a diff
artifact, a safetensors header — is stored under the BLAKE3-256 hash of its own
bytes. Objects never change. History is a Merkle DAG over them, so verifying a
lineage is walking the DAG and re-hashing, and injecting a modified block is
detectable because its hash no longer matches the name it is stored under.

That much is Git. The three things that are not Git are below.

---

## 2. The manifest describes a *file*, not a model

This is the single most important decision in the format, and it came out of a
demo that failed in front of the whole team.

We had a manifest that listed layers, each with its tensor's hash. Every tensor
round-tripped bit-identically. The **file** did not:

```
file              24,636 bytes
header             1,136 bytes   (4.6% of the file)
trailing spaces        4   <- legal padding, part of the bytes

reconstruction attempt                                      bytes  match
----------------------------------------------------------------------------
alphabetical keys, pretty JSON, no metadata, no padding    24,732  NO
original key order, pretty JSON                            24,771  NO
original key order, compact JSON, no padding               24,632  NO

every tensor is bit-identical to the target.  the FILE is not.
```

Three things that are invisible until you try:

- `safetensors` wrote the keys in an order nobody would guess
  (`1.num_batches_tracked`, `5.num_batches_tracked`, *then* the weights).
- `__metadata__` differs between producers, and is supposed to.
- Trailing padding spaces are legal and are part of the bytes. The third
  attempt above is off by exactly four of them.

**The fix.** The manifest stores:

1. `file.header_block` — the verbatim `[8-byte LE length][JSON header]` prefix,
   as its own content-addressed object;
2. `buffer` — every tensor in the file, **in buffer order**, with offset and
   length, whether or not the topology models it;
3. `file.sha256` — what reconstruction must produce.

Reconstruction stops being serialisation and becomes concatenation:

```
out = block(header_block) || concat(read_range(e.group, e.off, e.nbytes) for e in buffer)
```

Nothing to choose, so nothing to get wrong.

The obvious objection — "it's wasteful to store the header per commit" — is
answered by measurement: 4.6% on a 24 KB fixture, roughly 0.0005% on a 7B
checkpoint, a hundred unique headers is ten megabytes. And content addressing
gives the reuse for free: identical header bytes hash identically and `put`
writes nothing.

The other objection — "just reuse the base commit's header" — fails because
`__metadata__` legitimately changes between checkpoints, producers differ, and
the tensor set is not stable across a lineage.

---

## 3. Verification granularity equals read granularity

The prototype's block store verified the whole block's hash on every read. That
is correct and it makes tamper detection (10% of the grade) and mmap throughput
(8%) mutually exclusive:

```
block 32 MiB · chunk 64 KiB · 2000 random 4 KiB reads

strategy                       hashed / read   total time    read throughput
------------------------------------------------------------------------------
verify whole block (today)         33.554 MB     430.4 ms             0.6 MB/s
verify touched chunk (fix)          0.066 MB      44.5 ms           183.9 MB/s
no verification (ceiling)           0.000 MB       1.5 ms          5375.3 MB/s

whole block: caught -> corrupt
touched chunk: caught -> corrupt chunk 137
```

They only conflict because the two granularities differ. Every block therefore
carries per-chunk digests (64 KiB), and there are two read paths:

- `read_range(oid, off, len)` — verifies only the chunks it touches. The mount's
  fault path.
- `verify_block(oid)` — verifies everything. Peer ingestion, `sfs verify`, post-crash.

300× the throughput, and the chunked version gives *better* diagnostics: it
names the corrupt chunk instead of condemning the block.

Paired with the per-frame digests in §4, this gives something the problem
statement does not ask for: **tamper detection that survives the reconstruction
path**, at a cost proportional to what was actually read. A corrupted base
block, a corrupted residual and a wrong permutation all fail the same check.

---

## 4. Residuals are stored in frames

One 32 MiB layer, delta chain depth 5, serving a single 4 KiB read:

| Strategy | Bytes decompressed | Wall time | Peak RAM |
|---|---|---|---|
| Whole layer (naive) | 201.3 MB | 1347.8 ms | 134.2 MB |
| Frames | 0.8 MB | 0.8 ms | 0.4 MB |

1.3 seconds for one page fault. So the residual payload is a list of
**independently decompressible zstd frames**, each covering a contiguous range
of output units, indexed in the artifact header, each with a digest of its
reconstructed target bytes.

"But permutation destroys locality, so we can't chunk both sides at the same
offsets." Correct, and not a problem: the mapping is known exactly, and the
natural atomic unit — one output unit, a linear row or a conv filter — is
contiguous on *both* sides. A target frame covering units `[a, b)` needs base
units `p[a:b]`, which the reader computes and fetches as runs.

Recursion is per frame, so peak memory is `frame × depth` instead of
`layer × depth`. That is what keeps the daemon inside the RSS budget at 7B.

---

## 5. Framing, so that a type confusion is a hash mismatch

```
frame(kind, payload) = "synapsefs." || kind || " " || decimal(len) || 0x00 || payload
oid                  = "b3:" || hex(BLAKE3_256(frame(kind, payload)))
```

The kind is *inside* the hashed bytes. Without that, the same byte string read
as a tensor group and as a diff artifact has the same address, and a peer can
hand you a block that validates as both. With it, reading an object at the
wrong kind is a hash mismatch, not a silent cast.

---

## 6. Crash safety

```
create tmp/<random> → write → fsync file → rename → fsync parent directory
```

and then the ordering rule: **objects before refs**. Write and fsync every
object, then compare-and-swap the ref.

The consequence is the sentence to remember: *the failure mode of a crashed
commit is wasted disk, not a broken repository.* Nothing points at the orphaned
objects, `verify` does not walk them, `gc` collects them.

The parent-directory fsync is the step people skip. Without it, a power failure
can lose the rename while keeping the data.

Two operations touch more than one ref-like file — `merge` and `gc --pack` — and
those get a journal record written before the first mutation and removed after
the last. On open, a leftover record means recovery: replay if it is complete
and idempotent, otherwise **refuse and say so**. The PS explicitly allows
refusing, and refusing is better than guessing.

`tests/crash_matrix.cpp` kills at every stage of `commit`, `push` and `merge`
and asserts the repository is clean or explicitly refusing. It runs in a loop
and the iteration counter is a presentation slide.

---

## 7. What each object looks like

Field-by-field tables are in [SPEC 10](spec/10-object-model.md). The shapes:

```jsonc
// commit
{"type":"synapsefs.commit","format_version":1,
 "parents":["b3:…"],"manifest":"b3:…","topology":"b3:…",
 "timestamp":"2026-08-29T10:15:00Z","author":"…","message":"…"}

// manifest
{"type":"synapsefs.manifest","format_version":1,"hash_algo":"blake3",
 "file":{"name":"model.safetensors","header_block":"b3:…",
         "total_bytes":44236928,"sha256":"…"},
 "buffer":[{"tensor":"0.weight","off":0,"nbytes":432,"group":"g0"}],
 "groups":{"g0":{"mode":"delta","base":{"commit":"b3:…","group":"g0"},
                 "diff_block":"b3:…","chain_depth":4},
           "g4":{"mode":"full","block":"b3:…","chain_depth":0}}}

// diff artifact:  [8-byte LE len][JSON header][payload]
{"magic":"SYNDIFF","format_version":1,"group":"g0","codec":"zstd",
 "permutation":{"kind":"identity"},
 "tensors":[{"name":"0.weight","shape":[8,3,3,3],"dtype":"F16",
             "residual":"xor_after_permute",
             "frames":[{"units":[0,4],"off":0,"len":168,"digest":"7c1f…"}]}],
 "alignable":true,
 "alignment":{"method":"weight_matching_lap","cost_raw":0.0021,"cost_normalized":0.031}}
```

Notice what is *absent*, because each absence was a bug we removed:

- No `parent` alongside `parents` — they disagreed with each other.
- No `branch` on a commit — refs own that, and a commit can be on many branches.
- No `commit_hash` field — it is the storage key, and a self-referential field
  cannot be verified.
- No `base_commit` inside the diff artifact — a position in history inside a
  content-addressed object means identical diffs never deduplicate.
- No `unchanged_reuse` mode — copying the parent's entry verbatim costs zero
  new bytes *and* does not lengthen the chain for layers that never change.

---

## 8. Two bounds, not one

| Parameter | Default | Bounds |
|---|---|---|
| `max_chain_depth` | 5 | **time** — a hundred 0.1% deltas cost a hundred hops on every read |
| `snapshot_alpha` | 0.5 | **space** — one badly aligned group can produce a delta *larger* than the full block, because the XOR of unrelated fp16 is high-entropy noise |

Neither implies the other, which is why both exist. Git makes exactly this pair
of choices in its packfiles.
