# SPEC 10 - Object model

This document defines the five object kinds SynapseFS stores, how they are
framed and addressed, and the invariants a repository must satisfy. `docs/spec/11-repo-layout.md` says where the
objects live on disk and `docs/spec/12-residual-format.md` defines the binary
payload of a diff artifact.

---

## 1. Addressing

### 1.1 Object identifier

Every object is addressed by the BLAKE3-256 digest of its **framed bytes**
(§1.3). The identifier is written in text as:

```
b3:<64 lowercase hex characters>
```

The `b3:` prefix is part of the string wherever an identifier appears in JSON,
in refs files, on the wire and in CLI output. In binary contexts (the packfile
index, the wire protocol) the raw 32 bytes are used and the prefix is implied.

Rationale for BLAKE3 over SHA-256 is in `docs/adr/0002-blake3-over-sha256.md`,
with the measurement. Changing the hash function invalidates every stored
object, so the algorithm identifier is deliberately *not* negotiable per
repository: a repository written by one build must be readable by another.

### 1.2 Abbreviation

Implementations MAY display abbreviated identifiers (first 12 hex characters
after the prefix) in human-facing output. They MUST NOT accept an abbreviated
identifier anywhere a stored object references another object.

### 1.3 Block framing

Before hashing, every payload is framed:

```
frame(kind, payload) = "synapsefs." || kind || " " || decimal(len(payload)) || 0x00 || payload
oid(kind, payload)   = "b3:" || hex(BLAKE3_256(frame(kind, payload)))
```

`kind` MUST be one of:

| kind       | payload                                                     |
|------------|-------------------------------------------------------------|
| `raw`      | uncompressed tensor-group bytes exactly as they appear in the reconstructed file |
| `diff`     | a diff artifact (SPEC 12)                                    |
| `header`   | the verbatim `[8-byte little-endian length][JSON header]` prefix of a `.safetensors` file |
| `manifest` | UTF-8 canonical JSON (§4)                                    |
| `commit`   | UTF-8 canonical JSON (§3)                                    |
| `topology` | UTF-8 canonical JSON (SPEC 13)                               |
| `tree`     | UTF-8 canonical JSON (§6a) - format version 2 only           |

The kind is inside the hashed bytes. This is not decoration: without it, the
same byte string read as a tensor group and as a diff artifact would have the
same address, and a peer could hand us a block that validates as both. With it,
a type confusion is a hash mismatch.

**Implementations MUST verify the kind at read time**, not merely record it.
Reading an object of the wrong kind is `ErrKind::ObjectKindMismatch`, not a
silent cast.

### 1.4 Canonical JSON

JSON-valued objects (`commit`, `manifest`, `topology`, `tree`) MUST be serialised
canonically, because the serialisation *is* the address:

1. UTF-8, no BOM.
2. Object keys sorted by Unicode code point, ascending.
3. No insignificant whitespace: no spaces after `:` or `,`, no indentation, no
   trailing newline.
4. Integers serialised without exponent, sign or leading zeros. Numbers that
   are not exactly representable as an IEEE-754 double MUST NOT appear; sizes
   and offsets are within 2^53 for every fixture size in scope.
5. Floating-point values (`cost_raw`, `cost_normalized`) are serialised with
   `%.17g` and are **advisory only** - see §4.4.
6. No duplicate keys.

A conforming reader MUST reject an object whose re-serialisation does not
reproduce its own address. That check is what makes `verify` meaningful for
JSON objects, and it is cheap.

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
   file.header_block │ groups[].block / .diff_block
        ┌───────────┴────────────┐
        ▼                        ▼
   ┌────────┐              ┌──────────┐   base.commit ──► (an ancestor commit)
   │ header │              │   diff   │
   └────────┘              └──────────┘
                                 │ resolves against
                                 ▼
                            ┌────────┐
                            │  raw   │
                            └────────┘
```

Everything reachable from a ref is live. Everything else is garbage
(SPEC 11 §6).

---

## 3. Commit

```jsonc
{
  "type": "synapsefs.commit",
  "format_version": 1,
  "parents": ["b3:c4de41a1…"],     // [] root · [x] normal · [x, y] merge
  "manifest": "b3:9a8b7c6d…",
  "topology": "b3:1f2e3d4c…",
  "timestamp": "2026-08-29T10:15:00Z",
  "author": "rohan <rohan@example.org>",
  "message": "fine-tune epoch 3"
}
```

| Field            | Type      | Req | Notes |
|------------------|-----------|-----|-------|
| `type`           | string    | ✔   | Exactly `"synapsefs.commit"`. |
| `format_version` | integer   | ✔   | `1`. A reader MUST refuse a version it does not implement. |
| `parents`        | [oid]     | ✔   | Ordered. Empty for a root commit. Length 2 for a merge; `parents[0]` is the branch that was checked out. Length > 2 MUST be rejected. |
| `manifest`       | oid       | ✔   | Kind `manifest`. |
| `topology`       | oid       | ✔   | Kind `topology`. Present so that a **pulled** repository can decode itself without the producer's config file. |
| `timestamp`      | string    | ✔   | RFC 3339, UTC, second precision, always `Z`. |
| `author`         | string    | ✔   | Free text. Not authenticated - see `docs/threat_model.md`. |
| `message`        | string    | ✔   | Free text, may be empty. |

**Deliberately absent.** `parent` (singular) - it disagreed with `parents` in
the prototype. `branch` - refs own branch membership, and a commit can be on
many branches. `commit_hash` - it is the storage key, and a self-referential
field cannot be verified.

---

## 4. Manifest

The manifest is the object that makes byte-exactness achievable. It describes
**a file**, not a model.

```jsonc
{
  "type": "synapsefs.manifest",
  "format_version": 1,
  "hash_algo": "blake3",
  "file": {
    "name": "model.safetensors",
    "header_block": "b3:1a2b3c…",
    "total_bytes": 44236928,
    "sha256": "3394556288…"
  },
  "buffer": [
    {"tensor": "1.num_batches_tracked", "off": 0,  "nbytes": 8,   "group": "s1"},
    {"tensor": "0.weight",              "off": 16, "nbytes": 432, "group": "g0"}
  ],
  "groups": {
    "g0": {"mode": "delta", "base": {"commit": "b3:c4de…", "group": "g0"},
           "diff_block": "b3:aa11…", "chain_depth": 4},
    "g4": {"mode": "full",  "block": "b3:cc33…", "chain_depth": 0}
  }
}
```

### 4.1 `file`

| Field          | Type    | Req | Notes |
|----------------|---------|-----|-------|
| `name`         | string  | ✔   | The name the file takes at checkout and in the mount. No path separators. |
| `header_block` | oid     | ✔   | Kind `header`. The **verbatim** `[8-byte LE length][JSON header]` prefix, including any trailing padding spaces the producer wrote. |
| `total_bytes`  | integer | ✔   | Size of the reconstructed file. MUST equal `len(header) + Σ buffer[i].nbytes`. |
| `sha256`       | string  | ✔   | SHA-256 of the whole reconstructed file, 64 lowercase hex. This is what reconstruction must produce, and the one place SHA-256 appears - it is a *witness for humans and for the PS*, not an address. |

Storing the header verbatim rather than regenerating it is the single most
important decision in this spec. `safetensors` writes keys in an order that is
an artifact of the writer, `__metadata__` differs between producers, and legal
trailing padding is part of the file. Regeneration produced a file that was
four bytes wrong with every tensor bit-identical. Reconstruction is
concatenation, so there is nothing to choose and nothing to get wrong.

The header deduplicates for free: identical header bytes hash identically and
`put` writes nothing. Cost is 4.6% on a 24 KB fixture and ~0.0005% on a 7B
checkpoint.

### 4.2 `buffer`

An ordered array covering the file's data section **exactly**, with no gaps and
no overlaps, in **buffer order** - which is not key order and not topology
order.

| Field    | Type    | Req | Notes |
|----------|---------|-----|-------|
| `tensor` | string  | ✔   | Name as it appears in the safetensors header. |
| `off`    | integer | ✔   | Offset from the start of the data section (i.e. from the end of the header block). |
| `nbytes` | integer | ✔   | Length in bytes. |
| `group`  | string  | ✔   | Permutation-group id from the topology, or a singleton group. |

Invariants a reader MUST check:

- `buffer[0].off == 0`
- `buffer[i+1].off == buffer[i].off + buffer[i].nbytes`
- `Σ nbytes + len(header_block) == file.total_bytes`

Every tensor in the file appears here, including ones the topology does not
model (`num_batches_tracked` and friends). An unmodelled tensor gets its own
singleton group with an identity permutation and round-trips unchanged. A
tensor that is in the file and not in the buffer is data loss that no test of
"does the model still load" would catch.

Reconstruction is then:

```
out = block(file.header_block)
   || concat( read_range(entry.group, entry.off_within_group, entry.nbytes)
              for entry in buffer )
```

### 4.3 `groups`

A map from group id to how that group's bytes are stored.

| Field         | Type    | Req | Applies to | Notes |
|---------------|---------|-----|------------|-------|
| `mode`        | string  | ✔   | both       | `"full"` or `"delta"`. |
| `block`       | oid     | ✔   | `full`     | Kind `raw`. The group's bytes, possibly compressed per SPEC 11 §3. |
| `base`        | object  | ✔   | `delta`    | `{"commit": oid, "group": string}` - where to resolve the base from. |
| `diff_block`  | oid     | ✔   | `delta`    | Kind `diff`. |
| `chain_depth` | integer | ✔   | both       | `0` for `full`; `base.chain_depth + 1` for `delta`. Stored, not computed, so the policy check is O(1). |

**Invariants.**

- `chain_depth` MUST be ≤ `max_chain_depth` (default 5, SPEC 11 §5).
- **Ancestor invariant:** for every `delta` group, `base.commit` MUST be an
  ancestor of the commit that contains this manifest. `verify` checks this.
  Without it, `push` can transfer a manifest whose base is unreachable on the
  receiving side, and a repository that passes a per-object hash check is still
  unreadable.
- A group unchanged since the parent commit is represented by **copying the
  parent's entry verbatim**, including its `base` and `chain_depth`. This costs
  zero new bytes and, importantly, does not lengthen the chain for layers that
  never change. There is no `unchanged_reuse` mode; it was a third code path
  for something the two existing modes already express.

### 4.4 What is not in the manifest

No top-level `base_commit`: each group owns its own base, which is what lets a
commit mix full and delta groups against different ancestors. No
`block_hash` field overloaded across modes: `block` and `diff_block` are
distinct field names with distinct kinds, so a reader cannot be tricked into
resolving one as the other.

---

## 5. Diff artifact

Framing is `[8-byte LE header length][JSON header][binary payload]`; the JSON
header is described here and the payload in SPEC 12.

```jsonc
{
  "magic": "SYNDIFF",
  "format_version": 1,
  "group": "g0",
  "codec": "zstd",
  "permutation": {"kind": "explicit", "n": 8, "dtype": "u16", "off": 312, "len": 16},
  "tensors": [{
    "name": "0.weight", "shape": [8, 3, 3, 3], "dtype": "F16",
    "residual": "xor_after_permute",
    "frames": [
      {"units": [0, 4], "off": 0,   "len": 168, "digest": "7c1f…"},
      {"units": [4, 8], "off": 168, "len": 144, "digest": "b904…"}
    ]
  }],
  "alignable": true,
  "alignment": {"method": "weight_matching_lap", "cost_raw": 0.0021, "cost_normalized": 0.031}
}
```

| Field            | Req | Notes |
|------------------|-----|-------|
| `magic`          | ✔   | `"SYNDIFF"`. Mismatch is a hard error, not a fallback. |
| `format_version` | ✔   | `1`. |
| `group`          | ✔   | The permutation group this artifact reconstructs. |
| `codec`          | ✔   | `"zstd"` or `"none"`. Per-frame, uniform within an artifact. |
| `permutation`    | ✔   | `{"kind": "identity"}` - zero payload bytes, the common case for a fine-tune - or `{"kind": "explicit", n, dtype, off, len}` locating the permutation array in the payload. |
| `tensors`        | ✔   | One entry per tensor in the group, with its frame index. See SPEC 12. |
| `alignable`      | ✔   | `false` means the aligner found no meaningful correspondence; the containing manifest entry MUST then be `mode: "full"`. |
| `alignment`      | ✔   | Advisory provenance: method, raw and normalised cost. Reconstruction MUST NOT depend on these values. |

**Deliberately absent.** `base_commit` and `base_layer_source`. Putting a
position in history inside a content-addressed object means two identical
diffs never deduplicate. The base lives in the manifest entry that points at
the artifact, which is where history belongs.

The permutation lives in the binary payload, not in JSON. A JSON integer array
for an 8192-unit layer is ~40 KB of parsing on every hop of a chain, and the
identity case - which is most fine-tunes - costs zero bytes.

---

## 6. Topology

Defined in `docs/spec/13-topology-config.md`. Summary of the contract with this
document: it maps each tensor axis to a permutation group id and a blocking
factor, and it is stored as an object so a pulled repository is
self-describing.

---

## 6a. Tree - sharded checkpoints (format version 2)

A format-version-1 commit names exactly one file, so `commit.manifest`
addresses a `manifest` directly and there is no tree object. A sharded
checkpoint (`model-00001-of-00003.safetensors` alongside an `index.json`) is a
*set* of files and needs one more level of indirection.

A `tree` is UTF-8 canonical JSON:

```json
{"entries":[{"manifest":"b3:a1…","name":"model-00001-of-00003.safetensors"}],"format_version":2}
```

- `format_version` MUST be `2`. A version-1 reader that resolves
  `commit.manifest` and finds a `tree` MUST fail with
  `ErrKind::UnsupportedFormatVersion`, not attempt to interpret it.
- `entries` MUST be non-empty and sorted by `name`, strictly ascending. The
  serialisation is the address, so an unsorted tree is a second address for the
  same content and MUST be rejected rather than silently reordered.
- `name` is a plain file name as it appears at checkout and in the mount: no
  `/` or `\`, no NUL or other control bytes, not `.` and not `..`. A tree that
  could name a path outside the checkout directory is a malformed object.
- `manifest` addresses an object of kind `manifest`. Kind is verified at read
  time (§1.3); a tree entry pointing at a non-manifest is
  `ErrKind::ObjectKindMismatch`.

Everything below a manifest - header, buffer layout, groups, diffs, raw blocks
- is unchanged by sharding. A tree adds a level to the graph and changes
nothing else about it.

Implemented in `modules/format/{include/synapsefs/format/tree.hpp,src/tree.cpp}`.
**Not yet wired to `Commit`**: see §7.

---

## 7. Versioning and compatibility

- Every JSON object carries `format_version`. A reader MUST refuse an unknown
  major version with `ErrKind::UnsupportedFormatVersion` rather than
  best-effort parsing.
- Unknown *fields* MUST be rejected, not ignored. Canonical JSON means an
  ignored field still changes the address, so silently tolerating one produces
  two objects that disagree about their own identity.
- Format version 2 (`tree`) is **specified and implemented as an object, but no
  commit in this build points at one**. `Commit.manifest` always addresses a
  `manifest`. Writing trees requires a `Commit` that can address either, plus
  the checkout, mount and verify paths to walk the extra level; until that
  lands, a repository written by this build is version 1 throughout.
- Until 1 October 2026 the format version is not stable and no backward
  compatibility is offered. This is a course project; say so rather than
  implying a compatibility promise we will not keep.

---

## 8. Test hooks

The following are asserted by `tests/golden/` and by CI, and are the concrete
form of everything above:

| Assertion | Test |
|---|---|
| Canonical JSON round-trips to the same oid | `modules/format/tests/test_canonical_bytes.cpp` |
| Framing rejects kind confusion | `modules/store/tests/test_block_store.cpp` |
| Buffer layout covers the file with no gaps | `modules/format/tests/test_st_roundtrip.cpp` |
| `sha256(reconstruct(A, diff(A,B))) == sha256(B)` | `tests/byte_identity.cpp` |
| Ancestor invariant | `modules/store/tests/test_dag_walk.cpp` |
| Tree sorting, name rules and round-trip | `modules/format/tests/test_tree.cpp` |
| Golden objects still parse | `tests/golden/*.json` + `validate.py` in CI |
