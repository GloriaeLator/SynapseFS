# Storage format

This details how data is structured and persisted in storage.

## Object identity

Every object is addressed by a **framed BLAKE3-256** digest
(`core::compute_oid`, `modules/core/src/oid.cpp`):

```
frame = "synapsefs.<kind>" " " <decimal payload length> 0x00 <payload>
oid   = "b3:" + hex(BLAKE3_256(frame))
```

`<kind>` is one of `raw`, `diff`, `header`, `manifest`, `commit`, `topology`
(`ObjectKind::{Raw,Diff,Header,Manifest,Commit,Topology}`). Hashing the kind
and length into the frame means the same bytes stored as two different kinds
never collide on an address. A **separate, unframed** BLAKE3 digest (`core::digest`) is used
for chunk digests and journal-record digests; those digests "address
nothing" and are never confusable with an object oid.

`Oid` is 32 raw bytes, printed as `b3:` + 64 lowercase hex characters.
`Oid::parse` requires the full `b3:` prefix and all 64 hex characters -
abbreviations are a display/lookup convenience (`abbrev()`, first 12 hex
chars) resolved by `RefStore::rev_parse` scanning the object fan-out
directories for a unique prefix, never accepted by `Oid::parse` itself.

## Loose object container

Every object - commit, manifest, topology, raw tensor bytes, diff artifact,
verbatim safetensors header - is stored the same way, one file per object,
at `.synapsefs/objects/<first-2-hex>/<remaining-62-hex>`:

```
offset  size   field
0       8      magic = {'S','F','S','O','B','J', 0x00, 0x01}
8       1      kind          (ObjectKind)
9       1      compression   (0 = None, 1 = Zstd)
10      4      chunk_log2    (LE u32; default 16 -> 64 KiB chunks)
14      8      payload_len   (LE u64, uncompressed)
22      8      stored_len    (LE u64, on-disk size)
30      4      chunk_count   (LE u32)
34      6      reserved, zero on encode
──── header = 40 bytes ────
40      32×N   chunk digests - one unframed BLAKE3 digest per
               chunk_bytes-sized slice of the payload (last chunk short)
40+32N  …      payload bytes
```

The entire object is not compressed directly rather we a different approach is used:

- **`Raw`-kind objects** (whole tensors, stored via `blocks.put(ObjectKind::Raw, ...)`
  in `commit_planner.cpp`) really are byte-for-byte uncompressed -
  `RepoConfig::compress_raw` defaults to `false`, and the rationale is that fp16 weight bytes don't compress usefully anyway.
- **`Diff`-kind objects** - the actual output of aligning and diffing two
  checkpoints, and the common case for anything after the first commit -
  are genuinely compressed on disk, just not through this header field.
  `codec::encode_group` (`modules/codec/src/diff_encoder.cpp`) builds the
  diff artifact by running each residual frame through
  `codec::compress_frame` (real `ZSTD_compress2`, `EncodeOptions::codec`
  defaulting to `format::Codec::Zstd` - see
  [`spec/12-residual-format.md`](spec/12-residual-format.md) §6) *before*
  that artifact ever reaches `format::encode_object`. `encode_object` then
  wraps those already-compressed bytes as its `payload`, this means the container header is not where to look for whether a `Diff` object's contents are compressed - they are, one layer up, frame by frame, inside the payload itself.

`chunk_bytes` must be a power of two and defaults to
`RepoConfig::chunk_bytes` = 64 KiB, baked into the object at write time -
changing the config afterward does not touch objects already on disk.

Two independent verification paths read this container
(`store::LooseStore`, implementing `core::IBlockStore`):

- **`get(oid)`** - reads the whole object, decodes the header, re-hashes the
  entire payload against the framed oid. Full verification, one hash.
- **`read_range(oid, offset, out)`** - reads only the header and the chunk
  digest table, then `pread`s and re-verifies (`format::verify_chunk`) only
  the chunks that intersect `[offset, offset+len)`. This is the path the
  FUSE mount and `checkout` actually take on every read.

`verify_block()` (used by `sfs verify --full`) is implemented as "call
`get()` and discard the payload" - a full re-hash, not a chunk-by-chunk
walk. (see [`threat_model.md`](threat_model.md)).

Objects are content-addressed and therefore write-once: both
`LooseStore::put` and the streaming `BlockStore::begin_put`/`commit` no-op
(discard the write, return the existing oid) if the destination path already
exists.

## JSON objects: Commit, Manifest, DiffHeader

Commit and Manifest use **canonical JSON**: `nlohmann::json`'s default
map-backed type serializes object keys in sorted order, which combined with
no whitespace and no trailing newline gives a deterministic byte
representation. `Commit::parse`/`Manifest::parse` re-serialize whatever they
just parsed and byte-compare it against the input; any difference -
including a key that doesn't round-trip, like an unrecognized extra field -
is rejected with `ErrKind::CanonicalizationMismatch`. **`DiffHeader::parse`
does not do this round-trip check**; a diff artifact's JSON header is parsed
field-by-field with no canonicalization requirement.

`Commit` fields: `format_version, parents[] (≤2), manifest, topology,
timestamp (RFC3339 UTC, second precision), author, message`. Deliberately
absent: a singular `parent` (disagrees with `parents`), a `branch` field (a
commit can be on many branches), and a self-referential `commit_hash` (the
storage address already is one).

`Manifest` fields: `format_version, hash_algo ("blake3"), file{name,
header_block, total_bytes, sha256}, buffer[{tensor, off, nbytes, group}],
groups{name -> {mode: "full"|"delta", block?, base?{commit,group},
diff_block?, chain_depth}}`. `file.sha256` is explicitly a *witness*, not an
address - the one place SHA-256 appears in the object model; every address
in the system is BLAKE3. `groups` is a `std::map` specifically so its
iteration order is already sorted for canonicalization.

The golden fixtures are under `tests/golden/` (`commit.json`, `manifest.json`) - see [`testing.md`](testing.md)

## Diff artifacts (residuals)

Layout: `[8-byte LE header_len][JSON header][payload]`, magic string
`"SYNDIFF"` inside the JSON - mirroring the safetensors `[len][json][data]`
shape on purpose, "so the same reader primitive serves both" (code comment).
All offsets inside the header are relative to byte 0 of the payload section.
Full field-level layout is in
[`spec/12-residual-format.md`](spec/12-residual-format.md).

The residual itself is never a floating-point subtraction. Three kinds
(`ResidualKind`), chosen per tensor by the encoder:

- **`Raw`** - the frame bytes *are* the target bytes; no base is read.
- **`XorAfterPermute`** - `target[i] = base[perm[i]] ^ residual[i]`, a plain
  bytewise XOR over the dtype's raw bit pattern.
- **`ZigzagAfterPermute`** (the encoder's default) - `target = base +
  zigzag_decode(residual)`, i.e. an unsigned-wraparound integer subtraction
  of the raw bit pattern, zigzag-packed. Code comments describe this as the
  winner of a six-candidate ratio/throughput experiment over XOR and two
  byte-level transforms (`BytePlane`, `Bitshuffle`) that can additionally be
  applied before zstd compression.

Both `XorAfterPermute` and `ZigzagAfterPermute` are bijective by
construction, so reconstruction is always exact - there is no lossy
approximation anywhere in this path.

A permutation reference (`PermutationRef`) is either `Identity` (zero
payload bytes - "the common case for a fine-tune") or `Explicit` (an array
of u16 or u32 indices, width chosen by whether the group has more than
65536 units). Every explicit permutation read from a diff artifact is
validated as a true bijection (`core::is_valid_permutation`) *before* it is
used to index anything.

Each residual frame carries a BLAKE3 digest of the **reconstructed target
bytes** for that frame (post-decompress, post-residual-apply), so tamper
detection covers the full reconstruction path, not just what was stored at
rest.

## Crash safety

Two distinct mechanisms are actually used, not one:

**Atomic rename** (`util::atomic_write`, `modules/util/src/atomic_io.cpp`)
covers almost everything: loose objects, ref files, HEAD, and journal
records themselves. Write to a randomly-named temp file in the target
directory, optionally `fsync` its contents, `rename()` over the destination
(atomic on the same filesystem), optionally `fsync` the parent directory.
Object writes use `overwrite=false` (a no-op if the destination already
exists - this *is* the content-addressed idempotence); ref writes use
`overwrite=true` because compare-and-swap is enforced separately by reading
the current value first.

`CommitStore::commit_and_advance()`'s invariant: write and fsync the commit
object first, *then* CAS the branch ref. A crash between those two steps
leaves an unreferenced object. `verify` never walks unreferenced objects.

**A write-ahead journal** (`modules/store/src/journal.cpp`) is used only for
the one operation that touch more than one file atomically: `merge`
(commit object + branch ref together). Journal record format:

```
[4-byte LE payload_len][JSON payload][32-byte BLAKE3 digest of the JSON]
```

one file per pending record at `.synapsefs/journal/j.<seq>`, written with
both `fsync_contents` and `fsync_parent` set, deleted (with a directory
fsync) once the guarded mutation completes. A length or digest mismatch on
decode is reported as `ErrKind::JournalTorn` rather than partially
interpreted - "detect, don't guess." `Journal::recover()` (invoked by `sfs
verify --repair`, and only there) re-checks whether a `Merge` record's
mutation already landed (ref already equals `ref_new` -> just delete the
record) and otherwise re-attempts the same compare-and-swap.

A **fast-forward** merge takes no journal record at all: HEAD is already
symbolic, so only the branch ref file moves, and a single-file rename is
already atomic.

## Locking

`RepoLock` (`modules/store/src/lockfile.cpp`) is an advisory `flock()` on
`.synapsefs/index.lock`, non-blocking, exclusive or shared. `commit` and
`merge` take the exclusive lock; `checkout` and `verify` take no
lock at all, since objects are immutable and content-addressed, so
concurrent reads are safe by construction. Failing to acquire the lock
surfaces as `ErrKind::RepositoryLocked` -> CLI exit code 6.

## Merge

`store::plan_merge` (`modules/store/src/merge_logic.cpp`) is a real,
tested, per-tensor-group three-way merge:

1. `ours == theirs` and ancestor checks first: a true fast-forward either
   way is detected via a real DAG walk (`is_ancestor`), not a heuristic.
2. The merge base is the first commit reached by breadth-first search from
   `theirs` that is also an ancestor of `ours`. No common ancestor ->
   refuses outright with `MergeConflict` (no synthetic/octopus base).
3. The two checkpoints' safetensors headers (`file.header_block`) must be
   identical, or the merge refuses entirely - this merge operates on one
   checkpoint file shape, not arbitrary files.
4. Each tensor group is resolved independently
   (`store::resolve_group(base, ours, theirs)`): unchanged on both sides ->
   keep; changed on exactly one side relative to base -> take that side;
   changed on both sides to the *same* content -> keep (Full groups compare
   the stored block oid; Delta groups compare the diff-block oid **and**
   the delta base - the same diff bytes against a different base counts as
   different content); changed on both sides to different content ->
   conflict.
5. There is no numeric blending of tensor weights anywhere in this path.
6. On conflict, `--ours`/`--theirs` auto-resolve; the default strategy
   refuses and writes nothing.
7. A clean merge writes a new manifest, then a two-parent commit, following
   the journal-guarded write-object-then-CAS-ref sequence above.
