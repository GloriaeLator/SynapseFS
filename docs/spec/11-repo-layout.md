# SPEC 11 — Repository layout and on-disk invariants

**Status:** normative · **Format version:** 1

Where objects live, how they are written so that `kill -9` can never corrupt
history, and how storage is reclaimed. SPEC 10 defines what the objects *are*;
this document defines the bytes on disk and the ordering rules around them.

---

## 1. Directory layout

```
<repo>/
  .synapsefs/
    format                     # "synapsefs 1\n" — refuse to open anything else
    config                     # canonical JSON, repository-local settings (§5)
    objects/
      <xx>/<remaining 62 hex>  # loose objects, fan-out on the first byte
      pack/
        pack-<oid>.sfspack     # packfile (§4)
        pack-<oid>.sfsidx      # its index
      tmp/                     # in-flight writes; anything here is garbage
    refs/
      heads/<branch>           # one line: "b3:<64 hex>\n"
      remotes/<remote>/<branch>
    HEAD                       # "ref: refs/heads/main\n"  or a detached oid
    journal/
      <seq>.jrnl               # intent records for multi-file updates (§3.3)
    index.lock                 # flock() advisory lock, whole-repo writer lock
    incoming/                  # blocks received from a peer, pre-verification
```

Everything under `.synapsefs/` is owned by SynapseFS. The working tree — the
checked-out `.safetensors` file — lives beside it and is never read back as a
source of truth.

`format` is checked on every open. A repository written by a future format
version MUST be refused with a clear message, not opened optimistically.

---

## 2. Object storage

### 2.1 Loose objects

Path is derived from the identifier: `objects/<hex[0:2]>/<hex[2:64]>`, where
`hex` is the identifier without the `b3:` prefix. The two-character fan-out
keeps directory sizes reasonable on ext4 without a second level.

A loose object file contains, in order:

```
  0 ..  7   magic         "SFSOBJ\0\1"
  8 ..  8   kind          enum: 0 raw · 1 diff · 2 header · 3 manifest · 4 commit · 5 topology
  9 ..  9   compression   enum: 0 none · 1 zstd
 10 .. 11   reserved      zero
 12 .. 15   chunk_log2    log2 of the chunk-digest size in bytes (default 16 → 64 KiB)
 16 .. 23   payload_len   uncompressed payload length, LE u64
 24 .. 31   stored_len    on-disk payload length after compression, LE u64
 32 .. 35   chunk_count   LE u32 = ceil(payload_len / chunk_size)
 36 .. 39   reserved      zero
 40 .. 40+32*chunk_count  chunk digests, BLAKE3-256 of each UNCOMPRESSED chunk
 …          payload
```

The object's identifier is **not** the hash of this file. It is the hash of
`frame(kind, payload)` per SPEC 10 §1.3, over the *uncompressed* payload. The
container above is a local storage detail; two repositories that compress
differently still agree on every address.

### 2.2 Chunk digests, and why they are here

`chunk_size` is 64 KiB by default. Each chunk digest covers the uncompressed
bytes of that chunk.

This is what lets verification granularity equal read granularity. Two
distinct read paths:

```
read_range(oid, off, len)   // fast path: verify only the chunks touched
verify_block(oid)           // slow path: verify the whole object and its identifier
```

`read_range` is what the mount calls on a page fault. Verifying the whole
object on every read makes tamper detection and mmap throughput mutually
exclusive; they are only in conflict because the two granularities differ.
Measured on the prototype: 0.6 MB/s whole-block versus 183.9 MB/s chunked, with
the chunked path additionally *naming* the corrupt chunk. See
`docs/adr/0005-residual-encoding.md` and `docs/tradeoffs.md`.

`verify_block` MUST be used when: ingesting a block from a peer, running
`sfs verify`, and recovering after a crash.

Compression interacts with range reads, so it is constrained: when
`compression = zstd`, the payload MUST be written as independently
decompressible zstd frames aligned to chunk boundaries. An object that cannot
satisfy that is stored uncompressed. `raw` tensor-group blocks are normally
stored uncompressed — fp16 weights do not compress usefully and the mount pays
for it on every fault.

### 2.3 Packfiles

Loose objects cost one inode and one `open()` each. A repository with a few
hundred commits over a 7B checkpoint has enough of them to matter.
`sfs gc --pack` rewrites reachable loose objects into packfiles:

```
pack-<oid>.sfspack:  [ header ][ object records… ][ trailer ]
pack-<oid>.sfsidx:   sorted (oid → offset, length, kind) + a fan-out table
```

Objects in a packfile keep their chunk digests, so `read_range` behaves
identically whether an object is loose or packed. `<oid>` in the filename is
the digest of the sorted list of contained identifiers, which makes packfile
names themselves content-addressed and safe to sync.

Packfiles are append-only and immutable. `gc` writes a new pack and deletes the
old one only after the new one is fsynced and its index is in place.

See `docs/adr/0006-packfiles-vs-loose-objects.md`.

---

## 3. Crash safety

The requirement from the PS: a crash or `kill -9` mid-write must never leave
history corrupt. The repository must recover cleanly or **safely refuse to
proceed**. Not "usually recover".

### 3.1 The one primitive

Every single-file write goes through `sfs::util::atomic_write`:

```
1. create  .synapsefs/tmp/<random>          (O_CREAT|O_EXCL, mode 0644)
2. write   the full contents
3. fsync   the temp file                     ← the data is durable
4. rename  temp → final path                 ← atomic on the same filesystem
5. fsync   the PARENT DIRECTORY              ← the rename is durable
```

Step 5 is the one that gets skipped and the one that matters. Without it, the
rename can be lost across a power failure while the data survives, which
presents as a missing object rather than a corrupt one — recoverable, but only
because everything else here is careful.

Consequences that follow directly:

- A partially written object is never visible under its final name, so a reader
  never sees a truncated object. It sees no object.
- `.synapsefs/tmp/` may accumulate junk after a crash. Everything in it is
  garbage by construction and `gc` deletes it. It is never read.
- Objects are immutable and content-addressed, so a concurrent writer producing
  the *same* object is harmless: both write to distinct temp files and both
  rename to the same final name with identical content.

### 3.2 Ordering: objects before refs

For any operation that creates objects and then moves a ref:

```
write all objects (each atomically, each fsynced)
   → fsync the objects directory
   → atomically update the ref
```

Crash before the ref update leaves unreferenced objects. Those are garbage, not
corruption: nothing points at them, `verify` does not walk them, `gc` collects
them. This is the property that makes the whole design tolerable — **the
failure mode of a crashed commit is wasted disk, not a broken repository.**

### 3.3 The journal

Two operations touch more than one ref-like file and therefore cannot be made
atomic by rename alone:

- `merge`, which updates a branch ref and `HEAD`;
- `gc --pack`, which adds a packfile and removes loose objects.

For those, an intent record is written to `.synapsefs/journal/<seq>.jrnl`
(atomically, per §3.1) *before* the first mutation, and removed after the last
one. On open, a non-empty journal directory means a previous process died
mid-operation. Recovery is: replay if the record is complete and its effects
are idempotent, otherwise **refuse to proceed** and tell the user to run
`sfs verify --repair`. Refusing is an allowed outcome under the PS and a
correct one; guessing is not.

Journal records are themselves framed and digested, so a torn journal record is
detected rather than replayed.

### 3.4 Locking

`.synapsefs/index.lock` carries a `flock(LOCK_EX)` for the duration of any
mutating operation. Readers — `log`, `verify`, `checkout`, the mount daemon —
take `LOCK_SH` or no lock at all, since objects are immutable. The lock is
advisory and process-scoped; it is not a defence against a hostile local
process, which is out of scope (`docs/threat_model.md`).

The mount daemon MUST NOT hold the exclusive lock. A mounted repository stays
writable.

---

## 4. Refs

A ref file contains exactly one line: `b3:<64 hex>\n`. It is updated with
`atomic_write`, never with `O_TRUNC` + write.

`HEAD` contains either `ref: refs/heads/<name>\n` or a bare identifier
(detached, which `checkout <oid>` produces).

`branch` creates and lists only. Switching branches is `checkout <branch-name>`
— pre-2.23 git semantics, as the PS specifies. See SPEC 15.

A ref update is a compare-and-swap: the caller states the identifier it expects
to find, and the update fails if the file no longer holds it. This is what
makes concurrent `sfs commit` invocations safe under the repository lock and
what makes `pull` refuse a non-fast-forward instead of silently discarding.

---

## 5. Configuration

`.synapsefs/config` is canonical JSON, repository-local, and is **not** an
object (it is machine-local policy, not history):

```jsonc
{
  "format_version": 1,
  "chunk_bytes": 65536,          // verification granularity   (§2.2)
  "frame_bytes": 131072,         // residual frame target      (SPEC 12)
  "max_chain_depth": 5,          // bounds read LATENCY
  "snapshot_alpha": 0.5,         // bounds SPACE: snapshot when delta > α × full
  "compress_raw": false,
  "cache_bytes": 1073741824,     // mount LRU budget
  "listen": "127.0.0.1:9418"     // `sfs serve` default
}
```

`chunk_bytes` and `frame_bytes` are baked into every object written while they
are in effect, so changing them does not rewrite existing objects — readers
take the value from the object header, not from config. `max_chain_depth` and
`snapshot_alpha` affect only future writes and can be changed freely.

Why both a depth bound and a size ratio: depth bounds **time** (a hundred 0.1%
deltas cost a hundred hops on every read), α bounds **space** (one badly
aligned group can produce a delta larger than the full block, because the XOR
of unrelated fp16 is high-entropy noise). Neither implies the other. Git makes
exactly this pair of choices in its packfiles.

---

## 6. Garbage collection

Reachability: start at every ref and at `HEAD`, walk commits through `parents`,
each commit to its `manifest` and `topology`, each manifest to its
`header_block` and to every group's `block` or `diff_block`, each diff artifact
to nothing (its base is named by the manifest entry, and that manifest is
reachable by the ancestor invariant).

Unreachable objects, `.synapsefs/tmp/*` and `.synapsefs/incoming/*` older than
the current process's start time are collectable.

`gc` MUST take the exclusive lock, and MUST NOT run while a mount daemon is
attached to the repository — the daemon holds open object handles by identifier
and an unlinked-but-open file is a silent correctness trap on the next
remount. The daemon registers itself in `.synapsefs/config`-adjacent runtime
state; `gc` refuses if that registration is live.

---

## 7. Concurrency summary

| Actor | Lock | Notes |
|---|---|---|
| `commit`, `merge`, `pull`, `gc` | `LOCK_EX` | One writer at a time, repository-wide. |
| `checkout` | `LOCK_SH` | Reads objects, writes only the working tree. |
| `verify`, `log` | none | Objects are immutable; a concurrently added object is simply not walked. |
| mount daemon | none | Never holds the write lock; see §6 for the `gc` interaction. |
| `serve` | `LOCK_SH` while streaming | Prevents `gc` from removing an object mid-transfer. |

Multiple concurrent readers of the mount are a graded requirement and are
handled inside the daemon (SPEC 16), not by repository locking.
