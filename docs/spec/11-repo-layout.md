# SPEC 11 - Repository layout and on-disk invariants

## 1. Directory layout

```
<root>/.synapsefs/
  config              flat key=value text file (NOT JSON - see §2)
  HEAD                "ref: refs/heads/<name>\n"  or  "b3:<64 hex>\n"
  refs/heads/<name>    one line: "b3:<64 hex>\n"
  objects/<xx>/<62hex> one file per object, loose only (see §3, §5)
  pack/                path exists as a helper (RepoPaths::pack()); nothing
                        ever creates a file here - see §5
  tmp/                 atomic_write scratch; anything left here after a
                        crash is garbage, swept unconditionally by `sfs gc`
  incoming/            reserved for in-progress network transfers; also
                        swept unconditionally by `sfs gc`
  journal/j.<seq>      pending write-ahead records for merge (see §4)
  index.lock           advisory flock() target
  mount-daemon.pid      written by a running `sfs mount`; read by `sfs gc`
                        to refuse while a mount is attached
```

`RepoPaths::discover(start)` walks upward from `start` looking for a
directory containing `.synapsefs`, failing with `ErrKind::NotARepository`
at the filesystem root.

## 2. Repository config

`.synapsefs/config` is a flat text file, one `key=value` pair per line -
**not JSON**, unlike every other structured object in this system. A
missing file is treated as "defaults are valid," not an error. Unknown keys
are silently ignored on load. Numeric fields are parsed with
`std::stoul`/`std::stoull`/`std::stod`, which throw uncaught on a malformed
value - a hand-corrupted config file can crash the process rather than
produce a clean `Error`, since this parse path has no surrounding
try/catch.

| Key | Default | Meaning |
|---|---|---|
| `format_version` | 1 | |
| `chunk_bytes` | 65536 (64 KiB) | object chunk-digest granularity, baked into each object at write time |
| `frame_bytes` | 131072 (128 KiB) | residual frame target size |
| `max_chain_depth` | 5 | delta-chain depth cap, enforced at both commit and read time |
| `snapshot_alpha` | 0.5 | delta must be ≤ this fraction of full size or it's stored Full instead |
| `compress_raw` | false | parsed and round-tripped by `RepoConfig`, but never read anywhere else in the codebase - a dead knob regardless of its value. Not to be confused with `Diff`-object payloads, which are genuinely Zstd-compressed independent of this setting (see SPEC 10 §2, SPEC 12 §6) |
| `cache_bytes` | 1 GiB | mount daemon's frame-cache LRU budget |
| `listen` | `127.0.0.1:9418` | documented default for `sfs serve`. The CLI's own `-p/--port` default was fixed to match this value (`9418`). - see SPEC 14 §5 |

`validate()` requires `format_version != 0`, `chunk_bytes`/`frame_bytes`/
`max_chain_depth` all nonzero, and `0 < snapshot_alpha ≤ 1.0`.

## 3. Refs

`HEAD` is either symbolic (`ref: refs/heads/<name>\n`) or detached
(`b3:<64hex>\n`), written atomically. Branch refs are one-line files at
`refs/heads/<name>`. `RefStore::update()` is a real compare-and-swap: an
`expected` value of empty means "must not currently exist"; any mismatch
between the on-disk content and `expected` fails with `ErrKind::RefRaceLost`.

`rev_parse()` resolves, in order: the literal string `"HEAD"` (recursive
symbolic resolution), `"refs/heads/<name>"` or a bare branch name (direct
file read), a full `b3:`-prefixed 64-hex oid (direct parse), or a shorter
`b3:`-prefixed abbreviation (resolved by scanning the objects fan-out
directories for a unique prefix - ambiguous is a hard error, none is
`RefNotFound`).

`delete_branch(name, force)`: the `force` flag is a no-op inside
`RefStore` itself (`(void)force;`) - "is this branch reachable from
elsewhere" is checked by the CLI layer (`sfs branch -d`), not enforced
inside the ref store, specifically to avoid a circular dependency on the
DAG-walk code.

## 4. Crash safety mechanisms

**Atomic rename** (`util::atomic_write`) is the default for every single-file
mutation: write a randomly-named temp file in the target directory,
optionally fsync its contents, `rename()` over the destination, optionally
fsync the parent directory. Covers loose objects (`overwrite=false`,
content-addressed idempotence), refs and HEAD (`overwrite=true`, CAS
enforced by the caller comparing content first), and journal records.

**Write-ahead journal** (`.synapsefs/journal/j.<seq>`) is used only for
operations that must move more than one file together: `merge` (a new
commit object plus a branch-ref CAS) and `gc --pack` (currently
unreachable - see SPEC 10/§7 and below). Record format:

```
[4-byte LE payload_len][JSON payload][32-byte BLAKE3 digest of the JSON]
```

JSON payload fields: `format_version, op ("merge"|"pack"), seq, timestamp,
ref_name, ref_old, ref_new, pack_name, subsumed[]`. Written *before* the
guarded mutation begins, deleted (with a directory fsync) *after* it
completes. A length or digest mismatch on decode is `ErrKind::JournalTorn`
- corruption is reported, never partially interpreted.

`sfs verify --repair` runs `Journal::recover()` before anything else. For a
pending `Merge` record: if the ref already reads `ref_new`, the mutation
already completed and the record is simply deleted; otherwise the same
compare-and-swap is retried (a `RefRaceLost` during recovery - someone else
already finished it - is tolerated). For a pending `Pack` record: recovery
always fails with `ErrKind::NotImplemented`, since no code path in this
build can currently create one.

A fast-forward merge writes **no** journal record - HEAD stays symbolic, so
only the branch ref file itself moves, and that single rename is already
atomic on its own.

## 5. Object storage: loose only

Every object lives at `objects/<first-2-hex>/<remaining-62-hex>`. **Packed
storage is not implemented.** `store::packfile.hpp` declares a
`Packfile`/`PackWriter` API, but there is no corresponding `.cpp` file, no
build target compiles it, and no code anywhere in the tree constructs
either type. `sfs gc --pack` returns `ErrKind::NotImplemented`
unconditionally. Treat `pack/` as a reserved, currently-unused path.

## 6. Locking

`.synapsefs/index.lock`, advisory `flock()`, non-blocking by default.
`commit`, `merge`, and `gc` take it exclusively. `checkout` and `verify`
take no lock - reads of immutable, content-addressed objects are safe
without one. Failure to acquire -> `ErrKind::RepositoryLocked` -> CLI exit
code 6.
