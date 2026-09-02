# ADR 0007 - Atomic rename everywhere; a journal only for multi-file updates

## Context

A crash or `kill -9` mid-write must never leave history corrupt. The
system must recover cleanly **or safely refuse to proceed**. "Refuse" is an
allowed outcome, which is worth noticing - it means we can choose the design
with the smallest number of recoverable states rather than the one that tries
hardest to recover.

## Decision

**Rename as the primitive, journal only where rename is insufficient.**

**The primitive** (`sfs::util::atomic_write`):

```
create .synapsefs/tmp/<random> (O_CREAT|O_EXCL)
write contents
fsync file            ← data durable
rename -> final path   ← atomic within the filesystem
fsync parent dir      ← the rename is durable
```

The parent-directory fsync is the step that gets skipped and the one that
matters: without it a power failure can lose the rename while keeping the data,
which presents as a missing object rather than a corrupt one.

**The ordering rule:** objects before refs. Write and fsync every object, then
update the ref. A crash before the ref update leaves unreferenced objects,
which are garbage, not corruption - nothing points at them, `verify` does not
walk them.

**The journal** exists for exactly two operations that touch more than one
ref-like file: `merge` (branch ref + `HEAD`). An intent record is written atomically before the first mutation and removed after the last. On open, a leftover record means a
previous process died mid-operation: replay if the record is complete and its
effects are idempotent, otherwise **refuse and tell the user to run
`sfs verify --repair`**. Journal records are framed and digested, so a torn
record is detected rather than replayed.

## Consequences

- `.synapsefs/tmp/` accumulates junk after a crash. Everything in it is garbage
  by construction; it is never read.
- Objects are immutable and content-addressed, so two writers producing the same
  object is harmless: distinct temp files, identical final content.
- Ref updates are compare-and-swap, which is what makes concurrent `commit` safe
  under the repository lock and what makes `pull` refuse a non-fast-forward
  rather than silently discard.
