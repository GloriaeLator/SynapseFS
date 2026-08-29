# ADR 0006 — Loose objects first, packfiles behind `gc --pack`

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

A repository over a 7B checkpoint with a few hundred commits holds a lot of
objects. Loose storage costs one inode, one `open()` and one `stat()` each.
Packfiles amortise that, at the cost of an index, a rewrite step and a second
read path.

## Options

1. **Loose only.** Simplest. Every object is a file; `read_range` is a `pread`.
2. **Packed only.** Every write goes into an append-only pack. Fast reads, but
   now a crash mid-append has to be reasoned about, and `gc` has to rewrite
   packs to reclaim anything.
3. **Loose on write, packed by an explicit `gc --pack`.** Git's answer.

## Decision

Option 3, and the ordering matters: **loose is implemented first and packfiles
are additive.** Concretely, `modules/store/src/packfile.cpp` may be entirely
unimplemented on Day 3 and the system is complete without it.

The reason is the crash-safety requirement. An atomic-rename write of a whole
file (ADR 0007) is trivially correct and easy to argue in a Q&A. An
append-to-pack write needs its own torn-write reasoning, and Module 2 is 20% of
the grade on *integrity*, not on inode efficiency. We buy the hard property
first and optimise afterwards.

Packfiles keep per-object chunk digests, so `read_range` behaves identically
whether an object is loose or packed and the mount does not know the
difference. Pack file names are themselves content-addressed (the digest of the
sorted list of contained identifiers), which makes them safe to name in a sync.

## Consequences

- Two read paths in `block_store.cpp`, behind one interface. The loose path is
  the reference; `test_block_store.cpp` runs the same assertions against both.
- `gc --pack` writes the new pack and its index, fsyncs both, and only then
  unlinks the loose objects. A crash leaves duplicates, never a gap.
- `gc` refuses while a mount daemon is attached: the daemon holds objects open
  by identifier, and an unlinked-but-open file is a trap that only fires on the
  next remount.
- If we fall behind, packfiles are on the cut list and the cut costs a
  performance number, not a feature.
