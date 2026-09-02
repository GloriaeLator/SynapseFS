# ADR 0006 - Loose objects first, packfiles

## Context

A repository over a 7B checkpoint with a few hundred commits holds a lot of
objects. Loose storage costs one inode, one `open()` and one `stat()` each.
Packfiles amortise that, at the cost of an index, a rewrite step and a second
read path.

## Decision

We implement based on git and the ordering matters: **loose is implemented first and packfiles
are additive.**

The reason is the crash-safety requirement. An
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
- If we fall behind, packfiles are on the cut list and the cut costs a
  performance number, not a feature.
