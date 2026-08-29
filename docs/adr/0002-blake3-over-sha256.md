# ADR 0002 — BLAKE3-256 as the content address

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

Every object is addressed by a digest, and `verify` time is 10% of the grade —
the same weight as tamper detection itself. At multi-GB scale the hash function
*is* the verification cost.

The digest is also the address. Changing it later invalidates every stored
object in every repository, so this is the most irreversible decision in the
project.

## Measurement

From the prototype's `demo_4_verify.py`, hashing 1 GB:

| Function | Throughput |
|---|---|
| SHA-256 | 0.38 GB/s |
| BLAKE3, 1 thread | 4.30 GB/s |
| BLAKE3, all cores | 8.37 GB/s |

**Caveat we state honestly:** the machine that produced these numbers may not
have SHA-NI. On hardware that does, SHA-256 runs 2–2.5 GB/s and the gap narrows
a lot. The table above must be regenerated on the demo machine
(`bench/verify_time.cpp`) and the result recorded in `docs/benchmarks.md`. If
SHA-256 wins there, "we measured and the difference did not justify a
non-stdlib dependency" is a better answer than following a blog post.

## Options

1. **SHA-256.** Ubiquitous, no dependency, hardware-accelerated where SHA-NI
   exists. Slow without it.
2. **BLAKE3-256.** 10× faster here, internally tree-structured so it
   parallelises within one object, and upstream ships runtime ISA dispatch.
3. **xxHash / non-cryptographic.** Fast and disqualifying: Module 2 is about
   detecting *malicious* block injection, and a non-cryptographic hash makes
   that trivial to defeat.

## Decision

BLAKE3-256, vendored from upstream at a pinned commit (`third_party/README.md`)
rather than taken from a package manager, so that the ISA dispatch survives and
so that the exact code producing our addresses is pinned.

Digests are written as `b3:<64 hex>` everywhere in text form. The prefix is not
a compatibility mechanism — we do not support a second algorithm — it is there
so a corrupt or foreign identifier is obviously not ours.

SHA-256 survives in exactly one place: `manifest.file.sha256`, a witness for
the whole reconstructed file. That is a value humans and the evaluator compare
by hand, and SHA-256 is what they will reach for.

## Consequences

- A vendored dependency with its own build. Accepted; it is small and
  self-contained.
- BLAKE3's tree structure means chunked verification (SPEC 11 §2.2) could in
  principle reuse subtree hashes. We do **not** do this — our chunk digests are
  independent BLAKE3 hashes of 64 KiB windows, which is simpler and lets us
  change the chunk size without touching the addressing scheme.
- If the demo machine has SHA-NI and the measured gap is under 2×, revisit.
  Log the revisit in `docs/tradeoffs.md`; do not silently keep it.
