# ADR 0002 - BLAKE3-256 as the content address

## Context

Every object is addressed by a digest, and `verify` time is 10% of the grade -
the same weight as tamper detection itself. At multi-GB scale the hash function
*is* the verification cost.

## Measurement

From the prototype we tested, hashing 1 GB:

| Function | Throughput |
|---|---|
| SHA-256 | 0.38 GB/s |
| BLAKE3, 1 thread | 4.30 GB/s |
| BLAKE3, all cores | 8.37 GB/s |

**Caveat we state honestly:** the machine that produced these numbers may not
have SHA-NI. On hardware that does, SHA-256 runs 2–2.5 GB/s and the gap narrows
a lot.

## Decision

BLAKE3-256, vendored from upstream (`third_party/README.md`)
rather than taken from a package manager, so that the exact code producing our addresses is pinned.

Digests are written as `b3:<64 hex>` everywhere in text form. The prefix is not
a compatibility mechanism - we do not support a second algorithm - it is there
so a corrupt or foreign identifier is obviously not ours.

SHA-256 survives in exactly one place: `manifest.file.sha256`, a witness for
the whole reconstructed file.

## Consequences

- Our chunk digests are independent BLAKE3 hashes of 64 KiB windows, which is simpler and lets us
  change the chunk size without touching the addressing scheme.
