# ADR 0005 — Framed residuals; encoding chosen by measurement

- **Status:** Accepted (frame design) / parameters pending the Day-2 measurement
- **Date:** 2026-08-29

## Context

Once units are aligned, we store the difference. Two questions: **how** the
difference is encoded, and **at what granularity** it is stored.

The second question turned out to matter far more than the first.

## Part 1 — Granularity: frames, not layers

The prototype stored one compressed blob per layer. Resolving a chain of depth
5 to serve a single 4 KiB page fault therefore decompressed the whole layer
five times.

Measured, 32 MiB layer, chain depth 5, 128 KiB frames:

| Strategy | Bytes decompressed | Wall time | Peak RAM |
|---|---|---|---|
| Whole layer | 201.3 MB | 1347.8 ms | 134.2 MB |
| Frames | 0.8 MB | 0.8 ms | 0.4 MB |

**1.3 seconds for one page fault**, on a layer that is small by fixture
standards. This is the decision that makes Module 3's throughput number
possible at all.

**Decision:** the residual payload is a list of independently decompressible
zstd frames, each covering a contiguous range of output units, indexed in the
artifact header (SPEC 12 §4). Recursion is per frame, so peak memory is
`frame × depth` rather than `layer × depth`.

The objection to answer is "we can't chunk both sides at the same offsets,
because permutation destroys locality". True, and irrelevant: the mapping is
known exactly, and the natural atomic unit — one output unit, a linear row or a
conv filter — is contiguous on *both* sides. A target frame covering units
`[a,b)` needs base units `p[a:b]`, which the reader computes.

Frame size 128 KiB, rounded to whole output units. It is baked into every
artifact, so it is decided once, together with `max_chain_depth` and not
separately: depth bounds read latency, and small frames are what make a deep
chain tolerable.

Each frame carries a digest of its **reconstructed target bytes**, so tamper
detection survives the reconstruction path at a cost proportional to what was
read. That property is not asked for by the PS and is our strongest
original-thinking claim in Module 2.

## Part 2 — Encoding: to be decided by measurement

Candidates, all bijective and all exact (no floating-point arithmetic is
performed on weights anywhere in this system):

| Residual | Transform |
|---|---|
| `a ^ b` | none |
| `a ^ b` | byte-plane split |
| `a ^ b` | bitshuffle |
| `zigzag(b - a)` | none |
| `zigzag(b - a)` | byte-plane split |
| `zigzag(b - a)` | bitshuffle |

**Decision procedure, timeboxed to 90 minutes on Day 2:** measure all six for
**both** compression ratio and decompression throughput, on a real fine-tune
pair, and record all six numbers in `docs/tradeoffs.md` — not just the winner.

Ratio is 7% of the grade and mmap throughput is 8%. Choosing on ratio alone is
choosing the smaller number over the larger one.

Intuition, to be confirmed or refuted rather than assumed: fp16 weights that
differ slightly share exponent and high mantissa bits, so both residuals are
mostly zero in the high bytes and noisy in the low ones. A byte-plane split
should group the zeros together and help zstd; bitshuffle should help more and
cost more to undo, which is exactly the ratio-versus-throughput trade the
measurement exists to settle.

Whatever wins is recorded per artifact (SPEC 12 §5), not assumed by the reader,
so a later change does not invalidate existing objects.

## Consequences

- Frame size is immutable per artifact and readers take it from the header.
- The XOR/zigzag kernels are the hot loop of the read path and are the reason
  ADR 0011 exists.
- A group whose delta compresses to more than `α × full` is stored full
  instead (SPEC 12 §7). The XOR of two *unrelated* fp16 tensors is
  high-entropy noise that compresses to larger than the input, so without this
  bound a badly aligned group can make the repository grow faster than storing
  full copies.

## How we would know we were wrong

If the measured residual ratio on a real fine-tune pair is worse than plain
zstd of the target, alignment is not helping and the problem is upstream in
ADR 0004 — not here.
