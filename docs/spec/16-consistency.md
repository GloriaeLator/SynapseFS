# SPEC 16 - Consistency and the filesystem contract

## 1. One reconstructor

`codec::reconstruct::read_range()` is the only function in this codebase
that turns stored objects back into checkpoint bytes. `sfs checkout
--output`, the FUSE mount's `read()` callback, and `sfs verify --full`'s
byte-level re-hash all call it. There is no second, independent
reconstruction implementation that could silently diverge from it - "byte
identical" between checkout and mount is a property of the call graph, not
a coincidence that happens to be tested.

## 2. Delta chain resolution

Recursion down a delta chain (Delta group -> base commit's manifest -> that
group's own entry, repeated) happens **per residual frame**, not per layer
of the chain. Peak memory for serving one read is bounded by
`frame_bytes × chain_depth` - at defaults, 128 KiB × up to 5 = 640 KiB, not
proportional to the full checkpoint size or to the number of intermediate
commits' full tensors. `RepoConfig::max_chain_depth` (default 5) is
enforced both when planning a new commit's storage decision
(`codec::snapshot_policy::decide`) and when reading
(`ReadCtx::max_depth`).

Every frame touched during reconstruction has its BLAKE3 digest (of the
*reconstructed* bytes - see [SPEC 12](12-residual-format.md#4-frames))
checked before those bytes are returned to the caller. A secondary
(non-leading-axis) permutation dependency is resolved by looking up the
owning tensor's own diff artifact within the same commit
(`resolve_secondary_permutation`); row-gather and column-gather are applied
in a way that commutes, so evaluation order does not affect correctness.

## 3. Checkout

`sfs checkout --output <path>` writes through `stio::StWriter`: bytes are
appended strictly in order (no scatter-write support - this is a
concatenation of already-verbatim ranges, never a re-serialization), a
running SHA-256 is tracked, and `finish()` requires both the total byte
count to match `manifest.file.total_bytes` **and** (if
`manifest.file.sha256` is non-empty) the SHA-256 to match, checked *before*
the temp file is renamed into place. A mismatch on either check deletes the
temp file and leaves the destination untouched - a failed checkout never
produces a partially-correct file at the requested path.

## 4. Mount

`mount::SynapseFs` (FUSE3 low-level) serves reads through a bounded LRU
`FrameCache` keyed by `(artifact oid, tensor index, frame index)`, calling
`read_range` to fill a cache miss and never pre-materializing or eagerly
warming the whole file - `daemon.hpp`'s explicit contract: no reconstructed
file is ever written to disk, and no group is ever fully reconstructed to
serve a partial read. The mount is unconditionally read-only (`open()`
rejects any non-`O_RDONLY` flag with `EROFS`); there is no write path in
the FUSE op table at all.

Sequential-read detection (`PrefetchState`, default: 3 consecutive
sequential reads trigger warming up to 4 frames ahead) exists purely as a
latency optimization and never changes correctness - a prefetch failure is
swallowed silently, never surfaced as a read error.

## 5. What is and is not covered by "byte-for-byte"

Reconstruction is exact at the byte level for the safetensors container:
the verbatim header block plus concatenated (possibly delta-reconstructed)
tensor-group bytes reproduce the original file exactly, which is why
`safetensors.torch.load_file()` can open a mounted or checked-out file
unmodified. This guarantee is anchored by two independent checks: the
per-frame BLAKE3 digest (integrity of the reconstruction path itself) and,
on `checkout`, the whole-file SHA-256 witness recorded at commit time
(integrity of the *original-to-final* claim). **This guarantee applies to
files reconstructed from objects fetched through `sfs commit`/`checkout`/
`mount`/`verify` - it does not currently extend to bytes received via `sfs
push`/`pull`**, whose transfer protocol performs no hash verification of
its own (see [SPEC 14](14-wire-protocol.md#4-resumability--size-based-not-content-verified)
and [`threat_model.md`](../threat_model.md)). A pulled repository should be
verified (`sfs verify --full`) before it is trusted.
