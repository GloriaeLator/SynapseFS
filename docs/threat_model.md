# Threat model

## What is detected

**Every stored object re-hashes to its own address.** A loose object's
oid is a framed BLAKE3-256 digest of its payload; `store::LooseStore::get()`
re-hashes the whole payload on every full read and fails with
`ErrKind::HashMismatch` on any difference. `read_range()` verifies only the
chunk digests intersecting the requested range (`format::verify_chunk`,
also BLAKE3-based) - cheaper, still exact for the bytes actually served.

**Reconstruction is verified independently of storage.** Every residual
frame carries a BLAKE3 digest of the *reconstructed* target bytes (post
decompress, post residual-apply), checked on every read. This means tamper
detection covers the whole reconstruction path - a corrupted base object,
a corrupted diff artifact, or a bit flip introduced during decompression
are all caught the same way, at the point the bytes are about to be
handed to a caller.

**A repository-wide integrity walk exists standalone.** `sfs verify` needs
no checkout and no mount; it walks every reachable commit, checks each
commit and manifest's own self-hash and canonicalization, checks the
"ancestor invariant" (a delta's base commit must actually be an ancestor of
the commit referencing it - `ErrKind::AncestorInvariantViolated`), checks
chain-depth bookkeeping consistency, and (with `--full`) re-hashes every
referenced object's chunks. It exits with a distinct code (`Integrity = 4`)
so scripts can tell "the data is wrong" apart from "the program is wrong."

**Untrusted permutation data cannot cause an out-of-bounds read.** Every
explicit permutation parsed from a diff artifact is validated as a true
bijection over `[0, n)` before it is used to index anything
(`core::is_valid_permutation`); a non-bijective permutation is a clean
`ErrKind::InvalidPermutation`, not a memory-safety incident.

**No exception crosses a module boundary.** `core::Error`/`Result<T>`
propagate failures explicitly everywhere; the design comment in
`core/error.hpp` is explicit about why - an exception escaping a libfuse
callback would hang the mount rather than return an error to the kernel.

**A written safetensors file is cross-checked twice.** `checkout`'s writer
requires the emitted byte count to match the manifest's recorded total, and
(when the manifest carries a non-empty SHA-256 witness) the emitted bytes'
SHA-256 to match, *before* the temp file is renamed into place - a failed
checkout never leaves a partially-correct file at the destination.

## What is explicitly out of scope

**No authentication or encryption on the network protocol.** `sfs
serve`/`push`/`pull` run over plain, unauthenticated TCP with no crypto -
this matches the project's own stated scope decision.

## Summary

The content-addressed storage core - commit, manifest, diff artifact,
loose-object container, reconstruction - has real, layered, independently
tested integrity checking: hash-of-payload at rest, hash-of-reconstructed-
bytes on read, and a standalone whole-repository walk. The network layer is
architecturally separate from all of that. Object payloads crossing
the wire are re-verified against their own BLAKE3 address before being
committed to disk, and a failed push/pull is now visible at the shell exit
code rather than silently swallowed. It is still a best-effort file copier
in the ways that matter most - refs/HEAD/journal sync on raw content/size
equality with no cryptographic check at all. A push cannot
learn whether the receiver's own integrity check passed. Treat it as a
verified-at-the-object-level, unverified-overall transport until
`modules/net` is rebuilt on top of `modules/store`'s object model (which is
what `store::have_probe`/`reachable_objects` already exist to support,
unused, today) - and run `sfs verify --full` after any `pull` before
trusting the result.
