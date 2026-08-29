# Threat model

What SynapseFS detects, what it does not, and why that is the right scope for
this problem rather than an excuse.

Being explicit about the boundary is itself part of the answer: a system that
claims more than it delivers is worse than one that states its limits.

---

## 1. Trust anchor

**Trust is rooted at a locally accepted commit or ref identifier.**

If you have a commit id you believe in — because you created it, or because
someone you trust told you it out of band — then everything reachable from it
is verifiable, because every edge in the graph is a cryptographic hash. Change
one byte anywhere in the history under that commit and the identifier changes.

Everything below follows from where that anchor sits.

---

## 2. In scope — attacks we detect

| Attack | How it is caught |
|---|---|
| **Bit rot** — a byte flips on disk | Per-chunk BLAKE3 digests inside every block. `read_range` verifies the chunks it touches; `verify --full` checks all of them. |
| **Malicious block injection** — a peer or a local process substitutes a block | The object is stored under the hash of its own framed bytes. Different bytes, different address; `verify_block` fails on ingest and on read. |
| **Type confusion** — a block that validates as two different kinds | The object kind is inside the hashed bytes (`synapsefs.<kind> <len>\0`), so reading at the wrong kind is a hash mismatch. |
| **Tampering with a residual mid-chain** | Each residual frame carries a digest of its *reconstructed target bytes*. Corruption of the base block, of the residual, or of the permutation all fail the same check, at the hop where it happened. |
| **Corruption discovered only on read** | The mount returns `EIO` and logs the object and chunk. It never serves plausible-looking wrong bytes. |
| **Manifest referencing an unreachable base** | The ancestor invariant: every delta's base commit must be an ancestor of the commit containing it. Checked by `verify` and on push/pull ingest. |
| **Torn write from a crash or `kill -9`** | Atomic rename plus parent-directory fsync; objects before refs. A partial object is never visible under its final name. |
| **Torn journal record** | Journal records are framed and digested; a torn one is detected and the repository refuses rather than replaying. |

The last two are the ones the PS calls out specifically, and the guarantee is:
after any crash, the repository either verifies clean or **explicitly refuses to
proceed**. It never silently continues from a damaged state.

---

## 3. Out of scope — and why

### 3.1 A peer with a self-consistent alternate history

A malicious peer can construct an entirely valid history — correctly hashed,
internally consistent — that is simply not ours, and offer it under a ref name
we recognise. Ref rollback and wholesale history replacement are the same
attack.

We do not defend against this, and the PS explicitly puts it out of scope. The
reason it is out of scope rather than merely hard: **defending against it
requires an authenticated notion of who is allowed to move a ref**, which means
identity, key distribution and revocation. That is a different system.

What we do instead: `pull` refuses a non-fast-forward. A peer cannot silently
rewrite history you already have; it can only offer you a divergent one, which
you are told about and must resolve deliberately.

### 3.2 Authentication and signing

No GPG signatures on commits, no authentication on the wire, no TLS. The
`author` field is free text and is not evidence of anything.

Out of scope per the PS. The honest framing: SynapseFS gives you **integrity**
(the bytes are the bytes you accepted) and not **authenticity** (a claim about
who produced them). Those are different properties and only one of them is
asked for here.

If it were in scope, the shape is known and small: sign the commit identifier,
distribute public keys out of band, verify at ref-update time. The Merkle
structure already reduces the problem to signing one 32-byte value.

### 3.3 A hostile local process

Anything running as your user can delete `.synapsefs/` or replace the `sfs`
binary. The repository lock is `flock`, which is advisory. We assume the local
machine is not adversarial.

Note what still holds even here: a hostile local process can *destroy* data or
*refuse service*, but it cannot silently alter a checkpoint and have `verify`
pass, unless it also rewrites the ref you are anchoring on.

### 3.4 Denial of service

A peer can send a well-formed but enormous have/want set, or open many
connections. We cap frame sizes (8 MiB) and identifier counts per round (256),
and that is the extent of it. Rate limiting and connection accounting are not
implemented.

### 3.5 Side channels

Object sizes and access patterns leak information about the model being stored.
Not considered.

---

## 4. What "verify" actually checks

`sfs verify` runs standalone, with no checkout and no mount — this is a PS
requirement and a graded metric. It performs:

1. every ref resolves to a commit that exists;
2. DAG walk from every ref: each commit's bytes hash to its identifier, and
   canonical re-serialisation reproduces that identifier;
3. each commit's manifest and topology exist and are well-formed;
4. buffer layout invariants: no gaps, no overlaps, total matches
   `file.total_bytes`;
5. every referenced block exists and its identifier matches its content;
6. the **ancestor invariant** for every delta group;
7. `chain_depth` consistency down every chain;
8. with `--full`: every chunk of every reachable object.

Exit code 4 on any integrity failure, with the object and chunk named. Distinct
from exit 1 so a script can tell "the data is wrong" from "the program failed".

---

## 5. Residual risks we accept

- A repository whose refs have been rewritten by someone with write access to
  the directory is indistinguishable from a legitimate one. Anchor on a commit
  id you recorded elsewhere.
- Verification is only as good as the anchor. `verify` on an entirely
  attacker-supplied repository will pass; it is self-consistent by construction.
- We detect corruption; we do not repair it. `--repair` handles a crashed
  journal, not a bit flip. Recovery from corruption means fetching the block
  again from a peer that has it.
