# SPEC 14 — Wire protocol

**Status:** normative · **Protocol version:** 1

Two peers synchronising history over TCP, transferring only the blocks the
receiver does not have.

The PS is explicit that the transport is a free choice and is **not graded**,
while content-addressed block diffing **must be implemented by the team** and
not delegated to `rsync` or `rclone`. This document is therefore deliberately
plain about the transport and precise about the negotiation.

---

## 1. Transport

A single TCP connection, length-prefixed binary frames, one request/response
exchange at a time per connection. No TLS, no authentication — see
`docs/threat_model.md` for why that is the right scope here and what it costs.

Default listen address `127.0.0.1:9418`, overridable by `.synapsefs/config`
(`listen`) and by `sfs serve --listen host:port`. The README documents the
invocation, which is a listed deliverable.

---

## 2. Framing

```
┌────────────┬──────────┬──────────────────────┐
│ u32 LE len │ u8  type │ payload (len bytes)  │
└────────────┴──────────┴──────────────────────┘
```

`len` counts the payload only. Maximum frame payload is 8 MiB; a peer
announcing more is a protocol error and the connection is closed. Block
payloads larger than that are split across `BLOCK_DATA` frames (§4.3).

| type | name | direction |
|------|------|-----------|
| 0x01 | `HELLO`       | → |
| 0x02 | `HELLO_ACK`   | ← |
| 0x10 | `REF_LIST`    | → / ← |
| 0x11 | `HAVE`        | → |
| 0x12 | `WANT`        | ← |
| 0x20 | `BLOCK_HDR`   | → / ← |
| 0x21 | `BLOCK_DATA`  | → / ← |
| 0x22 | `BLOCK_END`   | → / ← |
| 0x30 | `REF_UPDATE`  | → |
| 0x7e | `DONE`        | → / ← |
| 0x7f | `ERROR`       | → / ← |

All multi-byte integers are little-endian. Object identifiers travel as raw 32
bytes without the `b3:` prefix.

---

## 3. Negotiation

The interesting part, and the part that is graded.

### 3.1 Naive approaches, and why not

Dumping every block identifier the sender holds is O(repository), which at 7B
with a few hundred commits is millions of identifiers to exchange in order to
transfer one delta. Dumping every identifier the *receiver* holds is the same
problem mirrored.

### 3.2 What we do: walk the commit DAG

```
1. Client sends REF_LIST (its refs) and the server replies with REF_LIST (its refs).
2. Client sends HAVE: the identifiers of commits it already has, walked
   backwards from its own refs, in exponentially widening strides
   (1, 2, 4, 8, … generations back), capped at 256 identifiers per round.
3. Server intersects. If it finds a common ancestor, it replies WANT with the
   set of objects reachable from the requested ref but NOT reachable from the
   common ancestor. If not, the client sends another HAVE round, deeper.
4. After 8 rounds with no intersection, the client declares no common history
   and the server sends the full closure.
```

The stride pattern is the same idea git uses, and for the same reason: the
common case is two peers a handful of commits apart, and that costs one round
trip with a few identifiers.

Reachability is computed exactly as in SPEC 11 §6 — commit → manifest +
topology, manifest → header block and every group's `block`/`diff_block`. The
**ancestor invariant** (SPEC 10 §4.3) is what makes this correct: because a
delta's base is guaranteed to be an ancestor of the commit containing it, the
closure over "commits the receiver lacks" is guaranteed to include every base
those commits' deltas resolve against. Without the invariant, a receiver could
accept a manifest whose base block lives on a branch it does not have, pass
every per-object hash check, and be unreadable.

The sender MUST additionally filter the WANT set against the receiver's
declared `HAVE` blocks for the objects it names, so that a block shared between
two branches is not sent twice.

---

## 4. Block transfer

### 4.1 `BLOCK_HDR`

```
oid[32] │ u8 kind │ u64 payload_len │ u32 chunk_count │ chunk_digest[32] × chunk_count
```

The chunk digests are sent up front, before the data. That is what makes an
interrupted transfer resumable at chunk granularity rather than block
granularity, and it lets the receiver reject a block early.

### 4.2 `BLOCK_DATA`

```
oid[32] │ u64 offset │ bytes
```

Offsets are into the uncompressed payload and MUST be chunk-aligned except for
the final chunk.

### 4.3 `BLOCK_END`

```
oid[32]
```

On receipt the receiver:

1. verifies every chunk against the digests from `BLOCK_HDR`;
2. verifies `oid == blake3(frame(kind, payload))` — the full `verify_block`
   path, not the fast path, because this is untrusted input;
3. writes the object with `atomic_write` (SPEC 11 §3.1) from
   `.synapsefs/incoming/` into `objects/`.

A failure at any step discards the block and sends `ERROR`. Nothing partial
becomes visible.

### 4.4 `REF_UPDATE` (push only)

```
u8 ref_len │ ref_name │ oid_expected[32] │ oid_new[32]
```

Sent **after** every object it depends on has been acknowledged. The server
applies it as a compare-and-swap (SPEC 11 §4) and rejects a non-fast-forward
unless `--force` was passed, which sets `oid_expected` to zeroes.

Objects before refs, on the wire exactly as on disk. A crash mid-push leaves
the receiver with unreferenced objects, which are garbage rather than
corruption.

---

## 5. Resumability

The PS requires an interrupted sync to resume without re-transferring and
without corrupting either side.

**There is no transfer journal, and that is a design decision, not an
omission.** Content addressing already gives us everything a journal would:

- Objects fully received are in `objects/` and are self-identifying.
- Objects partially received are in `.synapsefs/incoming/<oid>.part` with their
  chunk digests from `BLOCK_HDR`. On resume, the receiver verifies the chunks
  it already holds, discards the first bad one and everything after it, and
  asks for the remainder from that offset.
- The want set is **recomputed** on resume from the receiver's current state.
  Anything that arrived is now a `HAVE` and is not requested again.

The state that a journal would have tracked is derivable from the store, and
derived state cannot disagree with reality the way recorded state can. The
prototype had a `resume.py` doing this bookkeeping; deleting it removed a whole
class of "the journal says we have it but we don't" bugs.

`resume_token` in `modules/net/include/synapsefs/net/resume.hpp` is therefore a
thin thing: `{session_nonce, ref_name, oid_target}`. It exists so a resumed
connection can skip re-negotiation when nothing changed, and correctness does
not depend on it.

---

## 6. Errors

```
ERROR: u16 code │ u8 msg_len │ msg
```

| Code | Meaning |
|------|---------|
| 1 | Protocol version mismatch |
| 2 | Malformed frame |
| 3 | Unknown object requested |
| 4 | Hash mismatch on received block |
| 5 | Ref update rejected (non-fast-forward) |
| 6 | Repository locked |
| 7 | Ancestor invariant violated by received manifest |

Codes map onto `sfs::core::ErrKind` (see `docs/interfaces/errors.md`) so a
network failure surfaces to the user in the same vocabulary as a local one.

---

## 7. Test hooks

| Assertion | Test |
|---|---|
| Frame encode/decode round trip, including the 8 MiB boundary | `modules/net/tests/test_framing.cpp` |
| Have/want finds the common ancestor in one round for near peers | `modules/net/tests/test_havewant.cpp` |
| A block on both branches is sent once | `modules/net/tests/test_havewant.cpp` |
| Kill mid-transfer, resume, no re-transfer, both sides verify | `tests/sync_interrupt.cpp` |
| Received manifest violating the ancestor invariant is rejected | `modules/store/tests/test_dag_walk.cpp` |
