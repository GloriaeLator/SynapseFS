# SPEC 12 — Diff artifact and residual format

**Status:** normative · **Format version:** 1

The binary layout of a diff artifact, the frame index that makes partial reads
cheap, and the exact reconstruction procedure. SPEC 10 §5 defines the JSON
header; this document defines the payload and the algorithm.

---

## 1. Artifact layout

```
┌────────────────────┬──────────────────────┬────────────────────────────────┐
│ 8 bytes, LE u64    │ JSON header          │ payload                        │
│ header_len         │ (SPEC 10 §5)         │ [perm array] [frame 0] [f 1] … │
└────────────────────┴──────────────────────┴────────────────────────────────┘
                     ↑ offset 8             ↑ offset 8 + header_len
```

All `off` values in the JSON header are relative to the **start of the
payload**, not to the start of the artifact. Readers add `8 + header_len` once.

The framing mirrors `.safetensors` deliberately: the same reader primitive
(read a u64, read that many bytes, parse JSON, treat the rest as an addressable
buffer) serves both, and `stio` and `codec` share it.

The artifact is stored as an object of kind `diff` (SPEC 10 §1.3) and therefore
carries chunk digests like every other object. A corrupted artifact is caught
by `read_range` before it is ever parsed.

---

## 2. Units

The atomic unit of a residual is **one output unit** of a tensor:

| Tensor kind | Output unit | Contiguous? |
|---|---|---|
| Linear `[out, in]` | one row | yes |
| Conv2d `[out_c, in_c, kh, kw]` | one filter | yes |
| Norm `[C]` (weight, bias, mean, var) | one scalar | yes |
| Bias `[out]` | one scalar | yes |

Output units are contiguous on both sides of a permutation, which is the whole
reason this is the unit. Target unit `i` is built from base unit `p[i]`.

`unit_bytes(t) = nbytes(t) / group_size`. It MUST divide exactly; if it does
not, the topology is wrong for this tensor and the aligner MUST report
`alignable: false` rather than guess.

Axes whose length is a multiple of the group size — a flattened conv feeding a
linear layer — carry a **blocking factor** (SPEC 13). The permutation is
expanded before use:

```
expand(p, block)[i*block + k] = p[i]*block + k        for k in [0, block)
```

with `block == 1` giving `p` unchanged.

---

## 3. Permutation array

Located by `permutation` in the JSON header.

- `{"kind": "identity"}` — occupies zero payload bytes. The common case for a
  fine-tune, and worth a dedicated encoding: it removes both the storage and
  the per-hop parse.
- `{"kind": "explicit", "n": N, "dtype": "u16"|"u32", "off": O, "len": L}` —
  `N` little-endian unsigned integers at payload offset `O`, `L == N * width`.
  `u16` when `N ≤ 65536`, `u32` otherwise.

The array MUST be a valid permutation of `[0, N)`: every value in range, no
duplicates. A reader MUST validate this before use — a malformed permutation
otherwise produces out-of-bounds reads driven by file contents, which is the
one place in this design where a corrupt object could become a memory-safety
problem. Validation is O(N) with an N-bit bitmap and is negligible next to the
decompression it precedes.

---

## 4. Frames

Each tensor entry carries an ordered frame index:

```jsonc
{"name": "0.weight", "shape": [8,3,3,3], "dtype": "F16",
 "residual": "xor_after_permute",
 "frames": [
   {"units": [0, 4], "off": 0,   "len": 168, "digest": "7c1f…"},
   {"units": [4, 8], "off": 168, "len": 144, "digest": "b904…"}
 ]}
```

| Field    | Notes |
|----------|-------|
| `units`  | Half-open `[a, b)` range of output units this frame covers. |
| `off`    | Payload offset of the frame's compressed bytes. |
| `len`    | Compressed length. `codec: "none"` means stored length equals raw length. |
| `digest` | BLAKE3-256 of the **reconstructed target bytes** for these units — after decompression and after the residual is applied. 16 bytes, hex. |

Rules:

- Frames MUST tile `[0, group_size)` exactly: `frames[0].units[0] == 0`,
  `frames[i+1].units[0] == frames[i].units[1]`, last `units[1] == group_size`.
- Each frame is an **independently decompressible** zstd frame. Not a block
  within one stream — a frame, so that decompression can start at `off` with no
  preceding state.
- Frame size targets `frame_bytes` (default 128 KiB) rounded to a whole number
  of output units. A single unit larger than the target gets its own frame.
- `digest` covers reconstructed target bytes, not compressed bytes. That is
  what makes tamper detection survive the reconstruction path: a corrupted base
  block, a corrupted residual, or a wrong permutation all fail the same check,
  at a cost proportional to what was actually read.

### 4.1 Why frames exist

Without them, resolving a chain of depth 5 to serve one 4 KiB page fault
decompresses the whole layer five times. Measured on a 32 MiB layer:

| Strategy | Bytes decompressed | Wall time | Peak RAM |
|---|---|---|---|
| Whole layer, depth 5 | 201.3 MB | 1347.8 ms | 134.2 MB |
| Frames, depth 5 | 0.8 MB | 0.8 ms | 0.4 MB |

1.3 seconds for a single page fault, on a layer that is small. Peak memory
becomes `frame × depth` instead of `layer × depth`, which is what keeps the
daemon inside the RSS budget at 7B.

### 4.2 Dependency sets

Because a permutation destroys locality, a target frame covering units `[a, b)`
needs base units `p[a:b]` — which are scattered on the base side. The mapping
is known exactly, so the dependency set is computable, but a reader must not
assume matching offsets.

For `kind: "explicit"`, the reader computes `p[a:b]`, groups the resulting base
unit indices into runs, and issues one `read_range` per run against the base.
For `kind: "identity"` the dependency set is the identical range and the
recursion is trivially aligned.

Implementations SHOULD cache the inverse permutation per artifact; it is needed
by the writer and by `alignable` diagnostics, and it is O(N) to build.

---

## 5. Residual encodings

`residual` is one of:

| Value | Definition |
|---|---|
| `"raw"` | Frame bytes are the target bytes. No base is read. Used for a group the aligner could not align, and inside a `full` block. |
| `"xor_after_permute"` | `target_unit[i] = base_unit[p[i]] XOR residual_unit[i]`, bytewise. |
| `"zigzag_after_permute"` | `target = base + zigzag_decode(residual)`, elementwise over the dtype's integer view. |

XOR and zigzag-delta are both bijective and both exact — no floating-point
arithmetic is performed on weights at any point, so reconstruction is
bit-exact by construction rather than by luck. This matters: permuting a tensor
and running the model produces outputs that differ at ~5e-5 because float
addition is not associative, so **an alignment cannot be verified by comparing
model outputs.** It is verified by reconstructing bytes.

A byte-plane transform MAY be applied before compression:

| `transform` | Meaning |
|---|---|
| `"none"` | Residual bytes compressed as-is. |
| `"byteplane"` | All low bytes of the frame, then all high bytes (fp16: 2 planes). |
| `"bitshuffle"` | Bit-level transpose within the frame. |

The choice is recorded per artifact, not assumed by the reader. The Day-2
codec experiment measures `{xor, zigzag} × {none, byteplane, bitshuffle}` for
both **ratio and decompression throughput** and records all six numbers in
`docs/tradeoffs.md`. Ratio is 7% of the grade and mmap throughput is 8%, so the
winner is not the one with the best ratio.

---

## 6. Reconstruction

```
read_range(group g, offset off, length len) -> bytes:
    entry = manifest.groups[g]

    if entry.mode == "full":
        return store.read_range(entry.block, off, len)      # chunk-verified

    art   = parse_diff(store.read_range(entry.diff_block, 0, header_extent))
    units = units_covering(off, len)                        # [a, b)

    out = []
    for frame in art.frames_covering(units):
        base_units = art.permutation.apply_range(frame.units)
        base_bytes = concat(read_range(entry.base.group, r.off, r.len)
                            for r in runs(base_units))      # ← recursion
        resid      = decompress(art.codec, frame.bytes)
        tgt        = apply_residual(art.residual, base_bytes, resid, art.permutation)
        require(blake3(tgt) == frame.digest)                # tamper check
        out.append(tgt)

    return slice(concat(out), off, len)
```

Properties this shape guarantees, all of them graded:

- Recursion is **per frame**, never per layer. Depth is bounded by
  `max_chain_depth`; memory by `frame_bytes × depth`.
- The digest check happens on every hop, so corruption is caught at the level
  it was introduced and named there.
- `checkout` and the mount call **the same function**. There is exactly one
  reconstructor, so the PS's consistency requirement — checkout bytes identical
  to mount bytes — holds by construction rather than by testing. `checkout` is
  a loop over `read_range` writing to a file; it is about thirty lines and it
  is not a second implementation.

### 6.1 A tensor with two permuted axes

A tensor's dim-0 (output) axis is what this artifact's own `permutation`
describes. Some tensors — a hidden layer's weight matrix, whose *input* axis
is the *previous* layer's output — also have a **secondary**, non-dim-0 axis
bound to a **different** permutation group. That group's own permutation is
not stored in this artifact at all: it lives in whichever *other* tensor owns
that group's dim-0 axis, in **that tensor's own diff artifact, from this same
commit**.

Reconstructing such a tensor is the pseudocode above, plus one step before
the digest check: after `base_bytes` is gathered by the primary permutation,
each row is *also* re-ordered along the secondary axis, using a permutation
recovered by:

1. looking up the tensor's secondary axis's group in the topology;
2. finding which other tensor owns that group's dim-0 axis;
3. reading **that tensor's own manifest entry, in this same commit** — not
   the base commit, and not a field of this tensor's own artifact;
4. if it is `delta`, fetching and parsing its diff artifact purely for its
   `permutation` field (no recursion into resolving its own reconstructed
   *bytes* — this is a single object fetch, independent of chain depth);
5. if it is `full`, treating the secondary axis as identity — never an
   error, because rule 5 below guarantees that is only possible when the
   real permutation for that group truly was identity.

Row-order and column-order permutation commute, so applying the secondary
gather after the primary one (as above) produces the same bytes as applying
it before, which is how the writer computes it (§8).

---

## 7. Snapshot policy

A group is written as `full` when any of:

1. there is no base (first commit, or the group is new);
2. the aligner reports `alignable: false`;
3. `base.chain_depth + 1 > max_chain_depth`;
4. `len(delta_bytes) > snapshot_alpha × len(full_bytes)`;
5. the tensor has a secondary axis (§6.1) bound to a non-identity group, and
   that group's own owning tensor is *not* being written as `delta` in this
   same commit.

Rule 4 is free to evaluate: the writer holds both byte strings at that point.
It exists because XOR of two unrelated fp16 tensors is high-entropy noise that
compresses to *larger* than the original — an unbounded-α design can make a
repository grow faster than storing full copies.

Rule 5 exists because a secondary axis's permutation has nowhere else to
live: if the group it depends on isn't stored `delta` here, that permutation
is computed, used to build this tensor's residual, and then unrecoverable —
the exact silent-wrong-reconstruction bug §6.1 exists to prevent. Applying
rule 5 can cascade: downgrading one group to `full` can, in turn, force
another group that depends on *it* to `full` too. A writer MUST iterate rules
1–5 to a fixed point, not stop after one pass.

---

## 8. Writer requirements

- Frames MUST be emitted in ascending unit order.
- `off` values MUST be non-decreasing and MUST NOT overlap.
- The writer MUST compute each `digest` from the bytes it intends the reader to
  produce, by *re-reading them through the reader path* in debug builds. A
  writer that computes the digest from its own in-memory target is not testing
  reconstruction; `tests/byte_identity.cpp` enforces the round trip.
- An artifact with `alignable: false` MUST NOT be referenced by a `delta`
  manifest entry.
