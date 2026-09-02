# SPEC 12 - Diff artifact and residual format

## 1. Artifact layout

```
[8-byte LE header_len][JSON header, header_len bytes][payload]
```

deliberately mirroring the safetensors `[8-byte len][JSON][data]` shape, so
the same low-level reader primitive parses both. Every `off`/byte-range
field inside the JSON header is relative to byte 0 of the payload section
(i.e. to `8 + header_len`), not to the start of the artifact.

`kDiffMagic = "SYNDIFF"`. A missing or mismatched `"magic"` field is a hard
`ErrKind::MalformedObject` - there is no fallback interpretation.

## 2. `DiffHeader` JSON

```json
{
  "magic": "SYNDIFF",
  "format_version": 1,
  "group": "<permutation group name>",
  "codec": "none" | "zstd",
  "permutation": { "kind": "identity" }
                | { "kind": "explicit", "n": N, "dtype": "u16"|"u32", "off": N, "len": N },
  "tensors": [
    {
      "name": "<tensor name>",
      "shape": [ ... ],
      "dtype": "F16" | "BF16" | "F32" | "F64" | "I8" | ... | "BOOL",
      "residual": "raw" | "xor_after_permute" | "zigzag_after_permute",
      "transform": "none" | "byteplane" | "bitshuffle",
      "frames": [
        { "units": [begin, end], "off": N, "len": N, "digest": "<64 hex>" }
      ]
    }
  ],
  "alignable": true | false,
  "alignment": { "method": "weight_matching_lap", "cost_raw": F, "cost_normalized": F }
}
```

## 3. Permutation encoding

`PermutationRef { kind, n, width, off, len }`:

- `kind = "identity"` - **zero payload bytes**. The common case for a
  fine-tune where no meaningful reordering was recovered (or none was
  needed).
- `kind = "explicit"` - `n` little-endian integers of `width` bytes each
  (`width = 2` / `dtype: "u16"` if `n ≤ 65536`, else `width = 4` /
  `dtype: "u32"`), `len` must equal `n × width`, and `[off, off+len)` must
  fit inside the payload.

On read, an explicit permutation is validated as a true bijection over
`[0, n)` (`core::is_valid_permutation`) *before* any code indexes with it.
A non-bijective permutation is rejected with `ErrKind::InvalidPermutation`
rather than being used - this is the one place in the format where a
corrupt object could otherwise turn into an out-of-bounds read instead of a
clean, detectable failure.

## 4. Frames

`FrameIndexEntry { unit_begin, unit_end, off, len, digest }` - `unit_begin`/
`unit_end` are a half-open range of permutation units (rows), not bytes;
`off`/`len` locate the frame's (possibly compressed) bytes inside the
payload; `digest` is a 32-byte BLAKE3 digest, hex-encoded to 64 characters
in JSON, of the **fully reconstructed target bytes** for that frame - i.e.
computed *after* decompression and *after* the residual (XOR or zigzag) has
been applied. This is deliberate: it means tamper detection covers the
entire reconstruction path for a frame, not just what was written to disk.

Frames for one tensor must tile `[0, group_size)` contiguously with no gaps
or overlaps (`TensorDiff::validate_tiling`); frame boundaries are placed at
whole-unit granularity, sized to approximately `frame_bytes` (config
default 128 KiB) - `frame_units = max(1, frame_bytes / unit_bytes)`, so a
single oversized unit still gets its own frame. This is **fixed-size**
tiling by target byte count, not content-defined chunking.

## 5. Residual kinds - never a floating-point subtraction

`ResidualKind`:

- **`raw`** - the frame's bytes are the target bytes verbatim; no base
  tensor is read at all.
- **`xor_after_permute`** - `target[i] = base[permutation[i]] XOR
  residual[i]`, a plain bytewise XOR over the dtype's raw bit
  representation. Self-inverse, so encode and decode share one code path.
- **`zigzag_after_permute`** - the encoder's default. `target = base +
  zigzag_decode(residual)`, where the delta is computed as an unsigned
  wraparound subtraction of the raw integer bit pattern and then
  zigzag-packed (`z = (delta << 1) XOR signmask`). Chosen, per code
  comments, as the best of a six-candidate ratio/throughput comparison over
  `{xor, zigzag} × {none, byteplane, bitshuffle}`.

Both non-`raw` kinds are exact bijections over the raw bits - there is no
lossy approximation or float re-encoding anywhere in this path. Fp16/bf16
values are only ever widened to float for *alignment cost computation*
(`modules/align`); reconstruction never performs that conversion.

`Transform` (`none`/`byteplane`/`bitshuffle`) is an optional, lossless,
same-size byte rearrangement applied before zstd compression, chosen and
recorded per-artifact - a reader must not assume a fixed transform.

## 6. Compression

`Codec::Zstd` frames are compressed with one `ZSTD_compress2`/
`ZSTD_decompress` call per frame - no streaming context is kept across
frames, specifically so a range read can start decompression at an
arbitrary frame boundary without touching earlier frames. Zstd's own
checksum is disabled (`checksum=false`); the frame digest above is the
system's only integrity check for this data. `Codec::None` stores frame
bytes uncompressed. `EncodeOptions::codec` defaults to `Codec::Zstd`, so
this is the compression that actually runs for a `Diff` object's payload -
distinct from, and not gated by, the unimplemented `Compression` field on
the loose-object *container* (see [SPEC 10](10-object-model.md#2-loose-object-container-formatobjectheader)
§2 and [`storage_format.md`](../storage_format.md#loose-object-container)).

## 7. Chunking is unrelated to residual frames - two different mechanisms

Do not confuse this section's frames with the loose-object chunk-digest
table in [SPEC 10](10-object-model.md#2-loose-object-container-formatobjectheader):
object chunking is a fixed 64 KiB (default) integrity-verification
granularity applied to *every* stored object regardless of kind; residual
framing is a fixed ~128 KiB (default) *decompression/read* granularity
applied only inside a diff artifact's payload. Both are fixed-size; neither
is content-defined.
