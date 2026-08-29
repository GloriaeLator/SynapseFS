# SPEC 13 — Topology and permutation groups

**Status:** normative · **Format version:** 1

How a checkpoint's architecture is normalised into permutation groups, what the
stored topology object contains, and the rules a parser must follow.

---

## 1. What problem this solves

A permutation of layer *l*'s output units, matched by the same permutation on
layer *l+1*'s input units, leaves the network's function unchanged and changes
100% of the file's bytes. To exploit that we need to know, for every tensor
axis, **which permutation applies to it**.

The prototype expressed this with pairwise links (`permute_input_from`,
`permute_dim`, `permutable`, `permute_source`). Two things broke it:

**Pairwise links cannot express a shared group.** A ResNet block adds a skip
connection onto the main path, so the block's output channels, the shortcut's
output channels and everything downstream must all carry the *same*
permutation. That is a set, not a chain of pairs.

**Pairwise links carry no blocking factor.** After the last pool, a
`conv2d_4` with 16 output channels feeds a `linear_9` whose input axis is 1024
(16 channels × 8 × 8). Applying a 16-element permutation to a 1024-wide axis is
legal numpy: `W[:, p]` silently returns a `(10, 16)` array. No exception, no
warning, a model that computes garbage — measured output error 113.667 against
4.96e-05 for the correct expansion.

The fix is one group id per tensor axis plus a blocking factor. It is the
output of a union-find over parameter axes, which the parser has to build
anyway.

---

## 2. The topology object

Stored as an object of kind `topology` (SPEC 10 §1.3) and referenced by every
commit, so a **pulled** repository can decode itself without the producer's
`config.json`.

```jsonc
{
  "type": "synapsefs.topology",
  "format_version": 1,
  "source": {"kind": "hf_config", "arch": "resnet", "digest": "b3:5b6a…"},
  "perm_groups": {
    "in":  {"size": 3,  "pinned": true},
    "g0":  {"size": 8,  "pinned": false},
    "g4":  {"size": 16, "pinned": false},
    "out": {"size": 10, "pinned": true},
    "s1":  {"size": 1,  "pinned": true}
  },
  "tensors": {
    "0.weight": {"axes": [{"dim": 0, "group": "g0",  "block": 1},
                          {"dim": 1, "group": "in",  "block": 1}]},
    "0.bias":   {"axes": [{"dim": 0, "group": "g0",  "block": 1}]},
    "1.weight": {"axes": [{"dim": 0, "group": "g0",  "block": 1}]},
    "1.num_batches_tracked": {"axes": [{"dim": 0, "group": "s1", "block": 1}]},
    "9.weight": {"axes": [{"dim": 0, "group": "out", "block": 1},
                          {"dim": 1, "group": "g4",  "block": 64}]}
  }
}
```

### 2.1 `perm_groups`

| Field    | Type    | Req | Notes |
|----------|---------|-----|-------|
| `size`   | integer | ✔   | Number of units in the group. Every axis referencing it must have length `size × block`. |
| `pinned` | boolean | ✔   | `true` means the identity permutation is the only legal one. |

Pinned groups are the ones where unit identity carries external meaning: the
input channels (a network's first layer sees RGB in a fixed order) and the
classifier output (class 3 means the same class in both checkpoints). Permuting
either produces a file that reconstructs correctly and a model that is wrong,
and no byte-level test would catch it. Pinning is a semantic assertion, so it
is explicit in the object and not inferred.

### 2.2 `tensors`

One entry per tensor **that the topology models**. `axes` lists only the axes
that carry a permutation; unlisted axes are untouched.

| Field   | Type    | Req | Notes |
|---------|---------|-----|-------|
| `dim`   | integer | ✔   | Axis index into the tensor's shape. |
| `group` | string  | ✔   | Key into `perm_groups`. |
| `block` | integer | ✔   | Blocking factor, ≥ 1. `shape[dim] == perm_groups[group].size × block`. |

Applying a group permutation to an axis is then one function, for every case:

```cpp
// A group permutation over an axis whose entries are `block`-sized runs.
std::vector<uint32_t> expand(std::span<const uint32_t> perm, uint32_t block) {
    if (block == 1) return {perm.begin(), perm.end()};
    std::vector<uint32_t> out(perm.size() * block);
    for (size_t i = 0; i < perm.size(); ++i)
        for (uint32_t k = 0; k < block; ++k)
            out[i * block + k] = perm[i] * block + k;
    return out;
}
```

Everything falls out of that:

| Structure | Expression |
|---|---|
| Residual add | Both branches get the **same group id**. |
| Flatten into a linear layer | `block = axis_len / group_size`, **derived**, never hardcoded. |
| Grouped / depthwise conv | Smaller groups, same mechanism. |
| Classifier, input stem | `pinned: true`. |
| BatchNorm following a conv | Same group as the conv's output, `block = 1`. |
| A tensor nothing models | Its own singleton group, `size = 1`, `pinned: true`, identity. |

---

## 3. Coverage rule

**Every tensor present in the checkpoint MUST appear in the manifest's buffer
layout (SPEC 10 §4.2), whether or not the topology models it.**

The topology and the buffer layout are different things and it is worth being
explicit about why. The topology says what may be permuted. The buffer layout
says what bytes exist. On the CNN fixture, 16 tensors are in the checkpoint and
14 are in the topology; the two missing ones are
`1.num_batches_tracked` and `5.num_batches_tracked`. A design where the
manifest is derived from the topology loses them silently, and the loss is
invisible to any test that only asks whether the model still loads.

Unmodelled tensors are assigned a singleton pinned group by the parser so they
round-trip through the identical code path as everything else. No special case
in the reconstructor.

---

## 4. Parser requirements

The parser takes a checkpoint plus its `config.json` and produces a topology
object. It is our own code; the PS explicitly expects teams to write one
against "this general shape" rather than depend on a library.

1. **Read the safetensors header first.** The set of tensors and their shapes
   come from the file, not from the config. The config supplies structure.
2. **Union-find over axes.** Create a set per (tensor, axis). Union an axis with
   another whenever the architecture forces them to share a permutation:
   consecutive layers, norm parameters with their preceding conv/linear, both
   branches of a residual add, and a flatten's grouped input axis with the
   producing conv's output axis.
3. **Derive blocking factors**, never hardcode them. For an axis of length `L`
   joined to a group of size `S`: `block = L / S`, and `L % S != 0` is a parse
   error naming both tensors.
4. **Pin** the first layer's input axis and the final classifier's output axis.
5. **Assign singleton groups** to every tensor in the file not otherwise
   covered.
6. **Validate**: every axis reference resolves; every group is non-empty; for
   every listed axis `shape[dim] == size × block`; no tensor is listed twice.
7. Emit canonical JSON (SPEC 10 §1.4) and store it.

The parser MUST fail loudly on anything it cannot model. Silently omitting a
tensor from the topology is safe (it becomes a singleton); silently guessing a
group is not.

### 4.1 Scope

MLPs and CNNs of ResNet shape — conv, batch/layer norm, linear, residual adds,
pooling. Transformers are explicitly out of scope in the PS and are not in the
graded fixtures; attention head permutations and RoPE interleaving would need
their own group semantics and we do not claim them. See
`docs/tradeoffs.md`.

---

## 5. Test hooks

| Assertion | Test |
|---|---|
| Both sample configs parse | `modules/align/tests/test_topology_parser.cpp` |
| `linear_9` comes out as `{"dim": 1, "group": "g4", "block": 64}` | same |
| Every tensor in the fixture appears in the buffer layout | `modules/format/tests/test_st_roundtrip.cpp` |
| A mismatched `size × block` is a parse error, not a wrong answer | `modules/align/tests/test_topology_parser.cpp` |
| Golden topology still parses | `tests/golden/topology_cnn.json` |
