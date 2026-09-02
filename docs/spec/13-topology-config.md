# SPEC 13 - Topology and permutation groups

## 1. Two representations

There are two related-but-different schemas here, and the doc structure
matters because they live in different modules:

- **`core::Topology`** - the in-memory value type and its structural
  validator, defined in `modules/core`. No JSON (de)serialization code for
  it lives in `core` or `format`.
- **The `config.json` a user supplies to `sfs commit --topology`, and the
  `Topology` object that gets written to the repo** - both parsed/produced
  only by `modules/align/src/topology_parser.cpp`. Its wire schema does
  **not** use the same field names as the in-memory `core::Topology`
  struct's own member names; see §3.

## 2. Input: `config.json`

A plain layer-chain description, matching a straightforward
`nn.Sequential`-style model:

```json
{
  "arch": "example-mlp",
  "layers": [
    { "type": "linear", "weight": "0.weight", "bias": "0.bias", "in": 8, "out": 16 },
    { "type": "relu" },
    { "type": "linear", "weight": "2.weight", "bias": "2.bias", "in": 16, "out": 10 }
  ]
}
```

Recognized `layers[].type` values (`topology_parser.cpp`): `linear`,
`conv2d`, `batchnorm2d`, `layernorm`, `relu`, `maxpool2d`, `flatten`,
`dropout`, `avgpool2d`, `adaptiveavgpool2d`. In strict mode (the default),
any other type is a hard parse error. A layer entry supplies the tensor
names it owns (`weight`/`bias`/etc.) plus enough shape information for the
parser to bind axes.

`config.json` is optional. If none is supplied and none is auto-discovered
next to the checkpoint file, `sfs commit` stores a literal `"{}"` as the
topology object and every tensor becomes its own pinned singleton group
(no alignment is attempted). If a file *is* found but fails to parse, that
is a hard commit failure - absence is tolerated, corruption is not.

There is no structural inference of groups from tensor names or shapes
alone; the parser only acts on what `config.json` states explicitly.

## 3. Output: the stored `Topology` object

The `Topology` object written to the repo (`ObjectKind::Topology`) has this
JSON shape (confirmed against `tests/golden/topology_cnn.json`):

```json
{
  "format_version": 1,
  "source": { "kind": "hf_config", "arch": "...", "digest": "b3:..." },
  "perm_groups": {
    "<group name>": { "size": N, "pinned": true|false }
  },
  "tensors": {
    "<tensor name>": { "axes": [ { "dim": N, "group": "<group name>", "block": N } ] }
  }
}
```

Note the JSON key is **`perm_groups`**, not `groups` - the in-memory
`core::Topology::groups` field is renamed at this JSON boundary. There is
no `core`/`format` code that reads or writes this shape directly; only
`topology_parser.cpp` does.

## 4. How groups and axis bindings are built

For each parameterized layer the parser encounters, it unions axis handles
in a weighted union-find (`align::AxisUnionFind`, path halving + union by
rank):

- a layer's input axis is unioned with the *previous* layer's output axis
  (sequential dependency),
- a bias vector is unioned with its own layer's output axis,
- BatchNorm/LayerNorm affine parameters are unioned with the preceding
  layer's output axis,
- the very first layer's input axis and the last parameterized layer's
  output axis are **pinned** (only the identity permutation is legal),
- any axis not touched by the config gets a fresh pinned singleton group,
  so unmodelled tensors (e.g. `num_batches_tracked`) still round-trip
  through the identical reconstruction path.

`finalize()` sets a group's `size` to the minimum axis length among its
members, and each member axis's `block` factor to `axis_length / size`. A
non-integer result is a hard `BlockFactorMismatch` - never silently
truncated or rounded.

## 5. `AxisBinding.block` and blocked permutations

`AxisBinding { dim, group, block }` requires `shape[dim] == group.size *
block`. `core::expand_permutation(perm, block)` expands a group-level
(unit-level) permutation into an element-level one: for each output
position `p` in the permutation, it emits `block` consecutive output
indices `p*block .. p*block+block-1`. Every structural case the aligner
handles - residual-block sharing, a flatten before a linear layer, grouped
convolution, a pinned classifier head - reduces to this one function; there
is no separate code path per case.

## 6. Validation

`Topology::validate(shapes)` checks: every group non-empty; every tensor
the topology references exists in the given `shapes` map
(`ErrKind::TopologyIncomplete` otherwise); every axis `dim` is in range and
names a group that exists (`ErrKind::TopologyParse` otherwise); and
`shape[dim] == group.size * block` exactly for every binding
(`ErrKind::BlockFactorMismatch` otherwise). `core::is_valid_permutation`
(a `[0,n)` bijection check) must be applied to any permutation parsed from
an untrusted diff artifact before it is used to index anything - this is a
memory-safety boundary, not just a correctness check, and is enforced in
`format::read_permutation` (see [SPEC 12](12-residual-format.md#3-permutation-encoding)).
