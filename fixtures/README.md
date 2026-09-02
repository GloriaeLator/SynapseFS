# Fixtures

Checkpoints are **generated, never committed**. That is a listed deliverable in
the problem statement, and CI fails the `lint` job if a `.safetensors` shows up
in git.

```bash
make fixtures-small     # MLP + ResNet, a few hundred MB
make fixtures           # adds the 7B pair — large and slow
```

`manifest.toml` is the catalogue: what each fixture is and why it exists.

Two things worth knowing about how these are written:

**The header is deliberately awkward.** `generate_synthetic_checkpoint.py`
writes `__metadata__` and lets safetensors choose its own key order, which is
not alphabetical and not the order we would pick. That is the point: a
round-trip test against a tidy file we generated ourselves proves nothing,
because the evaluator's checkpoints did not come from our writer.

**`permute.py` produces the demo.** It applies a valid permutation to a
checkpoint, giving a file that computes exactly the same function and shares
almost no bytes. That pair is the headline number in the presentation, so
generate it early and know what the byte-difference count is.

**Two different "topology" files, two different consumers.** `gen_mlp.py`
and `gen_cnn.py` each write two sidecars per architecture, and they are
NOT interchangeable:

- `<prefix>_topology.json` — the spec 13 §2 `perm_groups`/`tensors` shape
  (axis -> permutation-group mapping). Consumed by `permute.py` and
  `bench/residual_codec.cpp` only.
- `<prefix>_layers_config.json` — the plain `{"layers": [...]}` shape that
  `align/topology_parser.cpp` actually implements. This is what you pass to
  **`sfs commit --topology`**. Passing `<prefix>_topology.json` there fails
  with `topology parse error: config.json has no 'layers' array`, since that
  file has no top-level `layers` key.
