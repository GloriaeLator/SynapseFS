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
