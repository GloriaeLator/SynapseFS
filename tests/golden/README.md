# Golden objects

One canonical example of each object kind. They are stored **pretty-printed for
reading**; `validate.py` canonicalises before checking, so the files here are
examples and the canonical form is what the checks are about.

Two jobs:

1. They must still parse — a regression test against accidental format drift.
2. Canonicalisation must be idempotent and stable, since the serialisation *is*
   the address (SPEC 10 §1.4).

`validate.py` runs in CI's cheap `lint` job, before anything is compiled. If you
change a format, these files change **in the same commit** as the spec and the
tests.

The hashes in these files are illustrative and are not real digests of real
content. Anything that needs a real digest belongs in a C++ test with a store.
