# Testing

Four things must never break. Everything else is diagnostics that tell you
*where* one of them broke.

1. `sha256(reconstruct(A, diff(A, B))) == sha256(B)` — byte-exactness.
2. `mmap` works against the mount.
3. `verify` runs standalone and catches every single-byte tamper, and names it.
4. `kill -9` at any point leaves the repository clean or explicitly refusing.

These are the four whose failure makes a solution fail outright rather than
score lower. They are green in CI on every push, from Day 1, even while
everything around them is stubbed — byte-exactness drifts silently, and you
want to know the hour it breaks, not on Day 6.

---

## Layers

| Layer | Where | Runs |
|---|---|---|
| Unit | `modules/*/tests/` | every push, seconds |
| Integration | `tests/` | every push |
| End-to-end (Python) | `tests/e2e.py` | every push, needs fixtures |
| Crash matrix | `tests/crash_matrix.cpp` | nightly + continuously from Day 4 |
| Tamper matrix | `tests/tamper.cpp` | every push |
| Concurrency | `tests/concurrent_readers.cpp`, `test_blockcache_race.cpp` | every push; under TSan nightly |
| Scale | fixture-size run with RSS assertions | nightly |
| Benchmarks | `bench/` | manual, from the `release` preset only |

One Catch2 executable per test **file**, not per module: a crash in a FUSE test
should not take the other results with it.

Labels: `unit`, `e2e`, `concurrency`, `slow`, plus the module name.

```bash
ctest --preset unit
ctest --preset dev -L concurrency
ctest --preset dev -R 'align\.' --output-on-failure
```

---

## The test ladder for the mount

Rungs, in order. Do not skip one because the next passes.

1. `getattr` reports the right size; `readdir` lists the file.
2. Sequential `read()` of the whole file matches a checkout, byte for byte.
3. Random `read()` at arbitrary offsets and sizes matches.
4. Deliberate edge cases: the header/buffer boundary, a tensor boundary, a
   1-byte read, a read at EOF, a read past EOF.
5. `mmap` of the whole file matches.
6. `safetensors.torch.load_file()` succeeds and returns correct tensors.
7. 4–8 concurrent loaders, all byte-identical, no crash.
8. Cold page cache, peak RSS within budget.

Rung 4 is where most bugs live and it is the one people skip.

---

## Fault injection

The seams exist so that failures can be *caused*, not waited for
([interfaces](interfaces/interfaces.md)):

| Double | Enables |
|---|---|
| `FaultInjectingStore` | Flip any byte in any object at any chain position → `tests/tamper.cpp` |
| `CrashingStore` | Fail the Nth write, or die between rename and directory fsync → `tests/crash_matrix.cpp` |
| `FlakyTransport` | Cut a transfer at byte N, deterministically → `tests/sync_interrupt.cpp` |

Deterministic fault injection beats racing a real `kill -9`, because a failing
case is reproducible. The crash *matrix* uses the double; the crash *harness*
uses a real `kill -9` in a loop, and both are needed — the double proves each
stage is handled, the harness finds the stage nobody thought of.

## The tamper matrix

Corrupt one byte in each object kind at each position in a chain, and assert
every case is detected **and named**:

| Object kind | Depth 0 | Mid-chain | Tip |
|---|---|---|---|
| `raw` block | ✓ | ✓ | ✓ |
| `diff` artifact header | ✓ | ✓ | ✓ |
| `diff` frame payload | ✓ | ✓ | ✓ |
| `header` block | ✓ | — | — |
| `manifest` | ✓ | ✓ | ✓ |
| `commit` | ✓ | ✓ | ✓ |
| `topology` | ✓ | ✓ | ✓ |

"Named" means the error identifies the object and, where applicable, the chunk
or frame index. `verify` exiting 4 with "something is wrong" is a fail.

## The crash matrix

Kill at every stage of `commit`, `push` and `merge`; each must recover cleanly
or refuse to proceed. Tabulate. Keep the iteration counter — it is a
presentation slide, and "we ran 40,000 iterations" is a different claim from "we
handle crashes".

## Golden objects

`tests/golden/` holds one canonical example of each object kind. Two jobs:

- they must still parse (regression against accidental format drift);
- they must round-trip to their own identifier (canonical JSON, SPEC 10 §1.4).

`tests/golden/validate.py` runs in CI's cheap `lint` job, before anything is
compiled.

## What CI runs

| Job | Gate |
|---|---|
| `lint` | clang-format, no committed checkpoint blobs, golden objects valid |
| `build-and-test` | `dev` and `no-simd` presets, unit tests, `sfs --version` |
| `round-trip` | init → commit → checkout byte-exact; checkout == mount; tamper matrix |
| `sanitizers` | asan on push, tsan on the concurrency label |
| `docker` | clean-environment build from nothing |

A PR that fails `round-trip` does not merge, whatever else it fixes.
