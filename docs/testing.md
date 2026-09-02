# Testing

This describes what actually exists in `tests/` and `modules/*/tests/`.

## Top-level integration tests

Only two files exist directly under `tests/`:

- **`tests/byte_identity.cpp`** - 3 cases, tagged `[integration][byte_identity]`:
  a hand-constructed multi-axis-permutation scenario (a tensor whose two
  axes belong to two different groups, one already resolved) round-trips
  byte-exact through `plan_commit_groups` + `read_range`, and asserts a
  pinned-axis tensor correctly stays `Full` while alignable ones become
  `Delta`; an oversized/unaligned delta correctly falls back to `Full`; a
  group whose secondary dependency has no available info forces `Full` and
  correctly cascades that decision to dependents. These tests build a
  `align::MatchReport` by hand rather than calling the real
  `align::Matcher` (which needs LibTorch).
- **`tests/byte_identity_cnn.cpp`** - 1 case, `[integration][byte_identity][cnn]`:
  a two-conv CNN with a non-pinned secondary axis (dimension 1, not the
  last), using the *real* `align::topology_parser` on a literal
  `config.json`. Documented as a regression test for a real bug where the
  multi-axis permutation machinery had only ever been exercised against
  rank-2 (Linear) tensors and failed on conv weights with a short-read
  error before a fix landed in `commit_planner.cpp`/`reconstruct.cpp`.

## What `tests/CMakeLists.txt` expects but does not have

The test harness's CMake glob and comments originally referenced five
missing integration tests and an end-to-end Python suite -
`tests/tamper.cpp`, `tests/crash_matrix.cpp`, `tests/concurrent_readers.cpp`,
`tests/sync_interrupt.cpp`, `tests/e2e.py` - with `TEST_PREFIX "e2e."`,
`slow`/`concurrency` labels, and conditional Python3 registration already
wired up for them. **`tests/tamper.cpp` now exists** (see below); the other
four still don't (confirmed repo-wide, not just under `tests/`). A sixth,
separate phantom reference also exists: `modules/mount/src/daemon.cpp`'s
comments cite `test_blockcache_race.cpp` for concurrent-reader coverage -
`modules/mount` has no `tests/` directory at all, and that file doesn't
exist either.

The supporting harness code - `tests/common/temprepo.hpp`'s `TempRepo` and
`tests/common/fault_inject.hpp`'s `FaultInjectingStore`/`CrashingStore` -
had **zero implementation and zero callers anywhere in the repository**
until this pass: `tests/common/` had only the four headers plus
`harness.cpp`, no `temprepo.cpp`/`fault_inject.cpp`, and every existing
test built its own throwaway repo/fakes inline instead of going through
them. `tests/common/harness.hpp`/`harness.cpp`'s synthetic-checkpoint
builders (`write_synthetic_checkpoint`, `plant_permutation`, `perturb`,
`compare_files`, `sha256_file`) were real and complete but, despite being
compiled into `sfs_test_common` all along, had **zero real callers either**
- `modules/codec/tests/test_byte_identity.cpp` only *mentions*
`harness.hpp`'s rationale in a comment, it never `#include`s it or calls
anything in it, and `tests/byte_identity.cpp` (the top-level test of the
same name) builds its own in-memory fakes too. (An earlier pass of this
file claimed `test_byte_identity.cpp` was a real caller - it was not; this
correction supersedes that.) `tests/tamper.cpp` is now the first real
caller of `harness.hpp`, `temprepo.hpp`, and `fault_inject.hpp` all three.

Treat "crash-window recovery," "concurrent-reader safety," and "sync
interruption" as still **exercised only indirectly** or not at all -
`tamper.cpp`'s two `CrashingStore` cases are a first building block toward
`crash_matrix.cpp`, not a substitute for it (see the note in `tamper.cpp`
on what a wrapper of `core::IBlockStore` can and cannot simulate about the
real write/rename/fsync crash windows). `concurrent_readers.cpp` and
`sync_interrupt.cpp` remain entirely unwritten, as does `e2e.py`
(`scripts/e2e.sh` covers adjacent but non-overlapping ground - see
`docs/known-gaps.md`).

## Golden fixtures (`tests/golden/`)

`commit.json`, `manifest.json`, `diff_artifact.json`, `topology_cnn.json`
are pretty-printed **examples**, not literal canonical wire bytes - the
accompanying `validate.py` (dependency-free, run in CI's lint job before
compilation) canonicalizes before comparing, and its own README states the
oids shown are illustrative, not real digests.

**This drift is now fixed.** `commit.json` and `manifest.json` used to
carry a top-level `"type"` field (e.g. `"synapsefs.commit"`) that
`validate.py` required, even though
`format::Commit::to_canonical_json()`/`Manifest::to_canonical_json()` never
emit one and `Commit::parse()`/`Manifest::parse()` never require one - the
mandatory round-trip byte-comparison those parsers perform would have
rejected the fixture's own canonicalized bytes with
`ErrKind::CanonicalizationMismatch`, because the re-serialized JSON (no
`"type"`) would not match the input (which had one). The field has been
removed from both fixtures, and `validate.py`'s check was flipped: it now
actively *fails* if `"type"` is present, so this can't silently drift back.
Separately, `validate.py` used to assert `chain_depth ≥ 1` for every delta
group, which `Manifest::validate()` does not enforce - that assertion is
removed, replaced with a comment pointing at the real rule. Both fixtures
and `validate.py` were re-checked directly against `Manifest::validate()`
(`modules/format/src/manifest.cpp`) and `Commit::parse()`
(`modules/format/src/commit.cpp`) while fixing this, and
`python3 tests/golden/validate.py` now passes clean (4/4 golden objects
valid) - `make format-check`/the lint job is a real gate again.

`diff_artifact.json` and `topology_cnn.json` do **not** have this problem -
their field shapes (magic, `format_version`, `group`, `codec`,
`permutation`, `tensors[]`, `alignable`, `alignment` for the diff artifact;
`perm_groups`/`tensors` for topology) line up with
`format::residual_hdr.cpp` and `align::topology_parser.cpp` respectively.

## Module-level unit tests

- **`modules/align/tests/`**: `test_known_permutation.cpp` (plants a known
  permutation on a small MLP and asserts `Matcher::run()` recovers it
  exactly - caught a real sign-convention bug in `confidence::assess`
  during development), `test_mlp_end_to_end.cpp` (real `.safetensors`
  files and the real topology parser - regression test for a bug where
  only *incoming* evidence reordered a row's other axis, not outgoing),
  `test_sparse_match.cpp` (exercises the large-group fingerprint + Jacobi
  auction path above `sparse_crossover`, against real checkpoint files).
- **`modules/codec/tests/`**: `test_byte_identity.cpp` (encode -> artifact ->
  `read_range`'s Delta branch reproduces target bytes exactly, plus a
  tamper-detection case), `test_chunk_ranges.cpp` (chunk-range arithmetic
  and digest-failure attribution), `test_compress_roundtrip.cpp` (zstd +
  byteplane/bitshuffle transforms round-trip exactly at deliberately odd
  sizes), `test_kernel_equivalence.cpp` (every enabled SIMD ISA is
  byte-identical to the scalar oracle, including unaligned tails, at sizes
  `{0,1,3,15,17,31,32,33,63,64,65,127,128,129,1000,100003}` chosen to
  exercise vector-width tails), `test_permutation_validate.cpp`
  (encode/decode round-trip, identity = zero payload bytes, the u16/u32
  width boundary at `n = 65536`, rejection of a malformed permutation
  before any indexing happens), `test_snapshot_policy.cpp` (the four
  storage-decision rules, in order: no base, not alignable, chain too
  deep, delta too large relative to `snapshot_alpha`).
- **`modules/store/tests/`**: `test_gc.cpp` (the gc/mount-daemon interlock
  - a live PID marker blocks gc, a stale one does not, a malformed one
  refuses rather than guessing), `test_merge_logic.cpp` (an exhaustive
  base×ours×theirs truth table for `resolve_group`, including the
  "same diff bytes against a different base counts as different content"
  edge case, runnable without a live repository).

## Running the suite

`make test-unit` and `make format-check` are what CI runs before merge, per
the top-level README. Catch2 3 is required (`SFS_BUILD_TESTS` CMake
option, on by default) and registered via CTest/`include(Catch)`.
