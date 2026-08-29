# ADR 0001 — C++23, GCC 14, CMake + Ninja, Linux only

- **Status:** Accepted
- **Date:** 2026-08-29
- **Supersedes:** the Python prototype, which is retained in git history and in
  `research/` as the source of the measurements quoted throughout these docs.

## Context

The prototype was Python with C where measured. Two things pushed the rewrite:
the mount daemon's read path is the throughput metric (8% of the grade) and was
already heading for a C extension, and the alignment engine's inner loops at 7B
are not something to run under an interpreter with a 16 GB RAM cap.

Rewriting mid-project is expensive and we are only doing it because the
on-disk formats are frozen and carry over unchanged (SPEC 10–16). The rewrite
is of code, not of design.

## Options

1. **Keep Python, push hot paths into C extensions.** Cheapest short-term.
   Costs: two languages, two build systems, GIL contention in the FUSE daemon
   with concurrent readers, and a peak-RSS number dominated by the interpreter.
2. **Rust.** Excellent fit (slices, no aliasing bugs, good FUSE bindings). The
   team does not know it. On a four-day clock, that settles it.
3. **C++23.** The team knows C++. `std::span`, `std::expected`,
   `std::byteswap`, `std::mdspan` and designated initializers make the
   serialisation code readable, and the mount daemon is a plain threaded
   program with no runtime in the way.

## Decision

C++23. GCC 14 as the reference compiler (Ubuntu 24.04's `g++-14`), Clang 18
supported for `clang-tidy` and the sanitizers. CMake ≥ 3.25 with presets,
Ninja. **Linux only** — the PS scopes the target OS, and portability shims we
cannot test are worse than an honest `FATAL_ERROR` in `CMakeLists.txt`.

Specific C++23 features we depend on, so that the toolchain requirement is
justified rather than fashionable:

| Feature | Where | Why it matters here |
|---|---|---|
| `std::expected<T, Error>` | every fallible API | Error handling without exceptions across the FUSE boundary, where an escaping exception is a kernel-visible hang. |
| `std::span<const std::byte>` | all block/frame APIs | Non-owning byte ranges are the entire vocabulary of this project. |
| `std::byteswap`, `std::endian` | `core/endian.hpp` | On-disk little-endian, checked at compile time. |
| `if consteval`, `constexpr` containers | format constants | Frame sizes and magic numbers validated at compile time. |
| `std::jthread` + `stop_token` | daemon, thread pool | Clean shutdown of the mount daemon without a bespoke flag. |

## Consequences

- The evaluator needs GCC 14 or Clang 18. The `Dockerfile` provides both a
  clean-environment build and the proof it works; CI builds it every push.
- No exceptions across module boundaries. `std::expected` everywhere; see
  `docs/interfaces/errors.md`.
- We lose the prototype's `scipy.optimize.linear_sum_assignment` and must
  implement the LAP solver ourselves (ADR 0004). That is a real cost and is
  budgeted for.
- Python does not disappear: fixtures, the e2e harness and `research/` stay
  Python, because that is where `safetensors` and `torch` live and the PS
  requires `load_file()` to work against our mount unmodified.

## How we would know we were wrong

If by end of Day 2 the mount daemon is not serving a real checkpoint, the
rewrite is behind where the prototype already was and we should ship the
prototype with a C extension instead. That is a real checkpoint, not a
rhetorical one.
