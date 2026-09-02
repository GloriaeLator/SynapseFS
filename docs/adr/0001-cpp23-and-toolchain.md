# ADR 0001 - C++23, GCC 14, CMake + Ninja, Linux only

## Context

The initial prototype was Python with C where measured. The mount daemon's read path is the throughput metric (8% of the grade) and was
already heading for a C extension, and the alignment engine's inner loops at 7B
are not something to run under an interpreter with a 16 GB RAM cap.

## Decision

C++23. GCC 14 as the reference compiler (Ubuntu 24.04's `g++-14`), Clang 18
supported for `clang-tidy` and the sanitizers. CMake ≥ 3.25 with presets,
Ninja. **Linux only** - the PS scopes the target OS.

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

- The project needs GCC 14 or Clang 18. The `Containerfile` provides both a
  clean-environment build.
- No exceptions across module boundaries. `std::expected` everywhere; see
  `docs/interfaces/errors.md`.
- We lose the prototype's `scipy.optimize.linear_sum_assignment` and must
  implement the LAP solver ourselves (ADR 0004).
- Python does not disappear: fixtures, the e2e harness stay
  Python, because that is where `safetensors` and `torch` live and the PS
  requires `load_file()` to work against our mount unmodified.
