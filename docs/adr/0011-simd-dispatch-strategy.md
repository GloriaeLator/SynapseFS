# ADR 0011 - Compile every ISA, choose at runtime from CPUID

## Context

The residual kernel - XOR or zigzag-add over a frame, plus the byte-plane or
bitshuffle inverse - is the innermost loop of every read. It runs on every page
fault, at every level of a delta chain. It is embarrassingly vectorisable.

It also runs on a machine we do not own. Building with `-mavx512f` produces a
binary that SIGILLs on anything older than Skylake-SP.

## Decision

**Compile each kernel in its own translation unit with its own ISA flags; pick at runtime from CPUID.**

- `modules/codec/src/kernels/residual_scalar.cpp` - no ISA flags, and compiled
  with `-fno-tree-vectorize` so it stays a literal reference implementation.
- `residual_avx2.cpp` - `-mavx2 -mfma -mbmi -mbmi2`.
- `residual_avx512.cpp` - `-mavx512f -mavx512bw -mavx512vl -mavx512dq`.
- `dispatch.cpp` - no ISA flags. Reads CPUID once (`util/cpuid.hpp`), including
  `xgetbv` to confirm the OS actually saves the wide state, and publishes a
  function pointer.

`cmake/SimdKernels.cmake` attaches the flags to **source files**, never to a
target and never globally. That constraint is the whole ADR: one stray global
`-mavx2` and the binary is unshippable, and nothing in the build will tell you.

Dispatch happens **once per frame**, not per unit, so the indirect call is
noise next to the `ZSTD_decompress` beside it.

AVX-512 additionally checks for downclocking-prone hosts only by measurement,
not by heuristic: `bench/residual_codec.cpp` reports throughput per ISA, and if
AVX-512 is not actually faster on the demo machine, `SFS_FORCE_ISA=avx2`
pins it and the number goes in `docs/benchmarks.md`.

## Consequences

- `SFS_ENABLE_SIMD=OFF` and the `no-simd` preset build the scalar path only.
  CI builds both, because `test_kernel_equivalence.cpp` treats the scalar
  kernel as the oracle and asserts every enabled ISA produces **identical
  bytes** on random inputs including unaligned tails.
- `SFS_FORCE_ISA=scalar|avx2|avx512` overrides dispatch at runtime, for
  benchmarking and for bisecting a suspected kernel bug.
- Someone will eventually put an ISA flag on a target. The reviewer's job on
  any `cmake/` change is to check that. It is written here so the check has a
  reference.
- BLAKE3 does its own dispatch internally, which is one reason it is vendored
  from upstream rather than taken from a package manager (ADR 0002).
