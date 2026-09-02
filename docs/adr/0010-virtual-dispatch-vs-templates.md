# ADR 0010 — Virtual interfaces at module seams, templates inside

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

Eight people, nine modules, four days, and a hot read path. Two forces pull
apart: the team needs stable seams to work behind in parallel, and
`read_range` is called on every page fault.

## Options

1. **Concrete types everywhere.** Fastest, no indirection, and every module
   change recompiles and re-breaks every other module. With three teams landing
   code simultaneously this is the option that loses days.
2. **Virtual interfaces everywhere**, including the inner loops. Clean seams,
   and a virtual call per frame or per unit in the residual kernels.
3. **Virtual at module seams, templates and concrete types inside.**

## Decision

Option 3, with the boundary drawn explicitly.

**Virtual** (`modules/core/include/synapsefs/core/interfaces.hpp`):

| Interface | Why it is virtual |
|---|---|
| `IBlockStore` | Loose vs packed vs an in-memory test double. Called once per frame, not per byte. |
| `IObjectSource` | Lets `checkout`, mount and `verify` share `read_range` while tests inject faults. |
| `ITensorSource` | Real safetensors file vs synthetic fixture. |
| `ILapSolver` | Exact JV vs greedy+2-swap, chosen at runtime by size (ADR 0004). |
| `ITransport` | TCP vs an in-process pipe, so `test_havewant.cpp` needs no sockets. |

The cost of a virtual call at these granularities is unmeasurable next to the
`pread` or `ZSTD_decompress` it wraps, and the ability to inject a
fault-injecting `IBlockStore` is what makes the tamper and crash matrices
possible at all.

**Not virtual — templates or plain functions:**

- The residual kernels. Dispatched once per frame through a function pointer
  chosen from CPUID (ADR 0011), then a tight loop with no indirection inside.
- `expand(perm, block)`, endian conversion, chunk hashing: free functions,
  header-inline, `constexpr` where possible.
- The interval table lookup in the mount: a concrete `std::vector` and
  `std::ranges::lower_bound`. It is on the fault path and it has exactly one
  implementation.

**Rule of thumb for the team:** if a call happens once per frame or less often,
a virtual call is free — use the interface. If it happens once per unit or per
byte, it must not be virtual.

## Consequences

- Every interface gets a test double in `tests/common/`, and the fault-injecting
  store is one of them.
- Interfaces live in `core` and depend on nothing else, so a module can be
  compiled and tested against stubs before its collaborators exist. On Day 1
  that is the difference between eight people working and three people waiting.
- We accept some ugliness where a virtual boundary sits close to a hot path
  (`IBlockStore::read_range` returning into a caller-owned `std::span` rather
  than a `std::vector`, to keep allocation out of the fault path).
