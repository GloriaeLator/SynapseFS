# Architecture decision records

One file per decision that would be expensive to reverse. Each records the
context at the time, the options considered, what we chose, and — where one
exists — the measurement that decided it.

These are written for the Q&A. The rubric says a feature nobody can explain
counts as not implemented, and that a worse solution whose authors can defend
it scores above a better one whose authors cannot. An ADR is the artifact that
makes "why" answerable by whoever is standing there.

| # | Decision | Status |
|---|---|---|
| [0001](0001-cpp23-and-toolchain.md) | C++23, GCC 14, CMake + Ninja, Linux only | Accepted |
| [0002](0002-blake3-over-sha256.md) | BLAKE3-256 as the content address | Accepted |
| [0003](0003-fuse-lowlevel-vs-highlevel.md) | FUSE low-level API | Accepted |
| [0004](0004-weight-matching-vs-activation-vs-ot.md) | Weight matching by LAP, not activation matching or OT | Accepted |
| [0005](0005-residual-encoding.md) | Framed residuals; encoding chosen by measurement | Accepted (parameters pending Day-2 measurement) |
| [0006](0006-packfiles-vs-loose-objects.md) | Loose objects first, packfiles behind `gc --pack` | Accepted |
| [0007](0007-crash-safety-journal-vs-rename.md) | Atomic rename everywhere; a journal only for multi-file updates | Accepted |
| [0008](0008-out-of-core-streaming.md) | Streaming alignment is the only path, not a fallback | Accepted |
| [0009](0009-vcpkg-vs-fetchcontent.md) | vcpkg manifest mode | Accepted |
| [0010](0010-virtual-dispatch-vs-templates.md) | Virtual interfaces at module seams, templates inside | Accepted |
| [0011](0011-simd-dispatch-strategy.md) | Compile every ISA, choose at runtime from CPUID | Accepted |

**Template:** context → options → decision → consequences → how we would know
we were wrong.
