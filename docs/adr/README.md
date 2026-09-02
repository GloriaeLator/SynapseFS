# Architecture decision records

| # | Decision | Status |
|---|---|---|
| [0001](0001-cpp23-and-toolchain.md) | C++23, GCC 14, CMake + Ninja, Linux only 
| [0002](0002-blake3-over-sha256.md) | BLAKE3-256 as the content address 
| [0003](0003-fuse-lowlevel-vs-highlevel.md) | FUSE low-level API 
| [0004](0004-weight-matching-vs-activation-vs-ot.md) | Alignment solver selection, sparse pipeline, and dispatch policy 
| [0005](0005-residual-encoding.md) | Framed residuals; encoding chosen by measurement
| [0006](0006-packfiles-vs-loose-objects.md) | Loose objects first, packfiles 
| [0007](0007-crash-safety-journal-vs-rename.md) | Atomic rename everywhere; a journal only for multi-file updates 
| [0008](0008-out-of-core-streaming.md) | Streaming alignment is the only path, not a fallback 
| [0009](0009-libtorch-for-large-groups.md) | libtorch for the large-group sparse alignment path
| [0010](0010-virtual-dispatch-vs-templates.md) | Virtual interfaces at module seams, templates inside 
| [0011](0011-simd-dispatch-strategy.md) | Compile every ISA, choose at runtime from CPUID 
