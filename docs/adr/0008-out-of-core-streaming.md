# ADR 0008 - Streaming alignment is the only path, not a fallback

## Context

Fixtures go to ~7B parameters in fp16/bf16, and the grading environment caps
RAM at 16 GB and VRAM at 8 GB. A 7B fp16 checkpoint is ~14 GB.

## Decision

We chose to **Stream always.** - One code path, slower on tiny inputs, correct everywhere. **The streaming path is the only path, used even on the 1M-parameter
MLP fixture.**.

Mechanics:

- Tensors are read lazily through `stio` (`safe_open`-equivalent + ranged
  reads), never by loading a whole file.
- The cost matrix for a group of size *n* is `n × n` floats - 256 MB at
  n = 8192, which is fine; the group size, not the parameter count, bounds it.
  Groups larger than a configurable cap use the tiled accumulation below.
- Cost accumulation is **tiled**: for each pair of unit tiles, read the two
  slices, accumulate their contribution, release. Neither checkpoint is ever
  fully resident. Peak RSS is `tile × 2 + cost_matrix`.
- The residual pass streams too: one frame of target units at a time, pulling
  only the base units that frame depends on.

`bench/scripts/peak_rss.sh` reads `VmHWM` and the scale test asserts a hard
ceiling, so a regression fails CI rather than surfacing on the evaluator's
machine.

## Consequences

- Alignment wall-clock (8% of the grade) is measured on the streaming path, so
  I/O is in the number. Tile size is therefore a tuning parameter, not a
  detail: `bench/align_time.cpp` sweeps it.
- `stio` must support ranged reads on a compressed-nothing file - safetensors
  is uncompressed, so this is just `pread` at a computed offset, which is why
  the format is pleasant to work with.
