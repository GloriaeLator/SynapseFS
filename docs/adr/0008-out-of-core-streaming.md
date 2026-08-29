# ADR 0008 — Streaming alignment is the only path, not a fallback

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

Fixtures go to ~7B parameters in fp16/bf16, and the grading environment caps
RAM at 16 GB and VRAM at 8 GB. A 7B fp16 checkpoint is ~14 GB. Two of them
resident is an instant OOM, and the PS is explicit: **OOM at fixture size means
the metric is failed, even if it passes locally.**

## Options

1. **Load both, fall back to streaming when memory is tight.** Two code paths,
   and the streaming one is exercised only at fixture scale — which is where
   we cannot debug it.
2. **Stream always.** One code path, slower on tiny inputs, correct everywhere.
3. **Memory-map both and let the kernel page.** Attractive and wrong: the
   cost-matrix accumulation touches base units in permuted order, so the access
   pattern is scattered and the kernel thrashes. It also makes peak RSS a
   function of kernel policy rather than of our code, which is the number we
   are graded on.

## Decision

Option 2. **The streaming path is the only path, used even on the 1M-parameter
MLP fixture.** This is a deliberate cost: it is slower than necessary on small
inputs and it is the thing that guarantees there is no retrofit on Day 4.

Mechanics:

- Tensors are read lazily through `stio` (`safe_open`-equivalent + ranged
  reads), never by loading a whole file.
- The cost matrix for a group of size *n* is `n × n` floats — 256 MB at
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

- The tiny fixtures run slower than they would with everything in RAM. Accepted;
  the whole point is that the code we test is the code that runs.
- `stio` must support ranged reads on a compressed-nothing file — safetensors
  is uncompressed, so this is just `pread` at a computed offset, which is why
  the format is pleasant to work with.
- Alignment wall-clock (8% of the grade) is measured on the streaming path, so
  I/O is in the number. Tile size is therefore a tuning parameter, not a
  detail: `bench/align_time.cpp` sweeps it.
- We never call `torch`. The commit path reads bytes.

## How we would know we were wrong

If peak RSS at 7B is dominated by the cost matrix rather than by tiles, the
group cap is wrong and the fix is a blocked LAP, not more memory.
