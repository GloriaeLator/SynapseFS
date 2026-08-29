# ADR 0003 — FUSE low-level API

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

Module 3 is 25% of the grade, split POSIX compliance 10, mmap throughput 8,
peak RSS 7. The PS suggests FUSE but does not mandate it, and requires
`open`, `read`, `mmap`, `lseek`, `stat`, `close` to work against an unmodified
`safetensors.torch.load_file()`.

## Options

1. **FUSE high-level API** (`fuse.h`, path-based). Simplest: implement
   `getattr`/`open`/`read` on paths and you have a filesystem. libfuse handles
   inodes, and the callback signature hands you a `char* buf` to fill.
2. **FUSE low-level API** (`fuse_lowlevel.h`, inode-based). More code: you own
   the inode table, lookup counts, and reply buffers.
3. **NFS loopback / 9p / a custom block device.** Avoids FUSE's per-request
   overhead entirely, at a cost in implementation and in "does it work on the
   evaluator's machine".

## Decision

Low-level. Three concrete reasons, all of them tied to graded metrics:

**Open flags.** `FOPEN_KEEP_CACHE` and `FOPEN_DIRECT_IO` are set per-open in
`fuse_reply_open`. We need `KEEP_CACHE` on (a commit is immutable, so the
kernel's page cache never goes stale) and `DIRECT_IO` **off** (with direct I/O
enabled, `mmap` does not work at all). The high-level API's `fuse_file_info`
exposes these too, but the low-level path makes them explicit at the one place
they are decided, and mmap is 8% of the grade.

**Reply without a copy.** `fuse_reply_buf` lets us hand back a buffer we
already have. On the fast path — a cached, already-decompressed frame — that is
one copy instead of two, on every page fault.

**Inode identity.** We serve one file, so the inode table is small, but we own
it. That means `stat` reports a size we computed from the manifest, and lookup
counts are ours to reason about when a reader holds the file open across a
`gc`.

The cost is roughly 200 extra lines in `modules/mount/src/fuse_ll.cpp` and
having to get `forget` right. That is a good trade for a module worth a quarter
of the grade.

Options 3 are rejected on risk: they may well be faster, and none of them is
something we can debug on Day 5 if the evaluator's kernel disagrees with ours.

## Consequences

- `FUSE_USE_VERSION=34`, libfuse ≥ 3.10, `_FILE_OFFSET_BITS=64` (set in
  `cmake/FindFUSE3.cmake` — getting it wrong truncates `off_t` and only shows
  up at multi-GB, which is exactly the fixture size).
- `--foreground` is required under sanitizers, and is what the demo uses so
  `strace` output stays visible.
- `max_read` and readahead raised to 128 KiB so that `safetensors`' large
  sequential reads arrive as fewer, larger requests.
- We must handle `forget` and lookup counts correctly or leak inode entries.
  Covered by `modules/mount/tests/test_inode_table.cpp`.

## How we would know we were wrong

If profiling on Day 4 shows FUSE request overhead — not decompression —
dominating the mmap benchmark, the answer is larger `max_read` and writeback
tuning, not a different filesystem mechanism. If that fails to move the number,
record it as a measured limitation rather than rewriting on Day 5.
