# ADR 0003 - FUSE low-level API

## Context

Module 3 is 25% of the grade, split POSIX compliance 10, mmap throughput 8,
peak RSS 7. The PS suggests FUSE but does not mandate it, and requires
`open`, `read`, `mmap`, `lseek`, `stat`, `close` to work against an unmodified
`safetensors.torch.load_file()`.

## Decision

We use a Low-level FUSE. Three concrete reasons:

**Open flags.** `FOPEN_KEEP_CACHE` and `FOPEN_DIRECT_IO` are set per-open in
`fuse_reply_open`. We need `KEEP_CACHE` on (a commit is immutable, so the
kernel's page cache never goes stale) and `DIRECT_IO` **off** (with direct I/O
enabled, `mmap` does not work at all). The high-level API's `fuse_file_info`
exposes these too, but the low-level path makes them explicit at the one place
they are decided.

**Reply without a copy.** `fuse_reply_buf` lets us hand back a buffer we
already have. On the fast path - a cached, already-decompressed frame - that is
one copy instead of two, on every page fault.

**Inode identity.** We serve one file, so the inode table is small, but we own
it. That means `stat` reports a size we computed from the manifest, and lookup
counts are ours to reason about when a reader holds the file open across a
`gc`.

## Consequences

- `FUSE_USE_VERSION=34`, libfuse ≥ 3.10, `_FILE_OFFSET_BITS=64` (set in
  `cmake/FindFUSE3.cmake` - getting it wrong truncates `off_t` and only shows
  up at multi-GB, which is exactly the fixture size).
- `max_read` and readahead raised to 128 KiB so that `safetensors`' large
  sequential reads arrive as fewer, larger requests.
- We must handle `forget` and lookup counts correctly or leak inode entries.
