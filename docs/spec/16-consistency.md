# SPEC 16 — Consistency and the filesystem contract

**Status:** normative

What a reader of the mount is guaranteed, what the daemon may and may not do,
and how the PS's consistency requirement is satisfied by construction rather
than by testing.

---

## 1. The requirement

> A checkpoint restored via checkout and the same checkpoint read through the
> mount must be byte-identical.

and

> Strictly no pre-materialization — tensors are built dynamically in RAM/VRAM
> as faults arrive; benchmarked from a cold page cache.

These pull in opposite directions if you implement them twice. They do not if
you implement them once.

---

## 2. One reconstructor

There is exactly one function that turns a (group, offset, length) request into
bytes: `read_range` (SPEC 12 §6).

```
sfs checkout        →  for each buffer entry: read_range(...) → write(fd)
mount, read()       →  interval lookup → read_range(...) → copy to fuse buffer
mount, mmap fault   →  same path, via the kernel's read of the backing pages
sfs verify          →  read_range over every reachable object
```

`checkout` is a loop of about thirty lines. It is not a second reconstructor,
and a design where it *is* one has already failed the consistency requirement —
it just does not know it yet.

The consequence worth stating in the presentation: consistency here is not a
test that passes, it is a property of the call graph. `tests/byte_identity.cpp`
asserts it anyway, at every fixture scale, because a property you do not assert
is a property somebody refactors away on Day 4.

---

## 3. The mount's view

### 3.1 What is exposed

```
<mountpoint>/
  model.safetensors        # manifest.file.name from the mounted commit
```

Read-only. `open()` with any write flag returns `EROFS`. No directory
hierarchy, no branches as directories — a mount is of one commit.

`stat()` reports `st_size == manifest.file.total_bytes` and `st_mtime` from the
commit's timestamp. Getting `st_size` right matters more than it sounds:
`safetensors` reads the 8-byte header length, then the header, then trusts
offsets; a wrong size surfaces as a truncated tensor much later.

### 3.2 The interval table

Built once at `open()` from the manifest's buffer layout, never rebuilt:

```
[0, header_len)                        → header block
[header_len + off, +nbytes) per entry  → (group, offset within group)
```

`read(fd, buf, off, len)` is a binary search into that table, then one or more
`read_range` calls, then a copy. There is no per-read parsing of anything.

### 3.3 Graded syscalls

`open`, `read`, `mmap`, `lseek`, `stat`, `close`. In practice that means:

- FUSE low-level API (`fuse_lowlevel.h`), not the high-level path — we need
  control over open flags and reply buffers. See
  `docs/adr/0003-fuse-lowlevel-vs-highlevel.md`.
- `FOPEN_KEEP_CACHE` **set**: pages the kernel already has stay valid, because
  a commit is immutable.
- `FOPEN_DIRECT_IO` **unset**: with direct I/O on, `mmap` does not work. This
  is the single most common way to fail Module 3 while every `read()` test
  passes.
- `max_read` and readahead raised (128 KiB), so `safetensors`' large sequential
  reads arrive as fewer, larger requests.

---

## 4. No pre-materialisation

The daemon MUST NOT, at any point:

- write the reconstructed file anywhere;
- reconstruct a group in full before serving a partial read of it;
- populate a cache eagerly at mount time.

Demonstrable: `strace -f -e trace=write,openat sfs mount --foreground` during a
full `load_file()` shows reads of objects and no creation of a checkpoint-sized
file. That trace is a presentation slide.

What the daemon *may* hold is a bounded LRU of **decompressed frames**
(`cache_bytes`, default 1 GiB). A frame is at most `frame_bytes`, so the cache
is a cache of small things, and peak RSS is bounded by
`cache_bytes + frame_bytes × max_chain_depth × concurrent_readers` plus the
interval table.

---

## 5. Concurrency

Multiple concurrent readers must be correct — graded.

- **Immutability.** Objects are immutable; a mounted commit is immutable.
  There is no invalidation problem, only a fill problem.
- **Single-flight fill.** Two readers faulting the same frame must not both
  decompress it. A per-frame in-flight map with a condition variable: the first
  arrival decompresses, the rest wait and then read the published entry.
- **Publication.** A cache entry becomes visible only after it is fully
  populated, published with release semantics and read with acquire. A reader
  never observes a half-filled frame.
- **Eviction.** An entry with a non-zero reader refcount is never evicted;
  eviction picks the LRU entry with refcount zero. Under sustained pressure
  from more concurrent readers than the budget allows, the daemon serves
  correctly and slowly rather than incorrectly.

`modules/mount/tests/test_blockcache_race.cpp` runs under TSan, and
`tests/concurrent_readers.cpp` runs 4–8 simultaneous loaders and compares every
result byte-for-byte against a checkout.

---

## 6. Interaction with the rest of the repository

| Event during a mount | Behaviour |
|---|---|
| `commit` on the mounted branch | Mount is of a commit, not a branch. Unaffected. |
| `gc` | Refused while a daemon is attached (SPEC 11 §6). |
| `pull` adding objects | Harmless; objects are add-only. |
| Repository write lock held | Mount does not take it and is not blocked. |
| Underlying object tampered with mid-mount | Next `read_range` touching that chunk fails; the read returns `EIO` and the daemon logs the object and chunk. It does not serve wrong bytes. |

Returning `EIO` rather than plausible-looking bytes is the whole point of
putting the digest check on the read path.

---

## 7. Test hooks

| Assertion | Test |
|---|---|
| `safetensors.torch.load_file()` against the mount succeeds | `tests/e2e.py` |
| Mount bytes == checkout bytes, every fixture scale | `tests/byte_identity.cpp` |
| Random reads at arbitrary offsets/sizes match | `modules/mount/tests/test_inode_table.cpp` |
| Edge cases: header/buffer boundary, tensor boundary, 1-byte read, read at EOF, read past EOF | same |
| `mmap` works (direct I/O unset) | `tests/e2e.py` |
| Nothing written during mount | `scripts/e2e.sh --with-mount`, under `strace` |
| 4–8 concurrent loaders all byte-identical | `tests/concurrent_readers.cpp` |
| No data race under TSan | `modules/mount/tests/test_blockcache_race.cpp` |
| Peak RSS within budget at fixture scale | `bench/scripts/peak_rss.sh` |
