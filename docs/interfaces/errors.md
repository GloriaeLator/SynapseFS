# Errors

One error type, one taxonomy, no exceptions across module boundaries.

## Why not exceptions

An exception escaping into a libfuse callback does not produce a stack trace.
It produces a kernel-visible hang on a mounted filesystem, and the mount is 25%
of the grade. The same applies at the network read loop and inside the thread
pool.

So: every fallible API returns `std::expected<T, sfs::core::Error>`.
Exceptions may be used *within* a translation unit for genuinely exceptional
conditions (`std::bad_alloc`), and every module boundary catches and converts.

## The type

```cpp
namespace sfs::core {

enum class ErrKind : std::uint16_t {
    Ok = 0,

    // --- I/O and environment ------------------------------------------------
    Io,                       // errno-backed; see Error::sys_errno
    NoSuchFile,
    PermissionDenied,
    NoSpace,
    Interrupted,

    // --- repository ---------------------------------------------------------
    NotARepository,
    RepositoryLocked,
    UnsupportedFormatVersion,
    RefNotFound,
    RefRaceLost,             // compare-and-swap failed: someone else moved it
    NotFastForward,

    // --- objects ------------------------------------------------------------
    ObjectNotFound,
    ObjectKindMismatch,      // framing said `raw`, caller wanted `diff`
    MalformedObject,
    CanonicalizationMismatch,// re-serialised JSON does not reproduce its own oid

    // --- integrity (all of these map to exit code 4) -------------------------
    HashMismatch,
    ChunkDigestMismatch,
    FrameDigestMismatch,
    AncestorInvariantViolated,
    JournalTorn,

    // --- checkpoint / format ------------------------------------------------
    NotSafetensors,
    UnsupportedDType,
    ShapeMismatch,
    TensorNotInBufferLayout,

    // --- alignment ----------------------------------------------------------
    TopologyParse,
    TopologyIncomplete,      // an axis the parser could not assign to a group
    BlockFactorMismatch,     // shape[dim] % group_size != 0
    NotAlignable,            // normalised cost above threshold — not a failure
    InvalidPermutation,      // payload array is not a bijection on [0, n)

    // --- policy -------------------------------------------------------------
    ChainTooDeep,
    MergeConflict,

    // --- mount --------------------------------------------------------------
    MountFailed,
    ReadOnlyFilesystem,

    // --- network ------------------------------------------------------------
    ProtocolVersion,
    MalformedFrame,
    ConnectionLost,
    PeerError,

    // --- misc ---------------------------------------------------------------
    NotImplemented,
    Cancelled,
    Internal,
};

struct Error {
    ErrKind          kind = ErrKind::Internal;
    std::string      what;          // one line, human-facing
    std::string      context;       // path, oid, tensor name — whatever locates it
    int              sys_errno = 0; // set when kind == Io
    std::source_location where;

    [[nodiscard]] bool is_integrity() const noexcept;
    [[nodiscard]] int  exit_code()    const noexcept;  // SPEC 15 §3
};

template <class T> using Result = std::expected<T, Error>;
using Status = Result<void>;

}  // namespace sfs::core
```

## Conventions

**Construct with the macro, not by hand.** `SFS_ERR(kind, fmt, ...)` captures
`std::source_location` and formats the message. `SFS_TRY(expr)` propagates:

```cpp
Result<Manifest> load_manifest(IBlockStore& store, const Oid& oid) {
    auto bytes = SFS_TRY(store.get(oid, ObjectKind::Manifest));
    return Manifest::parse(bytes);
}
```

**`context` locates, `what` explains.** The CLI prints
`sfs: <command>: <what>: <context>`, so `what` should not repeat the path and
`context` should not be a sentence.

**`NotAlignable` is not a failure.** It is an answer. The aligner returns it,
the commit path reads it, and the group is stored `mode: full`. Do not log it
as an error.

**Integrity errors are their own class.** `is_integrity()` is true for
`HashMismatch`, `ChunkDigestMismatch`, `FrameDigestMismatch`,
`AncestorInvariantViolated` and `JournalTorn`. They exit 4, they are always
logged at error level with the object *and* the chunk index, and they are never
retried. "The data is wrong" is a different event from "the program failed",
and every harness we have depends on telling them apart.

## Mapping

| ErrKind group | CLI exit (SPEC 15 §3) | Wire code (SPEC 14 §6) |
|---|---|---|
| `Ok` | 0 | — |
| generic | 1 | — |
| usage (from CLI11) | 2 | — |
| `NotARepository`, `UnsupportedFormatVersion` | 3 | 1 |
| every `is_integrity()` kind | **4** | 4, 7 |
| `MergeConflict` | 5 | — |
| `RepositoryLocked` | 6 | 6 |
| `NotImplemented` | 7 | — |
| `ProtocolVersion`, `MalformedFrame`, `ConnectionLost`, `PeerError` | 8 | 1, 2, — |
| `ObjectNotFound` over the wire | 1 | 3 |
| `NotFastForward` | 1 | 5 |

## Logging

`util/log.hpp` — level, one line, structured key=value tail:

```
E 12:04:31.882 store  chunk digest mismatch oid=b3:9a8b7c6d chunk=137 expected=7c1f… got=b904…
```

Integrity failures log at `E` and include enough to reproduce. Everything on
the mount's fault path logs at `T` (trace) or not at all — a log statement per
page fault is a throughput bug.
