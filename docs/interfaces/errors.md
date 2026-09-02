# Errors

This describes the one error type, one taxonomy, no exceptions across module boundaries.

## The type

```cpp
enum class ErrKind : uint16_t { /* ~40 members, grouped by comment:
    I/O, repository, objects, integrity, checkpoint/format, alignment,
    policy, mount, network, misc */ };

struct Error {
    ErrKind kind;
    std::string what;
    std::string context;
    int sys_errno;
    std::source_location where;
};

template <class T> using Result = std::expected<T, Error>;
using Status = Result<void>;
```

`SFS_ERR(kind, ...)` constructs a `std::unexpected<Error>`; `SFS_TRY(expr)`
/`SFS_TRY_VOID(expr)` are statement-expression-based early-return macros
(a GCC/Clang extension) used throughout the codebase for `Result`
propagation without exceptions.

**Design rule, stated in the header comment**: no exception is allowed to
cross a module boundary. The concrete reason given is the FUSE mount - an
exception escaping a libfuse callback hangs the mount rather than returning
a clean error to the kernel. Every module's public API returns `Result<T>`/
`Status`, not `T` and a possible throw.

## Integrity kinds

```cpp
bool is_integrity(ErrKind);  // true exactly for:
//   HashMismatch, ChunkDigestMismatch, FrameDigestMismatch,
//   AncestorInvariantViolated, JournalTorn
```

Any of these five always maps to CLI exit code 4 - see below.

## `Error::exit_code()`

Hardcoded mapping (`error.cpp`), not a generic table:

| Condition | Exit code |
|---|---|
| `is_integrity(kind)` | 4 |
| `Ok` | 0 |
| `NotARepository` | 3 |
| `MergeConflict` | 5 |
| `RepositoryLocked` | 6 |
| `NotImplemented` | 7 |
| `ProtocolVersion` / `MalformedFrame` / `ConnectionLost` / `PeerError` | 8 |
| everything else | 1 |

This is the mapping `apps/sfs/exitcode.hpp`'s `exit_code_for()` delegates
to for any propagated `core::Error`. See
[`spec/15-cli-contract.md`](../spec/15-cli-contract.md) for the full
`ExitCode` enum and per-command notes.

## Constructing errors

```cpp
Error make_error(ErrKind, std::string what, std::string context = "");
Error from_errno(std::error_code, ...);
```

`from_errno` maps `no_such_file_or_directory` -> `NoSuchFile`,
`permission_denied` -> `PermissionDenied`, `no_space_on_device` -> `NoSpace`,
`interrupted` -> `Interrupted`, and everything else -> `Io`.
