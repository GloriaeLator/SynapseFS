#pragma once
/// \file error.hpp
/// One error type for the whole system. No exceptions cross a module boundary:
/// an exception escaping into a libfuse callback is a kernel-visible hang on a
/// mounted filesystem, not a stack trace.
///
/// Full taxonomy and the exit-code / wire-code mapping: docs/interfaces/errors.md

#include <cstdint>
#include <expected>
#include <source_location>
#include <string>
#include <string_view>
#include <system_error>

namespace sfs::core {

enum class ErrKind : std::uint16_t {
    Ok = 0,

    // I/O and environment
    Io, NoSuchFile, PermissionDenied, NoSpace, Interrupted,

    // repository
    NotARepository, RepositoryLocked, UnsupportedFormatVersion,
    RefNotFound, RefRaceLost, NotFastForward,

    // objects
    ObjectNotFound, ObjectKindMismatch, MalformedObject, CanonicalizationMismatch,

    // integrity — every one of these exits 4
    HashMismatch, ChunkDigestMismatch, FrameDigestMismatch,
    AncestorInvariantViolated, JournalTorn,

    // checkpoint / format
    NotSafetensors, UnsupportedDType, ShapeMismatch, TensorNotInBufferLayout,

    // alignment
    TopologyParse, TopologyIncomplete, BlockFactorMismatch,
    NotAlignable,          ///< an answer, not a failure — do not log as an error
    InvalidPermutation,

    // policy
    ChainTooDeep, MergeConflict,

    // mount
    MountFailed, ReadOnlyFilesystem,

    // network
    ProtocolVersion, MalformedFrame, ConnectionLost, PeerError,

    // misc
    NotImplemented, Cancelled, Internal,
};

[[nodiscard]] std::string_view to_string(ErrKind) noexcept;

/// True for the integrity kinds. These exit 4, are always logged with the
/// object and chunk, and are never retried. "The data is wrong" and "the
/// program failed" are different events and every harness depends on the
/// difference.
[[nodiscard]] bool is_integrity(ErrKind) noexcept;

struct Error {
    ErrKind     kind = ErrKind::Internal;
    std::string what;        ///< one line, human-facing; does not repeat `context`
    std::string context;     ///< a path, an oid, a tensor name — whatever locates it
    int         sys_errno = 0;
    std::source_location where = std::source_location::current();

    [[nodiscard]] bool is_integrity() const noexcept { return core::is_integrity(kind); }
    [[nodiscard]] int  exit_code() const noexcept;      ///< docs/spec/15-cli-contract.md §3
    [[nodiscard]] std::string to_string() const;
};

template <class T>
using Result = std::expected<T, Error>;
using Status = Result<void>;

[[nodiscard]] Error make_error(ErrKind, std::string what, std::string context = {},
                               std::source_location = std::source_location::current());

/// Wrap an errno-flavoured failure from `util`.
[[nodiscard]] Error from_errno(std::error_code, std::string context,
                               std::source_location = std::source_location::current());

}  // namespace sfs::core

/// SFS_ERR(ObjectNotFound, "no such object", oid.to_string())
#define SFS_ERR(kind_, ...) \
    ::std::unexpected(::sfs::core::make_error(::sfs::core::ErrKind::kind_, __VA_ARGS__))

/// Propagate on failure, unwrap on success.
///   auto bytes = SFS_TRY(store.get(oid, ObjectKind::Manifest));
#define SFS_TRY(expr)                                     \
    ({                                                    \
        auto sfs_try_result_ = (expr);                    \
        if (!sfs_try_result_) [[unlikely]]                \
            return ::std::unexpected(sfs_try_result_.error()); \
        ::std::move(sfs_try_result_).value();             \
    })

/// Same, for Status-returning expressions with no value.
#define SFS_TRY_VOID(expr)                                \
    do {                                                  \
        auto sfs_try_status_ = (expr);                    \
        if (!sfs_try_status_) [[unlikely]]                \
            return ::std::unexpected(sfs_try_status_.error()); \
    } while (0)
