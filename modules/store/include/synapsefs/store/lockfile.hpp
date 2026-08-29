#pragma once
/// \file lockfile.hpp
/// Advisory flock() on .synapsefs/index.lock. docs/spec/11-repo-layout.md §3.4.
///
/// Writers take LOCK_EX. Readers mostly take nothing, because objects are
/// immutable and a concurrently added object is simply not walked.
///
/// The mount daemon MUST NOT hold the exclusive lock: a mounted repository
/// stays writable.

#include <chrono>
#include <filesystem>

#include <synapsefs/core/error.hpp>
#include <synapsefs/util/file.hpp>

namespace sfs::store {

using core::Result;

enum class LockMode { Shared, Exclusive };

class RepoLock {
public:
    /// Non-blocking by default: a held lock is ErrKind::RepositoryLocked and
    /// exit code 6, which is a better user experience than an unexplained hang.
    [[nodiscard]] static Result<RepoLock> acquire(
        const std::filesystem::path& lock_path, LockMode,
        std::chrono::milliseconds wait = std::chrono::milliseconds{0});

    ~RepoLock();
    RepoLock(RepoLock&&) noexcept;
    RepoLock& operator=(RepoLock&&) noexcept;
    RepoLock(const RepoLock&) = delete;
    RepoLock& operator=(const RepoLock&) = delete;

    void release() noexcept;

private:
    RepoLock() = default;
    util::Fd fd_;
};

}  // namespace sfs::store
