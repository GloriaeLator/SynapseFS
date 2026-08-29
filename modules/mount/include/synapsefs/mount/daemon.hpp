#pragma once
/// \file daemon.hpp
/// The FUSE low-level daemon.
///
/// Low-level (fuse_lowlevel.h), not high-level, for three reasons that are all
/// graded metrics (docs/adr/0003):
///   * open flags: FOPEN_KEEP_CACHE ON (a commit is immutable),
///     FOPEN_DIRECT_IO OFF (with direct I/O, mmap does not work AT ALL — this
///     is the single most common way to fail Module 3 while every read() test
///     passes)
///   * fuse_reply_buf hands back a buffer we already have: one copy, not two
///   * inode identity is ours
///
/// STRICTLY NO PRE-MATERIALISATION. The daemon must never write the
/// reconstructed file anywhere, never reconstruct a group in full to serve a
/// partial read, and never populate the cache eagerly. Demonstrable under
/// `strace -f -e trace=write,openat`, which is a presentation slide.

#include <filesystem>
#include <memory>

#include <synapsefs/core/error.hpp>
#include <synapsefs/mount/fs.hpp>

namespace sfs::mount {

struct DaemonOptions {
    std::filesystem::path mountpoint;
    bool foreground = false;   ///< required under sanitizers; used by the demo
    bool debug      = false;
    FsOptions fs;
};

class Daemon {
public:
    [[nodiscard]] static core::Result<std::unique_ptr<Daemon>> start(
        std::unique_ptr<SynapseFs>, DaemonOptions);
    ~Daemon();

    /// Blocks until unmounted or stopped. libfuse owns the threads.
    [[nodiscard]] core::Status run();
    void stop() noexcept;

    /// Registers this daemon against the repository so that `gc` refuses while
    /// it is attached — the daemon holds objects open by id, and an
    /// unlinked-but-open file is a trap that fires on the next remount.
    [[nodiscard]] core::Status register_with_repo(const core::RepoPaths&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Daemon();
};

/// `sfs unmount`: waits for in-flight reads, falls back to `fusermount3 -u`.
[[nodiscard]] core::Status unmount(const std::filesystem::path& mountpoint);

}  // namespace sfs::mount
