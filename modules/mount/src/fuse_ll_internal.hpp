#pragma once
/// \file fuse_ll_internal.hpp
/// Private seam between fuse_ll.cpp (the libfuse callbacks) and daemon.cpp
/// (session lifecycle). Not under include/synapsefs/ -- nothing outside this
/// module includes fuse_lowlevel.h, so the FUSE_USE_VERSION / inode-table
/// details stay out of the public interface entirely (fs.hpp's whole point:
/// "so that the whole read path can be tested without mounting anything").

#include <atomic>
#include <cstdint>
#include <ctime>
#include <memory>

#include <synapsefs/mount/fs.hpp>
#include <synapsefs/mount/prefetch.hpp>
#include <synapsefs/mount/stats.hpp>

#if defined(__has_include)
#  if __has_include(<fuse3/fuse_lowlevel.h>)
#    include <fuse3/fuse_lowlevel.h>
#  else
#    include <fuse_lowlevel.h>
#  endif
#else
#  include <fuse_lowlevel.h>
#endif

namespace sfs::mount::detail {

/// Per-open-file-handle state, threaded through `fi->fh`. One per open(),
/// not shared -- two readers scanning different regions are two sequential
/// streams, not one random one (prefetch.hpp).
struct OpenFile {
    PrefetchState prefetch;
};

/// The FUSE session's userdata: everything the low-level callbacks need,
/// and nothing they don't. Constructed once in Daemon::start and outlives
/// the whole `fuse_session_loop`.
struct MountCtx {
    SynapseFs*     fs = nullptr;   // owned by Daemon::Impl, not here
    std::time_t    mount_time = 0; // st_mtime for both inodes: commit is immutable
    std::uint32_t  max_read = 128u * 1024;

    // Low-level API owns lookup counts (ADR 0003): kFileIno's count, so a
    // reader holding the file open across a `gc` on another process doesn't
    // become a dangling inode reference on our side. kRootIno needs no count
    // -- it is never forgotten (the kernel never releases the mount root).
    std::atomic<std::uint64_t> file_lookup_count{0};
};

/// The vtable fuse_ll.cpp builds and daemon.cpp passes to fuse_session_new.
[[nodiscard]] const fuse_lowlevel_ops& ll_ops() noexcept;

}  // namespace sfs::mount::detail
