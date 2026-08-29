#pragma once
/// \file gc.hpp
/// Garbage collection: repack, and remove what no ref reaches.
///
/// gc MUST take the exclusive lock, and MUST NOT run while a mount daemon is
/// attached: the daemon holds objects open by id, and an unlinked-but-open file
/// is a correctness trap that only fires on the next remount.
/// docs/spec/11-repo-layout.md §6.

#include <cstdint>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/store/dag.hpp>

namespace sfs::store {

struct GcOptions {
    bool pack    = false;   ///< rewrite reachable loose objects into a packfile
    bool prune   = false;   ///< delete unreachable objects
    bool dry_run = false;
};

struct GcReport {
    std::uint64_t objects_scanned = 0;
    std::uint64_t objects_pruned  = 0;
    std::uint64_t bytes_reclaimed = 0;
    std::uint64_t temp_files_removed = 0;
    std::uint64_t packed = 0;
};

[[nodiscard]] Result<GcReport> gc(core::IBlockStore&, CommitStore&, ManifestStore&,
                                  RefStore&, const core::RepoPaths&, const GcOptions& = {});

/// True if a mount daemon has registered itself against this repository.
/// gc refuses when it is.
[[nodiscard]] Result<bool> mount_attached(const core::RepoPaths&);

}  // namespace sfs::store
