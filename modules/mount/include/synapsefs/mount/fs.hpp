#pragma once
/// \file fs.hpp
/// The read-only filesystem view of one commit.
///
/// What is exposed:  <mountpoint>/<manifest.file.name>
/// Read-only. open() with any write flag returns EROFS. A mount is of a COMMIT,
/// not a branch, so nothing under it ever changes.
///
/// stat() reports st_size == manifest.file.total_bytes. Getting that wrong
/// matters more than it sounds: safetensors reads the 8-byte header length,
/// then the header, then trusts offsets — a wrong size surfaces as a truncated
/// tensor much later and much less obviously.

#include <cstdint>
#include <memory>
#include <string>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/error.hpp>
#include <synapsefs/mount/blockcache.hpp>
#include <synapsefs/mount/inode.hpp>

namespace sfs::mount {

struct FsOptions {
    std::uint64_t cache_bytes = 1024ull * 1024 * 1024;
    std::uint32_t max_read    = 128u * 1024;   ///< fewer, larger FUSE requests
    bool          allow_other = false;
};

/// Filesystem logic with no FUSE in it, so that the whole read path can be
/// tested without mounting anything. fuse_ll.cpp is a thin adapter over this.
class SynapseFs {
public:
    [[nodiscard]] static core::Result<std::unique_ptr<SynapseFs>> create(
        codec::ReadCtx, const format::Manifest&, FsOptions = {});
    ~SynapseFs();

    [[nodiscard]] std::string_view file_name() const noexcept;
    [[nodiscard]] std::uint64_t    file_size() const noexcept;

    /// The whole read path. Interval lookup, then read_range, then copy.
    /// Returns bytes written; short only at EOF. A digest failure surfaces as
    /// ErrKind::ChunkDigestMismatch, which the adapter turns into EIO — we
    /// never serve plausible-looking wrong bytes.
    [[nodiscard]] core::Result<std::size_t> read(std::uint64_t offset,
                                                 std::span<std::byte> out);

    [[nodiscard]] const IntervalTable& intervals() const noexcept;
    [[nodiscard]] FrameCache::Stats    cache_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SynapseFs();
};

}  // namespace sfs::mount
