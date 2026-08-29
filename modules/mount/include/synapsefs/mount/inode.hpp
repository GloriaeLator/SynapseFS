#pragma once
/// \file inode.hpp
/// The interval table: file offset -> (group, offset within group).
///
/// Built ONCE at open() from the manifest's buffer layout and never rebuilt.
/// read() is then a binary search plus one or more read_range calls plus a
/// copy. Nothing on the fault path parses anything.

#include <cstdint>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/format/manifest.hpp>

namespace sfs::mount {

struct Interval {
    std::uint64_t file_offset = 0;   ///< start, in the reconstructed file
    std::uint64_t length      = 0;
    std::uint64_t group_offset = 0;  ///< start, within the group's bytes
    std::uint32_t group_index = 0;   ///< into IntervalTable::groups
    bool          is_header = false; ///< the verbatim safetensors header block
};

/// Sorted, contiguous, covering [0, total_bytes).
class IntervalTable {
public:
    [[nodiscard]] static core::Result<IntervalTable> build(const format::Manifest&);

    /// The interval containing `offset`, or nullptr past EOF.
    [[nodiscard]] const Interval* find(std::uint64_t offset) const noexcept;
    [[nodiscard]] std::span<const Interval> intervals() const noexcept { return intervals_; }
    [[nodiscard]] std::string_view group_name(std::uint32_t index) const noexcept;
    [[nodiscard]] std::uint64_t total_bytes() const noexcept { return total_bytes_; }

private:
    std::vector<Interval>    intervals_;
    std::vector<std::string> groups_;
    std::uint64_t            total_bytes_ = 0;
};

/// FUSE inode numbers. We serve one file, so this is small — but it is ours,
/// which means stat() reports a size we computed from the manifest and lookup
/// counts are ours to reason about across a gc.
inline constexpr std::uint64_t kRootIno = 1;
inline constexpr std::uint64_t kFileIno = 2;

}  // namespace sfs::mount
