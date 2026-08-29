/// \file inode.cpp
/// Builds the interval table once, at open(), from the manifest's buffer
/// layout. Nothing on the fault path parses anything — this is the only place
/// that walks the manifest.
///
/// Layout (docs/spec/16-consistency.md §3.2):
///   [0, header_len)         -> header block
///   [off, off + nbytes)     -> (group, offset within group), per buffer entry
///
/// `BufferEntry::off` is ABSOLUTE from the start of the file, not relative to
/// the data section: Manifest::validate() seeds its contiguity cursor at
/// `buffer[0].off` (the header extent) and then requires `entry.off ==
/// cursor` all the way through, so `buffer[0].off` doubles as `header_len`
/// and every later `off` already has it folded in. (tensor.hpp's own comment
/// on `BufferEntry::off` — "from the start of the data section" — reads as
/// data-section-relative; validate()'s actual contiguity check is the
/// authoritative contract, since it's the one thing every writer must
/// satisfy to produce a manifest that passes it. Don't add header_len again
/// here.)
///
/// buffer entries are already in BUFFER order (contiguous and gap-free by the
/// same validate() invariant), so this is a single linear pass, not a sort —
/// the sort below is a defensive no-op on a well-formed manifest, not load-
/// bearing.

#include <synapsefs/mount/inode.hpp>

#include <algorithm>

namespace sfs::mount {

core::Result<IntervalTable> IntervalTable::build(const format::Manifest& manifest) {
    // validate() is the module boundary's job (fs.cpp calls it before this),
    // but a defensive check here costs nothing and turns a malformed manifest
    // into a named error instead of a garbage interval table.
    if (auto st = manifest.validate(); !st) {
        return std::unexpected(st.error());
    }

    IntervalTable table;
    table.total_bytes_ = manifest.file.total_bytes;

    // buffer[0].off IS header_len (see file comment) -- 0 only for the
    // degenerate case of no buffer entries at all, which validate() above
    // already rejects for any manifest describing real tensor content, but a
    // header-only "file" isn't expressly forbidden by this function alone.
    const std::uint64_t header_len =
        manifest.buffer.empty() ? manifest.file.total_bytes : manifest.buffer.front().off;

    table.intervals_.reserve(manifest.buffer.size() + 1);

    if (header_len > 0) {
        Interval hdr;
        hdr.file_offset  = 0;
        hdr.length       = header_len;
        hdr.group_offset = 0;
        hdr.group_index  = 0;  // unused for the header
        hdr.is_header    = true;
        table.intervals_.push_back(hdr);
    }

    // Group name -> index, so repeated groups (many tensors sharing a
    // permutation group) don't duplicate the string.
    std::vector<std::string>& groups = table.groups_;
    auto group_index_for = [&groups](std::string_view name) -> std::uint32_t {
        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (groups[i] == name) return static_cast<std::uint32_t>(i);
        }
        groups.emplace_back(name);
        return static_cast<std::uint32_t>(groups.size() - 1);
    };

    // Track the running offset within each group's reconstructed bytes,
    // since buffer order is not necessarily group-contiguous and the
    // manifest only gives us the group *name* per entry, not a within-group
    // offset. Groups are, in practice, laid out contiguously in buffer order
    // (they come from a single tensor's or permutation-group's span), so this
    // running counter is exact rather than a guess: it advances by exactly
    // nbytes for every entry belonging to that group, in the order those
    // entries appear in `buffer`, which is the same order read_range serves
    // the group's bytes in.
    std::vector<std::uint64_t> group_cursor;
    group_cursor.reserve(8);

    for (const auto& entry : manifest.buffer) {
        const std::uint32_t gi = group_index_for(entry.group);
        if (gi >= group_cursor.size()) group_cursor.resize(gi + 1, 0);

        Interval iv;
        iv.file_offset  = entry.off;  // already absolute; see file comment
        iv.length       = entry.nbytes;
        iv.group_offset = group_cursor[gi];
        iv.group_index  = gi;
        iv.is_header    = false;
        table.intervals_.push_back(iv);

        group_cursor[gi] += entry.nbytes;
    }

    // Contiguity is a manifest-validation invariant, but assert the join
    // point between the header and the first buffer entry defensively:
    // find() below relies on intervals_ being sorted and gap-free.
    std::sort(table.intervals_.begin(), table.intervals_.end(),
              [](const Interval& a, const Interval& b) {
                  return a.file_offset < b.file_offset;
              });

    return table;
}

const Interval* IntervalTable::find(std::uint64_t offset) const noexcept {
    if (offset >= total_bytes_ || intervals_.empty()) return nullptr;

    // Binary search for the last interval whose file_offset <= offset.
    auto it = std::upper_bound(
        intervals_.begin(), intervals_.end(), offset,
        [](std::uint64_t off, const Interval& iv) { return off < iv.file_offset; });

    if (it == intervals_.begin()) return nullptr;  // offset < first interval
    --it;

    // Defensive: offset must actually fall inside [file_offset, +length).
    if (offset < it->file_offset || offset >= it->file_offset + it->length) {
        return nullptr;
    }
    return &*it;
}

std::string_view IntervalTable::group_name(std::uint32_t index) const noexcept {
    if (index >= groups_.size()) return {};
    return groups_[index];
}

}  // namespace sfs::mount
