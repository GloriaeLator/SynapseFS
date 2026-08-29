#pragma once
/// \file tree.hpp
/// RESERVED — multi-file (sharded) checkpoints.
///
/// Format version 1 stores exactly one .safetensors file per commit, so a
/// commit points straight at a manifest and there is no tree object. Sharded
/// checkpoints (model-00001-of-00003.safetensors + an index.json) are a listed
/// known limitation in docs/tradeoffs.md §4, not a graded requirement.
///
/// The shape is written down here so that the extension is a small, obvious
/// change rather than a redesign: a Tree lists (name -> manifest oid), a commit
/// points at a Tree instead of a Manifest, and every other object is unchanged.
///
/// NOTHING IN FORMAT VERSION 1 READS OR WRITES THIS. If we reach the end of the
/// project without implementing sharded checkpoints, delete this file rather
/// than shipping a header nobody calls.

#include <cstdint>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::format {

struct TreeEntry {
    std::string name;        ///< file name as it appears at checkout and in the mount
    core::Oid   manifest;    ///< kind Manifest
};

struct Tree {
    std::uint32_t          format_version = 2;   ///< NOT 1 — see the file comment
    std::vector<TreeEntry> entries;              ///< sorted by name, for canonical JSON

    [[nodiscard]] std::vector<std::byte> to_canonical_json() const;
    [[nodiscard]] core::Oid oid() const;
    [[nodiscard]] static core::Result<Tree> parse(std::span<const std::byte>);
};

}  // namespace sfs::format
