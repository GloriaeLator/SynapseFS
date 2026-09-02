#pragma once
/// \file tree.hpp
/// Multi-file (sharded) checkpoints. docs/spec/10-object-model.md §5.
///
/// Format version 1 stores exactly one .safetensors file per commit, so a
/// version-1 commit points straight at a manifest and there is no tree object.
/// A sharded checkpoint (model-00001-of-00003.safetensors + an index.json) is
/// a set of files, so it needs one more level: a Tree lists (name -> manifest
/// oid), a version-2 commit points at a Tree instead of a Manifest, and every
/// other object in the model is unchanged.
///
/// This header used to be a reserved shape with no implementation. It is now
/// implemented: a Tree serialises, addresses, and parses exactly like every
/// other JSON object in the format, under its own ObjectKind::Tree so that a
/// tree payload and a manifest payload can never collide on one address
/// (oid.hpp).
///
/// READERS MUST STILL GATE ON format_version. A version-1 reader that is
/// handed a commit whose manifest field addresses a Tree has encountered a
/// repository it cannot interpret and must fail with UnsupportedFormatVersion,
/// not guess.
///
/// INVARIANT: `entries` is sorted by name, strictly ascending. The
/// serialisation IS the address, so two Trees with the same content in
/// different orders would otherwise be two different objects. Construct via
/// Tree::make(), which sorts and validates; to_canonical_json() does NOT sort
/// behind your back, because a writer that silently reorders cannot tell you
/// it disagreed with you.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::format {

using core::Oid;
using core::Result;
using core::Status;

/// The format_version a Tree-bearing repository declares. Named so that the
/// gate in a reader is a comparison against a constant rather than a literal 2
/// several call sites away from this file.
inline constexpr std::uint32_t kTreeFormatVersion = 2;

struct TreeEntry {
    std::string name;        ///< file name as it appears at checkout and in the mount
    Oid         manifest;    ///< kind Manifest

    friend bool operator==(const TreeEntry&, const TreeEntry&) noexcept = default;
};

struct Tree {
    std::uint32_t          format_version = kTreeFormatVersion;  ///< NOT 1
    std::vector<TreeEntry> entries;                              ///< sorted by name

    /// Structural checks only — no store access:
    ///   format_version == kTreeFormatVersion
    ///   at least one entry (an empty tree is a commit with no files)
    ///   names strictly ascending (so: sorted, and no duplicates)
    ///   names are plain file names: non-empty, no '/', no '\', no NUL or
    ///     other control bytes, not "." and not ".."
    ///   no entry addresses the null oid
    [[nodiscard]] Status validate() const;

    /// Sorts `entries` by name, then validates. This is the constructor to
    /// reach for; the aggregate is left public only so that parse() and tests
    /// can build one directly.
    [[nodiscard]] static Result<Tree> make(std::vector<TreeEntry> entries);

    [[nodiscard]] const TreeEntry* find(std::string_view name) const noexcept;

    /// Canonical JSON (docs/spec/10 §1.4): sorted keys, no whitespace, no
    /// trailing newline.
    [[nodiscard]] std::vector<std::byte> to_canonical_json() const;
    [[nodiscard]] Oid oid() const;

    /// Parses and, on success, verifies that re-serialising reproduces the
    /// same bytes — the same round-trip check Commit::parse and
    /// Manifest::parse apply.
    [[nodiscard]] static Result<Tree> parse(std::span<const std::byte>);
};

/// True if `name` is acceptable as a TreeEntry name. Exposed because the
/// checkout and mount paths need the same answer before they create anything
/// on a real filesystem.
[[nodiscard]] bool is_valid_tree_entry_name(std::string_view) noexcept;

}  // namespace sfs::format
