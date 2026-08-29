#pragma once
/// \file commit.hpp
/// The commit object. docs/spec/10-object-model.md §3.
///
/// Deliberately absent, each because it was a bug:
///   parent       - disagreed with `parents`
///   branch       - refs own branch membership; a commit can be on many
///   commit_hash  - it is the storage key, and a self-referential field cannot
///                  be verified

#include <cstdint>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::format {

using core::Oid;
using core::Result;

struct Commit {
    std::uint32_t    format_version = 1;
    std::vector<Oid> parents;      ///< [] root, [x] normal, [x,y] merge; >2 rejected
    Oid              manifest;
    Oid              topology;     ///< so a PULLED repo can decode itself
    std::string      timestamp;    ///< RFC 3339, UTC, second precision, always 'Z'
    std::string      author;       ///< free text, NOT authenticated
    std::string      message;

    [[nodiscard]] bool is_root()  const noexcept { return parents.empty(); }
    [[nodiscard]] bool is_merge() const noexcept { return parents.size() == 2; }

    /// Canonical JSON (docs/spec/10 §1.4). The serialisation IS the address, so
    /// this must be byte-stable: sorted keys, no whitespace, no trailing newline.
    [[nodiscard]] std::vector<std::byte> to_canonical_json() const;
    [[nodiscard]] Oid oid() const;

    /// Parses and, on success, verifies that re-serialising reproduces the same
    /// bytes. A reader that skips that check cannot claim `verify` means
    /// anything for JSON objects.
    [[nodiscard]] static Result<Commit> parse(std::span<const std::byte>);
};

/// RFC 3339 UTC, second precision.
[[nodiscard]] std::string now_timestamp();
[[nodiscard]] bool is_valid_timestamp(std::string_view);

}  // namespace sfs::format
