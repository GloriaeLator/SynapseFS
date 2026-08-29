#pragma once
/// \file manifest.hpp
/// The manifest describes a FILE, not a model. docs/spec/10-object-model.md §4.
///
/// This is the object that makes byte-exactness achievable: a verbatim header
/// block plus a buffer layout covering every byte, so that reconstruction is
/// concatenation rather than serialisation. Nothing to choose, nothing to get
/// wrong.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/tensor.hpp>

namespace sfs::format {

using core::BufferEntry;
using core::Oid;
using core::Result;
using core::Status;

enum class GroupMode : std::uint8_t { Full, Delta };

struct DeltaBase {
    Oid         commit;     ///< MUST be an ancestor of the commit holding this manifest
    std::string group;
};

struct GroupEntry {
    GroupMode                mode = GroupMode::Full;
    std::optional<Oid>       block;        ///< Full: kind Raw
    std::optional<DeltaBase> base;         ///< Delta
    std::optional<Oid>       diff_block;   ///< Delta: kind Diff
    std::uint32_t            chain_depth = 0;  ///< stored, so the policy check is O(1)
};

struct FileInfo {
    std::string   name;            ///< no path separators
    Oid           header_block;    ///< kind Header, verbatim including padding
    std::uint64_t total_bytes = 0;
    std::string   sha256;          ///< what reconstruction must produce; a witness,
                                   ///< not an address — the one place SHA-256 appears
};

struct Manifest {
    std::uint32_t                     format_version = 1;
    std::string                       hash_algo = "blake3";
    FileInfo                          file;
    std::vector<BufferEntry>          buffer;   ///< in BUFFER order; every tensor
    std::map<std::string, GroupEntry> groups;   ///< ordered, for canonical JSON

    /// Structural checks only — no store access:
    ///   buffer[0].off == 0, contiguous, sum + header == total_bytes
    ///   every buffer entry's group exists in `groups`
    ///   Full has `block`, Delta has `base` and `diff_block`
    ///   chain_depth == 0 for Full
    [[nodiscard]] Status validate() const;

    /// The ancestor invariant needs history, so it lives with the DAG walk in
    /// store/verify.hpp rather than here.

    [[nodiscard]] const GroupEntry* find_group(std::string_view) const noexcept;
    [[nodiscard]] std::uint32_t max_chain_depth() const noexcept;

    [[nodiscard]] std::vector<std::byte> to_canonical_json() const;
    [[nodiscard]] Oid oid() const;
    [[nodiscard]] static Result<Manifest> parse(std::span<const std::byte>);
};

}  // namespace sfs::format
