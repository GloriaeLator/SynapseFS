#pragma once
/// \file repo_config.hpp
/// .synapsefs/config — machine-local policy, NOT an object and not part of
/// history. docs/spec/11-repo-layout.md §5.
///
/// The two parameters that are baked into written objects (chunk_bytes,
/// frame_bytes) are read back from the object header, not from here, so
/// changing them does not invalidate anything already stored.

#include <cstdint>
#include <filesystem>
#include <string>

#include <synapsefs/core/error.hpp>

namespace sfs::core {

inline constexpr std::uint32_t kFormatVersion = 1;

struct RepoConfig {
    std::uint32_t format_version = kFormatVersion;

    /// Verification granularity. Baked into every block written while in effect.
    std::uint64_t chunk_bytes = 64u * 1024;

    /// Residual frame target, rounded to whole output units. Decided together
    /// with max_chain_depth, not separately: small frames are what make a deep
    /// chain tolerable.
    std::uint64_t frame_bytes = 128u * 1024;

    /// Bounds read LATENCY. A hundred 0.1% deltas cost a hundred hops.
    std::uint32_t max_chain_depth = 5;

    /// Bounds SPACE. Snapshot when a delta exceeds alpha x the full block: the
    /// XOR of unrelated fp16 is high-entropy noise that compresses to LARGER
    /// than its input, so without this a badly aligned group can make the
    /// repository grow faster than storing full copies.
    double snapshot_alpha = 0.5;

    bool compress_raw = false;   ///< fp16 weights do not compress usefully

    std::uint64_t cache_bytes = 1024ull * 1024 * 1024;  ///< mount LRU budget
    std::string   listen      = "127.0.0.1:9418";       ///< `sfs serve` default

    [[nodiscard]] static Result<RepoConfig> load(const std::filesystem::path& repo_root);
    [[nodiscard]] Status save(const std::filesystem::path& repo_root) const;
    [[nodiscard]] Status validate() const;
};

/// Standard paths under a repository root.
struct RepoPaths {
    std::filesystem::path root;          ///< the directory containing .synapsefs

    [[nodiscard]] std::filesystem::path dot() const;       ///< <root>/.synapsefs
    [[nodiscard]] std::filesystem::path objects() const;
    [[nodiscard]] std::filesystem::path pack() const;
    [[nodiscard]] std::filesystem::path tmp() const;
    [[nodiscard]] std::filesystem::path incoming() const;
    [[nodiscard]] std::filesystem::path refs_heads() const;
    [[nodiscard]] std::filesystem::path head() const;
    [[nodiscard]] std::filesystem::path journal() const;
    [[nodiscard]] std::filesystem::path lock() const;
    [[nodiscard]] std::filesystem::path object_path(const class Oid&) const;

    /// Search upward from `start` for a directory containing .synapsefs.
    [[nodiscard]] static Result<RepoPaths> discover(const std::filesystem::path& start);
};

}  // namespace sfs::core
