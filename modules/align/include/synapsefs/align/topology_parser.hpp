#pragma once
/// \file topology_parser.hpp
/// Builds a core::Topology from a checkpoint plus its config.json.
/// docs/spec/13-topology-config.md §4.
///
/// The parser is ours by design: the PS expects teams to write one against
/// "this general shape" rather than depend on a library.
///
/// Two rules it must never break:
///   * Blocking factors are DERIVED (block = axis_len / group_size), never
///     hardcoded. A non-integer result is a parse error naming both tensors —
///     the alternative is W[:, p] silently returning the wrong shape and a
///     model that computes garbage with no exception raised.
///   * Every tensor in the FILE gets a group, even ones the config does not
///     model (num_batches_tracked). Unmodelled tensors get singleton pinned
///     groups and round-trip through the identical code path.

#include <filesystem>
#include <string>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

using core::Result;

struct ParseOptions {
    /// Names whose output axis is pinned: class identity is fixed, and
    /// permuting the classifier produces a file that reconstructs correctly and
    /// a model that is wrong. Byte-level tests cannot catch that, so it is
    /// explicit.
    std::vector<std::string> pinned_output_tensors;
    std::vector<std::string> pinned_input_tensors;
    bool strict = true;   ///< fail on anything unmodellable rather than guessing
};

struct ParseDiagnostics {
    std::vector<std::string> unmodelled_tensors;   ///< got singleton groups
    std::vector<std::string> derived_blocks;       ///< "9.weight dim=1 block=64"
    std::uint32_t            group_count = 0;
};

/// `config` may be empty, in which case the parser infers structure from tensor
/// names and shapes alone — enough for the MLP fixtures, not for ResNet.
[[nodiscard]] Result<core::Topology> parse_topology(const core::ITensorSource&,
                                                    std::span<const std::byte> config_json,
                                                    const ParseOptions& = {},
                                                    ParseDiagnostics* diag = nullptr);

[[nodiscard]] Result<core::Topology> parse_topology_file(const core::ITensorSource&,
                                                         const std::filesystem::path& config,
                                                         const ParseOptions& = {});

}  // namespace sfs::align
