#pragma once
/// \file topology.hpp
/// Permutation groups: the value types for docs/spec/13-topology-config.md.
///
/// One group id per tensor axis plus a blocking factor. This replaced pairwise
/// links (permute_input_from and friends), which could not express "these five
/// layers share one permutation" (a residual block) and carried no blocking
/// factor (a flatten into a linear layer). The second failure is the dangerous
/// one: applying a 16-element permutation to a 1024-wide axis is legal numpy
/// that silently returns the wrong shape.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::core {

struct PermGroup {
    std::uint32_t size   = 0;      ///< number of units
    bool          pinned = false;  ///< identity is the only legal permutation
};

struct AxisBinding {
    std::uint32_t dim   = 0;
    std::string   group;
    std::uint32_t block = 1;       ///< shape[dim] == group.size * block
};

struct TensorAxes {
    std::vector<AxisBinding> axes;  ///< only the axes that carry a permutation
};

struct Topology {
    std::uint32_t format_version = 1;
    std::string   arch;                                  ///< advisory
    std::optional<Oid> source_digest;                    ///< of the config.json, advisory
    std::unordered_map<std::string, PermGroup>  groups;
    std::unordered_map<std::string, TensorAxes> tensors;

    /// Every axis reference resolves, every group is non-empty, and for every
    /// listed axis shape[dim] == size * block. Called by the parser and again
    /// by `verify`.
    [[nodiscard]] Status validate(
        const std::unordered_map<std::string, std::vector<std::uint64_t>>& shapes) const;

    [[nodiscard]] const PermGroup* find_group(std::string_view) const noexcept;
};

/// A group permutation over an axis whose entries are `block`-sized runs.
/// Every structural case in scope reduces to this one function: residual adds
/// share a group id, a flatten derives block = axis_len / group_size, grouped
/// convs use smaller groups, classifiers are pinned.
[[nodiscard]] std::vector<std::uint32_t> expand_permutation(
    std::span<const std::uint32_t> perm, std::uint32_t block);

/// Validate that `perm` is a bijection on [0, n). MUST be called on any
/// permutation read from a file before it is used to index anything — a
/// malformed one otherwise turns a corrupt object into an out-of-bounds read.
[[nodiscard]] bool is_valid_permutation(std::span<const std::uint32_t> perm,
                                        std::uint32_t n);

[[nodiscard]] std::vector<std::uint32_t> invert_permutation(
    std::span<const std::uint32_t> perm);

}  // namespace sfs::core
