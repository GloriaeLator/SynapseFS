#pragma once
/// Internal to modules/align/src -- shared by matcher.cpp and sparse_match.cpp
/// so "bring a row's other axis into the solved reference frame" has exactly
/// one implementation (see matcher.cpp's build_features for the original
/// bug this fixed: a row's OTHER axis needs reordering by that axis's own
/// group's solved permutation before two sides' rows are comparable).

#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/align/propagate.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align::detail {

/// The dim-0 binding of a tensor, if it has one.
inline const core::AxisBinding* dim0_binding(const core::TensorAxes& axes) {
    for (const auto& b : axes.axes) {
        if (b.dim == 0) return &b;
    }
    return nullptr;
}

/// The first binding on a different dim than `exclude_dim`, if any. Every
/// documented case (SPEC 13) has at most two grouped axes per tensor, so
/// "the other one" is unambiguous.
inline const core::AxisBinding* other_binding(const core::TensorAxes& axes, std::uint32_t exclude_dim) {
    for (const auto& b : axes.axes) {
        if (b.dim != exclude_dim) return &b;
    }
    return nullptr;
}

/// Reorders a [num_rows][row_width] buffer's columns by `group_perm`
/// (forward/scatter convention, per lap.cpp: assign[target_row] = base_col),
/// expanded by the block implied by row_width. Brings a row read from the
/// TARGET side into BASE's canonical frame along its other (non-group) axis,
/// once that axis's own group has a solved permutation; BASE's own rows are
/// never reordered (base is the fixed reference).
inline std::vector<float> reorder_columns(std::span<const float> data, std::uint64_t row_width,
                                          std::span<const std::uint32_t> group_perm) {
    const auto block = static_cast<std::uint32_t>(row_width / group_perm.size());
    const std::vector<std::uint32_t> full_perm = core::expand_permutation(group_perm, block);
    const std::vector<std::uint32_t> gather = core::invert_permutation(full_perm);
    std::vector<float> out(data.size());
    const std::uint64_t num_rows = data.size() / row_width;
    for (std::uint64_t r = 0; r < num_rows; ++r) {
        for (std::uint64_t k = 0; k < row_width; ++k) {
            out[r * row_width + k] = data[r * row_width + gather[k]];
        }
    }
    return out;
}

/// Reorders `raw` (TARGET's [count][row_width] chunk for `tensor`'s own
/// dim-0 evidence) in place if the tensor's other axis has a solved
/// permutation. No-op otherwise.
inline void reorder_if_solved(std::vector<float>& raw, const core::TensorAxes& axes, std::uint64_t row_width,
                              const PermutationMap& solved) {
    if (row_width <= 1) return;
    const core::AxisBinding* other = other_binding(axes, 0);
    if (other == nullptr) return;
    auto it = solved.find(other->group);
    if (it == solved.end() || it->second.empty() || row_width % it->second.size() != 0) return;
    raw = reorder_columns(raw, row_width, it->second);
}

}  // namespace sfs::align::detail
