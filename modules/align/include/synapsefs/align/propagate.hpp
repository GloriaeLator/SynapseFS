#pragma once
/// \file propagate.hpp
/// Coordinate descent across permutation groups.
///
/// A group's cost depends on its neighbours' permutations — an incoming column
/// slice must be permuted by the axis's own group before it can be compared —
/// so one pass is not enough. Sweep in topological order, re-solving each group
/// against the current state, until nothing changes or a cap is hit.
///
/// On fine-tune pairs convergence is typically 1-2 sweeps, because the answer
/// is close to identity. Log the count; it is worth showing.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

using PermutationMap = std::unordered_map<std::string, std::vector<std::uint32_t>>;

/// Groups in an order where a group's neighbours are settled before it, as far
/// as the graph allows. Cycles (residual connections) are broken arbitrarily
/// and resolved by the sweeps.
[[nodiscard]] std::vector<std::string> topological_group_order(const core::Topology&);

/// Groups whose cost matrices depend on `group`'s permutation.
[[nodiscard]] std::vector<std::string> dependents_of(const core::Topology&,
                                                     std::string_view group);

/// True when no permutation changed between two sweeps.
[[nodiscard]] bool converged(const PermutationMap& before, const PermutationMap& after);

}  // namespace sfs::align
