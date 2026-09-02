#pragma once
/// Internal to modules/align/src. Ties fingerprint.hpp + auction.hpp
/// together into one group solve for the large-group path (docs/adr/0012),
/// mirroring cpp/src/dispatch.cpp's "synapse-forward" branch: fingerprint +
/// top-K candidates + sparse true cost + Jacobi auction, widening K and
/// retrying if too many units end up unassigned, then an exact dense LAP
/// (reusing lap.cpp's solve_jv, not a separate port) on the small leftover.

#include <torch/torch.h>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/propagate.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

/// Squared weight-space distance on `cand`'s (n, K) candidate set. Reads
/// TARGET in row_tile-sized chunks; for each chunk, reads only the specific
/// BASE rows that chunk's candidates actually reference (deduplicated via
/// core::to_runs), never a whole tensor. `solved` reorders each TARGET row's
/// other axis into BASE's frame first, same convention as matcher.cpp's
/// build_features.
[[nodiscard]] core::Result<torch::Tensor> sparse_true_cost(core::ITensorSource& target, core::ITensorSource& base,
                                                           const core::Topology& topo, std::string_view group,
                                                           const PermutationMap& solved, const torch::Tensor& cand,
                                                           std::uint32_t row_tile);

struct NullRepairResult {
    std::int64_t n_repaired = 0;
    double       cost = 0.0;
};

/// Exact dense LAP (lap.cpp's solve_jv) on the auction's leftover block:
/// bidders auction left unassigned vs. objects nothing claimed. Mutates
/// `assign` in place. The leftover block is expected to be small (a
/// fraction of n after widen_on_null_rate retries), so a dense sub-solve
/// here is cheap regardless of how large the group itself is.
[[nodiscard]] core::Result<NullRepairResult> sparse_null_repair(core::ITensorSource& target,
                                                                core::ITensorSource& base,
                                                                const core::Topology& topo,
                                                                std::string_view group,
                                                                const PermutationMap& solved, torch::Tensor& assign);

/// One group's full sparse solve: fingerprint both sides, generate
/// candidates, score them, auction, widen-and-retry if needed, repair the
/// leftover, and package the result the same shape as the dense path's
/// GroupMatch.
[[nodiscard]] core::Result<GroupMatch> match_group_sparse(core::ITensorSource& target, core::ITensorSource& base,
                                                          const core::Topology& topo, std::string_view group,
                                                          std::uint32_t n, const PermutationMap& solved,
                                                          const SparseMatchOptions& opts,
                                                          const ConfidenceOptions& conf_opts);

}  // namespace sfs::align
