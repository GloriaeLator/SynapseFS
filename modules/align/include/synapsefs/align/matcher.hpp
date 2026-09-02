#pragma once
/// \file matcher.hpp
/// The alignment engine: cost, solve, propagate, decide.
///
/// Weight matching, not activation matching: a version control system must be
/// able to store a checkpoint given nothing but the checkpoint. Requiring a
/// dataset and a forward pass to commit is a different product, and it would
/// make diffs non-deterministic so that content addressing deduplicates
/// nothing. docs/adr/0004.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <synapsefs/align/confidence.hpp>
#include <synapsefs/align/cost.hpp>
#include <synapsefs/align/ooc_plan.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

/// Knobs for the large-group sparse path (fingerprint + candidate generation
/// + Jacobi auction, docs/adr/0012): used once a group's size reaches
/// `MatchOptions::sparse_crossover`, where a dense n x n cost matrix and an
/// O(n^2) greedy sort are no longer just slow but literally do not fit in
/// the PS's 16 GB RAM ceiling (docs/adr/0008). Defaults match the values the
/// project's earlier prototype (cpp/) shipped, not a live per-host
/// calibration -- see the ADR on why that's a reasonable v1 scope cut.
struct SparseMatchOptions {
    int64_t K              = 4;      ///< candidates per unit -- floor; match_group_sparse
                                     ///< scales the actual starting K up with group size
                                     ///< (roughly sqrt(n)), since the fingerprint's top-K
                                     ///< pruning -- not the auction, which is exact for
                                     ///< whatever candidate graph it gets -- is the sparse
                                     ///< path's only source of approximation error, and
                                     ///< the memory this path exists to save has orders of
                                     ///< magnitude of headroom left even at K in the
                                     ///< hundreds (docs/adr/0012).
    int     n_quantiles    = 16;     ///< fingerprint width for a "long" row
    int64_t short_row_D    = 512;    ///< rows at or below this width use up to 32 quantiles
    double  eta            = 10.0;   ///< private-null-object price multiplier
    double  eps_shrink     = 5.0;    ///< auction epsilon-scaling divisor
    int64_t bid_guard      = 400;    ///< rounds-per-phase cap
    double  widen_on_null_rate = 0.02;  ///< retry with a wider K above this unassigned fraction
    /// Ceiling for both the size-scaled starting K and the null-triggered
    /// widen-retry. 512 is still cheap: at n=57943 (ADR 0012's measured
    /// large-group case) the K=512 cost matrix is ~120 MB against the dense
    /// path's ~13.4 GB. A heuristic starting point, not an empirically
    /// calibrated one -- validate against real large checkpoints, the same
    /// caveat ConfidenceOptions::distinct_match_floor carries.
    int64_t max_K          = 512;
    std::uint32_t row_tile = 1024;   ///< rows per chunk when reading a huge tensor
};

struct MatchOptions {
    CostOptions   cost;
    MemoryBudget  budget;
    std::uint32_t max_sweeps   = 8;    ///< propagation cap
    std::uint32_t lap_crossover = 4096;
    /// Above this group size, match_group bypasses CostMatrix/ILapSolver
    /// entirely and uses the sparse fingerprint+auction path instead (see
    /// SparseMatchOptions) -- a dense path at this size doesn't just get
    /// slow, it doesn't fit in memory (docs/adr/0012).
    std::uint32_t sparse_crossover = 8192;
    SparseMatchOptions sparse;
    ConfidenceOptions confidence;
};

struct GroupMatch {
    std::string                group;
    std::vector<std::uint32_t> permutation;   ///< empty means identity
    bool   identity  = true;
    bool   alignable = true;
    double cost_raw = 0.0;
    double cost_normalized = 0.0;
    bool   exact_solver = false;
};

struct MatchReport {
    std::unordered_map<std::string, GroupMatch> groups;
    std::uint32_t sweeps = 0;         ///< 1-2 on fine-tune pairs; a nice number to show
    std::uint64_t peak_bytes = 0;
    double        wall_seconds = 0.0;
};

class Matcher {
public:
    Matcher(core::ITensorSource& base, core::ITensorSource& target,
            const core::Topology&, MatchOptions = {});
    ~Matcher();

    /// Streams both checkpoints; neither is ever fully resident.
    [[nodiscard]] core::Result<MatchReport> run();

    /// One group, for tests that plant a permutation and check recovery.
    [[nodiscard]] core::Result<GroupMatch> match_group(std::string_view group);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfs::align
