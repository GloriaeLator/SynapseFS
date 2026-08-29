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

struct MatchOptions {
    CostOptions   cost;
    MemoryBudget  budget;
    std::uint32_t max_sweeps   = 8;    ///< propagation cap
    std::uint32_t lap_crossover = 4096;
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
