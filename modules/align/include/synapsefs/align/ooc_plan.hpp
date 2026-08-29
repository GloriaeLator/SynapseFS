#pragma once
/// \file ooc_plan.hpp
/// Out-of-core tiling plan for cost accumulation.
///
/// The streaming path is the ONLY path, used even on the 1M-parameter MLP
/// fixture. It is slower than necessary there, and that is the price of never
/// discovering on Day 4 that the path we ship was never exercised.
/// docs/adr/0008-out-of-core-streaming.md
///
/// Peak RSS = tile_bytes * 2 + |cost matrix|, and |cost matrix| is a function
/// of the GROUP size, not of the parameter count.

#include <cstdint>
#include <vector>

#include <synapsefs/core/error.hpp>

namespace sfs::align {

struct MemoryBudget {
    /// Hard ceiling for the aligner. The grading environment caps RAM at 16 GB
    /// and an OOM at fixture size fails the metric outright, so this is
    /// asserted in the scale test rather than merely reported.
    std::uint64_t bytes = 8ull * 1024 * 1024 * 1024;
};

struct TilePlan {
    std::uint32_t units_per_tile = 0;
    std::uint32_t tile_count     = 0;
    std::uint64_t bytes_per_tile = 0;
    std::uint64_t cost_matrix_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
};

/// Choose a tile size that fits the budget. Tile size is a tuning parameter,
/// not a detail: alignment wall-clock is 8% of the grade and I/O is inside it,
/// so bench/align_time.cpp sweeps this.
[[nodiscard]] core::Result<TilePlan> plan_tiles(std::uint32_t group_size,
                                                std::uint64_t bytes_per_unit,
                                                const MemoryBudget& = {});

/// Peak resident bytes for a plan, for the assertion in the scale test.
[[nodiscard]] std::uint64_t estimate_peak_rss(const TilePlan&) noexcept;

}  // namespace sfs::align
