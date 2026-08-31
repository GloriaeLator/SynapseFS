#include <synapsefs/align/ooc_plan.hpp>

#include <algorithm>

namespace sfs::align {

core::Result<TilePlan> plan_tiles(std::uint32_t group_size, std::uint64_t bytes_per_unit,
                                  const MemoryBudget& budget) {
    if (group_size == 0) return SFS_ERR(Internal, "group_size must be > 0", "");
    if (bytes_per_unit == 0) return SFS_ERR(Internal, "bytes_per_unit must be > 0", "");

    const auto cost_matrix_bytes =
        static_cast<std::uint64_t>(group_size) * group_size * sizeof(float);
    if (cost_matrix_bytes >= budget.bytes) {
        // ADR 0008's own escape hatch: the group size, not the parameter
        // count, is what bounds this, and if IT overflows the budget the fix
        // is a blocked LAP, not a bigger tile.
        return SFS_ERR(Internal,
                      "group's n*n cost matrix alone exceeds the memory budget -- needs a "
                      "blocked LAP, not a smaller tile",
                      std::to_string(group_size));
    }

    const std::uint64_t remaining = budget.bytes - cost_matrix_bytes;
    const std::uint64_t max_units_per_tile = remaining / (2 * bytes_per_unit);
    if (max_units_per_tile == 0) {
        return SFS_ERR(Internal, "memory budget too small to hold even one unit per tile", "");
    }
    const auto units_per_tile =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(max_units_per_tile, group_size));

    TilePlan plan;
    plan.units_per_tile = units_per_tile;
    plan.tile_count = (group_size + units_per_tile - 1) / units_per_tile;
    plan.bytes_per_tile = static_cast<std::uint64_t>(units_per_tile) * bytes_per_unit;
    plan.cost_matrix_bytes = cost_matrix_bytes;
    plan.estimated_peak_bytes = plan.bytes_per_tile * 2 + cost_matrix_bytes;
    return plan;
}

std::uint64_t estimate_peak_rss(const TilePlan& plan) noexcept { return plan.estimated_peak_bytes; }

}  // namespace sfs::align
