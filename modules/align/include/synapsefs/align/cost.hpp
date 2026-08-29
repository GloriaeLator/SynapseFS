#pragma once
/// \file cost.hpp
/// The alignment cost matrix.
///
/// Cost(i, j) = -<normalised parameter slice of target unit i,
///                normalised parameter slice of base unit j>
/// gathered across EVERY tensor in the group and both sides of each unit:
/// outgoing rows/filters, incoming column slices permuted by their own axis's
/// current permutation, and folded norm statistics.
///
/// L2-normalising per unit matters: without it, a group whose units differ
/// wildly in magnitude has its assignment decided by scale rather than by
/// direction. The ablation is research/cost_ablation.py and the chosen form
/// belongs in docs/benchmarks.md with numbers.

#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

using core::Result;
using core::Status;

enum class CostMetric : std::uint8_t {
    NegInnerProduct,   ///< default
    CosineDistance,
    L2,
};

struct CostOptions {
    CostMetric metric = CostMetric::NegInnerProduct;
    bool normalize_units = true;
    bool include_incoming = true;
    bool include_norm_stats = true;   ///< BN statistics are highly discriminative
                                      ///< and cost one scalar per unit
};

/// Row-major n x n accumulator. Held in RAM: n is the GROUP size, not the
/// parameter count, so 8192 units is 256 MB and a 7B model is fine — that is
/// why out-of-core alignment works at all.
class CostMatrix {
public:
    explicit CostMatrix(std::uint32_t n);

    [[nodiscard]] std::uint32_t n() const noexcept { return n_; }
    [[nodiscard]] std::span<float> data() noexcept { return values_; }
    [[nodiscard]] std::span<const float> data() const noexcept { return values_; }

    [[nodiscard]] float& at(std::uint32_t i, std::uint32_t j) noexcept {
        return values_[static_cast<std::size_t>(i) * n_ + j];
    }

    /// Accumulate the contribution of one tile pair. Called from a thread pool,
    /// one task per tile; tiles are disjoint so no locking is needed.
    void accumulate_tile(std::span<const float> target_tile, std::uint32_t target_first,
                         std::uint32_t target_count, std::span<const float> base_tile,
                         std::uint32_t base_first, std::uint32_t base_count,
                         std::uint32_t vec_len, const CostOptions&);

    /// Cost of the identity assignment — the denominator for the normalised
    /// cost that decides `alignable`.
    [[nodiscard]] double identity_cost() const noexcept;
    [[nodiscard]] double assignment_cost(std::span<const std::uint32_t> perm) const noexcept;

private:
    std::uint32_t      n_ = 0;
    std::vector<float> values_;
};

}  // namespace sfs::align
