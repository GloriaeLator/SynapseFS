#include <synapsefs/align/lap.hpp>

#include <algorithm>
#include <utility>
#include <vector>

#include "rectangular_lsap.h"

namespace sfs::align {

using core::LapResult;
using core::Result;

Result<LapResult> solve_jv(std::span<const float> cost, std::uint32_t n) {
    if (n == 0) return LapResult{};
    if (cost.size() != static_cast<std::size_t>(n) * n) {
        return SFS_ERR(Internal, "cost matrix size does not match n*n", std::to_string(cost.size()));
    }

    std::vector<double> cost_d(cost.begin(), cost.end());
    std::vector<std::int64_t> a(n), b(n);
    const int ret = solve_rectangular_linear_sum_assignment(n, n, cost_d.data(), /*maximize=*/false,
                                                            a.data(), b.data());
    if (ret == RECTANGULAR_LSAP_INFEASIBLE) {
        return SFS_ERR(Internal, "LAP solver: infeasible cost matrix", "");
    }
    if (ret == RECTANGULAR_LSAP_INVALID) {
        return SFS_ERR(Internal, "LAP solver: invalid cost matrix", "");
    }

    LapResult result;
    result.assignment.assign(n, 0);
    double total = 0.0;
    for (std::uint32_t k = 0; k < n; ++k) {
        const auto row = static_cast<std::uint32_t>(a[k]);
        const auto col = static_cast<std::uint32_t>(b[k]);
        result.assignment[row] = col;
        total += static_cast<double>(cost[static_cast<std::size_t>(row) * n + col]);
    }
    result.cost_raw = total;
    result.exact = true;
    return result;
}

namespace {

class JvSolver final : public core::ILapSolver {
public:
    Result<LapResult> solve(std::span<const float> cost, std::uint32_t n) override {
        return solve_jv(cost, n);
    }
    [[nodiscard]] std::string_view name() const override { return "jonker-volgenant"; }
};

// Sort every (i, j) candidate pair ascending by cost, assign greedily
// (skipping a pair whose row or column is already taken), then repeatedly
// scan all assigned-row pairs for a 2-swap that lowers total cost, until a
// full pass finds none or `max_passes_` is hit. ADR 0004: "the crossover and
// the accuracy cost of the fallback are recorded", not waved at.
class GreedySolver final : public core::ILapSolver {
public:
    explicit GreedySolver(std::uint32_t max_passes) : max_passes_(max_passes) {}

    Result<LapResult> solve(std::span<const float> cost, std::uint32_t n) override {
        if (n == 0) return LapResult{};
        if (cost.size() != static_cast<std::size_t>(n) * n) {
            return SFS_ERR(Internal, "cost matrix size does not match n*n", std::to_string(cost.size()));
        }

        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
        pairs.reserve(static_cast<std::size_t>(n) * n);
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j < n; ++j) pairs.emplace_back(i, j);
        }
        std::sort(pairs.begin(), pairs.end(), [&](auto lhs, auto rhs) {
            return cost[static_cast<std::size_t>(lhs.first) * n + lhs.second] <
                   cost[static_cast<std::size_t>(rhs.first) * n + rhs.second];
        });

        std::vector<std::uint32_t> assignment(n, kUnassigned);
        std::vector<bool> row_taken(n, false);
        std::vector<bool> col_taken(n, false);
        std::uint32_t assigned = 0;
        for (const auto& [i, j] : pairs) {
            if (row_taken[i] || col_taken[j]) continue;
            assignment[i] = j;
            row_taken[i] = true;
            col_taken[j] = true;
            if (++assigned == n) break;
        }
        // A square cost matrix always admits a perfect greedy assignment by
        // this construction (every row eventually finds some free column),
        // so `assigned == n` here; no null-object fallback is needed the way
        // the sparse top-K auction design (not used here) would require one.

        std::uint32_t iterations = 0;
        bool improved = true;
        while (improved && iterations < max_passes_) {
            improved = false;
            ++iterations;
            for (std::uint32_t i1 = 0; i1 < n; ++i1) {
                for (std::uint32_t i2 = i1 + 1; i2 < n; ++i2) {
                    const std::uint32_t j1 = assignment[i1];
                    const std::uint32_t j2 = assignment[i2];
                    const float current = cost[static_cast<std::size_t>(i1) * n + j1] +
                                          cost[static_cast<std::size_t>(i2) * n + j2];
                    const float swapped = cost[static_cast<std::size_t>(i1) * n + j2] +
                                          cost[static_cast<std::size_t>(i2) * n + j1];
                    if (swapped < current) {
                        std::swap(assignment[i1], assignment[i2]);
                        improved = true;
                    }
                }
            }
        }

        LapResult result;
        result.assignment = std::move(assignment);
        double total = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) {
            total += static_cast<double>(cost[static_cast<std::size_t>(i) * n + result.assignment[i]]);
        }
        result.cost_raw = total;
        result.exact = false;
        result.iterations = iterations;
        return result;
    }

    [[nodiscard]] std::string_view name() const override { return "greedy+2swap"; }

private:
    static constexpr std::uint32_t kUnassigned = static_cast<std::uint32_t>(-1);
    std::uint32_t max_passes_;
};

class AutoSolver final : public core::ILapSolver {
public:
    AutoSolver(std::uint32_t crossover, std::unique_ptr<ILapSolver> exact,
              std::unique_ptr<ILapSolver> approx)
        : crossover_(crossover), exact_(std::move(exact)), approx_(std::move(approx)) {}

    Result<LapResult> solve(std::span<const float> cost, std::uint32_t n) override {
        return (n < crossover_) ? exact_->solve(cost, n) : approx_->solve(cost, n);
    }
    [[nodiscard]] std::string_view name() const override {
        return "auto(jonker-volgenant|greedy+2swap)";
    }

private:
    std::uint32_t crossover_;
    std::unique_ptr<ILapSolver> exact_;
    std::unique_ptr<ILapSolver> approx_;
};

}  // namespace

std::unique_ptr<ILapSolver> make_jv_solver() { return std::make_unique<JvSolver>(); }

std::unique_ptr<ILapSolver> make_greedy_solver(std::uint32_t max_passes) {
    return std::make_unique<GreedySolver>(max_passes);
}

std::unique_ptr<ILapSolver> make_auto_solver(std::uint32_t crossover) {
    return std::make_unique<AutoSolver>(crossover, make_jv_solver(), make_greedy_solver());
}

}  // namespace sfs::align
