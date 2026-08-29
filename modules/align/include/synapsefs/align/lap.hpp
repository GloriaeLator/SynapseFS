#pragma once
/// \file lap.hpp
/// Linear assignment. Implements core::ILapSolver.
///
/// Exact Jonker-Volgenant below a MEASURED crossover; greedy plus local 2-swap
/// refinement above it. The crossover and the accuracy it costs are numbers we
/// must be able to state — "we used greedy above n = X, and it costs 0.Y%
/// accuracy for a Z-times speedup" is an answer, "greedy is faster" is not.
/// research/lap_bench.py, docs/benchmarks.md §1.
///
/// The solver never sees a checkpoint, which makes test_lap.cpp a pure
/// algorithmic test against planted optima.

#include <cstdint>
#include <memory>

#include <synapsefs/core/interfaces.hpp>

namespace sfs::align {

using core::ILapSolver;
using core::LapResult;
using core::Result;

/// Exact. O(n^3) worst case, much better in practice on these matrices.
[[nodiscard]] std::unique_ptr<ILapSolver> make_jv_solver();

/// Greedy by ascending cost, then 2-swap until no swap improves or
/// `max_passes` is reached.
[[nodiscard]] std::unique_ptr<ILapSolver> make_greedy_solver(std::uint32_t max_passes = 8);

/// Picks by size. `crossover` defaults to the measured value; override from
/// config or from the benchmark harness.
[[nodiscard]] std::unique_ptr<ILapSolver> make_auto_solver(std::uint32_t crossover = 4096);

/// Free function form, for benchmarks and tests that do not want a solver
/// object. `cost` is row-major n x n.
[[nodiscard]] Result<LapResult> solve_jv(std::span<const float> cost, std::uint32_t n);

}  // namespace sfs::align
