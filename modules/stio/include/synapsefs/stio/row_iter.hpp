#pragma once
/// \file row_iter.hpp
/// Iteration over OUTPUT UNITS — one linear row, one conv filter, one norm
/// scalar. The atomic unit of everything: alignment costs, permutation,
/// residual frames.
///
/// Units are contiguous on BOTH sides of a permutation, which is the whole
/// reason this is the unit. Target unit i comes from base unit p[i], and each
/// is one memcpy. docs/spec/12-residual-format.md §2.

#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/tensor.hpp>

namespace sfs::stio {

using core::Result;

/// Reads a contiguous run of units from a tensor into a caller-owned buffer.
/// Never allocates per unit.
class UnitReader {
public:
    UnitReader(core::ITensorSource& src, std::string_view tensor, std::uint32_t dim);

    [[nodiscard]] std::uint64_t unit_count() const noexcept { return unit_count_; }
    [[nodiscard]] std::uint64_t unit_bytes() const noexcept { return unit_bytes_; }

    [[nodiscard]] Result<std::size_t> read(std::uint64_t first, std::uint64_t count,
                                           std::span<std::byte> out);

    /// Gather a scattered set of units (a frame's dependency set p[a:b]) into
    /// `out`, in the order given. Internally collapses `indices` into runs, so
    /// the cost is a few reads rather than one per unit.
    [[nodiscard]] Result<std::size_t> gather(std::span<const std::uint32_t> indices,
                                             std::span<std::byte> out);

private:
    core::ITensorSource* src_ = nullptr;
    std::string          tensor_;
    std::uint64_t        unit_count_ = 0;
    std::uint64_t        unit_bytes_ = 0;
};

/// How many whole units fit in a frame of `target_bytes`, minimum one.
/// Frames are sized in bytes and rounded to unit boundaries so that a frame is
/// never split across a unit — the permutation would make that meaningless.
[[nodiscard]] std::uint64_t units_per_frame(std::uint64_t unit_bytes,
                                            std::uint64_t target_bytes) noexcept;

/// Tile [0, unit_count) into frames of `units_per_frame` units.
[[nodiscard]] std::vector<core::UnitRun> plan_frames(std::uint64_t unit_count,
                                                     std::uint64_t units_per_frame);

}  // namespace sfs::stio
