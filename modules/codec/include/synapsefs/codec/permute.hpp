#pragma once
/// \file permute.hpp
/// Applying a permutation to output units.
///
/// Units are contiguous on both sides, so this is a gather of memcpys, not an
/// index shuffle. Target unit i comes from base unit p[i].

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/tensor.hpp>

namespace sfs::codec {

using core::Result;
using core::Status;

/// dst[i] = src[perm[i]], unit-wise. `dst` and `src` must not overlap.
void permute_units(std::span<std::byte> dst, std::span<const std::byte> src,
                   std::span<const std::uint32_t> perm, std::uint64_t unit_bytes) noexcept;

/// The base units a target frame covering [first, first+count) depends on,
/// collapsed into ascending runs so the frame costs a few reads rather than one
/// per unit. docs/spec/12-residual-format.md §4.2.
[[nodiscard]] std::vector<core::UnitRun> dependency_runs(
    std::span<const std::uint32_t> perm, std::uint64_t first, std::uint64_t count);

/// Expand a group permutation over a blocked axis. Identical to
/// core::expand_permutation, re-exported here because this is where callers
/// look for it.
[[nodiscard]] std::vector<std::uint32_t> expand(std::span<const std::uint32_t> perm,
                                                std::uint32_t block);

}  // namespace sfs::codec
