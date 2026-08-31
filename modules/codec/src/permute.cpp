#include <synapsefs/codec/permute.hpp>

#include <algorithm>
#include <cstring>

#include <synapsefs/core/topology.hpp>

namespace sfs::codec {

void permute_units(std::span<std::byte> dst, std::span<const std::byte> src,
                   std::span<const std::uint32_t> perm, std::uint64_t unit_bytes) noexcept {
    // dst[i] = src[perm[i]], unit-wise: a gather of memcpys, since output
    // units are contiguous on both sides (spec 12 §2).
    for (std::size_t i = 0; i < perm.size(); ++i) {
        std::memcpy(dst.data() + i * unit_bytes,
                   src.data() + static_cast<std::uint64_t>(perm[i]) * unit_bytes, unit_bytes);
    }
}

std::vector<core::UnitRun> dependency_runs(std::span<const std::uint32_t> perm,
                                           std::uint64_t first, std::uint64_t count) {
    // perm[first, first+count) is the TARGET-order slice; the dependency set
    // is the base indices it names, as a SET — which is what has to be
    // sorted before collapsing into runs. Permutation destroys locality (a
    // reversal maps an ascending target range to a descending base range),
    // but the set is still often contiguous, and core::to_runs only
    // collapses an already-ascending sequence. spec 12 §4.2.
    const auto slice = perm.subspan(first, count);
    std::vector<std::uint32_t> indices(slice.begin(), slice.end());
    std::sort(indices.begin(), indices.end());
    return core::to_runs(indices);
}

std::vector<std::uint32_t> expand(std::span<const std::uint32_t> perm, std::uint32_t block) {
    // Identical to core::expand_permutation; re-exported here per this
    // header's own doc comment, because this is where callers look for it.
    return core::expand_permutation(perm, block);
}

}  // namespace sfs::codec
