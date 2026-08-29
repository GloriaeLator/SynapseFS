#pragma once
/// \file graph.hpp
/// Union-find over tensor axes — the machinery behind permutation groups.
///
/// A ResNet block adds a skip connection onto the main path, so the block's
/// output channels, the shortcut's output channels and every downstream
/// consumer share ONE permutation. That is a set, not a chain of pairs, which
/// is exactly what pairwise links (permute_input_from and friends) could not
/// express.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

/// One (tensor, axis) pair.
struct AxisKey {
    std::string tensor;
    std::uint32_t dim = 0;
    friend bool operator==(const AxisKey&, const AxisKey&) noexcept = default;
};

struct AxisKeyHash {
    [[nodiscard]] std::size_t operator()(const AxisKey&) const noexcept;
};

class AxisUnionFind {
public:
    /// Register an axis of length `len`. Returns its handle.
    std::uint32_t add(const AxisKey&, std::uint64_t len);

    /// Union two axes into one permutation group. `block_a` and `block_b` are
    /// the blocking factors that relate each axis to the shared group size —
    /// pass 1 unless one side is a flattened axis.
    [[nodiscard]] core::Status unite(std::uint32_t a, std::uint32_t b);

    [[nodiscard]] std::uint32_t find(std::uint32_t) noexcept;
    void pin(std::uint32_t handle) noexcept;

    /// Collapse to group ids, deriving each axis's blocking factor as
    /// axis_len / group_size. A non-integer ratio is BlockFactorMismatch.
    [[nodiscard]] core::Result<core::Topology> finalize();

private:
    struct Node;
    std::vector<Node> nodes_;
    std::unordered_map<AxisKey, std::uint32_t, AxisKeyHash> index_;
};

/// Structural relations the parser feeds in. Each is a reason two axes must
/// share a permutation.
enum class Relation : std::uint8_t {
    Sequential,     ///< layer l's output axis with layer l+1's input axis
    NormFollows,    ///< BatchNorm/LayerNorm parameters with the layer they follow
    ResidualAdd,    ///< both branches of a skip connection
    Flatten,        ///< a conv's output channels with a linear's blocked input axis
    Bias,           ///< a bias vector with its layer's output axis
};

}  // namespace sfs::align
