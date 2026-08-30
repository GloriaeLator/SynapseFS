#include <synapsefs/core/topology.hpp>

#include <algorithm>

namespace sfs::core {

Status Topology::validate(
    const std::unordered_map<std::string, std::vector<std::uint64_t>>& shapes) const {
    for (const auto& [group_name, group] : groups) {
        if (group.size == 0)
            return SFS_ERR(TopologyParse, "empty permutation group", group_name);
    }

    for (const auto& [tensor_name, axes] : tensors) {
        auto shape_it = shapes.find(tensor_name);
        if (shape_it == shapes.end())
            return SFS_ERR(TopologyIncomplete, "topology references unknown tensor",
                           tensor_name);
        const auto& shape = shape_it->second;

        for (const auto& axis : axes.axes) {
            if (axis.dim >= shape.size())
                return SFS_ERR(TopologyParse, "axis out of range", tensor_name);

            auto group_it = groups.find(axis.group);
            if (group_it == groups.end())
                return SFS_ERR(TopologyParse, "axis references unknown group", axis.group);

            std::uint64_t expected =
                static_cast<std::uint64_t>(group_it->second.size) * axis.block;
            if (shape[axis.dim] != expected)
                return SFS_ERR(BlockFactorMismatch,
                               "shape[dim] != group.size * block for " + tensor_name,
                               axis.group);
        }
    }
    return {};
}

const PermGroup* Topology::find_group(std::string_view name) const noexcept {
    auto it = groups.find(std::string(name));
    return it == groups.end() ? nullptr : &it->second;
}

std::vector<std::uint32_t> expand_permutation(std::span<const std::uint32_t> perm,
                                              std::uint32_t block) {
    std::vector<std::uint32_t> out;
    out.reserve(perm.size() * block);
    for (auto p : perm)
        for (std::uint32_t b = 0; b < block; ++b) out.push_back(p * block + b);
    return out;
}

bool is_valid_permutation(std::span<const std::uint32_t> perm, std::uint32_t n) {
    if (perm.size() != n) return false;
    std::vector<bool> seen(n, false);
    for (auto p : perm) {
        if (p >= n || seen[p]) return false;
        seen[p] = true;
    }
    return true;
}

std::vector<std::uint32_t> invert_permutation(std::span<const std::uint32_t> perm) {
    std::vector<std::uint32_t> inv(perm.size());
    for (std::uint32_t i = 0; i < perm.size(); ++i) inv[perm[i]] = i;
    return inv;
}

}  // namespace sfs::core
