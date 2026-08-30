#include <synapsefs/align/graph.hpp>

#include <algorithm>
#include <limits>

namespace sfs::align {

std::size_t AxisKeyHash::operator()(const AxisKey& k) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(k.tensor);
    std::size_t h2 = std::hash<std::uint32_t>{}(k.dim);
    return h1 ^ (h2 + 0x9E3779B97F4A7C15ULL + (h1 << 6) + (h1 >> 2));
}

std::uint32_t AxisUnionFind::add(const AxisKey& key, std::uint64_t len) {
    if (auto it = index_.find(key); it != index_.end()) return it->second;
    const auto handle = static_cast<std::uint32_t>(nodes_.size());
    nodes_.push_back(Node{key, len, handle, 0, false});
    index_.emplace(key, handle);
    return handle;
}

std::uint32_t AxisUnionFind::find(std::uint32_t h) noexcept {
    // Path halving: cheap, iterative, no recursion depth concern on a graph
    // with tens of thousands of axes.
    while (nodes_[h].parent != h) {
        nodes_[h].parent = nodes_[nodes_[h].parent].parent;
        h = nodes_[h].parent;
    }
    return h;
}

core::Status AxisUnionFind::unite(std::uint32_t a, std::uint32_t b) {
    std::uint32_t ra = find(a);
    std::uint32_t rb = find(b);
    if (ra == rb) return {};

    if (nodes_[ra].rank < nodes_[rb].rank) std::swap(ra, rb);
    nodes_[rb].parent = ra;
    if (nodes_[ra].rank == nodes_[rb].rank) ++nodes_[ra].rank;
    nodes_[ra].pinned = nodes_[ra].pinned || nodes_[rb].pinned;
    return {};
}

void AxisUnionFind::pin(std::uint32_t handle) noexcept {
    nodes_[find(handle)].pinned = true;
}

core::Result<core::Topology> AxisUnionFind::finalize() {
    core::Topology topo;

    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> members_by_root;
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        members_by_root[find(i)].push_back(i);
    }

    // Group size is the SMALLEST axis length in the set: a flatten only ever
    // multiplies an axis (conv channels -> flattened features), never
    // divides, so the narrowest member is always the true unit count and
    // every other member is a multiple of it (SPEC 13 §2.2).
    std::unordered_map<std::uint32_t, std::string> root_to_group;
    std::uint32_t next_id = 0;
    for (const auto& [root, members] : members_by_root) {
        std::uint64_t min_len = std::numeric_limits<std::uint64_t>::max();
        bool pinned = false;
        for (std::uint32_t m : members) {
            min_len = std::min(min_len, nodes_[m].len);
            pinned = pinned || nodes_[m].pinned;
        }
        std::string name = "g" + std::to_string(next_id++);
        root_to_group.emplace(root, name);
        topo.groups.emplace(name, core::PermGroup{static_cast<std::uint32_t>(min_len), pinned});
    }

    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        const std::string& gname = root_to_group.at(find(i));
        const std::uint64_t group_size = topo.groups.at(gname).size;
        const Node& node = nodes_[i];
        if (group_size == 0 || node.len % group_size != 0) {
            return SFS_ERR(BlockFactorMismatch,
                          "axis length is not an integer multiple of its group's unit count",
                          node.key.tensor + " dim=" + std::to_string(node.key.dim) + " len=" +
                              std::to_string(node.len) + " group_size=" + std::to_string(group_size));
        }
        const auto block = static_cast<std::uint32_t>(node.len / group_size);
        topo.tensors[node.key.tensor].axes.push_back(core::AxisBinding{node.key.dim, gname, block});
    }

    return topo;
}

}  // namespace sfs::align
