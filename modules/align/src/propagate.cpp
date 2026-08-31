#include <synapsefs/align/propagate.hpp>

#include <algorithm>
#include <set>

namespace sfs::align {

namespace {

/// The dim-0 binding of a tensor, if it has one -- the axis that this
/// tensor's group "owns" (SPEC 13's tensors are named for their output axis
/// group; any other listed axis is an input the group's cost depends on).
const core::AxisBinding* dim0_binding(const core::TensorAxes& axes) {
    for (const auto& b : axes.axes) {
        if (b.dim == 0) return &b;
    }
    return nullptr;
}

}  // namespace

std::vector<std::string> topological_group_order(const core::Topology& topo) {
    std::unordered_map<std::string, std::set<std::string>> deps;
    for (const auto& [name, group] : topo.groups) deps[name] = {};

    for (const auto& [tensor, axes] : topo.tensors) {
        const core::AxisBinding* owner = dim0_binding(axes);
        if (owner == nullptr) continue;
        for (const auto& b : axes.axes) {
            if (b.dim != 0 && b.group != owner->group) deps[owner->group].insert(b.group);
        }
    }

    std::vector<std::string> order;
    std::vector<std::string> ready;
    auto remaining = deps;
    for (const auto& [name, d] : deps) {
        if (d.empty()) ready.push_back(name);
    }

    while (!ready.empty()) {
        std::sort(ready.begin(), ready.end());
        std::string p = ready.front();
        ready.erase(ready.begin());
        order.push_back(p);
        for (const auto& [name, _] : deps) {
            auto it = remaining[name].find(p);
            if (it == remaining[name].end()) continue;
            remaining[name].erase(it);
            const bool already_placed = std::find(order.begin(), order.end(), name) != order.end();
            const bool already_ready = std::find(ready.begin(), ready.end(), name) != ready.end();
            if (remaining[name].empty() && !already_placed && !already_ready) ready.push_back(name);
        }
    }

    if (order.size() != topo.groups.size()) {
        // A residual add creates a genuine cycle between a block's I/O group
        // and its inner group (docs/alignment_algorithm.md §2's ResNet
        // example) -- no linear order satisfies both edges. Fall back to a
        // deterministic order; the sweep loop, not the ordering, is what
        // actually resolves the cycle numerically.
        order.clear();
        for (const auto& [name, _] : topo.groups) order.push_back(name);
        std::sort(order.begin(), order.end());
    }
    return order;
}

std::vector<std::string> dependents_of(const core::Topology& topo, std::string_view group) {
    std::set<std::string> found;
    for (const auto& [tensor, axes] : topo.tensors) {
        const core::AxisBinding* owner = dim0_binding(axes);
        if (owner == nullptr || owner->group == group) continue;
        for (const auto& b : axes.axes) {
            if (b.dim != 0 && b.group == group) {
                found.insert(owner->group);
                break;
            }
        }
    }
    return {found.begin(), found.end()};
}

bool converged(const PermutationMap& before, const PermutationMap& after) {
    if (before.size() != after.size()) return false;
    for (const auto& [group, perm] : after) {
        auto it = before.find(group);
        if (it == before.end() || it->second != perm) return false;
    }
    return true;
}

}  // namespace sfs::align
