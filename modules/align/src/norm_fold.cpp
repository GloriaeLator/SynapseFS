#include <synapsefs/align/norm_fold.hpp>

namespace sfs::align {

std::vector<NormTensorRef> find_norm_tensors(const core::Topology& topo, std::string_view group) {
    // A norm tensor (BatchNorm/LayerNorm weight/bias/running_mean/running_var)
    // is rank-1 in the topology (exactly one axis binding, at dim 0) and
    // shares its layer's group. That shape alone does not distinguish it from
    // an ordinary layer's own bias, which is also rank-1 and in the same
    // group -- the topology carries no tensor-rank or role information beyond
    // axis bindings. The signal this uses instead: a group is norm-gauged
    // only if at least one of its rank-1 members is a running_mean or
    // running_var (which never come from anything but a norm layer); a
    // group with neither is presumed to have no norm layer, and its plain
    // bias is left for cost.cpp's own outgoing-evidence handling rather than
    // double-counted here.
    std::vector<NormTensorRef> members;
    bool has_running_stats = false;

    for (const auto& [tensor, axes] : topo.tensors) {
        if (axes.axes.size() != 1 || axes.axes.front().group != group) continue;
        NormRole role = NormRole::Other;
        if (tensor.ends_with(".running_mean")) {
            role = NormRole::RunningMean;
            has_running_stats = true;
        } else if (tensor.ends_with(".running_var")) {
            role = NormRole::RunningVar;
            has_running_stats = true;
        } else if (tensor.ends_with(".weight")) {
            role = NormRole::Weight;
        } else if (tensor.ends_with(".bias")) {
            role = NormRole::Bias;
        }
        members.push_back(NormTensorRef{tensor, role});
    }

    if (!has_running_stats) return {};
    return members;
}

void fold_norm_stats(std::span<float> features, std::uint32_t unit_count, std::uint32_t stride,
                     std::span<const float> norm_values, std::uint32_t feature_offset, float scale) {
    for (std::uint32_t u = 0; u < unit_count; ++u) {
        features[static_cast<std::size_t>(u) * stride + feature_offset] = norm_values[u] * scale;
    }
}

}  // namespace sfs::align
