#pragma once
/// \file norm_fold.hpp
/// Folding normalisation statistics into a unit's feature vector.
///
/// A BatchNorm following a conv carries per-channel weight, bias, running_mean
/// and running_var. They are highly discriminative between channels and cost
/// one scalar per unit per tensor, so leaving them out of the cost is giving up
/// accuracy for nothing.
///
/// Folding here means CONCATENATION into the cost feature vector, not the
/// BatchNorm-into-conv fusion of inference optimisation. We never rewrite
/// weights: reconstruction must be bit-exact.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

enum class NormRole : std::uint8_t { Weight, Bias, RunningMean, RunningVar, Other };

struct NormTensorRef {
    std::string tensor;
    NormRole    role = NormRole::Other;
};

/// Which tensors in `group` are per-unit normalisation statistics.
[[nodiscard]] std::vector<NormTensorRef> find_norm_tensors(const core::Topology&,
                                                           std::string_view group);

/// Append one scalar per unit per norm tensor onto `features`, which is
/// row-major [unit][feature]. Scaled so a norm scalar does not dominate a
/// thousand-element weight row.
void fold_norm_stats(std::span<float> features, std::uint32_t unit_count,
                     std::uint32_t stride, std::span<const float> norm_values,
                     std::uint32_t feature_offset, float scale = 1.0F);

}  // namespace sfs::align
