#include <synapsefs/align/cost.hpp>

#include <algorithm>
#include <cmath>

namespace sfs::align {

CostMatrix::CostMatrix(std::uint32_t n) : n_(n), values_(static_cast<std::size_t>(n) * n, 0.0F) {}

void CostMatrix::accumulate_tile(std::span<const float> target_tile, std::uint32_t target_first,
                                 std::uint32_t target_count, std::span<const float> base_tile,
                                 std::uint32_t base_first, std::uint32_t base_count,
                                 std::uint32_t vec_len, const CostOptions& opts) {
    // `+=`, not `=`: a group's evidence is usually spread across several
    // tensors (outgoing weight, incoming weight, folded norm scalars), and
    // the caller calls this once per tensor's contribution for the same
    // (target, base) tile pair. NegInnerProduct is the metric this is safe
    // for across repeated calls, since a dot product over a concatenated
    // vector is the sum of the dot products over its segments; L2 and
    // CosineDistance are exact only when called once with the full,
    // already-concatenated feature vector (their norm terms do not
    // decompose the same way), which is how matcher.cpp uses them.
    for (std::uint32_t ti = 0; ti < target_count; ++ti) {
        const float* a = &target_tile[static_cast<std::size_t>(ti) * vec_len];
        for (std::uint32_t bj = 0; bj < base_count; ++bj) {
            const float* b = &base_tile[static_cast<std::size_t>(bj) * vec_len];
            float contribution = 0.0F;

            switch (opts.metric) {
                case CostMetric::NegInnerProduct: {
                    float dot = 0.0F;
                    for (std::uint32_t k = 0; k < vec_len; ++k) dot += a[k] * b[k];
                    contribution = -dot;
                    break;
                }
                case CostMetric::L2: {
                    float sq = 0.0F;
                    for (std::uint32_t k = 0; k < vec_len; ++k) {
                        const float d = a[k] - b[k];
                        sq += d * d;
                    }
                    contribution = sq;
                    break;
                }
                case CostMetric::CosineDistance: {
                    float dot = 0.0F;
                    float na = 0.0F;
                    float nb = 0.0F;
                    for (std::uint32_t k = 0; k < vec_len; ++k) {
                        dot += a[k] * b[k];
                        na += a[k] * a[k];
                        nb += b[k] * b[k];
                    }
                    const float denom = std::sqrt(na) * std::sqrt(nb) + 1e-12F;
                    contribution = 1.0F - dot / denom;
                    break;
                }
            }
            at(target_first + ti, base_first + bj) += contribution;
        }
    }
}

double CostMatrix::identity_cost() const noexcept {
    double total = 0.0;
    for (std::uint32_t i = 0; i < n_; ++i) total += values_[static_cast<std::size_t>(i) * n_ + i];
    return total;
}

double CostMatrix::random_cost() const noexcept {
    if (n_ == 0) return 0.0;
    double total = 0.0;
    for (float v : values_) total += v;
    return total / static_cast<double>(n_);
}

double CostMatrix::assignment_cost(std::span<const std::uint32_t> perm) const noexcept {
    double total = 0.0;
    const std::uint32_t limit = std::min<std::uint32_t>(n_, static_cast<std::uint32_t>(perm.size()));
    for (std::uint32_t i = 0; i < limit; ++i) {
        total += values_[static_cast<std::size_t>(i) * n_ + perm[i]];
    }
    return total;
}

}  // namespace sfs::align
