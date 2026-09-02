#include <synapsefs/align/fingerprint.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <synapsefs/align/norm_fold.hpp>

#include "detail_device.hpp"
#include "detail_read_rows.hpp"

namespace sfs::align {

bool group_is_bn_gauged(const core::Topology& topo, std::string_view group) {
    for (const auto& norm : find_norm_tensors(topo, group)) {
        if (norm.role == NormRole::RunningMean || norm.role == NormRole::RunningVar) return true;
    }
    return false;
}

std::vector<std::int64_t> quantile_kth_indices(std::int64_t D, std::int64_t M) {
    const std::int64_t denom = std::max<std::int64_t>(M - 1, 1);
    std::set<std::int64_t> kth_set;
    for (std::int64_t i = 0; i < M; ++i) {
        kth_set.insert(static_cast<std::int64_t>(std::floor(static_cast<double>(i) * (D - 1) / denom)));
    }
    return {kth_set.begin(), kth_set.end()};
}

core::Result<torch::Tensor> fingerprint_group(core::ITensorSource& src, const core::Topology& topo,
                                              std::string_view group, std::uint32_t n, int n_quantiles,
                                              bool bn_gauged, int64_t short_row_D, std::uint32_t row_tile) {
    // Outgoing evidence only (dim-0 bindings) -- candidate generation is
    // deliberately one-sided, matching cpp/src/fingerprint.cpp. Sorted by
    // (tensor, dim) for determinism, mirroring the original's explicit
    // re-sort of ps.perm_to_axes[p] (iteration order over topo.tensors,
    // an unordered_map, is otherwise unspecified).
    std::vector<std::pair<std::string, core::AxisBinding>> members;
    for (const auto& [tensor, axes] : topo.tensors) {
        for (const auto& b : axes.axes) {
            if (b.dim == 0 && b.group == group) members.emplace_back(tensor, b);
        }
    }
    std::sort(members.begin(), members.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // torch::from_blob always wraps host memory (read_rows_as_float's
    // output), so every tensor built here starts on CPU; .to(device) moves
    // the actual compute (sort, index_select) onto CUDA when available.
    // Falls back to CPU with no code-path difference otherwise.
    const torch::Device device = detail::sparse_compute_device();

    std::vector<torch::Tensor> feats;
    for (const auto& [tensor, binding] : members) {
        const core::TensorMeta* meta = src.meta(tensor);
        if (meta == nullptr) return SFS_ERR(Internal, "topology tensor missing from source", tensor);

        if (meta->shape.size() <= 1) {
            // A bias-like tensor: `binding.block` raw scalars per unit
            // (usually 1, but not assumed -- unlike the old PermutationSpec
            // model, this one has no built-in reason a rank-1 tensor's
            // block factor must be 1). No sorting needed either way.
            const auto D = static_cast<int64_t>(binding.block);
            auto vals = SFS_TRY(
                detail::read_rows_as_float(src, tensor, 0, static_cast<std::uint64_t>(n) * binding.block, 1,
                                           meta->dtype));
            torch::Tensor t =
                torch::from_blob(vals.data(), {static_cast<int64_t>(n), D}, torch::kFloat32).clone().to(device);
            feats.push_back(std::move(t));
            continue;
        }

        const std::uint64_t row_width = meta->elem_count() / meta->shape[0];
        const std::uint32_t block = binding.block;
        const auto D = static_cast<int64_t>(block * row_width);
        const int64_t M = (D <= short_row_D) ? std::min<int64_t>(D, 32) : std::min<int64_t>(n_quantiles, D);
        const std::vector<int64_t> kth_vec = quantile_kth_indices(D, M);
        torch::Tensor kth =
            torch::from_blob(const_cast<int64_t*>(kth_vec.data()), {static_cast<int64_t>(kth_vec.size())},
                             torch::kLong)
                .clone()
                .to(device);

        torch::Tensor out_t =
            torch::empty({static_cast<int64_t>(n), M}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        for (std::uint32_t lo = 0; lo < n; lo += row_tile) {
            const std::uint32_t hi = std::min(lo + row_tile, n);
            const std::uint32_t chunk_units = hi - lo;
            // Peak memory here is O(row_tile * D), independent of n or the
            // tensor's total size -- the whole point of this path (docs/adr/0012).
            // (On CUDA, "memory" here means host RAM for the read plus one
            // row_tile-sized chunk of VRAM for the compute below, not the
            // whole group's VRAM footprint.)
            auto raw = SFS_TRY(detail::read_rows_as_float(src, tensor, static_cast<std::uint64_t>(lo) * block,
                                                          static_cast<std::uint64_t>(chunk_units) * block,
                                                          row_width, meta->dtype));
            torch::Tensor rows = torch::from_blob(raw.data(), {static_cast<int64_t>(chunk_units), D}, torch::kFloat32)
                                     .clone()
                                     .to(device);
            if (bn_gauged) {
                rows = rows / (rows.norm(2, 1, true) + 1e-8);
            }
            torch::Tensor sorted_rows = std::get<0>(torch::sort(rows, /*dim=*/1));
            out_t.slice(0, lo, hi) = sorted_rows.index_select(1, kth);
        }
        feats.push_back(std::move(out_t));
    }

    if (feats.empty()) {
        return torch::zeros({static_cast<int64_t>(n), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    return torch::cat(feats, /*dim=*/1);
}

std::pair<torch::Tensor, torch::Tensor> whiten_jointly(const torch::Tensor& F_a, const torch::Tensor& F_b) {
    torch::Tensor F = torch::cat({F_a, F_b}, /*dim=*/0);
    torch::Tensor mu = F.mean(/*dim=*/0);
    // Population std (unbiased=False, matching numpy's default), computed
    // directly rather than via torch::std's unbiased-bool overload, which
    // has shifted to a `correction` parameter across recent ATen versions --
    // this formulation is version-stable.
    torch::Tensor sigma = torch::sqrt(((F - mu).pow(2)).mean(/*dim=*/0)) + 1e-8;
    return {(F_a - mu) / sigma, (F_b - mu) / sigma};
}

torch::Tensor topk_candidates(const torch::Tensor& F_a_hat, const torch::Tensor& F_b_hat, int64_t K,
                              int64_t block) {
    const int64_t n = F_a_hat.size(0);
    K = std::min(K, n);

    if (n <= 64) {
        torch::Tensor d = (F_a_hat.unsqueeze(1) - F_b_hat.unsqueeze(0)).pow(2).sum(-1);
        return torch::argsort(d, /*stable=*/false, /*dim=*/1, /*descending=*/false)
            .slice(/*dim=*/1, 0, K)
            .to(torch::kLong);
    }

    torch::Tensor nb = (F_b_hat * F_b_hat).sum(1);
    torch::Tensor out = torch::empty({n, K}, torch::TensorOptions().dtype(torch::kLong).device(F_a_hat.device()));
    for (int64_t lo = 0; lo < n; lo += block) {
        const int64_t hi = std::min(lo + block, n);
        torch::Tensor fa = F_a_hat.slice(0, lo, hi);
        torch::Tensor d = (fa * fa).sum(1).unsqueeze(1) + nb.unsqueeze(0) - 2.0 * torch::matmul(fa, F_b_hat.t());
        torch::Tensor idx = std::get<1>(torch::topk(d, K, /*dim=*/1, /*largest=*/false, /*sorted=*/true));
        out.slice(0, lo, hi) = idx;
    }
    return out;
}

}  // namespace sfs::align
