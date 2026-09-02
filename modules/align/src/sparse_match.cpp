#include "sparse_match.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

#include <synapsefs/align/auction.hpp>
#include <synapsefs/align/fingerprint.hpp>
#include <synapsefs/align/lap.hpp>
#include <synapsefs/core/tensor.hpp>

#include "detail_device.hpp"
#include "detail_read_rows.hpp"
#include "detail_reorder.hpp"

namespace sfs::align {

namespace {

/// Outgoing evidence tensors for `group`, sorted by name -- the same list
/// fingerprint_group uses, kept in one place so the two can't diverge on
/// which tensors "belong" to a group.
std::vector<std::pair<std::string, core::AxisBinding>> outgoing_members(const core::Topology& topo,
                                                                        std::string_view group) {
    std::vector<std::pair<std::string, core::AxisBinding>> members;
    for (const auto& [tensor, axes] : topo.tensors) {
        for (const auto& b : axes.axes) {
            if (b.dim == 0 && b.group == group) members.emplace_back(tensor, b);
        }
    }
    std::sort(members.begin(), members.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return members;
}

/// Reads an arbitrary (not necessarily contiguous) set of raw unit indices
/// for `tensor`, in their ORIGINAL order, via core::to_runs-coalesced reads.
/// Small-set helper (null repair's leftover block); sparse_true_cost has its
/// own chunked variant for the full candidate-driven case.
core::Result<std::vector<float>> gather_units(core::ITensorSource& src, const std::string& tensor,
                                              std::span<const std::uint32_t> indices, std::uint32_t block,
                                              std::uint64_t row_width, core::DType dtype) {
    const auto D = static_cast<std::size_t>(block) * row_width;
    std::vector<std::uint32_t> sorted_idx(indices.begin(), indices.end());
    std::sort(sorted_idx.begin(), sorted_idx.end());
    const std::vector<core::UnitRun> runs = core::to_runs(sorted_idx);

    std::vector<float> pool(sorted_idx.size() * D);
    std::unordered_map<std::uint32_t, std::size_t> pos;
    pos.reserve(sorted_idx.size());
    std::size_t next = 0;
    for (const auto& run : runs) {
        auto raw = SFS_TRY(detail::read_rows_as_float(src, tensor, static_cast<std::uint64_t>(run.first) * block,
                                                      static_cast<std::uint64_t>(run.count) * block, row_width,
                                                      dtype));
        std::memcpy(pool.data() + next * D, raw.data(), raw.size() * sizeof(float));
        for (std::uint64_t i = 0; i < run.count; ++i) pos[static_cast<std::uint32_t>(run.first + i)] = next + i;
        next += run.count;
    }

    std::vector<float> out(indices.size() * D);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        std::memcpy(out.data() + i * D, pool.data() + pos.at(indices[i]) * D, D * sizeof(float));
    }
    return out;
}

}  // namespace

core::Result<torch::Tensor> sparse_true_cost(core::ITensorSource& target, core::ITensorSource& base,
                                             const core::Topology& topo, std::string_view group,
                                             const PermutationMap& solved, const torch::Tensor& cand,
                                             std::uint32_t row_tile) {
    const auto n = static_cast<std::uint32_t>(cand.size(0));
    const auto K = static_cast<std::uint32_t>(cand.size(1));
    const torch::Device device = detail::sparse_compute_device();
    torch::Tensor C = torch::zeros({static_cast<int64_t>(n), static_cast<int64_t>(K)},
                                   torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (const auto& [tensor, binding] : outgoing_members(topo, group)) {
        const core::TensorMeta* meta = target.meta(tensor);
        if (meta == nullptr) return SFS_ERR(Internal, "topology tensor missing from source", tensor);
        const core::TensorAxes& axes = topo.tensors.at(tensor);
        const std::uint64_t row_width = meta->shape.size() > 1 ? meta->elem_count() / meta->shape[0] : 1;
        const std::uint32_t block = binding.block;
        const auto D = static_cast<std::int64_t>(block * row_width);

        for (std::uint32_t lo = 0; lo < n; lo += row_tile) {
            const std::uint32_t hi = std::min(lo + row_tile, n);
            const std::uint32_t chunk_units = hi - lo;

            auto t_raw = SFS_TRY(detail::read_rows_as_float(target, tensor, static_cast<std::uint64_t>(lo) * block,
                                                            static_cast<std::uint64_t>(chunk_units) * block,
                                                            row_width, meta->dtype));
            detail::reorder_if_solved(t_raw, axes, row_width, solved);
            torch::Tensor T = torch::from_blob(t_raw.data(), {chunk_units, D}, torch::kFloat32).clone().to(device);

            // Union of candidate indices this chunk actually needs from
            // BASE, deduplicated -- read exactly those, not the whole
            // tensor (the mechanism that keeps this out-of-core). This is
            // host-side bookkeeping (std::set, unordered_map), so it always
            // needs `cand`'s values on CPU regardless of which device the
            // actual float compute below runs on -- accessor<>() throws on
            // a CUDA tensor.
            torch::Tensor cand_chunk = cand.slice(0, lo, hi).contiguous().cpu();
            auto cand_acc = cand_chunk.accessor<std::int64_t, 2>();
            std::set<std::uint32_t> uniq;
            for (std::uint32_t i = 0; i < chunk_units; ++i) {
                for (std::uint32_t k = 0; k < K; ++k) uniq.insert(static_cast<std::uint32_t>(cand_acc[i][k]));
            }
            const std::vector<std::uint32_t> uniq_idx(uniq.begin(), uniq.end());
            auto gathered = SFS_TRY(gather_units(base, tensor, uniq_idx, block, row_width, meta->dtype));
            torch::Tensor S_all =
                torch::from_blob(gathered.data(), {static_cast<int64_t>(uniq_idx.size()), D}, torch::kFloat32)
                    .clone()
                    .to(device);

            std::unordered_map<std::uint32_t, std::int64_t> pos_of;
            pos_of.reserve(uniq_idx.size());
            for (std::size_t i = 0; i < uniq_idx.size(); ++i) pos_of[uniq_idx[i]] = static_cast<std::int64_t>(i);

            // Built on CPU (accessor-filled), then moved to `device` as one
            // small index tensor before index_select needs to match S_all's
            // device.
            torch::Tensor pos_idx = torch::empty({chunk_units, static_cast<std::int64_t>(K)}, torch::kLong);
            auto pos_acc = pos_idx.accessor<std::int64_t, 2>();
            for (std::uint32_t i = 0; i < chunk_units; ++i) {
                for (std::uint32_t k = 0; k < K; ++k) {
                    pos_acc[i][k] = pos_of.at(static_cast<std::uint32_t>(cand_acc[i][k]));
                }
            }
            pos_idx = pos_idx.to(device);

            torch::Tensor Sc = S_all.index_select(0, pos_idx.reshape(-1)).reshape({chunk_units, static_cast<int64_t>(K), D});
            torch::Tensor tn = (T * T).sum(1);
            torch::Tensor sn_c = (Sc * Sc).sum(-1);
            torch::Tensor g = torch::einsum("bd,bkd->bk", {T, Sc});
            torch::Tensor contrib =
                ((tn.unsqueeze(1) + sn_c - 2.0 * g) / static_cast<double>(D)).to(torch::kFloat32);
            C.slice(0, lo, hi) += contrib;
        }
    }
    return C;
}

core::Result<NullRepairResult> sparse_null_repair(core::ITensorSource& target, core::ITensorSource& base,
                                                  const core::Topology& topo, std::string_view group,
                                                  const PermutationMap& solved, torch::Tensor& assign) {
    const auto n = static_cast<std::uint32_t>(assign.size(0));
    // `assign` comes from auction(), which runs on whatever device its
    // inputs were on -- accessor<>() requires CPU, and everything from here
    // to the end of match_group_sparse is host-side bookkeeping (small
    // leftover repair, final accessor-based extraction), so this is the one
    // point where the sparse path permanently leaves the GPU. Reassigning
    // (not just reading a local copy) updates the caller's tensor too, since
    // `assign` is taken by reference.
    assign = assign.cpu();
    auto acc = assign.accessor<std::int64_t, 1>();

    std::vector<std::uint32_t> free_bidders;
    std::vector<bool> is_taken(n, false);
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::int64_t a = acc[i];
        if (a < 0) {
            free_bidders.push_back(i);
        } else {
            is_taken[static_cast<std::uint32_t>(a)] = true;
        }
    }
    if (free_bidders.empty()) return NullRepairResult{};

    std::vector<std::uint32_t> free_objects;
    for (std::uint32_t j = 0; j < n; ++j) {
        if (!is_taken[j]) free_objects.push_back(j);
    }
    // A partial assignment is a matching: rows claimed == columns claimed
    // always, so the leftover sets are always the same size.
    if (free_objects.size() != free_bidders.size()) {
        return SFS_ERR(Internal, "auction left an unbalanced partial assignment", std::string(group));
    }
    const auto nb = static_cast<std::uint32_t>(free_bidders.size());

    std::vector<float> Csub(static_cast<std::size_t>(nb) * nb, 0.0F);
    for (const auto& [tensor, binding] : outgoing_members(topo, group)) {
        const core::TensorMeta* meta = target.meta(tensor);
        if (meta == nullptr) return SFS_ERR(Internal, "topology tensor missing from source", tensor);
        const core::TensorAxes& axes = topo.tensors.at(tensor);
        const std::uint64_t row_width = meta->shape.size() > 1 ? meta->elem_count() / meta->shape[0] : 1;
        const std::uint32_t block = binding.block;
        const auto D = static_cast<std::int64_t>(block * row_width);

        auto t_all = SFS_TRY(gather_units(target, tensor, free_bidders, block, row_width, meta->dtype));
        detail::reorder_if_solved(t_all, axes, row_width, solved);
        auto s_all = SFS_TRY(gather_units(base, tensor, free_objects, block, row_width, meta->dtype));

        for (std::uint32_t i = 0; i < nb; ++i) {
            for (std::uint32_t j = 0; j < nb; ++j) {
                double sq = 0.0;
                for (std::int64_t k = 0; k < D; ++k) {
                    const double d =
                        static_cast<double>(t_all[static_cast<std::size_t>(i) * D + k]) -
                        static_cast<double>(s_all[static_cast<std::size_t>(j) * D + k]);
                    sq += d * d;
                }
                Csub[static_cast<std::size_t>(i) * nb + j] += static_cast<float>(sq / static_cast<double>(D));
            }
        }
    }

    core::LapResult lap = SFS_TRY(solve_jv(Csub, nb));
    for (std::uint32_t i = 0; i < nb; ++i) {
        acc[free_bidders[i]] = static_cast<std::int64_t>(free_objects[lap.assignment[i]]);
    }
    return NullRepairResult{static_cast<std::int64_t>(nb), lap.cost_raw};
}

core::Result<GroupMatch> match_group_sparse(core::ITensorSource& target, core::ITensorSource& base,
                                            const core::Topology& topo, std::string_view group, std::uint32_t n,
                                            const PermutationMap& solved, const SparseMatchOptions& opts,
                                            const ConfidenceOptions& conf_opts) {
    GroupMatch gm;
    gm.group = std::string(group);

    const bool bn = group_is_bn_gauged(topo, group);
    torch::Tensor F_t = SFS_TRY(
        fingerprint_group(target, topo, group, n, opts.n_quantiles, bn, opts.short_row_D, opts.row_tile));
    torch::Tensor F_b =
        SFS_TRY(fingerprint_group(base, topo, group, n, opts.n_quantiles, bn, opts.short_row_D, opts.row_tile));
    auto [F_t_hat, F_b_hat] = whiten_jointly(F_t, F_b);

    // opts.K is a floor, not the actual starting width: a flat K=4 (the
    // ported prototype's default) is nowhere near enough once the group this
    // path exists for reaches tens of thousands of units, and the auction
    // below is exact for whatever candidate graph it's given -- the ONLY
    // source of this path's approximation error is the true best match
    // falling outside these K fingerprint-nearest candidates. Scale with
    // sqrt(n) so small groups stay cheap (nulls, if any, are covered by the
    // widen-retry below and by null_repair's exact dense solve on the
    // leftover) while large groups get meaningfully more candidates to work
    // with, capped at opts.max_K to keep memory bounded (SparseMatchOptions
    // doc comment has the ADR 0012 numbers this trades against).
    const auto K_scaled = static_cast<std::int64_t>(std::sqrt(static_cast<double>(n)) * 2.0);
    const std::int64_t K_ceiling = std::min(opts.max_K, static_cast<std::int64_t>(n));
    std::int64_t K = std::min(std::max(opts.K, K_scaled), K_ceiling);
    torch::Tensor cand = topk_candidates(F_t_hat, F_b_hat, K);
    torch::Tensor C = SFS_TRY(sparse_true_cost(target, base, topo, group, solved, cand, opts.row_tile));

    auto [assign, n_rounds] = auction(C, cand, opts.eta, opts.eps_shrink, opts.bid_guard);
    (void)n_rounds;

    // Safety valve: if too many units end up unassigned for the current K,
    // widen it and retry before falling back to a dense repair on a large
    // leftover block (cpp/src/dispatch.cpp's same widen_on_null_rate logic).
    std::int64_t n_free = (assign < 0).sum().item<std::int64_t>();
    const auto n_i64 = static_cast<std::int64_t>(n);
    while (static_cast<double>(n_free) / static_cast<double>(std::max<std::int64_t>(n_i64, 1)) >
              opts.widen_on_null_rate &&
          K < std::min(opts.max_K, n_i64)) {
        K = std::min({K * 4, opts.max_K, n_i64});
        cand = topk_candidates(F_t_hat, F_b_hat, K);
        C = SFS_TRY(sparse_true_cost(target, base, topo, group, solved, cand, opts.row_tile));
        auto retry = auction(C, cand, opts.eta, opts.eps_shrink, opts.bid_guard);
        assign = retry.first;
        n_free = (assign < 0).sum().item<std::int64_t>();
    }

    const NullRepairResult repair = SFS_TRY(sparse_null_repair(target, base, topo, group, solved, assign));

    std::vector<std::uint32_t> perm(n);
    {
        auto acc = assign.accessor<std::int64_t, 1>();
        for (std::uint32_t i = 0; i < n; ++i) perm[i] = static_cast<std::uint32_t>(acc[i]);
    }
    if (!core::is_valid_permutation(perm, n)) {
        return SFS_ERR(InvalidPermutation, "sparse path produced an invalid permutation", std::string(group));
    }

    // Achieved cost: sum of each row's cost against the candidate it was
    // actually auctioned to, plus null_repair's own resolved cost for the
    // leftover it fixed. Identity cost reuses sparse_true_cost with a
    // trivial 1-wide "candidate set" of {i} per row -- same function, no
    // separate code path to keep in sync.
    double achieved_cost = repair.cost;
    std::uint32_t distinct = 0;
    {
        // C and cand may still be on CUDA here (assign is already CPU,
        // moved inside sparse_null_repair above); accessor<>() needs CPU.
        torch::Tensor C_cpu = C.cpu();
        torch::Tensor cand_cpu = cand.cpu();
        auto c_acc = C_cpu.accessor<float, 2>();
        auto cand_acc = cand_cpu.accessor<std::int64_t, 2>();
        auto assign_acc = assign.accessor<std::int64_t, 1>();
        const auto Kc = static_cast<std::uint32_t>(cand.size(1));
        for (std::uint32_t i = 0; i < n; ++i) {
            float best = c_acc[i][0];
            std::uint32_t ties = 0;
            std::int64_t matched_k = -1;
            for (std::uint32_t k = 0; k < Kc; ++k) {
                if (cand_acc[i][k] == assign_acc[i]) matched_k = static_cast<std::int64_t>(k);
                if (c_acc[i][k] < best) {
                    best = c_acc[i][k];
                    ties = 0;
                } else if (c_acc[i][k] == best) {
                    ++ties;
                }
            }
            if (ties == 0) ++distinct;
            // matched_k < 0 means this row's final assignment came from
            // null_repair, not the candidate set -- its cost is already in
            // repair.cost above, so it does not get added again here.
            if (matched_k >= 0) achieved_cost += c_acc[i][static_cast<std::uint32_t>(matched_k)];
        }
    }

    torch::Tensor identity_cand = torch::arange(n_i64, torch::kLong).unsqueeze(1);
    torch::Tensor C_identity = SFS_TRY(sparse_true_cost(target, base, topo, group, solved, identity_cand, opts.row_tile));
    const double identity_cost = C_identity.sum().item<double>();

    // random_cost: Monte Carlo estimate of the "matched by chance" baseline
    // (CostMatrix::random_cost()'s dense-path exact mean is unavailable here
    // by design -- the whole point of this path is never materialising the
    // full n x n matrix, ADR 0012). A handful of random permutations, scored
    // with the same sparse_true_cost used for identity_cost above, is cheap
    // relative to the fingerprint/auction work already done and averages out
    // most of the per-permutation noise.
    constexpr int kRandomBaselineSamples = 4;
    double random_cost_sum = 0.0;
    for (int s = 0; s < kRandomBaselineSamples; ++s) {
        torch::Tensor random_cand = torch::randperm(n_i64, torch::kLong).unsqueeze(1);
        torch::Tensor C_random = SFS_TRY(sparse_true_cost(target, base, topo, group, solved, random_cand, opts.row_tile));
        random_cost_sum += C_random.sum().item<double>();
    }
    const double random_cost = random_cost_sum / static_cast<double>(kRandomBaselineSamples);

    Confidence conf = assess(achieved_cost, identity_cost, random_cost, n, distinct, conf_opts);

    gm.identity = false;
    gm.alignable = conf.verdict != Alignability::NotAlignable;
    gm.cost_raw = conf.cost_raw;
    gm.cost_normalized = conf.cost_normalized;
    gm.exact_solver = false;  // approximate by construction (top-K candidates, not all n)
    if (gm.alignable) gm.permutation = std::move(perm);
    return gm;
}

}  // namespace sfs::align
