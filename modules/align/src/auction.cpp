#include <synapsefs/align/auction.hpp>

#include <algorithm>
#include <cstdint>

namespace sfs::align {

std::pair<torch::Tensor, int64_t> auction(const torch::Tensor& C, const torch::Tensor& cand, double eta,
                                          double eps_shrink, int64_t bid_guard) {
    torch::Device device = C.device();
    const int64_t n = C.size(0);
    const int64_t K = C.size(1);

    double cmax = C.numel() > 0 ? C.max().item<double>() : 0.0;
    if (cmax <= 0) cmax = 1.0;
    const double c_null = eta * cmax;
    const double scale = static_cast<double>(1LL << 20) / c_null;
    torch::Tensor Ci = torch::round(C * scale).to(torch::kLong);
    const int64_t c_null_i = 1LL << 20;
    torch::Tensor S = -Ci;  // benefits, (n, K) int64

    const int64_t BIAS = 4 * c_null_i + 10;  // keeps (bid + BIAS) >= 0

    torch::Tensor prices = torch::zeros({n}, torch::TensorOptions().dtype(torch::kLong).device(device));
    torch::Tensor assign = torch::full({n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device));
    int64_t total_rounds = 0;

    int64_t eps = std::max<int64_t>(Ci.numel() > 0 ? Ci.max().item<int64_t>() / 4 : 0, 1);

    while (true) {
        assign.fill_(-1);
        torch::Tensor owner = torch::full({n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device));
        torch::Tensor unassigned = torch::ones({n}, torch::TensorOptions().dtype(torch::kBool).device(device));
        int64_t rounds_this_phase = 0;

        while (unassigned.any().item<bool>()) {
            torch::Tensor active = torch::nonzero(unassigned).squeeze(1);
            const int64_t na = active.size(0);

            torch::Tensor cand_active = cand.index_select(0, active);                      // (na, K)
            torch::Tensor prices_cand = prices.index_select(0, cand_active.reshape(-1)).reshape({na, K});
            torch::Tensor vals = S.index_select(0, active) - prices_cand;                  // (na, K)

            torch::Tensor v1, v2, best_k;
            if (K > 1) {
                auto top2 = torch::topk(vals, 2, /*dim=*/1);
                v1 = std::get<0>(top2).select(1, 0);
                v2 = std::get<0>(top2).select(1, 1);
                best_k = std::get<1>(top2).select(1, 0);
            } else {
                v1 = vals.select(1, 0);
                best_k = torch::zeros_like(active);
                v2 = torch::full_like(v1, -c_null_i);
            }
            // `defect = (-c_null_i) > v1`, i.e. strictly v1 < -c_null_i.
            torch::Tensor defect = v1 < -c_null_i;
            v2 = torch::maximum(v2, torch::full_like(v2, -c_null_i));

            torch::Tensor j = cand_active.gather(1, best_k.unsqueeze(1)).squeeze(1);
            torch::Tensor bid = prices.index_select(0, j) + (v1 - v2) + eps;

            torch::Tensor defect_idx = active.masked_select(defect);
            unassigned.index_fill_(0, defect_idx, false);
            assign.index_fill_(0, defect_idx, -1);

            torch::Tensor not_defect = defect.logical_not();
            torch::Tensor bidders = active.masked_select(not_defect);
            rounds_this_phase += 1;
            total_rounds += 1;
            if (bidders.numel() == 0) {
                if (rounds_this_phase > bid_guard) break;
                continue;
            }

            torch::Tensor bj = j.masked_select(not_defect);
            torch::Tensor bbid = bid.masked_select(not_defect);

            // Highest bid per contested object this round, tie-broken by
            // lowest bidder index via a strictly-unique composite key --
            // exactly one bidder wins each object.
            torch::Tensor combined = (bbid + BIAS) * (n + 1) + (n - bidders);
            torch::Tensor obj_best = torch::full({n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device));
            obj_best.scatter_reduce_(0, bj, combined, "amax", /*include_self=*/true);
            torch::Tensor win_mask = combined == obj_best.index_select(0, bj);
            torch::Tensor winners = bidders.masked_select(win_mask);
            torch::Tensor win_obj = bj.masked_select(win_mask);
            torch::Tensor win_price =
                torch::div(obj_best.index_select(0, win_obj), (n + 1), /*rounding_mode=*/"floor") - BIAS;

            torch::Tensor prev_owner = owner.index_select(0, win_obj);
            torch::Tensor has_prev = prev_owner >= 0;
            torch::Tensor evict = prev_owner.masked_select(has_prev);
            if (evict.numel() > 0) {
                assign.index_fill_(0, evict, -1);
                unassigned.index_fill_(0, evict, true);
            }

            owner.index_put_({win_obj}, winners);
            assign.index_put_({winners}, win_obj);
            unassigned.index_fill_(0, winners, false);
            prices.index_put_({win_obj}, win_price);

            if (rounds_this_phase > bid_guard) break;
        }
        if (eps <= 1) break;
        eps = std::max<int64_t>(static_cast<int64_t>(eps / eps_shrink), 1);
    }
    return {assign, total_rounds};
}

}  // namespace sfs::align
