#pragma once
/// \file auction.hpp
/// Jacobi (parallel) sparse auction with private null objects, for the
/// large-group sparse alignment path (docs/adr/0011).
///
/// Ported near-verbatim from cpp/src/auction.cpp -- the algorithm (Bertsekas
/// parallel auction, integer-quantised costs, private null objects at
/// eta*max(C) to guarantee termination on a top-K graph that need not admit
/// a perfect matching per Hall's theorem) operates purely on plain (n, K)
/// tensors and never touched the old PermutationSpec/ParamDict types, so
/// this port changes only the namespace.

#include <utility>

#include <torch/torch.h>

namespace sfs::align {

/// Returns (assign, n_rounds). assign[i] is the object won by bidder i, or
/// -1 if it holds its private null. `C` is (n, K) cost (lower is better,
/// pre-negated by the caller if the underlying metric is a similarity);
/// `cand` is the (n, K) int64 candidate-index matrix from topk_candidates.
[[nodiscard]] std::pair<torch::Tensor, int64_t> auction(const torch::Tensor& C, const torch::Tensor& cand,
                                                        double eta = 10.0, double eps_shrink = 5.0,
                                                        int64_t bid_guard = 400);

}  // namespace sfs::align
