#pragma once
/// \file fingerprint.hpp
/// Quantile-sketch fingerprints and candidate generation for the large-group
/// sparse alignment path (docs/adr/0012).
///
/// Ported from cpp/src/fingerprint.cpp with the algorithm unchanged: sorted-
/// quantile fingerprints (no summation, so bit-invariant under permutation),
/// joint z-score whitening, and blocked-GEMM top-K. The port only replaces
/// the old PermutationSpec/ParamDict-based I/O with core::ITensorSource,
/// reading a group's outgoing evidence tensors in row_tile-sized chunks
/// rather than expecting them already resident as in-memory tensors -- the
/// whole point of this path is never holding a huge layer's full weight
/// matrix at once.

#include <cstdint>
#include <string_view>
#include <utility>

#include <torch/torch.h>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>

namespace sfs::align {

/// True if `group`'s outgoing evidence includes a running_mean/running_var
/// tensor (a BatchNorm-gauged group) -- such groups get their per-unit rows
/// L2-normalised before fingerprinting, since raw BN scale is not
/// comparable across units the way it is for a conv/linear weight row.
[[nodiscard]] bool group_is_bn_gauged(const core::Topology& topo, std::string_view group);

/// Sorted-quantile-position sampling: which of a length-D row's sorted
/// positions get sampled as the M-quantile sketch. Deduplicated, ascending.
[[nodiscard]] std::vector<std::int64_t> quantile_kth_indices(std::int64_t D, std::int64_t M);

/// Per-unit descriptor for `group`'s `n` units, gathered from `src`'s
/// outgoing evidence tensors (dim-0 bindings only -- candidate generation is
/// deliberately one-sided, matching the original). Reads each evidence
/// tensor in `row_tile`-sized row chunks.
[[nodiscard]] core::Result<torch::Tensor> fingerprint_group(core::ITensorSource& src, const core::Topology& topo,
                                                            std::string_view group, std::uint32_t n,
                                                            int n_quantiles, bool bn_gauged, int64_t short_row_D,
                                                            std::uint32_t row_tile);

/// z-score using statistics fit on the union of both checkpoints' fingerprints.
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> whiten_jointly(const torch::Tensor& F_a,
                                                                     const torch::Tensor& F_b);

/// Top-K nearest fingerprint neighbours in B for each neuron of A, via a
/// blocked expansion of ||a-b||^2 (one BLAS-shaped matmul per block) -- the
/// mechanism that avoids ever materialising a full n x n distance matrix.
[[nodiscard]] torch::Tensor topk_candidates(const torch::Tensor& F_a_hat, const torch::Tensor& F_b_hat, int64_t K,
                                            int64_t block = 1024);

}  // namespace sfs::align
