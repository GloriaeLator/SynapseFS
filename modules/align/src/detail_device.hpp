#pragma once
/// Internal to modules/align/src. The large-group sparse path (fingerprint +
/// candidate generation + auction, docs/adr/0012) runs its tensor math on
/// CUDA when available and falls back to CPU otherwise -- the PS's own
/// stated 8 GB VRAM grading ceiling (docs/adr/0008, docs/benchmarks.md)
/// implies a GPU is actually present in that environment, not just a limit
/// to respect if one happens to exist.
///
/// Every torch::Tensor::accessor<>() call in this path REQUIRES a CPU
/// tensor -- it throws at runtime on a CUDA tensor, unlike .item<>() (a
/// device-to-host scalar copy, safe on any device). Any function that reads
/// a tensor element-by-element into plain C++ containers must call .cpu()
/// on it first, regardless of which device it originated on.

#include <torch/torch.h>

namespace sfs::align::detail {

/// The device this path's tensor compute should run on: CUDA if available,
/// CPU otherwise. Checked once per call site rather than cached, since the
/// cost of the check is negligible next to the actual tensor work and a
/// cached value could go stale across process lifetimes in tests.
inline torch::Device sparse_compute_device() {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
}

}  // namespace sfs::align::detail
