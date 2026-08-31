#pragma once
/// \file reconstruct.hpp
/// THE ONE RECONSTRUCTOR. docs/spec/12-residual-format.md §6.
///
/// `sfs checkout`, the FUSE read path, the mmap fault path and `sfs verify` all
/// call read_range(). checkout is a thirty-line loop over it, not a second
/// implementation — which is why "checkout bytes == mount bytes" is a property
/// of the call graph rather than a test that happens to pass.
///
/// Recursion down a delta chain is PER FRAME, never per layer:
///   peak memory  = frame_bytes * chain_depth
///   digest check = on every hop, so corruption is named where it happened
/// The alternative measured 1347.8 ms and 134.2 MB to serve one 4 KiB read.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/core/topology.hpp>
#include <synapsefs/format/manifest.hpp>

namespace sfs::codec {

class FrameCache;   // blockcache lives in mount/, but read_range takes one

struct ReadCtx {
    core::IBlockStore*      blocks   = nullptr;
    const format::Manifest* manifest = nullptr;
    core::IObjectSource*    history  = nullptr;   ///< to resolve a delta's base
    FrameCache*             cache    = nullptr;   ///< optional; the mount supplies one
    std::uint32_t           max_depth = 5;
    /// Needed only for a tensor whose SECONDARY (non-dim-0) axis is bound to
    /// a different, non-pinned permutation group — e.g. a hidden layer's
    /// weight matrix, whose input axis depends on the PREVIOUS layer's
    /// group. Reconstructing that axis means finding which other tensor
    /// owns that group (via this topology) and reading THAT tensor's own
    /// diff artifact — already in `manifest`, from this same commit — purely
    /// to recover its `permutation` field; see reconstruct.cpp's
    /// resolve_secondary_permutation. Left null and a tensor never needing
    /// it (the common case: one moved axis, or none) reconstructs exactly
    /// as before. commit_planner.cpp guarantees, on the write side, that a
    /// tensor is only ever planned as Delta with a secondary dependency when
    /// that dependency is itself resolvable this way — so a null topology
    /// reaching a tensor that DOES need one is a caller bug (checkout/mount
    /// failing to load and pass it), not a malformed-object condition.
    const core::Topology*   topology = nullptr;
};

/// Copy [offset, offset + out.size()) of `group`'s reconstructed bytes into
/// `out`. Writes into a caller-owned span so the fault path allocates nothing.
[[nodiscard]] core::Result<std::size_t> read_range(const ReadCtx&, std::string_view group,
                                                   std::uint64_t offset,
                                                   std::span<std::byte> out);

/// Whole file, streamed to a sink in buffer order. This is `checkout`.
[[nodiscard]] core::Status reconstruct_file(
    const ReadCtx&, const std::function<core::Status(std::span<const std::byte>)>& sink);

/// Depth of the chain behind `group`, for diagnostics and for `verify`.
[[nodiscard]] core::Result<std::uint32_t> chain_depth(const ReadCtx&, std::string_view group);

}  // namespace sfs::codec
