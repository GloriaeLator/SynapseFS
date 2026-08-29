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
#include <synapsefs/format/manifest.hpp>

namespace sfs::codec {

class FrameCache;   // blockcache lives in mount/, but read_range takes one

struct ReadCtx {
    core::IBlockStore*      blocks   = nullptr;
    const format::Manifest* manifest = nullptr;
    core::IObjectSource*    history  = nullptr;   ///< to resolve a delta's base
    FrameCache*             cache    = nullptr;   ///< optional; the mount supplies one
    std::uint32_t           max_depth = 5;
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
