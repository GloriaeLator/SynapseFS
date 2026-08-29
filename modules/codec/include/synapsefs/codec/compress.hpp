#pragma once
/// \file compress.hpp
/// zstd, constrained so that range reads stay possible.
///
/// The constraint: when an object or a residual is compressed, it is written as
/// INDEPENDENTLY DECOMPRESSIBLE frames aligned to chunk or unit boundaries —
/// not as blocks inside one stream. Decompression must be able to start at an
/// arbitrary frame with no preceding state, or the whole frame design collapses
/// back into decompressing a whole layer per page fault.

#include <cstddef>
#include <span>
#include <vector>

#include <synapsefs/core/error.hpp>

namespace sfs::codec {

using core::Result;

struct CompressOptions {
    int  level = 3;          ///< throughput matters as much as ratio here
    bool checksum = false;   ///< zstd's own; we have our own digests
};

[[nodiscard]] Result<std::vector<std::byte>> compress_frame(std::span<const std::byte>,
                                                            const CompressOptions& = {});

/// Decompress one independently decompressible frame into a caller-owned
/// buffer. Never allocates: this is on the mount's fault path.
[[nodiscard]] Result<std::size_t> decompress_frame(std::span<const std::byte> src,
                                                   std::span<std::byte> dst);

[[nodiscard]] std::size_t compress_bound(std::size_t input_size) noexcept;

/// Byte-plane split and bit-level transpose. Candidates in the Day-2 codec
/// experiment; whichever wins is recorded PER ARTIFACT, not assumed by readers.
void byteplane_split(std::span<const std::byte> src, std::span<std::byte> dst,
                     std::uint32_t elem_bytes) noexcept;
void byteplane_join(std::span<const std::byte> src, std::span<std::byte> dst,
                    std::uint32_t elem_bytes) noexcept;
void bitshuffle(std::span<const std::byte> src, std::span<std::byte> dst,
                std::uint32_t elem_bytes) noexcept;
void bitunshuffle(std::span<const std::byte> src, std::span<std::byte> dst,
                  std::uint32_t elem_bytes) noexcept;

}  // namespace sfs::codec
