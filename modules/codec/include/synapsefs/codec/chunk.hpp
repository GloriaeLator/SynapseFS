#pragma once
/// \file chunk.hpp
/// Chunk-digest helpers: the arithmetic that makes verification granularity
/// equal read granularity.
///
/// A read of [off, off+len) touches chunks [off/C, (off+len-1)/C]. Verifying
/// exactly those, and no more, is the difference between 0.6 MB/s and
/// 183.9 MB/s — and the chunked path additionally NAMES the corrupt chunk.

#include <cstdint>
#include <span>
#include <utility>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::codec {

struct ChunkRange {
    std::uint32_t first = 0;
    std::uint32_t last  = 0;   ///< inclusive
    [[nodiscard]] std::uint32_t count() const noexcept { return last - first + 1; }
};

[[nodiscard]] constexpr ChunkRange chunks_covering(std::uint64_t offset, std::uint64_t length,
                                                   std::uint64_t chunk_bytes) noexcept {
    if (length == 0) return {0, 0};
    return {static_cast<std::uint32_t>(offset / chunk_bytes),
            static_cast<std::uint32_t>((offset + length - 1) / chunk_bytes)};
}

[[nodiscard]] constexpr std::pair<std::uint64_t, std::uint64_t> chunk_extent(
    std::uint32_t index, std::uint64_t chunk_bytes, std::uint64_t total) noexcept {
    const std::uint64_t begin = static_cast<std::uint64_t>(index) * chunk_bytes;
    return {begin, begin + chunk_bytes > total ? total - begin : chunk_bytes};
}

/// Verify one chunk against the table; the error names the chunk index.
[[nodiscard]] core::Status verify_chunk_digest(std::span<const std::byte> chunk,
                                               std::span<const std::byte> digest_table,
                                               std::uint32_t index, const core::Oid& object);

}  // namespace sfs::codec
