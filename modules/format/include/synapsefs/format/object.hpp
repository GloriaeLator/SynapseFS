#pragma once
/// \file object.hpp
/// Object framing and the loose-object container.
///
/// Two different things, deliberately kept apart:
///   * the FRAME is what gets hashed and is identical everywhere
///     ("synapsefs.<kind> <len>\0" || payload)   -> docs/spec/10 §1.3
///   * the CONTAINER is local storage detail: compression, chunk digests
///                                               -> docs/spec/11 §2.1
/// Two repositories that compress differently still agree on every address.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::format {

using core::ObjectKind;
using core::Oid;
using core::Result;
using core::Status;

enum class Compression : std::uint8_t { None = 0, Zstd = 1 };

/// Fixed 40-byte prefix of a loose object file, then chunk digests, then the
/// payload. Field-by-field layout in docs/spec/11-repo-layout.md §2.1.
struct ObjectHeader {
    static constexpr std::size_t kSize        = 40;
    static constexpr std::uint8_t kMagic[8]   = {'S', 'F', 'S', 'O', 'B', 'J', 0, 1};

    ObjectKind    kind{};
    Compression   compression = Compression::None;
    std::uint32_t chunk_log2  = 16;    ///< 64 KiB
    std::uint64_t payload_len = 0;     ///< uncompressed
    std::uint64_t stored_len  = 0;     ///< on disk
    std::uint32_t chunk_count = 0;

    [[nodiscard]] std::uint64_t chunk_bytes() const noexcept { return 1ull << chunk_log2; }
    [[nodiscard]] std::size_t digests_offset() const noexcept { return kSize; }
    [[nodiscard]] std::size_t payload_offset() const noexcept {
        return kSize + std::size_t{chunk_count} * core::kOidBytes;
    }

    void encode(std::span<std::byte> out) const;   ///< needs kSize bytes
    [[nodiscard]] static Result<ObjectHeader> decode(std::span<const std::byte>);
};

/// Compute per-chunk digests of an uncompressed payload.
[[nodiscard]] std::vector<std::byte> compute_chunk_digests(std::span<const std::byte> payload,
                                                           std::uint64_t chunk_bytes);

/// Verify one chunk against the digest table. `chunk_index` is into the table.
[[nodiscard]] Status verify_chunk(std::span<const std::byte> chunk_bytes,
                                  std::span<const std::byte> digest_table,
                                  std::uint32_t chunk_index);

/// Serialise a complete loose-object file body for `payload`. Returns the
/// bytes to hand to util::atomic_write.
[[nodiscard]] Result<std::vector<std::byte>> encode_object(ObjectKind,
                                                           std::span<const std::byte> payload,
                                                           Compression,
                                                           std::uint64_t chunk_bytes);

}  // namespace sfs::format
