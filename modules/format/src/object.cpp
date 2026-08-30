#include <synapsefs/format/object.hpp>

#include <cstring>

#include <synapsefs/core/endian.hpp>
#include <synapsefs/util/bits.hpp>

namespace sfs::format {

void ObjectHeader::encode(std::span<std::byte> out) const {
    // [8-byte magic][1 kind][1 compression][4 chunk_log2]
    // [8 payload_len][8 stored_len][4 chunk_count][6 reserved] = 40 bytes.
    std::memcpy(out.data(), kMagic, 8);
    out[8] = static_cast<std::byte>(kind);
    out[9] = static_cast<std::byte>(compression);
    core::store_le<std::uint32_t>(out.data() + 10, chunk_log2);
    core::store_le<std::uint64_t>(out.data() + 14, payload_len);
    core::store_le<std::uint64_t>(out.data() + 22, stored_len);
    core::store_le<std::uint32_t>(out.data() + 30, chunk_count);
    std::memset(out.data() + 34, 0, 6);  // reserved
}

Result<ObjectHeader> ObjectHeader::decode(std::span<const std::byte> in) {
    if (in.size() < kSize)
        return SFS_ERR(MalformedObject, "loose object header truncated");
    if (std::memcmp(in.data(), kMagic, 8) != 0)
        return SFS_ERR(MalformedObject, "bad loose object magic");

    ObjectHeader h;
    h.kind = static_cast<ObjectKind>(in[8]);
    h.compression = static_cast<Compression>(in[9]);
    h.chunk_log2 = core::load_le<std::uint32_t>(in.data() + 10);
    h.payload_len = core::load_le<std::uint64_t>(in.data() + 14);
    h.stored_len = core::load_le<std::uint64_t>(in.data() + 22);
    h.chunk_count = core::load_le<std::uint32_t>(in.data() + 30);
    return h;
}

std::vector<std::byte> compute_chunk_digests(std::span<const std::byte> payload,
                                             std::uint64_t chunk_bytes) {
    std::vector<std::byte> out;
    if (payload.empty()) return out;

    std::uint64_t n = util::ceil_div<std::uint64_t>(payload.size(), chunk_bytes);
    out.resize(n * core::kOidBytes);
    for (std::uint64_t i = 0; i < n; ++i) {
        std::uint64_t off = i * chunk_bytes;
        std::uint64_t len = std::min<std::uint64_t>(chunk_bytes, payload.size() - off);
        auto d = core::digest(payload.subspan(off, len));
        std::memcpy(out.data() + i * core::kOidBytes, d.data(), core::kOidBytes);
    }
    return out;
}

Status verify_chunk(std::span<const std::byte> chunk_bytes, std::span<const std::byte> digest_table,
                    std::uint32_t chunk_index) {
    std::size_t off = static_cast<std::size_t>(chunk_index) * core::kOidBytes;
    if (off + core::kOidBytes > digest_table.size())
        return SFS_ERR(MalformedObject, "chunk index out of range");

    auto actual = core::digest(chunk_bytes);
    if (std::memcmp(actual.data(), digest_table.data() + off, core::kOidBytes) != 0)
        return SFS_ERR(ChunkDigestMismatch, "chunk digest mismatch", std::to_string(chunk_index));
    return {};
}

Result<std::vector<std::byte>> encode_object(ObjectKind kind, std::span<const std::byte> payload,
                                             Compression compression, std::uint64_t chunk_bytes) {
    if (compression != Compression::None) {
        // The scaffold's zstd::libzstd link is codec-module territory; the
        // day-1 vertical slice this port targets stores objects uncompressed
        // (fp16 weights do not compress usefully — RepoConfig::compress_raw
        // defaults to false). Compressed containers are a codec-module
        // extension, not implemented here.
        return SFS_ERR(NotImplemented, "object compression not implemented in this build");
    }

    if (!core::is_pow2(chunk_bytes))
        return SFS_ERR(Internal, "chunk_bytes must be a power of two");

    ObjectHeader hdr;
    hdr.kind = kind;
    hdr.compression = Compression::None;
    hdr.chunk_log2 = core::log2_exact(chunk_bytes);
    hdr.payload_len = payload.size();
    hdr.stored_len = payload.size();

    auto digests = compute_chunk_digests(payload, chunk_bytes);
    hdr.chunk_count = static_cast<std::uint32_t>(digests.size() / core::kOidBytes);

    std::vector<std::byte> out(ObjectHeader::kSize + digests.size() + payload.size());
    hdr.encode(std::span<std::byte>(out.data(), ObjectHeader::kSize));
    std::memcpy(out.data() + ObjectHeader::kSize, digests.data(), digests.size());
    std::memcpy(out.data() + ObjectHeader::kSize + digests.size(), payload.data(), payload.size());
    return out;
}

}  // namespace sfs::format
