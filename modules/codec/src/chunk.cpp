#include <synapsefs/codec/chunk.hpp>

#include <cstring>

namespace sfs::codec {

using core::Oid;
using core::Status;

Status verify_chunk_digest(std::span<const std::byte> chunk,
                           std::span<const std::byte> digest_table,
                           std::uint32_t index, const Oid& object) {
    const std::size_t off = static_cast<std::size_t>(index) * core::kOidBytes;
    if (off + core::kOidBytes > digest_table.size()) {
        return SFS_ERR(MalformedObject, "chunk index out of range for digest table",
                       object.to_string() + " chunk " + std::to_string(index));
    }

    // Unframed digest: chunk digests address nothing, they only witness
    // content, so this is core::digest() (not compute_oid()) — same
    // primitive format::verify_chunk uses on the write side.
    const auto actual = core::digest(chunk);
    if (std::memcmp(actual.data(), digest_table.data() + off, core::kOidBytes) != 0) {
        // ChunkDigestMismatch is an integrity kind (error.hpp: is_integrity()),
        // always logged with the object and the chunk that failed — the
        // context string here is what lets a caller several stack frames
        // away (read_range -> the mount's fault handler -> the daemon log)
        // report both without having to know the object at the point it
        // catches the error.
        return SFS_ERR(ChunkDigestMismatch, "chunk digest mismatch",
                       object.to_string() + " chunk " + std::to_string(index));
    }

    return {};
}

}  // namespace sfs::codec
