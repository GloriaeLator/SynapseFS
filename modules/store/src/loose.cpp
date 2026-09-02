#include <synapsefs/store/loose.hpp>

#include <fstream>

#include <synapsefs/util/atomic_io.hpp>
#include <synapsefs/util/file.hpp>
#include <cstring>

namespace fs = std::filesystem;

namespace sfs::store {

using core::Oid;
using core::ObjectKind;
using core::Result;
using core::Status;

LooseStore::LooseStore(fs::path objects_dir, fs::path tmp_dir, const core::RepoConfig& cfg)
    : objects_(std::move(objects_dir)), tmp_(std::move(tmp_dir)), cfg_(cfg) {}

namespace {
fs::path fanout_path(const fs::path& objects_dir, const Oid& oid) {
    return objects_dir / oid.fanout_path();
}
}  // namespace

Result<Oid> LooseStore::put(ObjectKind kind, std::span<const std::byte> payload) {
    Oid oid = core::compute_oid(kind, payload);
    fs::path dest = fanout_path(objects_, oid);

    std::error_code ec;
    if (fs::exists(dest, ec)) return oid;  // content-addressed: identical bytes, no-op

    auto encoded = format::encode_object(kind, payload, format::Compression::None,
                                         cfg_.chunk_bytes);
    if (!encoded) return std::unexpected(encoded.error());

    util::AtomicWriteOptions opts;
    opts.temp_dir = tmp_;
    opts.overwrite = false;
    if (auto r = util::atomic_write(dest, *encoded, opts); !r)
        return SFS_ERR(Io, "cannot write object", dest.string());

    return oid;
}

Result<std::vector<std::byte>> LooseStore::get(const Oid& oid, ObjectKind expected_kind) {
    fs::path p = fanout_path(objects_, oid);
    auto bytes = util::read_file(p);
    if (!bytes) return SFS_ERR(ObjectNotFound, "object not found", oid.to_string());

    auto hdr = format::ObjectHeader::decode(*bytes);
    if (!hdr) return std::unexpected(hdr.error());
    if (hdr->kind != expected_kind)
        return SFS_ERR(ObjectKindMismatch, "object kind mismatch", oid.to_string());
    if (hdr->compression != format::Compression::None)
        return SFS_ERR(NotImplemented, "compressed objects not supported in this build",
                       oid.to_string());

    std::size_t digest_bytes = static_cast<std::size_t>(hdr->chunk_count) * core::kOidBytes;
    std::size_t payload_off = format::ObjectHeader::kSize + digest_bytes;
    if (bytes->size() != payload_off + hdr->payload_len)
        return SFS_ERR(MalformedObject, "object size does not match header", oid.to_string());

    std::vector<std::byte> payload(bytes->begin() + static_cast<long>(payload_off), bytes->end());

    // Whole-object read verifies the full address, same as verify_block but
    // without re-reading from disk since we already have the bytes in hand.
    Oid actual = core::compute_oid(expected_kind, payload);
    if (actual != oid)
        return SFS_ERR(HashMismatch, "object content does not hash to its address",
                       oid.to_string());

    return payload;
}

Result<std::size_t> LooseStore::read_range(const Oid& oid, ObjectKind expected_kind,
                                           std::uint64_t offset, std::span<std::byte> out) {
    fs::path p = fanout_path(objects_, oid);
    auto fd_r = util::open_file(p, util::OpenMode::Read);
    if (!fd_r) return SFS_ERR(ObjectNotFound, "object not found", oid.to_string());

    std::array<std::byte, format::ObjectHeader::kSize> hdr_buf{};
    auto n = util::pread_all(fd_r->get(), hdr_buf, 0);
    if (!n || *n != hdr_buf.size())
        return SFS_ERR(MalformedObject, "cannot read object header", oid.to_string());
    auto hdr = format::ObjectHeader::decode(hdr_buf);
    if (!hdr) return std::unexpected(hdr.error());
    if (hdr->kind != expected_kind)
        return SFS_ERR(ObjectKindMismatch, "object kind mismatch", oid.to_string());
    if (hdr->compression != format::Compression::None)
        return SFS_ERR(NotImplemented, "compressed objects not supported in this build",
                       oid.to_string());

    if (offset >= hdr->payload_len) return std::size_t{0};
    std::size_t want = static_cast<std::size_t>(
        std::min<std::uint64_t>(out.size(), hdr->payload_len - offset));

    std::uint64_t chunk_bytes = std::uint64_t{1} << hdr->chunk_log2;
    std::size_t digest_bytes = static_cast<std::size_t>(hdr->chunk_count) * core::kOidBytes;
    std::size_t payload_off = format::ObjectHeader::kSize + digest_bytes;

    std::vector<std::byte> digests(digest_bytes);
    if (digest_bytes > 0) {
        auto dn = util::pread_all(fd_r->get(), digests, format::ObjectHeader::kSize);
        if (!dn || *dn != digest_bytes)
            return SFS_ERR(MalformedObject, "cannot read chunk digest table", oid.to_string());
    }

    std::size_t done = 0;
    while (done < want) {
        std::uint64_t abs_off = offset + done;
        std::uint32_t chunk_idx = static_cast<std::uint32_t>(abs_off / chunk_bytes);
        std::uint64_t chunk_start = static_cast<std::uint64_t>(chunk_idx) * chunk_bytes;
        std::uint64_t chunk_len =
            std::min<std::uint64_t>(chunk_bytes, hdr->payload_len - chunk_start);

        std::vector<std::byte> chunk_buf(static_cast<std::size_t>(chunk_len));
        auto cn = util::pread_all(fd_r->get(), chunk_buf, payload_off + chunk_start);
        if (!cn || *cn != chunk_buf.size())
            return SFS_ERR(Io, "short read on chunk", oid.to_string());

        if (auto vr = format::verify_chunk(chunk_buf, digests, chunk_idx); !vr)
            return std::unexpected(vr.error());

        std::uint64_t within = abs_off - chunk_start;
        std::size_t take = std::min<std::size_t>(want - done, chunk_len - within);
        ::memcpy(out.data() + done, chunk_buf.data() + within, take);
        done += take;
    }

    return done;
}

Status LooseStore::verify_block(const Oid& oid, ObjectKind expected_kind) {
    auto payload = get(oid, expected_kind);  // get() already does a full-object hash check
    if (!payload) return std::unexpected(payload.error());
    return {};
}

Result<bool> LooseStore::contains(const Oid& oid) const {
    std::error_code ec;
    return fs::exists(fanout_path(objects_, oid), ec);
}

Result<format::ObjectHeader> LooseStore::read_header(const Oid& oid) const {
    fs::path p = fanout_path(objects_, oid);
    auto fd_r = util::open_file(p, util::OpenMode::Read);
    if (!fd_r) return SFS_ERR(ObjectNotFound, "object not found", oid.to_string());
    std::array<std::byte, format::ObjectHeader::kSize> buf{};
    auto n = util::pread_all(fd_r->get(), buf, 0);
    if (!n || *n != buf.size())
        return SFS_ERR(MalformedObject, "cannot read object header", oid.to_string());
    return format::ObjectHeader::decode(buf);
}

Result<std::vector<Oid>> LooseStore::list_all() const {
    std::vector<Oid> out;
    std::error_code ec;
    if (!fs::exists(objects_, ec)) return out;
    for (const auto& top : fs::directory_iterator(objects_, ec)) {
        if (!top.is_directory()) continue;
        std::string top_hex = top.path().filename().string();
        for (const auto& leaf : fs::directory_iterator(top.path(), ec)) {
            if (!leaf.is_regular_file()) continue;
            auto oid_r = Oid::parse("b3:" + top_hex + leaf.path().filename().string());
            if (oid_r) out.push_back(*oid_r);
        }
    }
    return out;
}

Status LooseStore::unlink(const Oid& oid) {
    fs::path p = fanout_path(objects_, oid);
    std::error_code ec;
    fs::remove(p, ec);
    if (ec) return SFS_ERR(Io, "cannot unlink object", p.string());
    return {};
}

}  // namespace sfs::store
