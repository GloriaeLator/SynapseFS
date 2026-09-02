#include <synapsefs/store/block_store.hpp>

#include <atomic>
#include <cstring>
#include <unistd.h>

#include <synapsefs/store/loose.hpp>
#include <synapsefs/util/atomic_io.hpp>
#include <synapsefs/util/bits.hpp>
#include <synapsefs/util/file.hpp>
#include <synapsefs/util/log.hpp>
namespace fs = std::filesystem;

namespace sfs::store {

struct BlockStore::Impl {
    std::unique_ptr<LooseStore> loose;
    core::RepoPaths paths;
    core::RepoConfig cfg;
};

BlockStore::BlockStore() : impl_(std::make_unique<Impl>()) {}
BlockStore::~BlockStore() = default;

Result<std::unique_ptr<BlockStore>> BlockStore::open(const core::RepoPaths& paths,
                                                      const core::RepoConfig& cfg) {
    std::error_code ec;
    fs::create_directories(paths.objects(), ec);
    fs::create_directories(paths.tmp(), ec);

    auto self = std::unique_ptr<BlockStore>(new BlockStore());
    self->impl_->paths = paths;
    self->impl_->cfg = cfg;
    self->impl_->loose = std::make_unique<LooseStore>(paths.objects(), paths.tmp(), cfg);
    return self;
}

Result<Oid> BlockStore::put(ObjectKind kind, std::span<const std::byte> payload) {
    return impl_->loose->put(kind, payload);
}

Result<std::vector<std::byte>> BlockStore::get(const Oid& oid, ObjectKind kind) {
    return impl_->loose->get(oid, kind);
}

Result<std::size_t> BlockStore::read_range(const Oid& oid, ObjectKind kind, std::uint64_t offset,
                                           std::span<std::byte> out) {
    return impl_->loose->read_range(oid, kind, offset, out);
}

Status BlockStore::verify_block(const Oid& oid, ObjectKind kind) {
    return impl_->loose->verify_block(oid, kind);
}

Result<bool> BlockStore::contains(const Oid& oid) const { return impl_->loose->contains(oid); }

Result<std::uint64_t> BlockStore::size_of(const Oid& oid) const {
    auto hdr = impl_->loose->read_header(oid);
    if (!hdr) return std::unexpected(hdr.error());
    return hdr->payload_len;
}

Result<ObjectKind> BlockStore::kind_of(const Oid& oid) const {
    auto hdr = impl_->loose->read_header(oid);
    if (!hdr) return std::unexpected(hdr.error());
    return hdr->kind;
}

Result<std::vector<Oid>> BlockStore::list_all() const { return impl_->loose->list_all(); }

// ---------------------------------------------------------------------------
// Streaming writer: accumulate payload bytes in a temp file, chunk-digesting
// as they arrive, so a multi-GB `full` group never needs its whole content
// resident. On commit(), the header + digest table (known only once every
// byte has been seen) is written and the temp file is spliced into place.
// ---------------------------------------------------------------------------

namespace {

class StreamingWriter final : public BlockStore::Writer {
public:
    StreamingWriter(ObjectKind kind, std::uint64_t expected_len, fs::path objects_dir,
                    fs::path tmp_dir, core::RepoConfig cfg)
        : kind_(kind),
          expected_len_(expected_len),
          objects_(std::move(objects_dir)),
          tmp_(std::move(tmp_dir)),
          cfg_(std::move(cfg)) {}

    Status open() {
        std::error_code ec;
        fs::create_directories(tmp_, ec);
        static std::atomic<std::uint64_t> counter{0};
        tmp_path_ = tmp_ / ("wr." + std::to_string(::getpid()) + "." +
                            std::to_string(counter.fetch_add(1)));
        auto fd_r = util::open_file(tmp_path_, util::OpenMode::Write, /*create=*/true, 0644);
        if (!fd_r) return SFS_ERR(Io, "cannot open temp object", tmp_path_.string());
        fd_ = std::move(*fd_r);
        chunk_bytes_ = cfg_.chunk_bytes;
        hasher_.begin_frame(kind_, expected_len_);
        return {};
    }

    Status write(std::span<const std::byte> data) override {
        if (data.empty()) return {};
        auto n = util::pwrite_all(fd_.get(), data, written_);
        if (!n || *n != data.size())
            return SFS_ERR(Io, "write failed", tmp_path_.string());

        hasher_.update(data);

        // Chunk digests: fold data into the in-progress chunk buffer,
        // flushing a digest whenever a chunk boundary is crossed.
        std::size_t off = 0;
        while (off < data.size()) {
            std::size_t space = static_cast<std::size_t>(chunk_bytes_) - chunk_buf_.size();
            std::size_t take = std::min(space, data.size() - off);
            // span::iterator is contiguous; its difference_type is ptrdiff_t.
            // off and off + take are both bounded by data.size(), so the
            // narrowing to signed is provably safe.
            chunk_buf_.insert(chunk_buf_.end(), data.begin() + static_cast<std::ptrdiff_t>(off),
                              data.begin() + static_cast<std::ptrdiff_t>(off + take));
            off += take;
            if (chunk_buf_.size() == chunk_bytes_) flush_chunk();
        }

        written_ += data.size();
        return {};
    }

    Result<Oid> commit() override {
        if (!chunk_buf_.empty()) flush_chunk();
        if (written_ != expected_len_) {
            abort();
            return SFS_ERR(Internal, "streaming writer: written != declared length",
                           tmp_path_.string());
        }

        Oid oid = hasher_.finish();
        fs::path dest = objects_ / oid.fanout_path();

        std::error_code ec;
        if (fs::exists(dest, ec)) {
            // Content-addressed: identical content already stored.
            fd_.reset();
            fs::remove(tmp_path_, ec);
            return oid;
        }

        format::ObjectHeader hdr;
        hdr.kind = kind_;
        hdr.compression = format::Compression::None;
        hdr.chunk_log2 = util::log2_exact(chunk_bytes_);
        hdr.payload_len = expected_len_;
        hdr.stored_len = expected_len_;
        hdr.chunk_count = static_cast<std::uint32_t>(digests_.size() / core::kOidBytes);

        std::array<std::byte, format::ObjectHeader::kSize> hdr_bytes{};
        hdr.encode(hdr_bytes);

        // Final file layout is [header][digests][payload]. `payload` is
        // already sitting in tmp_path_ verbatim (we wrote it there directly),
        // so prepend header+digests via a second temp file rather than
        // rewriting the (potentially huge) payload.
        fs::path final_tmp = tmp_ / (dest.filename().string() + ".final");
        auto out_fd = util::open_file(final_tmp, util::OpenMode::Write, /*create=*/true, 0644);
        if (!out_fd) {
            abort();
            return SFS_ERR(Io, "cannot create final temp object", final_tmp.string());
        }
        std::uint64_t out_off = 0;
        auto w1 = util::pwrite_all(out_fd->get(), hdr_bytes, out_off);
        if (!w1 || *w1 != hdr_bytes.size()) {
            abort();
            return SFS_ERR(Io, "write header failed", final_tmp.string());
        }
        out_off += hdr_bytes.size();
        if (!digests_.empty()) {
            auto w2 = util::pwrite_all(out_fd->get(), digests_, out_off);
            if (!w2 || *w2 != digests_.size()) {
                abort();
                return SFS_ERR(Io, "write digest table failed", final_tmp.string());
            }
            out_off += digests_.size();
        }

        // Stream the payload from tmp_path_ into final_tmp.
        auto in_fd = util::open_file(tmp_path_, util::OpenMode::Read);
        if (!in_fd) {
            abort();
            return SFS_ERR(Io, "cannot reopen payload temp file", tmp_path_.string());
        }
        std::vector<std::byte> buf(1u << 20);
        std::uint64_t in_off = 0;
        while (in_off < written_) {
            std::size_t want = static_cast<std::size_t>(
                std::min<std::uint64_t>(buf.size(), written_ - in_off));
            auto r = util::pread_all(in_fd->get(), std::span(buf.data(), want), in_off);
            if (!r || *r != want) {
                abort();
                return SFS_ERR(Io, "copy failed", tmp_path_.string());
            }
            auto w = util::pwrite_all(out_fd->get(), std::span(buf.data(), want), out_off);
            if (!w || *w != want) {
                abort();
                return SFS_ERR(Io, "copy write failed", final_tmp.string());
            }
            in_off += want;
            out_off += want;
        }

        if (auto r = util::fsync_fd(out_fd->get()); !r) {
            abort();
            return SFS_ERR(Io, "fsync failed", final_tmp.string());
        }
        out_fd->reset();

        std::error_code rec;
        fs::create_directories(dest.parent_path(), rec);
        if (::rename(final_tmp.c_str(), dest.c_str()) != 0) {
            abort();
            return SFS_ERR(Io, "rename failed", dest.string());
        }
        if (auto r = util::fsync_dir(dest.parent_path()); !r) {
            // The rename already succeeded and the object's bytes are durable;
            // only the directory entry's durability is in question. Non-fatal
            // (the object is usable now), but it was previously discarded
            // outright despite fsync_dir being [[nodiscard]] — surface it.
            SFS_LOG_W("block_store", "fsync of objects directory failed after commit: {}",
                      r.error().message());
        }

        fd_.reset();
        fs::remove(tmp_path_, ec);
        return oid;
    }

    void abort() noexcept override {
        fd_.reset();
        std::error_code ec;
        if (!tmp_path_.empty()) fs::remove(tmp_path_, ec);
    }

private:
    void flush_chunk() {
        auto d = core::digest(chunk_buf_);
        std::size_t base = digests_.size();
        digests_.resize(base + core::kOidBytes);
        ::memcpy(digests_.data() + base, d.data(), core::kOidBytes);
        chunk_buf_.clear();
    }

    ObjectKind kind_;
    std::uint64_t expected_len_;
    fs::path objects_, tmp_, tmp_path_;
    core::RepoConfig cfg_;
    util::Fd fd_;
    core::Hasher hasher_;
    std::uint64_t written_ = 0;
    std::uint64_t chunk_bytes_ = 0;
    std::vector<std::byte> chunk_buf_;
    std::vector<std::byte> digests_;
};

}  // namespace

Result<std::unique_ptr<BlockStore::Writer>> BlockStore::begin_put(ObjectKind kind,
                                                                   std::uint64_t payload_len) {
    auto w = std::make_unique<StreamingWriter>(kind, payload_len, impl_->paths.objects(),
                                               impl_->paths.tmp(), impl_->cfg);
    if (auto st = w->open(); !st) return std::unexpected(st.error());
    return std::unique_ptr<BlockStore::Writer>(std::move(w));
}

}  // namespace sfs::store
