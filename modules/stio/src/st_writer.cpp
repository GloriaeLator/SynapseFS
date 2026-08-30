#include <synapsefs/stio/st_writer.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <cstring>

#include <synapsefs/util/atomic_io.hpp>
#include <synapsefs/util/file.hpp>

namespace sfs::stio {

// ---------------------------------------------------------------------------
// SHA-256. Vendored rather than linked: this is the ONE place SHA-256 appears
// in the system (manifest.hpp's comment on FileInfo::sha256), used purely as
// a witness that reconstruction produced what the PS calls "byte-for-byte
// identical" — never as a content address. BLAKE3 (core/oid.hpp) is the
// addressing hash everywhere else. Pulling in libssl for one 64-line
// algorithm would be a heavier dependency than the algorithm itself; the
// project already vendors BLAKE3 for the same reason (docs/adr/0002).
// Public-domain construction (FIPS 180-4), single translation unit.
// ---------------------------------------------------------------------------
namespace {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        h_[0] = 0x6a09e667u; h_[1] = 0xbb67ae85u; h_[2] = 0x3c6ef372u; h_[3] = 0xa54ff53au;
        h_[4] = 0x510e527fu; h_[5] = 0x9b05688cu; h_[6] = 0x1f83d9abu; h_[7] = 0x5be0cd19u;
        buf_len_ = 0;
        total_len_ = 0;
    }

    void update(const std::byte* data, std::size_t len) {
        total_len_ += len;
        while (len > 0) {
            std::size_t take = std::min<std::size_t>(64 - buf_len_, len);
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            len -= take;
            if (buf_len_ == 64) {
                process(buf_);
                buf_len_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finalize() {
        std::uint64_t bit_len = total_len_ * 8;
        std::uint8_t pad = 0x80;
        update(reinterpret_cast<const std::byte*>(&pad), 1);
        std::uint8_t zero = 0;
        while (buf_len_ != 56) update(reinterpret_cast<const std::byte*>(&zero), 1);
        std::uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) len_be[i] = static_cast<std::uint8_t>(bit_len >> (56 - 8 * i));
        update(reinterpret_cast<const std::byte*>(len_be), 8);

        std::array<std::uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<std::uint8_t>(h_[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(h_[i]);
        }
        return out;
    }

private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void process(const std::uint8_t* block) {
        static constexpr std::uint32_t kK[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(block[i*4]) << 24) | (std::uint32_t(block[i*4+1]) << 16) |
                   (std::uint32_t(block[i*4+2]) << 8) | std::uint32_t(block[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            std::uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            std::uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        std::uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],hh=h_[7];
        for (int i = 0; i < 64; ++i) {
            std::uint32_t s1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            std::uint32_t ch = (e & f) ^ (~e & g);
            std::uint32_t t1 = hh + s1 + ch + kK[i] + w[i];
            std::uint32_t s0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            std::uint32_t t2 = s0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h_[0]+=a; h_[1]+=b; h_[2]+=c; h_[3]+=d; h_[4]+=e; h_[5]+=f; h_[6]+=g; h_[7]+=hh;
    }

    std::uint32_t h_[8]{};
    std::uint8_t buf_[64]{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_len_ = 0;
};

std::string hex(std::span<const std::uint8_t> b) {
    static const char* kHex = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (auto v : b) {
        s.push_back(kHex[v >> 4]);
        s.push_back(kHex[v & 0xF]);
    }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sha256Stream — public pimpl wrapper around the vendored Sha256 above, so
// `sfs commit` can compute the same witness `sfs checkout` verifies.
// ---------------------------------------------------------------------------

struct Sha256Stream::Impl {
    Sha256 hasher;
};

Sha256Stream::Sha256Stream() : impl_(std::make_unique<Impl>()) {}
Sha256Stream::~Sha256Stream() = default;
Sha256Stream::Sha256Stream(Sha256Stream&&) noexcept = default;
Sha256Stream& Sha256Stream::operator=(Sha256Stream&&) noexcept = default;

void Sha256Stream::update(std::span<const std::byte> data) {
    impl_->hasher.update(data.data(), data.size());
}

std::string Sha256Stream::finish_hex() {
    auto digest = impl_->hasher.finalize();
    return hex(digest);
}

// ---------------------------------------------------------------------------
// StWriter
// ---------------------------------------------------------------------------

struct StWriter::Impl {
    std::filesystem::path tmp_path;
    std::filesystem::path dest_path;
    util::Fd fd;
    Sha256 hasher;
    bool verify = true;
    std::uint64_t written = 0;
    std::uint64_t expected_total = 0;
    std::string expected_sha256;
};

StWriter::StWriter() : impl_(std::make_unique<Impl>()) {}
StWriter::~StWriter() {
    // If finish() was never called, best-effort clean up the temp file so a
    // failed checkout does not leave garbage in the destination directory.
    if (impl_ && impl_->fd.valid()) {
        impl_->fd.reset();
        std::error_code ec;
        std::filesystem::remove(impl_->tmp_path, ec);
    }
}
StWriter::StWriter(StWriter&&) noexcept = default;
StWriter& StWriter::operator=(StWriter&&) noexcept = default;

Result<StWriter> StWriter::create(const std::filesystem::path& dest, const format::Manifest& m,
                                  StWriterOptions opts) {
    StWriter w;
    w.impl_->dest_path = dest;
    w.impl_->verify = opts.verify_sha256;
    w.impl_->expected_total = m.file.total_bytes;
    w.impl_->expected_sha256 = m.file.sha256;

    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    w.impl_->tmp_path = dest.parent_path() /
        (dest.filename().string() + ".sfs-tmp-" + std::to_string(::getpid()));

    auto fd = util::open_file(w.impl_->tmp_path, util::OpenMode::Write, /*create=*/true,
                              opts.file_mode);
    if (!fd) return SFS_ERR(Io, "cannot create temp file for checkout", dest.string());
    w.impl_->fd = std::move(*fd);

    return w;
}

Status StWriter::append(std::span<const std::byte> data) {
    if (data.empty()) return {};
    auto n = util::pwrite_all(impl_->fd.get(), data, impl_->written);
    if (!n || *n != data.size())
        return SFS_ERR(Io, "write failed during checkout", impl_->dest_path.string());
    if (impl_->verify) impl_->hasher.update(data.data(), data.size());
    impl_->written += data.size();
    return {};
}

Status StWriter::finish() {
    if (impl_->written != impl_->expected_total) {
        return SFS_ERR(MalformedObject,
                       "checkout wrote " + std::to_string(impl_->written) + " bytes, manifest says " +
                           std::to_string(impl_->expected_total));
    }

    if (auto r = util::fsync_fd(impl_->fd.get()); !r)
        return SFS_ERR(Io, "fsync failed", impl_->dest_path.string());

    // Verify BEFORE the temp file is renamed into place: a mismatch must
    // leave the destination untouched (interface contract), so the check has
    // to happen while we can still discard the tmp file instead of it.
    if (impl_->verify && !impl_->expected_sha256.empty()) {
        auto digest = impl_->hasher.finalize();
        std::string got = hex(digest);
        if (got != impl_->expected_sha256) {
            impl_->fd.reset();
            std::error_code ec;
            std::filesystem::remove(impl_->tmp_path, ec);
            return SFS_ERR(HashMismatch,
                           "reconstructed file sha256 does not match manifest witness",
                           impl_->dest_path.string());
        }
    }

    impl_->fd.reset();

    if (::rename(impl_->tmp_path.c_str(), impl_->dest_path.c_str()) != 0)
        return SFS_ERR(Io, "rename into place failed", impl_->dest_path.string());

    if (auto r = util::fsync_dir(impl_->dest_path.parent_path()); !r)
        return SFS_ERR(Io, "fsync parent dir failed", impl_->dest_path.parent_path().string());

    return {};
}

std::uint64_t StWriter::written() const noexcept { return impl_->written; }

}  // namespace sfs::stio
