#include <synapsefs/core/oid.hpp>

#include <blake3.h>

#include <cstdio>
#include <cstring>

namespace sfs::core {

namespace {

constexpr const char* kFramePrefix = "synapsefs.";

std::string_view kind_tag(ObjectKind k) noexcept {
    switch (k) {
        case ObjectKind::Raw:      return "raw";
        case ObjectKind::Diff:     return "diff";
        case ObjectKind::Header:   return "header";
        case ObjectKind::Manifest: return "manifest";
        case ObjectKind::Commit:   return "commit";
        case ObjectKind::Topology: return "topology";
        case ObjectKind::Tree:     return "tree";
    }
    return "?";
}

int hex_nibble(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

}  // namespace

std::string_view to_string(ObjectKind k) noexcept { return kind_tag(k); }

Result<ObjectKind> object_kind_from_string(std::string_view s) noexcept {
    if (s == "raw") return ObjectKind::Raw;
    if (s == "diff") return ObjectKind::Diff;
    if (s == "header") return ObjectKind::Header;
    if (s == "manifest") return ObjectKind::Manifest;
    if (s == "commit") return ObjectKind::Commit;
    if (s == "topology") return ObjectKind::Topology;
    if (s == "tree") return ObjectKind::Tree;
    return SFS_ERR(MalformedObject, "unknown object kind", std::string(s));
}

// ---------------------------------------------------------------------------
// Frame prefix: "synapsefs.<kind> <decimal-len>\0"
// ---------------------------------------------------------------------------

std::size_t write_frame_prefix(ObjectKind kind, std::uint64_t len, std::span<std::byte> out) {
    char buf[40];
    int n = std::snprintf(buf, sizeof(buf), "%s%.*s %llu", kFramePrefix,
                          static_cast<int>(kind_tag(kind).size()), kind_tag(kind).data(),
                          static_cast<unsigned long long>(len));
    if (n < 0) return 0;
    std::size_t total = static_cast<std::size_t>(n) + 1;  // + trailing NUL
    if (out.size() < total) return 0;
    std::memcpy(out.data(), buf, static_cast<std::size_t>(n));
    out[static_cast<std::size_t>(n)] = std::byte{0};
    return total;
}

// ---------------------------------------------------------------------------
// Oid
// ---------------------------------------------------------------------------

Result<Oid> Oid::parse(std::string_view s) {
    constexpr std::string_view kPrefix = "b3:";
    if (s.size() != kPrefix.size() + kOidHexChars || s.substr(0, kPrefix.size()) != kPrefix)
        return SFS_ERR(MalformedObject, "oid must be \"b3:<64 hex>\"", std::string(s));

    std::array<std::byte, kOidBytes> bytes{};
    std::string_view hex = s.substr(kPrefix.size());
    for (std::size_t i = 0; i < kOidBytes; ++i) {
        int hi = hex_nibble(hex[i * 2]);
        int lo = hex_nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0)
            return SFS_ERR(MalformedObject, "invalid hex digit in oid", std::string(s));
        bytes[i] = static_cast<std::byte>((hi << 4) | lo);
    }
    return Oid(bytes);
}

Result<Oid> Oid::from_bytes(std::span<const std::byte> b) {
    if (b.size() != kOidBytes)
        return SFS_ERR(MalformedObject, "oid must be exactly 32 bytes");
    std::array<std::byte, kOidBytes> bytes{};
    std::memcpy(bytes.data(), b.data(), kOidBytes);
    return Oid(bytes);
}

std::string Oid::to_string() const {
    static const char* kHex = "0123456789abcdef";
    std::string s;
    s.reserve(3 + kOidHexChars);
    s += "b3:";
    for (auto b : bytes_) {
        auto v = static_cast<unsigned char>(b);
        s.push_back(kHex[v >> 4]);
        s.push_back(kHex[v & 0xF]);
    }
    return s;
}

std::string Oid::abbrev() const {
    std::string full = to_string();
    // "b3:" + first (kAbbrevChars) hex chars.
    return full.substr(0, 3 + kAbbrevChars);
}

std::string Oid::fanout_path() const {
    static const char* kHex = "0123456789abcdef";
    std::string s;
    s.reserve(kOidHexChars + 1);
    auto b0 = static_cast<unsigned char>(bytes_[0]);
    s.push_back(kHex[b0 >> 4]);
    s.push_back(kHex[b0 & 0xF]);
    s.push_back('/');
    for (std::size_t i = 1; i < kOidBytes; ++i) {
        auto v = static_cast<unsigned char>(bytes_[i]);
        s.push_back(kHex[v >> 4]);
        s.push_back(kHex[v & 0xF]);
    }
    return s;
}

bool Oid::is_null() const noexcept {
    for (auto b : bytes_)
        if (b != std::byte{0}) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Hasher
// ---------------------------------------------------------------------------

struct Hasher::Impl {
    blake3_hasher state;
};

Hasher::Hasher() : impl_(std::make_unique<Impl>()) { blake3_hasher_init(&impl_->state); }
Hasher::~Hasher() = default;
Hasher::Hasher(Hasher&&) noexcept = default;
Hasher& Hasher::operator=(Hasher&&) noexcept = default;

void Hasher::begin_frame(ObjectKind kind, std::uint64_t payload_len) {
    blake3_hasher_init(&impl_->state);
    std::byte prefix[40];
    std::size_t n = write_frame_prefix(kind, payload_len, prefix);
    blake3_hasher_update(&impl_->state, prefix, n);
}

void Hasher::update(std::span<const std::byte> data) {
    blake3_hasher_update(&impl_->state, data.data(), data.size());
}

Oid Hasher::finish() {
    std::array<std::byte, kOidBytes> out{};
    blake3_hasher_finalize(&impl_->state, reinterpret_cast<std::uint8_t*>(out.data()),
                           kOidBytes);
    return Oid(out);
}

void Hasher::reset() { blake3_hasher_init(&impl_->state); }

Oid compute_oid(ObjectKind kind, std::span<const std::byte> payload) {
    Hasher h;
    h.begin_frame(kind, payload.size());
    h.update(payload);
    return h.finish();
}

std::array<std::byte, kOidBytes> digest(std::span<const std::byte> data) {
    blake3_hasher state;
    blake3_hasher_init(&state);
    blake3_hasher_update(&state, data.data(), data.size());
    std::array<std::byte, kOidBytes> out{};
    blake3_hasher_finalize(&state, reinterpret_cast<std::uint8_t*>(out.data()), kOidBytes);
    return out;
}

}  // namespace sfs::core

std::size_t std::hash<sfs::core::Oid>::operator()(const sfs::core::Oid& o) const noexcept {
    std::size_t v;
    std::memcpy(&v, o.raw().data(), sizeof(v));
    return v;
}
