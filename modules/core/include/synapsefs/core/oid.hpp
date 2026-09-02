#pragma once
/// \file oid.hpp
/// Object identifier: BLAKE3-256 over an object's FRAMED bytes.
///
///   frame(kind, payload) = "synapsefs." kind " " decimal(len) 0x00 payload
///   oid                  = "b3:" hex(BLAKE3_256(frame))
///
/// The kind is inside the hashed bytes on purpose: without it, the same byte
/// string read as a tensor group and as a diff artifact has the same address,
/// and a peer can hand us a block that validates as both.
/// docs/spec/10-object-model.md §1.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <synapsefs/core/error.hpp>

namespace sfs::core {

enum class ObjectKind : std::uint8_t {
    Raw      = 0,  ///< tensor-group bytes, exactly as in the reconstructed file
    Diff     = 1,  ///< diff artifact (docs/spec/12)
    Header   = 2,  ///< verbatim [8-byte len][JSON header] of a .safetensors file
    Manifest = 3,
    Commit   = 4,
    Topology = 5,
    Tree     = 6,  ///< sharded checkpoint: name -> manifest (format version 2)
};

[[nodiscard]] std::string_view to_string(ObjectKind) noexcept;
[[nodiscard]] Result<ObjectKind> object_kind_from_string(std::string_view) noexcept;

inline constexpr std::size_t kOidBytes    = 32;
inline constexpr std::size_t kOidHexChars = 64;
inline constexpr std::size_t kAbbrevChars = 12;   ///< human output only, never stored

class Oid {
public:
    constexpr Oid() = default;
    explicit constexpr Oid(std::array<std::byte, kOidBytes> b) noexcept : bytes_(b) {}

    /// Parse "b3:<64 hex>". The prefix is required — a bare hex string is not
    /// an oid, and abbreviated forms are never accepted here.
    [[nodiscard]] static Result<Oid> parse(std::string_view);
    /// Parse 32 raw bytes (wire format, packfile index).
    [[nodiscard]] static Result<Oid> from_bytes(std::span<const std::byte>);

    [[nodiscard]] std::string to_string() const;     ///< "b3:<64 hex>"
    [[nodiscard]] std::string abbrev() const;        ///< first 12 hex, display only
    [[nodiscard]] std::span<const std::byte, kOidBytes> raw() const noexcept { return bytes_; }

    /// Loose-object path fan-out: "<xx>/<remaining 62>".
    [[nodiscard]] std::string fanout_path() const;

    [[nodiscard]] bool is_null() const noexcept;

    friend bool operator==(const Oid&, const Oid&) noexcept = default;
    friend auto operator<=>(const Oid&, const Oid&) noexcept = default;

private:
    std::array<std::byte, kOidBytes> bytes_{};
};

/// Hash of the framed payload. This is THE addressing function; nothing else
/// may compute an oid.
[[nodiscard]] Oid compute_oid(ObjectKind, std::span<const std::byte> payload);

/// The frame prefix for a payload of `len` bytes, written into `out`.
/// Returns the number of bytes written. `out` needs 40 bytes.
[[nodiscard]] std::size_t write_frame_prefix(ObjectKind, std::uint64_t len,
                                             std::span<std::byte> out);

/// Streaming digest, so that a multi-GB object is hashed without being resident.
class Hasher {
public:
    Hasher();
    ~Hasher();
    Hasher(const Hasher&) = delete;
    Hasher& operator=(const Hasher&) = delete;
    Hasher(Hasher&&) noexcept;
    Hasher& operator=(Hasher&&) noexcept;

    /// Begin a framed object hash. Call before any update().
    void begin_frame(ObjectKind, std::uint64_t payload_len);
    void update(std::span<const std::byte>);
    [[nodiscard]] Oid finish();
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Unframed digest, for chunk and frame digests (which address nothing).
[[nodiscard]] std::array<std::byte, kOidBytes> digest(std::span<const std::byte>);

}  // namespace sfs::core

template <>
struct std::hash<sfs::core::Oid> {
    [[nodiscard]] std::size_t operator()(const sfs::core::Oid&) const noexcept;
};
