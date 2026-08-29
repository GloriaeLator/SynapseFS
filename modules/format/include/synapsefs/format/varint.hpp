#pragma once
/// \file varint.hpp
/// LEB128 varints for the binary parts of the diff artifact (frame indices in
/// compact form, run-length encoded permutations). JSON headers do not use
/// these; they are canonical JSON (docs/spec/10 §1.4).

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sfs::format {

inline constexpr std::size_t kMaxVarintBytes = 10;

/// Returns bytes written (<= kMaxVarintBytes), or 0 if `out` is too small.
[[nodiscard]] std::size_t encode_varint(std::uint64_t value, std::span<std::byte> out) noexcept;

/// Returns bytes consumed, or 0 on truncation or overlong encoding. Rejecting
/// overlong encodings matters: two byte sequences that decode to the same value
/// would give the same object two addresses.
[[nodiscard]] std::size_t decode_varint(std::span<const std::byte> in,
                                        std::uint64_t& out) noexcept;

void append_varint(std::vector<std::byte>& buf, std::uint64_t value);

[[nodiscard]] constexpr std::size_t varint_size(std::uint64_t v) noexcept {
    std::size_t n = 1;
    while (v >= 0x80) { v >>= 7; ++n; }
    return n;
}

}  // namespace sfs::format
