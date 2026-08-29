#pragma once
/// \file bits.hpp
/// Small bit and size helpers. Header-only, constexpr, no dependencies.
///
/// Owner: util (V1). See docs/interfaces/interfaces.md.

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace sfs::util {

/// Round `v` up to the next multiple of `align`. `align` must be a power of two.
template <std::unsigned_integral T>
[[nodiscard]] constexpr T align_up(T v, T align) noexcept {
    return (v + align - 1) & ~(align - 1);
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr T align_down(T v, T align) noexcept {
    return v & ~(align - 1);
}

/// Ceiling division. Used everywhere chunk and frame counts are computed.
template <std::unsigned_integral T>
[[nodiscard]] constexpr T ceil_div(T a, T b) noexcept {
    return (a + b - 1) / b;
}

[[nodiscard]] constexpr bool is_pow2(std::uint64_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

[[nodiscard]] constexpr std::uint32_t log2_exact(std::uint64_t v) noexcept {
    return static_cast<std::uint32_t>(std::countr_zero(v));
}

/// Zigzag encoding: maps signed values to unsigned so that small magnitudes,
/// positive or negative, become small unsigned numbers. Used by the varint
/// encoder and as one of the two candidate residual encodings
/// (docs/adr/0005-residual-encoding.md).
[[nodiscard]] constexpr std::uint64_t zigzag_encode(std::int64_t v) noexcept {
    return (static_cast<std::uint64_t>(v) << 1U) ^ static_cast<std::uint64_t>(v >> 63);
}

[[nodiscard]] constexpr std::int64_t zigzag_decode(std::uint64_t v) noexcept {
    return static_cast<std::int64_t>((v >> 1U) ^ (~(v & 1U) + 1U));
}

/// Human-readable byte counts for CLI output. Exact integers go in --json.
[[nodiscard]] std::size_t format_bytes(std::uint64_t n, char* out, std::size_t cap) noexcept;

}  // namespace sfs::util
