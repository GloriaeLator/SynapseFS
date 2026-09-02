#pragma once
/// \file bits.hpp
/// Small bit and size helpers. Header-only, constexpr, no dependencies.
///
/// Owner: util (V1). See docs/interfaces/interfaces.md.

#include <bit>
#include <concepts>
#include <cstdint>

namespace sfs::util {

/// Round `v` down to the previous multiple of `align`. `align` must be a
/// power of two. Used by Mmap to find the page-aligned offset a syscall
/// needs when the caller's requested offset isn't page-aligned itself.
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

}  // namespace sfs::util
