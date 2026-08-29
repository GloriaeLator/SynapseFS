#pragma once
/// \file endian.hpp
/// Everything on disk and on the wire is little-endian. These helpers make the
/// conversion explicit at every serialisation site, so that a big-endian host
/// is a compile-and-run question rather than a silent corruption question.

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace sfs::core {

static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big,
              "mixed-endian hosts are not supported");

template <std::unsigned_integral T>
[[nodiscard]] constexpr T to_le(T v) noexcept {
    if constexpr (std::endian::native == std::endian::little) return v;
    else return std::byteswap(v);
}

template <std::unsigned_integral T>
[[nodiscard]] constexpr T from_le(T v) noexcept {
    return to_le(v);
}

/// Read a little-endian integer from an unaligned byte range.
template <std::unsigned_integral T>
[[nodiscard]] inline T load_le(const std::byte* p) noexcept {
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return from_le(v);
}

template <std::unsigned_integral T>
inline void store_le(std::byte* p, T v) noexcept {
    const T le = to_le(v);
    std::memcpy(p, &le, sizeof(T));
}

/// Bounds-checked variants for parsing untrusted input (received blocks,
/// artifact headers). Returns false rather than reading past the end.
template <std::unsigned_integral T>
[[nodiscard]] inline bool try_load_le(std::span<const std::byte> buf, std::size_t off,
                                      T& out) noexcept {
    if (off + sizeof(T) > buf.size()) return false;
    out = load_le<T>(buf.data() + off);
    return true;
}

}  // namespace sfs::core
