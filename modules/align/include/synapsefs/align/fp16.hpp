#pragma once
/// \file fp16.hpp
/// fp16 and bf16 to float, in software.
///
/// Used ONLY for computing alignment costs. Reconstruction never converts:
/// residuals are XOR or zigzag over the integer view of the bits, which is
/// bijective and therefore exact. If a float conversion ever appears on the
/// reconstruction path, that is a bug, not an optimisation.

#include <bit>
#include <cstdint>

namespace sfs::align {

[[nodiscard]] constexpr float bf16_to_float(std::uint16_t v) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U);
}

[[nodiscard]] constexpr std::uint16_t float_to_bf16_rne(float f) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

/// IEEE-754 binary16 -> binary32. Handles subnormals and NaN/Inf; a checkpoint
/// with subnormal weights is unusual but not our business to normalise.
[[nodiscard]] float fp16_to_float(std::uint16_t) noexcept;
[[nodiscard]] std::uint16_t float_to_fp16_rne(float) noexcept;

/// Bulk conversion for a tile of the cost matrix. Vectorises; there is no
/// per-ISA dispatch here because this is off the fault path.
void fp16_to_float_n(const std::uint16_t* src, float* dst, std::size_t n) noexcept;
void bf16_to_float_n(const std::uint16_t* src, float* dst, std::size_t n) noexcept;

}  // namespace sfs::align
