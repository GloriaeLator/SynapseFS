#include <synapsefs/align/fp16.hpp>

namespace sfs::align {

float fp16_to_float(std::uint16_t v) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(v >> 15U) & 0x1U;
    std::uint32_t exp = static_cast<std::uint32_t>(v >> 10U) & 0x1FU;
    std::uint32_t mant = static_cast<std::uint32_t>(v) & 0x3FFU;
    std::uint32_t out_bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign << 31U;  // signed zero
        } else {
            // Subnormal fp16 -> normalise into fp32's wider exponent range.
            exp = 127 - 15 + 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1U;
                --exp;
            }
            mant &= 0x3FFU;
            out_bits = (sign << 31U) | (exp << 23U) | (mant << 13U);
        }
    } else if (exp == 0x1FU) {
        out_bits = (sign << 31U) | (0xFFU << 23U) | (mant << 13U);  // Inf / NaN
    } else {
        out_bits = (sign << 31U) | ((exp - 15U + 127U) << 23U) | (mant << 13U);
    }
    return std::bit_cast<float>(out_bits);
}

std::uint16_t float_to_fp16_rne(float f) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::int32_t exp = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    const std::uint32_t mant = bits & 0x7FFFFFU;

    if (exp >= 0x1F) {
        // Overflow or already Inf/NaN -> fp16 Inf (NaN payload is not preserved;
        // this path is off the reconstruction path, only used for cost input).
        return static_cast<std::uint16_t>(sign | 0x7C00U | (mant != 0 ? 0x200U : 0));
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);  // too small, flush to zero
        // Subnormal fp16: shift the implicit-1 mantissa right by the exponent deficit.
        const std::uint32_t m = (mant | 0x800000U) >> static_cast<std::uint32_t>(1 - exp);
        const std::uint32_t rounded = m + 0x1000U;  // round to nearest even, ties to even
        return static_cast<std::uint16_t>(sign | (rounded >> 13U));
    }
    const std::uint32_t rounded_mant = mant + 0x1000U;
    if ((rounded_mant & 0x800000U) != 0) {
        // Mantissa rounded up into the next exponent.
        return static_cast<std::uint16_t>(sign | ((static_cast<std::uint32_t>(exp) + 1) << 10U));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10U) | (rounded_mant >> 13U));
}

void fp16_to_float_n(const std::uint16_t* src, float* dst, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) dst[i] = fp16_to_float(src[i]);
}

void bf16_to_float_n(const std::uint16_t* src, float* dst, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_to_float(src[i]);
}

}  // namespace sfs::align
