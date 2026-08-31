#include <synapsefs/codec/residual_codec.hpp>

#include <cstring>

// -mavx512f -mavx512bw -mavx512vl -mavx512dq, this file only (ADR 0011). Same
// disabled-fallback contract as residual_avx2.cpp: always compiled, always
// callable, forwards to the scalar oracle when the compiler rejected the ISA
// flags (SFS_KERNEL_DISABLED).
#if !defined(SFS_KERNEL_DISABLED)
#include <immintrin.h>
#endif

namespace sfs::codec {

void xor_apply_avx512(std::byte* dst, const std::byte* base, const std::byte* resid,
                      std::size_t n) noexcept {
#if defined(SFS_KERNEL_DISABLED)
    xor_apply_scalar(dst, base, resid, n);
#else
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i b = _mm512_loadu_si512(base + i);
        const __m512i r = _mm512_loadu_si512(resid + i);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(b, r));
    }
    for (; i < n; ++i) dst[i] = base[i] ^ resid[i];  // unaligned tail
#endif
}

void xor_encode_avx512(std::byte* dst, const std::byte* base, const std::byte* target,
                       std::size_t n) noexcept {
    xor_apply_avx512(dst, base, target, n);  // XOR is self-inverse
}

namespace {

#if !defined(SFS_KERNEL_DISABLED)
// Same lane-parallel zigzag formula as residual_avx2.cpp's u16 path, 32
// elements per 512-bit register instead of 16. Requires AVX512BW for the
// epi16 ops (set/sub/srli on 16-bit lanes) — present per ADR 0011's flag set.
void zigzag_apply_avx512_u16(std::byte* dst, const std::byte* base, const std::byte* resid,
                             std::size_t count) noexcept {
    const __m512i one = _mm512_set1_epi16(1);
    std::size_t i = 0;
    for (; i + 32 <= count; i += 32) {
        const __m512i b = _mm512_loadu_si512(base + i * 2);
        const __m512i z = _mm512_loadu_si512(resid + i * 2);
        const __m512i low_bit = _mm512_and_si512(z, one);
        const __m512i mask = _mm512_sub_epi16(_mm512_setzero_si512(), low_bit);
        const __m512i delta_bits = _mm512_xor_si512(_mm512_srli_epi16(z, 1), mask);
        const __m512i t = _mm512_add_epi16(b, delta_bits);
        _mm512_storeu_si512(dst + i * 2, t);
    }
    if (i < count) {
        zigzag_apply_scalar(dst + i * 2, base + i * 2, resid + i * 2, (count - i) * 2, 2);
    }
}
#endif

}  // namespace

void zigzag_apply_avx512(std::byte* dst, const std::byte* base, const std::byte* resid,
                         std::size_t n, std::uint32_t elem_bytes) noexcept {
#if defined(SFS_KERNEL_DISABLED)
    zigzag_apply_scalar(dst, base, resid, n, elem_bytes);
#else
    if (elem_bytes == 2) {
        zigzag_apply_avx512_u16(dst, base, resid, n / 2);
        return;
    }
    zigzag_apply_scalar(dst, base, resid, n, elem_bytes);  // other widths: not vectorised here
#endif
}

}  // namespace sfs::codec
