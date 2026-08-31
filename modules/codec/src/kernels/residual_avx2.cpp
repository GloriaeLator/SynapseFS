#include <synapsefs/codec/residual_codec.hpp>

#include <cstring>

// -mavx2 -mfma -mbmi -mbmi2, attached to THIS file only by
// cmake/SimdKernels.cmake — never on the target, never globally (ADR 0011).
//
// cmake/SimdKernels.cmake still compiles this file even on a compiler that
// rejects -mavx2 (it drops the flag and defines SFS_KERNEL_DISABLED instead,
// rather than omitting the file), so every function here must exist and be
// callable either way. When disabled, each one forwards straight to the
// scalar oracle: correct, just not vectorised — dispatch.cpp can therefore
// always reference these symbols unconditionally without knowing at ITS
// compile time whether AVX2 was actually usable.
#if !defined(SFS_KERNEL_DISABLED)
#include <immintrin.h>
#endif

namespace sfs::codec {

void xor_apply_avx2(std::byte* dst, const std::byte* base, const std::byte* resid,
                    std::size_t n) noexcept {
#if defined(SFS_KERNEL_DISABLED)
    xor_apply_scalar(dst, base, resid, n);
#else
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i));
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(resid + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(b, r));
    }
    // Unaligned tail (n not a multiple of 32): the scalar oracle's own loop,
    // byte for byte, so equivalence holds trivially here regardless of n.
    for (; i < n; ++i) dst[i] = base[i] ^ resid[i];
#endif
}

void xor_encode_avx2(std::byte* dst, const std::byte* base, const std::byte* target,
                     std::size_t n) noexcept {
    // Same bytewise XOR as xor_apply — self-inverse, per residual_scalar.cpp.
    xor_apply_avx2(dst, base, target, n);
}

namespace {

#if !defined(SFS_KERNEL_DISABLED)
// Vectorised zigzag-apply for 16-bit elements (fp16/bf16 — the dtype every
// fixture in this repo actually uses). 16 elements per 256-bit register.
// Formula is exactly residual_scalar.cpp's zigzag_apply_width<uint16_t>,
// lane-parallel: mask = all-ones if the low bit of z is set, else 0;
// delta_bits = (z>>1) ^ mask; t = b + delta_bits (wraparound add).
void zigzag_apply_avx2_u16(std::byte* dst, const std::byte* base, const std::byte* resid,
                           std::size_t count) noexcept {
    const __m256i one = _mm256_set1_epi16(1);
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base) + i / 16);
        const __m256i z = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(resid) + i / 16);
        const __m256i low_bit = _mm256_and_si256(z, one);
        const __m256i mask = _mm256_sub_epi16(_mm256_setzero_si256(), low_bit);  // 0 or all-ones
        const __m256i delta_bits = _mm256_xor_si256(_mm256_srli_epi16(z, 1), mask);
        const __m256i t = _mm256_add_epi16(b, delta_bits);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst) + i / 16, t);
    }
    if (i < count) {
        // Tail: fewer than 16 elements left. Delegate to the oracle on the
        // remaining range — its width-generic loop is exactly this formula
        // done one lane at a time.
        zigzag_apply_scalar(dst + i * 2, base + i * 2, resid + i * 2, (count - i) * 2, 2);
    }
}
#endif

}  // namespace

void zigzag_apply_avx2(std::byte* dst, const std::byte* base, const std::byte* resid,
                       std::size_t n, std::uint32_t elem_bytes) noexcept {
#if defined(SFS_KERNEL_DISABLED)
    zigzag_apply_scalar(dst, base, resid, n, elem_bytes);
#else
    if (elem_bytes == 2) {
        zigzag_apply_avx2_u16(dst, base, resid, n / 2);
        return;
    }
    // Other widths (1/4/8-byte dtypes) aren't vectorised here — real
    // checkpoints in this project are fp16/bf16 almost exclusively
    // (docs/spec/13's own examples), so this is the width worth spending the
    // intrinsics on. Falling back to the oracle is still correct, just not
    // faster than scalar for those widths.
    zigzag_apply_scalar(dst, base, resid, n, elem_bytes);
#endif
}

}  // namespace sfs::codec
