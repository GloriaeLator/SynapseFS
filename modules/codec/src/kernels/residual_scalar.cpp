#include <synapsefs/codec/residual_codec.hpp>

#include <cstring>

// The reference implementation. No ISA flags are attached to this source
// file in cmake/SimdKernels.cmake, and it is built with -fno-tree-vectorize,
// so it stays a literal, portable, deliberately unoptimised statement of
// what every vectorised kernel must reproduce byte-for-byte
// (test_kernel_equivalence.cpp treats this as the oracle).

namespace sfs::codec {

void xor_apply_scalar(std::byte* dst, const std::byte* base, const std::byte* resid,
                      std::size_t n) noexcept {
    // target[i] = base[i] ^ residual[i], bytewise. spec 12 §5.
    for (std::size_t i = 0; i < n; ++i) dst[i] = base[i] ^ resid[i];
}

void xor_encode_scalar(std::byte* dst, const std::byte* base, const std::byte* target,
                       std::size_t n) noexcept {
    // XOR is its own inverse: the writer computes exactly the same
    // byte-for-byte operation to go target -> residual as the reader uses to
    // go residual -> target. Two names for one loop, kept separate because
    // the call sites (write path vs. read path) should never be confused
    // about which direction they're going.
    for (std::size_t i = 0; i < n; ++i) dst[i] = base[i] ^ target[i];
}

namespace {

// zigzag_after_permute is delta + zigzag over the dtype's INTEGER view — the
// bit pattern, never the float value (spec 12 §5: no floating-point
// arithmetic on weights anywhere). Addition/subtraction on the raw bits is
// done modulo 2^(8*sizeof(U)) via unsigned wraparound, which C++ defines
// exactly and which is why this is bit-exact regardless of what the bits
// mean numerically.
//
// Decoding a zigzag code back to a delta's two's-complement BIT PATTERN and
// then adding it to base as unsigned is equivalent to signed addition mod
// 2^n; there is no need to materialise a signed type at all.
template <typename U>
void zigzag_apply_width(std::byte* dst, const std::byte* base, const std::byte* resid,
                        std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        U b = 0, z = 0;
        std::memcpy(&b, base + i * sizeof(U), sizeof(U));
        std::memcpy(&z, resid + i * sizeof(U), sizeof(U));

        // util::zigzag_decode, generalised from a fixed 64-bit width to U:
        // signed_delta_bits = (z >> 1) ^ (~(z & 1) + 1)
        const U mask = static_cast<U>(~(z & U{1}) + U{1});  // all-ones if z odd, else 0
        const U delta_bits = static_cast<U>((z >> 1) ^ mask);

        const U t = static_cast<U>(b + delta_bits);  // wraparound add, mod 2^n
        std::memcpy(dst + i * sizeof(U), &t, sizeof(U));
    }
}

}  // namespace

void zigzag_apply_scalar(std::byte* dst, const std::byte* base, const std::byte* resid,
                         std::size_t n, std::uint32_t elem_bytes) noexcept {
    // n is the frame's total byte count, same convention as xor_apply_scalar;
    // it must be a whole multiple of elem_bytes (guaranteed by
    // dtype_size()'s range of {1,2,4,8} dividing unit_bytes exactly, per
    // spec 12 §2 — apply_residual() in residual_codec.cpp validates this
    // before ever reaching here).
    switch (elem_bytes) {
        case 1: zigzag_apply_width<std::uint8_t>(dst, base, resid, n / 1); return;
        case 2: zigzag_apply_width<std::uint16_t>(dst, base, resid, n / 2); return;
        case 4: zigzag_apply_width<std::uint32_t>(dst, base, resid, n / 4); return;
        case 8: zigzag_apply_width<std::uint64_t>(dst, base, resid, n / 8); return;
        default: return;  // unreachable: DType::dtype_size() never returns anything else
    }
}

}  // namespace sfs::codec
