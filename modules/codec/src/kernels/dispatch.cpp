#include <synapsefs/codec/residual_codec.hpp>

// No ISA flags on this file — see cmake/SimdKernels.cmake and ADR 0011. It
// must run on any machine in order to decide what the machine can run, so it
// is compiled exactly like any ordinary translation unit.

namespace sfs::codec {

// Not declared in residual_codec.hpp — only the scalar oracle is public API
// (docs/adr/0011). Defined in kernels/residual_avx2.cpp and
// kernels/residual_avx512.cpp, which always exist and are always compiled
// once present in the tree: cmake/SimdKernels.cmake still builds them (minus
// the -mavx2/-mavx512* flags, with SFS_KERNEL_DISABLED defined instead) even
// on a compiler that rejects the ISA, and each function internally forwards
// to the scalar oracle in that case — so these symbols are safe to reference
// unconditionally here, regardless of what THIS file's own compile flags are.
void xor_apply_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void xor_encode_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void zigzag_apply_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t,
                       std::uint32_t) noexcept;

void xor_apply_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void xor_encode_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void zigzag_apply_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t,
                         std::uint32_t) noexcept;

namespace {

struct DispatchTable {
    util::Isa     isa;
    XorApplyFn    xor_apply;
    XorEncodeFn   xor_encode;
    ZigzagApplyFn zigzag_apply;
};

const DispatchTable& table() noexcept {
    // Chosen once (function-local static init is thread-safe in C++11+) from
    // util::best_isa(), which itself honours SFS_FORCE_ISA. Dispatch happens
    // once per FRAME via the returned function pointers, never per unit —
    // the indirect call is noise next to the ZSTD_decompress beside it.
    static const DispatchTable selected = [] {
        switch (util::best_isa()) {
            case util::Isa::Avx512:
                return DispatchTable{util::Isa::Avx512, &xor_apply_avx512, &xor_encode_avx512,
                                     &zigzag_apply_avx512};
            case util::Isa::Avx2:
                return DispatchTable{util::Isa::Avx2, &xor_apply_avx2, &xor_encode_avx2,
                                     &zigzag_apply_avx2};
            case util::Isa::Scalar:
            default:
                return DispatchTable{util::Isa::Scalar, &xor_apply_scalar,
                                     &xor_encode_scalar, &zigzag_apply_scalar};
        }
    }();
    return selected;
}

}  // namespace

XorApplyFn xor_apply_dispatch() noexcept { return table().xor_apply; }
XorEncodeFn xor_encode_dispatch() noexcept { return table().xor_encode; }
ZigzagApplyFn zigzag_apply_dispatch() noexcept { return table().zigzag_apply; }
util::Isa active_isa() noexcept { return table().isa; }

}  // namespace sfs::codec
