#pragma once
/// \file residual_codec.hpp
/// The residual kernels and their runtime dispatch.
///
/// XOR and zigzag-delta are both BIJECTIVE and both exact — no floating-point
/// arithmetic is performed on weights anywhere, so reconstruction is bit-exact
/// by construction rather than by luck.
///
/// Dispatch is a plain function pointer chosen once from CPUID, called once per
/// FRAME. Not a virtual interface, and nothing per-unit
/// (docs/adr/0011-simd-dispatch-strategy.md). Every ISA is compiled; the flag
/// lives on the source file, never on a target — a stray global -mavx512f is a
/// SIGILL on the evaluator's machine and nothing in the build will warn you.

#include <cstddef>
#include <cstdint>
#include <span>

#include <synapsefs/core/dtype.hpp>
#include <synapsefs/format/residual_hdr.hpp>
#include <synapsefs/util/cpuid.hpp>

namespace sfs::codec {

using XorApplyFn   = void (*)(std::byte* dst, const std::byte* base,
                              const std::byte* resid, std::size_t n) noexcept;
using XorEncodeFn  = void (*)(std::byte* dst, const std::byte* base,
                              const std::byte* target, std::size_t n) noexcept;
using ZigzagApplyFn = void (*)(std::byte* dst, const std::byte* base,
                               const std::byte* resid, std::size_t n,
                               std::uint32_t elem_bytes) noexcept;

/// Selected once, honouring SFS_FORCE_ISA. Thread-safe.
[[nodiscard]] XorApplyFn    xor_apply_dispatch() noexcept;
[[nodiscard]] XorEncodeFn   xor_encode_dispatch() noexcept;
[[nodiscard]] ZigzagApplyFn zigzag_apply_dispatch() noexcept;
[[nodiscard]] util::Isa     active_isa() noexcept;

/// The scalar implementations. These are the ORACLE for
/// test_kernel_equivalence.cpp: every enabled ISA must produce byte-identical
/// output on random inputs, including unaligned tails.
void xor_apply_scalar(std::byte* dst, const std::byte* base, const std::byte* resid,
                      std::size_t n) noexcept;
void xor_encode_scalar(std::byte* dst, const std::byte* base, const std::byte* target,
                       std::size_t n) noexcept;
void zigzag_apply_scalar(std::byte* dst, const std::byte* base, const std::byte* resid,
                         std::size_t n, std::uint32_t elem_bytes) noexcept;

/// Apply a residual of the kind named in the artifact.
[[nodiscard]] core::Status apply_residual(format::ResidualKind, format::Transform,
                                          core::DType, std::span<const std::byte> base,
                                          std::span<const std::byte> residual,
                                          std::span<std::byte> out);

}  // namespace sfs::codec
