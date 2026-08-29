#pragma once
/// \file cpuid.hpp
/// Runtime CPU feature detection for residual kernel dispatch.
///
/// Read once, at startup, and cached. The kernels are compiled per-ISA in
/// separate translation units (cmake/SimdKernels.cmake); this decides which
/// one to call. Never put an ISA flag on a target — see
/// docs/adr/0011-simd-dispatch-strategy.md.

#include <cstdint>
#include <string_view>

namespace sfs::util {

enum class Isa : std::uint8_t {
    Scalar = 0,
    Avx2   = 1,
    Avx512 = 2,
};

struct CpuFeatures {
    bool sse2   = false;
    bool sse41  = false;
    bool avx    = false;
    bool avx2   = false;
    bool bmi2   = false;
    bool fma    = false;
    bool avx512f  = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool sha_ni = false;   ///< Decides whether ADR 0002's BLAKE3 margin holds here.
    /// True only if the OS actually saves the wide register state (xgetbv).
    /// Checking the CPUID bit alone is how you get a SIGILL on a machine that
    /// technically has the instructions.
    bool os_avx_enabled    = false;
    bool os_avx512_enabled = false;

    std::uint32_t cache_line_bytes = 64;
    std::uint32_t logical_cores    = 1;
};

/// Detected once on first call, then cached. Thread-safe.
[[nodiscard]] const CpuFeatures& cpu_features() noexcept;

/// Highest ISA usable on this machine, honouring the SFS_FORCE_ISA environment
/// variable ("scalar", "avx2", "avx512") for benchmarking and bisection.
[[nodiscard]] Isa best_isa() noexcept;

[[nodiscard]] std::string_view isa_name(Isa) noexcept;

}  // namespace sfs::util
