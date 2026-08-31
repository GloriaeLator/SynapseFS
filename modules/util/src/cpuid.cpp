#include <synapsefs/util/cpuid.hpp>

#include <cstdlib>
#include <string_view>

// cpuid.hpp is declared but had no implementation anywhere in the tree; the
// codec module's kernel dispatch (src/kernels/dispatch.cpp) needs it to pick
// an ISA, so it is written here rather than left as a second unimplemented
// module alongside align/.
//
// x86/x86-64 only: this is the only architecture the residual kernels are
// ever compiled for (docs/adr/0011-simd-dispatch-strategy.md names AVX2 and
// AVX-512 specifically). On any other architecture every feature reads false
// and best_isa() correctly settles on Scalar.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define SFS_CPUID_X86 1
#include <cpuid.h>
#else
#define SFS_CPUID_X86 0
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define SFS_CPUID_HAVE_SYSCONF 1
#else
#define SFS_CPUID_HAVE_SYSCONF 0
#endif

namespace sfs::util {

namespace {

#if SFS_CPUID_X86
// OS state-save check: CPUID alone only says the SILICON has AVX/AVX-512, not
// that the kernel enabled XSAVE for it. Trusting the CPUID bit alone is
// exactly how you get a SIGILL on an OS/VM that hasn't turned it on — this is
// the check the header's comment calls out by name.
std::uint64_t xgetbv(std::uint32_t index) noexcept {
    std::uint32_t eax = 0, edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
}
#endif

CpuFeatures detect() noexcept {
    CpuFeatures f;

#if SFS_CPUID_X86
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        f.sse2  = (edx & (1u << 26)) != 0;
        f.sse41 = (ecx & (1u << 19)) != 0;
        f.avx   = (ecx & (1u << 28)) != 0;
        f.fma   = (ecx & (1u << 12)) != 0;

        const bool os_xsave = (ecx & (1u << 27)) != 0;  // OSXSAVE
        if (os_xsave) {
            const std::uint64_t xcr0 = xgetbv(0);
            f.os_avx_enabled    = (xcr0 & 0x6) == 0x6;    // XMM (bit1) + YMM (bit2)
            f.os_avx512_enabled = (xcr0 & 0xE6) == 0xE6;  // + opmask/ZMM-lo/ZMM-hi
        }
    }

    unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        f.avx2     = (ebx & (1u << 5))  != 0;
        f.bmi2     = (ebx & (1u << 8))  != 0;
        f.avx512f  = (ebx & (1u << 16)) != 0;
        f.avx512bw = (ebx & (1u << 30)) != 0;
        f.avx512vl = (ebx & (1u << 31)) != 0;
        f.sha_ni   = (ebx & (1u << 29)) != 0;
    }

    // A feature the silicon has but the OS won't save is unusable: fold the
    // OS check into the feature bits so callers never have to remember to
    // check both.
    f.avx2     = f.avx2     && f.os_avx_enabled;
    f.avx512f  = f.avx512f  && f.os_avx512_enabled;
    f.avx512bw = f.avx512bw && f.os_avx512_enabled;
    f.avx512vl = f.avx512vl && f.os_avx512_enabled;
#endif

    f.cache_line_bytes = 64;
    f.logical_cores = 1;
#if SFS_CPUID_HAVE_SYSCONF
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) f.logical_cores = static_cast<std::uint32_t>(n);
#endif

    return f;
}

}  // namespace

const CpuFeatures& cpu_features() noexcept {
    static const CpuFeatures features = detect();
    return features;
}

Isa best_isa() noexcept {
    // SFS_FORCE_ISA overrides detection outright, for benchmarking
    // (bench/residual_codec.cpp --kernel-only) and for bisecting a suspected
    // kernel bug (ADR 0011).
    if (const char* forced = std::getenv("SFS_FORCE_ISA")) {
        const std::string_view v(forced);
        if (v == "scalar") return Isa::Scalar;
        if (v == "avx2")   return Isa::Avx2;
        if (v == "avx512") return Isa::Avx512;
        // Unrecognised value: fall through to real detection rather than
        // silently picking one, since a typo here should not be invisible.
    }

    const auto& f = cpu_features();
    if (f.avx512f && f.avx512bw && f.avx512vl) return Isa::Avx512;
    if (f.avx2 && f.fma && f.bmi2) return Isa::Avx2;
    return Isa::Scalar;
}

std::string_view isa_name(Isa isa) noexcept {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2:   return "avx2";
        case Isa::Avx512: return "avx512";
    }
    return "scalar";
}

}  // namespace sfs::util

#undef SFS_CPUID_X86
#undef SFS_CPUID_HAVE_SYSCONF
