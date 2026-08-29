# Per-translation-unit ISA flags for the residual kernels.
#
# The rule this file exists to enforce: the ISA flag is attached to ONE source
# file, never to a target and never globally. A binary built with -mavx512f
# everywhere SIGILLs on the evaluator's machine if it turns out to be a Zen 2
# or a Skylake-client. We compile every kernel, and choose at runtime from
# CPUID (see modules/util/include/synapsefs/util/cpuid.hpp and
# docs/adr/0011-simd-dispatch-strategy.md).

include(CheckCXXCompilerFlag)

set(SFS_AVX2_FLAGS   -mavx2 -mfma -mbmi -mbmi2)
set(SFS_AVX512_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512dq)

check_cxx_compiler_flag("-mavx2"     SFS_COMPILER_HAS_AVX2)
check_cxx_compiler_flag("-mavx512bw" SFS_COMPILER_HAS_AVX512)

# sfs_add_isa_source(<target> <source> <isa>)
#   isa: scalar | avx2 | avx512
function(sfs_add_isa_source target source isa)
    target_sources(${target} PRIVATE ${source})

    if(isa STREQUAL "scalar")
        # The scalar kernel is the correctness oracle for
        # test_kernel_equivalence. Do not let the compiler auto-vectorise it
        # into something with different rounding or different UB.
        set_source_files_properties(${source} PROPERTIES
            COMPILE_OPTIONS "-fno-tree-vectorize")
        return()
    endif()

    if(NOT SFS_ENABLE_SIMD)
        set_source_files_properties(${source} PROPERTIES
            COMPILE_DEFINITIONS "SFS_KERNEL_DISABLED=1")
        return()
    endif()

    if(isa STREQUAL "avx2")
        if(NOT SFS_COMPILER_HAS_AVX2)
            message(WARNING "Compiler rejects -mavx2; AVX2 residual kernel disabled")
            set_source_files_properties(${source} PROPERTIES
                COMPILE_DEFINITIONS "SFS_KERNEL_DISABLED=1")
            return()
        endif()
        set_source_files_properties(${source} PROPERTIES
            COMPILE_OPTIONS "${SFS_AVX2_FLAGS}"
            COMPILE_DEFINITIONS "SFS_HAVE_AVX2=1")
    elseif(isa STREQUAL "avx512")
        if(NOT SFS_COMPILER_HAS_AVX512)
            message(STATUS "Compiler rejects -mavx512bw; AVX-512 residual kernel disabled")
            set_source_files_properties(${source} PROPERTIES
                COMPILE_DEFINITIONS "SFS_KERNEL_DISABLED=1")
            return()
        endif()
        set_source_files_properties(${source} PROPERTIES
            COMPILE_OPTIONS "${SFS_AVX512_FLAGS}"
            COMPILE_DEFINITIONS "SFS_HAVE_AVX512=1")
    else()
        message(FATAL_ERROR "sfs_add_isa_source: unknown ISA '${isa}'")
    endif()
endfunction()
