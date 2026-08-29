# modules/codec
#
# The one module whose CMakeLists is not boilerplate: the residual kernels are
# compiled once per ISA, each in its own translation unit with its own flags,
# and selected at runtime from CPUID. The flag goes on the SOURCE FILE, never
# on the target and never globally — one stray global -mavx512f and the binary
# SIGILLs on the evaluator's machine with nothing in the build to warn you.
# docs/adr/0011-simd-dispatch-strategy.md

file(GLOB SFS_CODEC_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_CODEC_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(codec
    SOURCES ${SFS_CODEC_SOURCES}
    HEADERS ${SFS_CODEC_HEADERS}
    DEPS    synapsefs::format synapsefs::util
    PRIVATE_DEPS zstd::libzstd)

# Kernels, one ISA per file. dispatch.cpp gets NO ISA flags: it must be able to
# run on any machine in order to decide what the machine can run.
set(SFS_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels")
if(EXISTS "${SFS_KERNEL_DIR}/dispatch.cpp")
    target_sources(sfs_codec PRIVATE "${SFS_KERNEL_DIR}/dispatch.cpp")
endif()
foreach(pair "residual_scalar.cpp;scalar" "residual_avx2.cpp;avx2" "residual_avx512.cpp;avx512")
    list(GET pair 0 kfile)
    list(GET pair 1 kisa)
    if(EXISTS "${SFS_KERNEL_DIR}/${kfile}")
        sfs_add_isa_source(sfs_codec "${SFS_KERNEL_DIR}/${kfile}" ${kisa})
    endif()
endforeach()

file(GLOB SFS_CODEC_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_CODEC_TESTS)
    sfs_add_module_test(codec ${t})
endforeach()
