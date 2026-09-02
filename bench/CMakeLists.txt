# bench/ — the five graded numbers.
#
# Build these from the `release` preset and no other. A RelWithDebInfo number
# is not a result, and docs/benchmarks.md says so.

file(GLOB SFS_BENCHES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src IN LISTS SFS_BENCHES)
    get_filename_component(name ${src} NAME_WE)
    add_executable(bench_${name} ${src})
    target_link_libraries(bench_${name} PRIVATE
        synapsefs::core synapsefs::util synapsefs::store
        synapsefs::codec synapsefs::stio)
    # Include-only, not linked, by default: most benches don't call compiled
    # align:: code (sfs_store already went through this exact change -- see
    # modules/store/CMakeLists.txt). A hard link here would force every
    # bench target through sfs_align's Torch dependency for headers whose
    # symbols (align::MatchReport, GroupMatch, etc.) are only ever used as
    # plain data, never called.
    target_include_directories(bench_${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/modules/align/include)
    # Exception: residual_codec.cpp calls the REAL, compiled
    # align::parse_topology_file() (not just plain-data types) to build its
    # fixtures' topology structurally, same reasoning as
    # tests/CMakeLists.txt's byte_identity_cnn exception -- genuinely needs
    # the linked library, and therefore Torch, to build at all.
    if(name STREQUAL "residual_codec")
        target_link_libraries(bench_${name} PRIVATE synapsefs::align)
    endif()
    # align_time.cpp runs the real align::Matcher/parse_topology_file end to
    # end (it IS the alignment wall-clock/accuracy benchmark), and reads real
    # .safetensors fixtures via align's own SimpleStSource tool
    # (modules/align/tools/simple_st_source.hpp) rather than duplicating a
    # tensor-source adapter here.
    if(name STREQUAL "align_time")
        target_link_libraries(bench_${name} PRIVATE synapsefs::align sfs_align_tools)
    endif()
    # sparse_bench.cpp calls Matcher::match_group() directly (routes to the
    # real sparse fingerprint+auction path above sparse_crossover) and reads
    # fixtures via SimpleStSource, same reasoning as align_time above.
    if(name STREQUAL "sparse_bench")
        target_link_libraries(bench_${name} PRIVATE synapsefs::align sfs_align_tools)
    endif()
    # lap_bench.cpp benchmarks align::make_jv_solver()/make_greedy_solver()
    # directly against synthetic cost matrices -- no checkpoint involved, but
    # still the real, compiled align:: solver code, so it needs the real
    # link (and therefore Torch) same as the two exceptions above.
    if(name STREQUAL "lap_bench")
        target_link_libraries(bench_${name} PRIVATE synapsefs::align)
    endif()
    if(SFS_BUILD_MOUNT)
        target_link_libraries(bench_${name} PRIVATE synapsefs::mount)
    endif()
    set_target_properties(bench_${name} PROPERTIES OUTPUT_NAME ${name})
    sfs_set_warnings(bench_${name})
    # Deliberately NOT sanitized: a sanitized benchmark measures the sanitizer.
endforeach()
