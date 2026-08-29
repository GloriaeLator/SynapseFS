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
        synapsefs::codec synapsefs::align synapsefs::stio)
    if(SFS_BUILD_MOUNT)
        target_link_libraries(bench_${name} PRIVATE synapsefs::mount)
    endif()
    set_target_properties(bench_${name} PROPERTIES OUTPUT_NAME ${name})
    sfs_set_warnings(bench_${name})
    # Deliberately NOT sanitized: a sanitized benchmark measures the sanitizer.
endforeach()
