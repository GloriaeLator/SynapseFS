# modules/mount
#
# Only built when SFS_BUILD_MOUNT is on; the root CMakeLists guards the
# add_subdirectory. libfuse3 comes from the system, not from vcpkg: it is tied
# to the kernel's FUSE ABI and to the installed fusermount3.
# FUSE3::FUSE3 carries FUSE_USE_VERSION=34 and _FILE_OFFSET_BITS=64 — getting
# the latter wrong truncates off_t, which only shows up at multi-GB, which is
# exactly the fixture size.

file(GLOB SFS_MOUNT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_MOUNT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(mount
    SOURCES ${SFS_MOUNT_SOURCES}
    HEADERS ${SFS_MOUNT_HEADERS}
    DEPS    synapsefs::store synapsefs::codec
    PRIVATE_DEPS FUSE3::FUSE3 Threads::Threads)

file(GLOB SFS_MOUNT_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_MOUNT_TESTS)
    # The cache race test is the reason the tsan preset exists.
    get_filename_component(tn ${t} NAME_WE)
    if(tn STREQUAL "test_blockcache_race")
        sfs_add_module_test(mount ${t} LABELS concurrency slow)
    else()
        sfs_add_module_test(mount ${t})
    endif()
endforeach()
