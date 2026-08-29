# FindFUSE3
# ---------
# Locates libfuse3 (>= 3.10, for FUSE_CAP_* we rely on) and defines the
# imported target FUSE3::FUSE3.
#
# Prefers pkg-config, which is what libfuse actually ships and what carries
# the required -D_FILE_OFFSET_BITS=64.
#
# Variables set: FUSE3_FOUND, FUSE3_VERSION, FUSE3_INCLUDE_DIRS, FUSE3_LIBRARIES

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_FUSE3 QUIET fuse3)
endif()

find_path(FUSE3_INCLUDE_DIR
    NAMES fuse_lowlevel.h
    HINTS ${PC_FUSE3_INCLUDEDIR} ${PC_FUSE3_INCLUDE_DIRS}
    PATH_SUFFIXES fuse3)

find_library(FUSE3_LIBRARY
    NAMES fuse3
    HINTS ${PC_FUSE3_LIBDIR} ${PC_FUSE3_LIBRARY_DIRS})

set(FUSE3_VERSION ${PC_FUSE3_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FUSE3
    REQUIRED_VARS FUSE3_LIBRARY FUSE3_INCLUDE_DIR
    VERSION_VAR   FUSE3_VERSION
    FAIL_MESSAGE  "libfuse3 development files not found. Install libfuse3-dev (Debian/Ubuntu) or fuse3-devel (Fedora), or configure with -DSFS_BUILD_MOUNT=OFF.")

if(FUSE3_FOUND AND NOT TARGET FUSE3::FUSE3)
    add_library(FUSE3::FUSE3 UNKNOWN IMPORTED)
    set_target_properties(FUSE3::FUSE3 PROPERTIES
        IMPORTED_LOCATION             "${FUSE3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FUSE3_INCLUDE_DIR}"
        # libfuse mandates these two. Getting _FILE_OFFSET_BITS wrong produces
        # a silently truncated off_t, which on multi-GB checkpoints is a bug
        # you find at fixture scale and nowhere else.
        INTERFACE_COMPILE_DEFINITIONS "FUSE_USE_VERSION=34;_FILE_OFFSET_BITS=64")
endif()

set(FUSE3_INCLUDE_DIRS ${FUSE3_INCLUDE_DIR})
set(FUSE3_LIBRARIES    ${FUSE3_LIBRARY})
mark_as_advanced(FUSE3_INCLUDE_DIR FUSE3_LIBRARY)
