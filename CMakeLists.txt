cmake_minimum_required(VERSION 3.25)

# See docs/adr/0009-vcpkg-vs-fetchcontent.md.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "vcpkg toolchain file")
endif()

project(synapsefs
  VERSION 0.1.0
  DESCRIPTION "Permutation-aware, verifiable version control and virtual filesystem for neural network checkpoints"
  LANGUAGES C CXX)

# Global settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "SynapseFS targets Linux only (see docs/adr/0001-cpp23-and-toolchain.md). "
    "Detected: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(SFS_BUILD_TESTS      "Build unit and integration tests"      ON)
option(SFS_BUILD_BENCH      "Build benchmark executables"           ON)
option(SFS_BUILD_MOUNT      "Build the FUSE mount module and daemon" ON)
option(SFS_ENABLE_SIMD      "Build AVX2/AVX-512 residual kernels"   ON)
option(SFS_ENABLE_LTO       "Enable interprocedural optimisation in release builds" OFF)
option(SFS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"   OFF)

set(SFS_SANITIZER "none" CACHE STRING "Sanitizer: none|address|thread|undefined|address+undefined")
set_property(CACHE SFS_SANITIZER PROPERTY STRINGS none address thread undefined address+undefined)

include(CompilerWarnings)
include(Sanitizers)
include(SimdKernels)
include(SfsAddModule)

# Dependencies
#
# zstd::libzstd is what every module here links against, but the target
# name a CONFIG package actually exports is not standardised: vcpkg's zstd
# port exports zstd::libzstd directly, while Debian/Ubuntu's system
# libzstd-dev package (found via find_package(zstd CONFIG) on a plain apt
# install, no vcpkg involved) exports zstd::libzstd_shared /
# zstd::libzstd_static instead. Normalise whichever was found to the one
# name the rest of the build expects, falling back to pkg-config only if
# neither CONFIG target exists at all.
find_package(zstd CONFIG QUIET)
if(TARGET zstd::libzstd)
  # Already the expected name (vcpkg and some distros' packaging).
elseif(TARGET zstd::libzstd_shared)
  add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  add_library(zstd::libzstd ALIAS zstd::libzstd_static)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
  add_library(zstd::libzstd ALIAS PkgConfig::zstd)
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Torch REQUIRED)
pkg_check_modules(json-c REQUIRED IMPORTED_TARGET json-c)

if(SFS_BUILD_MOUNT)
  find_package(FUSE3 REQUIRED)
endif()

if(SFS_BUILD_TESTS)
  find_package(Catch2 3 CONFIG REQUIRED)
  include(CTest)
  include(Catch)
endif()

# See docs/adr/0002-blake3-over-sha256.md.
add_subdirectory(third_party/blake3 EXCLUDE_FROM_ALL)

# Modules, in dependency order
add_subdirectory(modules/util)
add_subdirectory(modules/core)
add_subdirectory(modules/format)
add_subdirectory(modules/stio)
add_subdirectory(modules/codec)   
add_subdirectory(modules/store)
add_subdirectory(modules/align)
if(SFS_BUILD_MOUNT)
  add_subdirectory(modules/mount)
endif()
add_subdirectory(modules/net)

add_subdirectory(apps/sfs)

if(SFS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(SFS_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# Summary
message(STATUS "")
message(STATUS "SynapseFS ${PROJECT_VERSION}")
message(STATUS "  build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  sanitizer      : ${SFS_SANITIZER}")
message(STATUS "  tests / bench  : ${SFS_BUILD_TESTS} / ${SFS_BUILD_BENCH}")
message(STATUS "  mount (FUSE3)  : ${SFS_BUILD_MOUNT}")
message(STATUS "  simd kernels   : ${SFS_ENABLE_SIMD}")
message(STATUS "")
