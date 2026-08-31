# Point to the upstream BLAKE3 C implementation
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../blake3-upstream/c blake3 EXCLUDE_FROM_ALL)
add_library(sfs_blake3 ALIAS blake3)
