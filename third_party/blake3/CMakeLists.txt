# Placeholder until the upstream submodule is added; see ../README.md.
#
# Once third_party/blake3-upstream exists, replace the body of this file with:
#
#   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../blake3-upstream/c blake3 EXCLUDE_FROM_ALL)
#   add_library(sfs_blake3 ALIAS blake3)
#
# The stub below lets the tree configure before the submodule lands, so that a
# docs-only PR does not break `cmake --preset dev` for everyone else.

add_library(blake3 INTERFACE)
target_compile_definitions(blake3 INTERFACE SFS_BLAKE3_STUB=1)
message(WARNING
  "third_party/blake3 is a stub. Hashing will not work until the upstream "
  "submodule is added - see third_party/README.md.")
