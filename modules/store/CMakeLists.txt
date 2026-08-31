# modules/store
#
# Sources are globbed with CONFIGURE_DEPENDS deliberately. Eight people are
# adding files to nine modules over four days; an explicitly listed source set
# means a merge conflict in this file every time someone adds a .cpp, and a
# build that silently omits a file when someone resolves it wrong. Ninja
# re-runs CMake when the glob result changes, so the usual objection does not
# apply here. Revisit if this outlives the project.
file(GLOB SFS_STORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_STORE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(store
    SOURCES ${SFS_STORE_SOURCES}
    HEADERS ${SFS_STORE_HEADERS}
    # align added for commit_planner.hpp: the align<->codec glue that turns
    # an align::MatchReport into per-tensor storage decisions belongs beside
    # merge_logic.hpp, not duplicated into apps/sfs (which only orchestrates
    # I/O -- repo config, refs, locking -- around calls into here).
    DEPS    synapsefs::format;synapsefs::codec;synapsefs::align
    PRIVATE_DEPS )

file(GLOB SFS_STORE_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_STORE_TESTS)
    sfs_add_module_test(store ${t})
endforeach()
