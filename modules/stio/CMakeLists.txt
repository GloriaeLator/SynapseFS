# modules/stio
#
# Sources are globbed with CONFIGURE_DEPENDS deliberately. Eight people are
# adding files to nine modules over four days; an explicitly listed source set
# means a merge conflict in this file every time someone adds a .cpp, and a
# build that silently omits a file when someone resolves it wrong. Ninja
# re-runs CMake when the glob result changes, so the usual objection does not
# apply here. Revisit if this outlives the project.
file(GLOB SFS_STIO_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_STIO_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(stio
    SOURCES ${SFS_STIO_SOURCES}
    HEADERS ${SFS_STIO_HEADERS}
    DEPS    synapsefs::format
    PRIVATE_DEPS )

file(GLOB SFS_STIO_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_STIO_TESTS)
    sfs_add_module_test(stio ${t})
endforeach()
