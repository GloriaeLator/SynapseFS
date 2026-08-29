# modules/format
#
# Sources are globbed with CONFIGURE_DEPENDS deliberately. Eight people are
# adding files to nine modules over four days; an explicitly listed source set
# means a merge conflict in this file every time someone adds a .cpp, and a
# build that silently omits a file when someone resolves it wrong. Ninja
# re-runs CMake when the glob result changes, so the usual objection does not
# apply here. Revisit if this outlives the project.
file(GLOB SFS_FORMAT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_FORMAT_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(format
    SOURCES ${SFS_FORMAT_SOURCES}
    HEADERS ${SFS_FORMAT_HEADERS}
    DEPS    synapsefs::core
    PRIVATE_DEPS nlohmann_json::nlohmann_json)

file(GLOB SFS_FORMAT_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_FORMAT_TESTS)
    sfs_add_module_test(format ${t})
endforeach()
