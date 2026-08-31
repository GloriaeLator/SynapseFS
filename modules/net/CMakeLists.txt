# modules/net
#
# Sources are globbed with CONFIGURE_DEPENDS deliberately. Eight people are
# adding files to nine modules over four days; an explicitly listed source set
# means a merge conflict in this file every time someone adds a .cpp, and a
# build that silently omits a file when someone resolves it wrong. Ninja
# re-runs CMake when the glob result changes, so the usual objection does not
# apply here. Revisit if this outlives the project.
cmake_minimum_required(VERSION 3.25)

file(GLOB SFS_NET_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_NET_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(net
    SOURCES ${SFS_NET_SOURCES}
    HEADERS ${SFS_NET_HEADERS}
    DEPS    synapsefs::store
    PRIVATE_DEPS OpenSSL::Crypto;PkgConfig::json-c)

file(GLOB SFS_NET_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_NET_TESTS)
    sfs_add_module_test(net ${t})
endforeach()
