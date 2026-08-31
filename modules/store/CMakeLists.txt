# modules/store

file(GLOB SFS_STORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_STORE_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(store
    SOURCES ${SFS_STORE_SOURCES}
    HEADERS ${SFS_STORE_HEADERS}
    DEPS    synapsefs::format;synapsefs::codec
    PRIVATE_DEPS )

target_include_directories(sfs_store PUBLIC
    ${CMAKE_SOURCE_DIR}/modules/align/include)

file(GLOB SFS_STORE_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_STORE_TESTS)
    sfs_add_module_test(store ${t})
endforeach()
