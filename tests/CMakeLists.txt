# tests/ — integration and end-to-end. Module unit tests live beside their
# module; these are the ones that need more than one module in the room.
#
# The four that must never break (docs/testing.md):
#   byte_identity      sha256(reconstruct(A, diff(A,B))) == sha256(B), and
#                      checkout bytes == mount bytes
#   e2e.py             mmap works, load_file() works, unmodified
#   tamper             every single-byte tamper detected AND named
#   crash_matrix       kill -9 anywhere leaves clean or explicitly refusing

add_library(sfs_test_common INTERFACE)
target_include_directories(sfs_test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(sfs_test_common INTERFACE
    synapsefs::core synapsefs::store synapsefs::codec Catch2::Catch2WithMain)

file(GLOB SFS_INTEGRATION_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src IN LISTS SFS_INTEGRATION_TESTS)
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE
        sfs_test_common synapsefs::align synapsefs::stio synapsefs::net)
    if(SFS_BUILD_MOUNT)
        target_link_libraries(${name} PRIVATE synapsefs::mount)
    endif()
    sfs_set_warnings(${name})
    sfs_apply_sanitizers(${name})

    set(labels e2e)
    if(name STREQUAL "crash_matrix" OR name STREQUAL "sync_interrupt")
        list(APPEND labels slow)
    endif()
    if(name STREQUAL "concurrent_readers")
        list(APPEND labels concurrency)
    endif()
    catch_discover_tests(${name} TEST_PREFIX "e2e." PROPERTIES LABELS "${labels}")
endforeach()

# The PS requires safetensors.torch.load_file() to work UNMODIFIED against the
# mount. Only Python can assert that, so this is not optional decoration.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME e2e.python
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/e2e.py
                     --sfs $<TARGET_FILE:sfs>)
    set_tests_properties(e2e.python PROPERTIES LABELS "e2e;slow")
endif()
