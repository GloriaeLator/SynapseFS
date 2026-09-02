# tests/ — integration and end-to-end. Module unit tests live beside their
# module; these are the ones that need more than one module in the room.
#
# The four that must never break (docs/testing.md):
#   byte_identity      sha256(reconstruct(A, diff(A,B))) == sha256(B), and
#                      checkout bytes == mount bytes
#   e2e.py             mmap works, load_file() works, unmodified
#   tamper             every single-byte tamper detected AND named
#   crash_matrix       kill -9 anywhere leaves clean or explicitly refusing

# STATIC, not INTERFACE: common/harness.cpp needs somewhere to actually be
# compiled. temprepo.hpp/fault_inject.hpp are still header-only-declared (no
# .cpp yet) and stay unaffected by this either way.
add_library(sfs_test_common STATIC common/harness.cpp)
target_include_directories(sfs_test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(sfs_test_common PUBLIC
    synapsefs::core synapsefs::format synapsefs::stio synapsefs::store synapsefs::codec
    Catch2::Catch2WithMain)
sfs_set_warnings(sfs_test_common)

file(GLOB SFS_INTEGRATION_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src IN LISTS SFS_INTEGRATION_TESTS)
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE sfs_test_common synapsefs::stio)
    # Most integration tests only need align::MatchReport/GroupMatch as
    # hand-built plain data (never a real align::Matcher run) and no net::
    # symbols at all -- include-only, not linked, so this whole directory
    # stays buildable without Torch by default (same reasoning as
    # modules/store's and bench/'s own align decoupling).
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/modules/align/include
        ${CMAKE_SOURCE_DIR}/modules/net/include)
    # Exception: byte_identity_cnn.cpp calls the REAL, compiled
    # align::parse_topology() (not just its plain-data types), so it
    # genuinely needs the linked library -- and therefore Torch -- to build
    # at all. Add any future test with the same need here, not to the
    # blanket link above.
    if(name STREQUAL "byte_identity_cnn")
        target_link_libraries(${name} PRIVATE synapsefs::align)
    endif()
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
