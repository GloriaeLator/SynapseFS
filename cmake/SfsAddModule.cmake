# sfs_add_module(<name>
#     [SOURCES  <src>...]
#     [HEADERS  <hdr>...]        # for IDE/install only
#     [DEPS     <target>...]     # PUBLIC link deps
#     [PRIVATE_DEPS <target>...]
#     [INTERFACE])               # header-only
#
# Creates target sfs_<name>, alias synapsefs::<name>, with
# modules/<name>/include on the PUBLIC include path, warnings and sanitizers
# applied, and installation wired up.
#
# sfs_add_module_test(<module> <test-source> [LABELS <label>...])
#   Creates one Catch2 executable per test file and registers it with CTest.
#   One executable per file, not one per module: a crash in a FUSE test should
#   not take the other tests' results with it.

include(CompilerWarnings)
include(Sanitizers)

function(sfs_add_module name)
    cmake_parse_arguments(ARG "INTERFACE" "" "SOURCES;HEADERS;DEPS;PRIVATE_DEPS" ${ARGN})

    set(target sfs_${name})
    set(inc "${CMAKE_CURRENT_SOURCE_DIR}/include")

    if(ARG_INTERFACE OR NOT ARG_SOURCES)
        add_library(${target} INTERFACE)
        target_include_directories(${target} INTERFACE
            $<BUILD_INTERFACE:${inc}>
            $<INSTALL_INTERFACE:include>)
        if(ARG_DEPS)
            target_link_libraries(${target} INTERFACE ${ARG_DEPS})
        endif()
    else()
        add_library(${target} STATIC ${ARG_SOURCES} ${ARG_HEADERS})
        target_include_directories(${target} PUBLIC
            $<BUILD_INTERFACE:${inc}>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(${target} PUBLIC cxx_std_23)
        if(ARG_DEPS)
            target_link_libraries(${target} PUBLIC ${ARG_DEPS})
        endif()
        if(ARG_PRIVATE_DEPS)
            target_link_libraries(${target} PRIVATE ${ARG_PRIVATE_DEPS})
        endif()
        sfs_set_warnings(${target})
        sfs_apply_sanitizers(${target})
        if(SFS_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
            set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    endif()

    add_library(synapsefs::${name} ALIAS ${target})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${name})

    install(TARGETS ${target} EXPORT synapsefsTargets)
    install(DIRECTORY ${inc}/ DESTINATION include)
endfunction()

function(sfs_add_module_test module source)
    if(NOT SFS_BUILD_TESTS)
        return()
    endif()
    cmake_parse_arguments(ARG "" "" "LABELS;DEPS" ${ARGN})

    get_filename_component(tname ${source} NAME_WE)
    set(target ${module}_${tname})

    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        synapsefs::${module} Catch2::Catch2WithMain ${ARG_DEPS})
    sfs_set_warnings(${target})
    sfs_apply_sanitizers(${target})

    set(labels unit ${module})
    if(ARG_LABELS)
        list(APPEND labels ${ARG_LABELS})
    endif()

    catch_discover_tests(${target}
        TEST_PREFIX "${module}."
        PROPERTIES LABELS "${labels}")
endfunction()
