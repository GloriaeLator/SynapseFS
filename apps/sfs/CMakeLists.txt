# apps/sfs — the single CLI binary.
#
# Every subcommand lives in its own file under cmd/ so that eleven people are
# not editing one main.cpp. A command that is wired but unimplemented exits 7
# with a message naming what is missing; silently succeeding is worse than
# failing, and during a build week an honest exit 7 is what keeps the
# integration script meaningful. docs/spec/15-cli-contract.md §3.

file(GLOB SFS_APP_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmd/*.cpp")

add_executable(sfs ${SFS_APP_SOURCES})

target_link_libraries(sfs PRIVATE
    synapsefs::core synapsefs::util synapsefs::format synapsefs::stio
    synapsefs::store synapsefs::align synapsefs::codec synapsefs::net
    CLI11::CLI11)

if(SFS_BUILD_MOUNT)
    target_link_libraries(sfs PRIVATE synapsefs::mount)
    target_compile_definitions(sfs PRIVATE SFS_WITH_MOUNT=1)
endif()

target_include_directories(sfs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
sfs_set_warnings(sfs)
sfs_apply_sanitizers(sfs)

# `sfs --version` prints this. A version without a commit is not useful when
# someone reports a bug from a demo machine.
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE SFS_GIT_SHA
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT SFS_GIT_SHA)
    set(SFS_GIT_SHA "unknown")
endif()
target_compile_definitions(sfs PRIVATE
    SFS_VERSION="${PROJECT_VERSION}"
    SFS_GIT_SHA="${SFS_GIT_SHA}")

install(TARGETS sfs RUNTIME DESTINATION bin)
