# modules/align
#
# Sources are globbed with CONFIGURE_DEPENDS deliberately. Eight people are
# adding files to nine modules over four days; an explicitly listed source set
# means a merge conflict in this file every time someone adds a .cpp, and a
# build that silently omits a file when someone resolves it wrong. Ninja
# re-runs CMake when the glob result changes, so the usual objection does not
# apply here. Revisit if this outlives the project.
file(GLOB SFS_ALIGN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SFS_ALIGN_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

sfs_add_module(align
    SOURCES ${SFS_ALIGN_SOURCES}
    HEADERS ${SFS_ALIGN_HEADERS}
    DEPS    synapsefs::stio;synapsefs::core
    # nlohmann_json and Torch are both private: no public align header
    # exposes a json or torch::Tensor type (docs/adr/0011) -- the sparse
    # large-group path is an internal detail of matcher.cpp/sparse_match.cpp,
    # not part of align's interface.
    PRIVATE_DEPS nlohmann_json::nlohmann_json;${TORCH_LIBRARIES})
target_compile_options(sfs_align PRIVATE ${TORCH_CXX_FLAGS})
# ${TORCH_LIBRARIES} (find_package(Torch)'s legacy variable form, not an
# imported target) carries library paths only -- unlike linking
# Torch::Torch/`torch` directly, it does NOT propagate include directories,
# so torch/torch.h was never reachable from sfs_align's actual compile
# command: "fatal error: torch/torch.h: No such file or directory" even
# with find_package(Torch REQUIRED) succeeding.
target_include_directories(sfs_align PRIVATE ${TORCH_INCLUDE_DIRS})

# tools/ is align's own test-support code, not the module's public library:
# a whole-file-resident ITensorSource over a real .safetensors file, used to
# exercise align end-to-end before stio's real (streaming) reader exists.
# See tools/simple_st_source.hpp.
add_library(sfs_align_tools STATIC tools/simple_st_source.cpp)
target_include_directories(sfs_align_tools PUBLIC tools)
target_link_libraries(sfs_align_tools PUBLIC synapsefs::core PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(sfs_align_tools PUBLIC cxx_std_23)

add_executable(sfs_align_demo tools/align_demo.cpp)
target_link_libraries(sfs_align_demo PRIVATE synapsefs::align sfs_align_tools nlohmann_json::nlohmann_json)

file(GLOB SFS_ALIGN_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(t IN LISTS SFS_ALIGN_TESTS)
    sfs_add_module_test(align ${t} DEPS sfs_align_tools)
endforeach()
