# Warning set applied to every first-party target via sfs_set_warnings().
# Deliberately not applied to third_party/ or to vcpkg-installed headers.

function(sfs_set_warnings target)
    set(gcc_clang
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        # We serialize structs to disk by hand; padding bugs are format bugs.
        -Wno-padded
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND gcc_clang
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast
            -Wsuggest-override)
    endif()

    if(SFS_WARNINGS_AS_ERRORS)
        list(APPEND gcc_clang -Werror
            # -Wpadded is advisory: it fires on layouts we chose deliberately.
            -Wno-error=padded)
    endif()

    target_compile_options(${target} PRIVATE ${gcc_clang})
endfunction()
