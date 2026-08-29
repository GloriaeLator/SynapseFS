# Sanitizer wiring driven by the SFS_SANITIZER cache variable.
#
# Note for the mount module: TSan and FUSE coexist badly under some kernels
# because the daemon forks. Always run the tsan preset with `--foreground`.

function(sfs_apply_sanitizers target)
    if(SFS_SANITIZER STREQUAL "none")
        return()
    endif()

    set(flags "")
    if(SFS_SANITIZER STREQUAL "address")
        set(flags -fsanitize=address)
    elseif(SFS_SANITIZER STREQUAL "undefined")
        set(flags -fsanitize=undefined -fno-sanitize-recover=undefined)
    elseif(SFS_SANITIZER STREQUAL "address+undefined")
        set(flags -fsanitize=address,undefined -fno-sanitize-recover=undefined)
    elseif(SFS_SANITIZER STREQUAL "thread")
        set(flags -fsanitize=thread)
    else()
        message(FATAL_ERROR "Unknown SFS_SANITIZER='${SFS_SANITIZER}'")
    endif()

    target_compile_options(${target} PRIVATE ${flags} -fno-omit-frame-pointer -g)
    target_link_options(${target} PRIVATE ${flags})
endfunction()
