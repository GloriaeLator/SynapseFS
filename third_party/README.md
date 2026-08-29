# third_party

Vendored dependencies that vcpkg does not carry in a form we want.

## blake3

Upstream: https://github.com/BLAKE3-team/BLAKE3 (CC0-1.0 / Apache-2.0 dual licensed)

We vendor the reference C implementation (`c/` subtree) rather than taking it
from vcpkg, for two reasons:

1. Upstream's own CMake already does per-file ISA dispatch (SSE2/SSE4.1/AVX2/
   AVX-512 with runtime CPUID selection). That is exactly the property we need
   and exactly the thing that gets lost when a package manager builds it with a
   single baseline ISA.
2. The digest is the address of every object in the store. Pinning the exact
   commit that produced our on-disk hashes is a correctness requirement, not a
   convenience.

Add it as a submodule, pinned:

```bash
git submodule add https://github.com/BLAKE3-team/BLAKE3.git third_party/blake3-upstream
git -C third_party/blake3-upstream checkout <PINNED_TAG>
```

then point `third_party/blake3/CMakeLists.txt` at `third_party/blake3-upstream/c`.

Record the pinned tag in `docs/adr/0002-blake3-over-sha256.md` when you set it.

**Do not** upgrade this after the store format is frozen without re-reading
that ADR: a different digest is a different address space, and every existing
repository becomes unreadable.
