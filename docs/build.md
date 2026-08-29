# Building

Linux only. If you want to skip all of this, the `Dockerfile` builds from
nothing and CI proves it on every push:

```bash
docker build -t synapsefs . && docker run --rm synapsefs --version
```

---

## Prerequisites

| Need | Minimum | Ubuntu 24.04 |
|---|---|---|
| C++23 compiler | GCC 14 or Clang 18 | `sudo apt install g++-14` |
| CMake | 3.25 | `sudo apt install cmake` |
| Ninja | any | `sudo apt install ninja-build` |
| pkg-config | any | `sudo apt install pkg-config` |
| libfuse3 | 3.10 | `sudo apt install libfuse3-dev fuse3` |
| Python | 3.11 | `sudo apt install python3 python3-venv` |
| vcpkg | any recent | see below |

`libfuse3` comes from the system deliberately — it is tied to the kernel's FUSE
ABI and to the installed `fusermount3`. See
[ADR 0009](adr/0009-vcpkg-vs-fetchcontent.md).

Everything else (zstd, nlohmann_json, CLI11, Catch2) comes from vcpkg, pinned by
`vcpkg.json`. BLAKE3 is vendored — see `third_party/README.md`.

## vcpkg

```bash
git clone --depth 1 https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg          # put this in your shell rc
```

The presets read `$VCPKG_ROOT`. Without it, configure fails with a missing
toolchain file.

## Configure and build

```bash
git clone --recurse-submodules <repo-url> synapsefs && cd synapsefs
cmake --preset dev
cmake --build --preset dev -j"$(nproc)"
./build/dev/apps/sfs/sfs --version
```

Or just `make`, which does the same thing.

### Presets

| Preset | For |
|---|---|
| `dev` | Day-to-day. RelWithDebInfo, warnings as errors, tests and benchmarks. |
| `debug` | `-O0 -g`, assertions. |
| `release` | **The only configuration published benchmark numbers may come from.** LTO on. |
| `asan` | Address + UB sanitizers. |
| `tsan` | Thread sanitizer — run the mount with `--foreground`. |
| `no-simd` | Scalar residual kernels only. The correctness oracle for `test_kernel_equivalence`. |

```bash
cmake --preset release && cmake --build --preset release -j"$(nproc)"
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `SFS_BUILD_TESTS` | ON | |
| `SFS_BUILD_BENCH` | ON | |
| `SFS_BUILD_MOUNT` | ON | Turn off to build without libfuse3. |
| `SFS_ENABLE_SIMD` | ON | AVX2/AVX-512 kernels; runtime dispatch either way. |
| `SFS_ENABLE_LTO` | OFF | Release only. |
| `SFS_WARNINGS_AS_ERRORS` | OFF | ON in the `dev` preset and in CI. |
| `SFS_SANITIZER` | `none` | `address` / `thread` / `undefined` / `address+undefined` |

## Fixtures

Checkpoints are **generated, never committed** — a listed deliverable, and CI
fails if a `.safetensors` appears in git.

```bash
make fixtures-small     # MLP + small CNN, a few hundred MB
make fixtures           # adds the 7B pair. Large and slow.
```

## Running the tests

```bash
make test               # everything
make test-unit          # fast, no fixtures needed
make test-e2e           # needs fixtures
make asan               # sanitizers
ctest --preset dev -R align --output-on-failure    # one module
```

`tests/e2e.py` needs `safetensors` and `torch`; `make fixtures-small` creates
the venv it uses.

## Common failures

**`Could NOT find FUSE3`** — install `libfuse3-dev`, or configure with
`-DSFS_BUILD_MOUNT=OFF` if you are not working on the mount.

**`fusermount3: option allow_other only allowed if 'user_allow_other' is set`** —
either drop `--allow-other` or uncomment that line in `/etc/fuse.conf`.

**`Transport endpoint is not connected`** at a mountpoint — a daemon died.
`fusermount3 -u <mountpoint>` and check the log.

**vcpkg baseline errors** — `builtin-baseline` in `vcpkg.json` must be a real
vcpkg commit sha. It ships as a placeholder in the scaffold and is meant to fail
loudly until someone sets it.

**`mmap` returns zeros through the mount** — `FOPEN_DIRECT_IO` is set. It must
not be. See [SPEC 16 §3.3](spec/16-consistency.md).

**Compiler too old** — `g++ --version` must be 14+. `g++-13` does not have
`std::expected` in a form we rely on.
