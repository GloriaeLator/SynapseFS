# Building

Linux only. If you want to skip all of this, the `Containerfile` build is mentioned below :

```bash
podman build -t sfs .
podman run --rm --cap-add SYS_ADMIN --device /dev/fuse -v $(pwd):/workspace --device "nvidia.com/gpu=all" sfs <subcommand> # CMD is sfs
podman run -it --rm --cap-add SYS_ADMIN --device /dev/fuse -v $(pwd):/workspace --device "nvidia.com/gpu=all" sfs /bin/bash # interactive mode inside the container
```

`--cap-add SYS_ADMIN --device /dev/fuse` is not optional if you intend to
run `sfs mount` inside the container: without it, `fuse_session_mount()`
fails outright (deterministically -- confirmed by testing both ways
side by side, not a timing thing), and a script polling for the mounted
file to appear just times out looking like a hang. Anything that doesn't
touch `mount` (`init`/`commit`/`checkout`/`verify`/`push`/`pull`/`serve`)
works fine without these flags.

Recommended to use `distrobox`

```bash
#after building the container image
distrobox create --name sfs_cont --nvidia -a " --device /dev/fuse  --cap-add SYS_ADMIN " --image sfs  #for most distros
distrobox create --name sfs_cont --image sfs -a " --device "nvidia.com/gpu=all" --device /dev/fuse  --cap-add SYS_ADMIN " #for nix-os 
distrobox enter sfs -- bash 
```

---

## Prerequisites

```bash
sudo apt update
sudo apt install -y \
  g++-14 gcc-14 cmake ninja-build pkg-config git \
  libzstd-dev nlohmann-json3-dev libcli11-dev \
  libssl-dev libjson-c-dev \
  libfuse3-dev fuse3 \
  catch2 \
  python3 python3-venv
```

If `g++-14` isn't in your Ubuntu release's default repos yet, add the toolchain
PPA first:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install -y g++-14 gcc-14
```

If `catch2` doesn't resolve (older distro), that's worth checking manually:
`apt search catch2` — you want a package providing Catch2 v3.

## Configure and build

```bash
git clone --recurse-submodules https://github.com/GloriaeLator/SynapseFS synapsefs && cd synapsefs
```

BLAKE3 comes in as a submodule — a plain zip download of the repo won't have
it, so `--recurse-submodules` isn't optional.

### PyTorch with CUDA

Make sure your Torch install actually has CUDA support *and* ships its CMake
files — apt's `python3-torch` is CPU-only and won't have either. Install it in
a venv:

```bash
python3 -m venv ~/torch-venv
source ~/torch-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy
```

Confirm the CMake export actually exists before moving on:

```bash
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
```

If it's empty, or your install lacks CUDA, get a build from the PyTorch
repository that has both instead.


```
bash
# NVIDIA's own repo, not Ubuntu's 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-2
```
Then make sure nvcc and the toolkit paths are visible to CMake:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
```

### Environment variables

The presets read these three:

```bash
export SFS_CC=gcc-14
export SFS_CXX=g++-14
export SFS_TORCH_CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

Put these in `~/.bashrc` (after the venv activation line, or re-export them
each session) so you're not retyping them every time.

### Configure, build, test

```bash
cmake --preset dev
cmake --build --preset dev -j"$(nproc)"
ctest --preset dev
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

**Missing dev headers (`CLI11`, `OpenSSL`, `json-c`, ...)** — these come from
`apt`, not vcpkg. Re-run the package install block above; the usual culprit is
`libcli11-dev`, `libssl-dev`, or `libjson-c-dev` not being installed.

**`mmap` returns zeros through the mount** — `FOPEN_DIRECT_IO` is set. It must
not be. See [SPEC 16 §3.3](spec/16-consistency.md).

**Compiler too old** — `g++ --version` must be 14+. `g++-13` does not have
`std::expected` in a form we rely on.

**Torch CMake config missing / no CUDA** — `python3 -c "import torch; print(torch.utils.cmake_prefix_path)"`
returns nothing useful, or `SFS_TORCH_CMAKE_PREFIX_PATH` points at a CPU-only
build. Reinstall Torch from the `cu128` index in a clean venv (see above), or
grab a CUDA build with CMake files from the PyTorch repository directly.