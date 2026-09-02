#   docker build -f Dockerfile.pytorch -t synapsefs .
#   docker run --rm -it --gpus all --cap-add SYS_ADMIN --device /dev/fuse synapsefs

ARG TORCH_IMAGE=docker.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel
ARG TORCH_RUNTIME_IMAGE=docker.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

# ---------------------------------------------------------------- build stage
FROM ${TORCH_IMAGE} AS build

ARG BLAKE3_TAG=1.8.7
ARG ENABLE_LTO=OFF
ARG BUILD_MOUNT=ON
ARG BUILD_TESTS=OFF
# Default to $(nproc) via empty string below, but override on a
# memory-constrained host: each translation unit including <torch/torch.h>
# can use 1-2+ GB by itself, and launching one per core can overcommit a
# WSL2/Docker Desktop VM's RAM allocation badly enough to thrash (swap)
# rather than actually compile faster -- a real build that "hangs" for
# hours, not minutes, almost always means this, not a genuinely slow build.
ARG BUILD_JOBS=

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config \
      git ca-certificates \
      libzstd-dev libssl-dev libjson-c-dev libfuse3-dev \
      nlohmann-json3-dev libcli11-dev catch2 \
    && rm -rf /var/lib/apt/lists/*

RUN cmake --version \
    && printf 'cmake_minimum_required(VERSION 3.25)\n' > /tmp/vcheck.cmake \
    && cmake -P /tmp/vcheck.cmake \
    && rm /tmp/vcheck.cmake

RUN git clone --depth 1 --branch "${BLAKE3_TAG}" \
      https://github.com/BLAKE3-team/BLAKE3.git /opt/blake3-upstream

WORKDIR /src
COPY . .

RUN rm -rf third_party/blake3-upstream \
    && cp -a /opt/blake3-upstream third_party/blake3-upstream

RUN PY="$(command -v python || command -v python3)" \
    && TORCH_CMAKE="$("$PY" -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
    && TORCH_PREFIX="$(dirname "$(dirname "${TORCH_CMAKE}")")" \
    && test -f "${TORCH_CMAKE}/Torch/TorchConfig.cmake" \
    && ln -sfn "${TORCH_PREFIX}" third_party/libtorch \
    && printf '%s\n' "${TORCH_PREFIX}" > /opt/torch_prefix

ENV LIBTORCH_ROOT=/src/third_party/libtorch
ENV Torch_DIR=/src/third_party/libtorch/share/cmake/Torch
ENV CMAKE_PREFIX_PATH=/src/third_party/libtorch
ENV LD_LIBRARY_PATH=/src/third_party/libtorch/lib:${LD_LIBRARY_PATH}

RUN printf 'find_package(Torch REQUIRED)\nmessage(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}")\n' \
      > /opt/sfs_find_torch.cmake

RUN cmake -S . -B build/release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="${Torch_DIR}" \
      -DCMAKE_PREFIX_PATH="${LIBTORCH_ROOT}" \
      -DCMAKE_INSTALL_RPATH="$(cat /opt/torch_prefix)/lib" \
      -DCMAKE_PROJECT_synapsefs_INCLUDE=/opt/sfs_find_torch.cmake \
      -DCMAKE_DISABLE_FIND_PACKAGE_zstd=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DSFS_BUILD_TESTS="${BUILD_TESTS}" \
      -DSFS_BUILD_BENCH=OFF \
      -DSFS_BUILD_MOUNT="${BUILD_MOUNT}" \
      -DSFS_ENABLE_SIMD=ON \
      -DSFS_ENABLE_LTO="${ENABLE_LTO}" \
    && cmake --build build/release -j "${BUILD_JOBS:-$(nproc)}" \
    && cmake --install build/release --prefix /out

# -------------------------------------------------------------- runtime stage

FROM ${TORCH_RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      libzstd1 libjson-c5 fuse3 \
    && rm -rf /var/lib/apt/lists/*

RUN PY="$(command -v python || command -v python3)" \
      && "$PY" -c 'import torch.utils, os; print(os.path.join(os.path.dirname(os.path.dirname(torch.utils.cmake_prefix_path)), "lib"))' \
      > /etc/ld.so.conf.d/libtorch.conf \
      && test -s /etc/ld.so.conf.d/libtorch.conf

COPY --from=build /out /usr/local
RUN ldconfig

RUN sfs --help > /dev/null

ENTRYPOINT ["sfs"]
CMD ["--help"]
