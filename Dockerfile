# Clean-environment build. This file IS the answer to "does it build from
# nothing" - CI builds it on every push, and the submission checklist requires
# a fresh clone to build here before we ship.
#
#   docker build -t synapsefs .
#   docker run --rm -it --cap-add SYS_ADMIN --device /dev/fuse synapsefs
#
# SYS_ADMIN + /dev/fuse are needed only for `sfs mount`; everything else
# (commit, checkout, verify, push, pull) runs in a plain container.

FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++-14 cmake ninja-build git curl zip unzip tar pkg-config \
      libfuse3-dev fuse3 python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14

# vcpkg in manifest mode; the manifest pins the versions.
ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone --depth 1 https://github.com/microsoft/vcpkg "$VCPKG_ROOT" \
    && "$VCPKG_ROOT"/bootstrap-vcpkg.sh -disableMetrics
ENV PATH="$VCPKG_ROOT:$PATH"

WORKDIR /src
COPY . .
RUN cmake --preset release && cmake --build --preset release -j "$(nproc)"
RUN cmake --install build/release --prefix /out

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libstdc++6 fuse3 python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out /usr/local
RUN sfs --version
ENTRYPOINT ["sfs"]
CMD ["--help"]
