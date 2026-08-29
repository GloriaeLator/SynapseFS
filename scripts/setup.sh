#!/usr/bin/env bash
# One-shot developer setup on Ubuntu 24.04. Prints what it would do and asks
# before installing anything system-wide.
set -euo pipefail

PKGS="g++-14 cmake ninja-build pkg-config libfuse3-dev fuse3 python3 python3-venv git curl zip unzip tar"

echo "SynapseFS setup"
echo
echo "System packages needed:"
echo "  $PKGS"
echo
read -rp "Install them with apt? [y/N] " ans
if [[ "${ans,,}" == "y" ]]; then
    sudo apt-get update
    sudo apt-get install -y $PKGS
fi

: "${VCPKG_ROOT:=$HOME/vcpkg}"
if [[ ! -d "$VCPKG_ROOT" ]]; then
    echo "Cloning vcpkg into $VCPKG_ROOT"
    git clone --depth 1 https://github.com/microsoft/vcpkg "$VCPKG_ROOT"
    "$VCPKG_ROOT"/bootstrap-vcpkg.sh -disableMetrics
fi
export VCPKG_ROOT
echo
echo "Add this to your shell rc if it is not there already:"
echo "  export VCPKG_ROOT=$VCPKG_ROOT"
echo
echo "Then:  cmake --preset dev && cmake --build --preset dev -j\$(nproc)"
