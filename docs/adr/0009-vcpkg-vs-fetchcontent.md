# ADR 0009 — vcpkg manifest mode

- **Status:** Accepted
- **Date:** 2026-08-29

## Context

The deliverable includes "clean-environment build instructions" and the
evaluator will build from nothing. The prototype's equivalent failure was
instructive: `requirements.txt` disagreed with `pyproject.toml`, `typer` was
missing, and the documented setup path produced a CLI that failed on import.
The dependency story is graded through the README (5%) and through whether
anything runs at all.

Dependencies: zstd, nlohmann_json, CLI11, Catch2, plus system libfuse3 and
vendored BLAKE3.

## Options

1. **System packages only** (`apt install libzstd-dev nlohmann-json3-dev …`).
   Zero build infrastructure, but versions vary by distro and Catch2 v3 is not
   everywhere.
2. **CMake `FetchContent`.** No external tool. Downloads and builds sources at
   configure time — slow on every fresh configure, no binary caching, and it
   makes the build depend on GitHub being reachable mid-configure.
3. **vcpkg manifest mode.** `vcpkg.json` pins versions with a baseline;
   binaries cache in `~/.cache/vcpkg/archives`; CMake integration is one
   toolchain file.

## Decision

vcpkg manifest mode, with a **pinned `builtin-baseline`**, plus two exceptions:

- **libfuse3 comes from the system.** It is tied to the kernel's FUSE ABI and
  the running `fusermount3` binary; a vcpkg-built copy is a way to get a
  mismatch that only appears at mount time. `cmake/FindFUSE3.cmake` locates it
  through pkg-config.
- **BLAKE3 is vendored** (ADR 0002), so its per-file ISA dispatch survives and
  the exact code producing our addresses is pinned.

The root `CMakeLists.txt` also accepts a plain `find_package` fallback for zstd
via pkg-config, so someone with system packages can build without vcpkg.

## Consequences

- `VCPKG_ROOT` must be set, or the presets' toolchain path fails. `docs/build.md`
  and the `Dockerfile` both do this, and the Dockerfile is the authoritative
  clean-environment answer.
- CI caches `~/.cache/vcpkg/archives` keyed on `vcpkg.json`, so a normal push
  does not rebuild dependencies.
- `builtin-baseline` must be set to a real vcpkg commit before this is merged —
  it is a placeholder in the scaffold and CI will fail loudly until it is
  filled in. That is intended: a silently floating dependency set is exactly
  the failure this ADR exists to prevent.
- The submission ZIP does not contain vcpkg. The Dockerfile bootstraps it.
