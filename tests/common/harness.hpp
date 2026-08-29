#pragma once
/// \file harness.hpp
/// Shared helpers: synthetic checkpoints, planted permutations, byte comparison.
///
/// Tests build their own tiny checkpoints in memory rather than depending on
/// fixtures/ where possible, so that `ctest --preset unit` runs in a clean
/// checkout with no Python and no downloads.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <synapsefs/core/topology.hpp>

namespace sfs::test {

struct SyntheticSpec {
    std::vector<std::uint32_t> layer_widths{8, 16, 10};
    bool     with_batchnorm = false;
    bool     with_conv = false;
    std::uint64_t seed = 42;
    /// Write a __metadata__ block and unusual key order, so that a test which
    /// passes is actually exercising the verbatim-header path rather than a
    /// tidy file we generated ourselves.
    bool     awkward_header = true;
};

/// Build a .safetensors file at `dest`. Returns its SHA-256.
[[nodiscard]] std::string write_synthetic_checkpoint(const std::filesystem::path& dest,
                                                     const SyntheticSpec&);

/// Copy `src` applying `perm` to `group`'s units, writing `dest`. The result is
/// a function-identical, byte-different checkpoint — the exact thing alignment
/// is supposed to collapse to nothing.
[[nodiscard]] std::vector<std::uint32_t> plant_permutation(
    const std::filesystem::path& src, const std::filesystem::path& dest,
    const core::Topology&, std::string_view group, std::uint64_t seed);

/// Perturb every weight slightly, as a fine-tune step would.
void perturb(const std::filesystem::path& src, const std::filesystem::path& dest,
             float scale, std::uint64_t seed);

/// Byte-compare, reporting the FIRST differing offset — "files differ" is not
/// a useful failure message when the answer is four bytes of padding.
struct ByteDiff {
    bool          identical = true;
    std::uint64_t first_difference = 0;
    std::uint64_t size_a = 0;
    std::uint64_t size_b = 0;
};
[[nodiscard]] ByteDiff compare_files(const std::filesystem::path& a,
                                     const std::filesystem::path& b);

[[nodiscard]] std::string sha256_file(const std::filesystem::path&);

}  // namespace sfs::test
