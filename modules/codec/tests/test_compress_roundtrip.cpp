/// zstd wrapper and the two byte-level transforms round-trip exactly,
/// including sizes that don't divide evenly by the element width or by 8
/// elements (bitshuffle's group size) — the "trailing bytes copied verbatim"
/// paths in compress.cpp.

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <synapsefs/codec/compress.hpp>

using namespace sfs;

namespace {
std::vector<std::byte> random_bytes(std::size_t n, std::mt19937_64& rng) {
    std::vector<std::byte> v(n);
    for (auto& b : v) b = std::byte(rng());
    return v;
}
}  // namespace

TEST_CASE("compress_frame/decompress_frame round-trip", "[codec][compress]") {
    std::mt19937_64 rng(10);
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{100}, std::size_t{70000}}) {
        const auto src = random_bytes(n, rng);
        auto comp = codec::compress_frame(src);
        REQUIRE(comp.has_value());

        std::vector<std::byte> out(n);
        auto dn = codec::decompress_frame(*comp, out);
        REQUIRE(dn.has_value());
        REQUIRE(*dn == n);
        REQUIRE(out == src);
    }
}

TEST_CASE("compress_frame is one independently decompressible frame per call", "[codec][compress]") {
    // Two separately compressed frames, concatenated, must each decompress
    // on their own with no shared state — spec 12 §4's "no preceding state"
    // requirement, which is what lets a reader start at an arbitrary offset.
    std::mt19937_64 rng(11);
    const auto a = random_bytes(500, rng);
    const auto b = random_bytes(500, rng);
    auto ca = codec::compress_frame(a);
    auto cb = codec::compress_frame(b);
    REQUIRE(ca.has_value());
    REQUIRE(cb.has_value());

    std::vector<std::byte> out_a(500), out_b(500);
    REQUIRE(codec::decompress_frame(*ca, out_a).value() == 500);
    REQUIRE(codec::decompress_frame(*cb, out_b).value() == 500);
    REQUIRE(out_a == a);
    REQUIRE(out_b == b);
}

TEST_CASE("byteplane_split/join round-trip at every element width", "[codec][compress]") {
    std::mt19937_64 rng(20);
    for (std::uint32_t ew : {1u, 2u, 4u, 8u}) {
        // Sizes deliberately include a partial trailing element (not a
        // multiple of ew) to exercise the verbatim-tail path.
        for (std::size_t n : {std::size_t{0}, std::size_t(ew), std::size_t(ew * 5),
                              std::size_t(ew * 5 + 1), std::size_t(ew * 100 + 3)}) {
            const auto src = random_bytes(n, rng);
            std::vector<std::byte> split(n), joined(n);
            codec::byteplane_split(src, split, ew);
            codec::byteplane_join(split, joined, ew);

            INFO("elem_bytes = " << ew << ", n = " << n);
            REQUIRE(joined == src);
        }
    }
}

TEST_CASE("bitshuffle/bitunshuffle round-trip, same size as input", "[codec][compress]") {
    std::mt19937_64 rng(30);
    for (std::uint32_t ew : {1u, 2u, 4u, 8u}) {
        // Sizes spanning: empty, less than one group of 8 elements, exactly
        // one group, several groups plus a partial group, plus a partial
        // trailing element.
        for (std::size_t n : {std::size_t{0}, std::size_t(ew * 3), std::size_t(ew * 8),
                              std::size_t(ew * 20 + ew / 2 + 1), std::size_t(ew * 100 + 5)}) {
            if (n == 0 && ew != 1) continue;  // avoid duplicate empty cases
            const auto src = random_bytes(n, rng);
            std::vector<std::byte> shuffled(n), restored(n);
            codec::bitshuffle(src, shuffled, ew);
            codec::bitunshuffle(shuffled, restored, ew);

            INFO("elem_bytes = " << ew << ", n = " << n);
            REQUIRE(restored == src);
            REQUIRE(shuffled.size() == src.size());  // same-size, lossless permutation
        }
    }
}
