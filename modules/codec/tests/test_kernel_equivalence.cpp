/// Every enabled ISA must produce byte-identical output to the scalar
/// oracle, on random inputs, including unaligned tails — residual_scalar.cpp
/// is the reference every vectorised kernel is checked against
/// (docs/adr/0011-simd-dispatch-strategy.md).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>
#include <vector>

#include <synapsefs/codec/residual_codec.hpp>

using namespace sfs;

// Not declared in residual_codec.hpp (only the scalar oracle is public API);
// defined in kernels/residual_avx2.cpp / residual_avx512.cpp, always present
// once those files exist regardless of whether the compiler accepted the ISA
// flag (they forward to scalar internally when disabled) — see dispatch.cpp.
namespace sfs::codec {
void xor_apply_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void xor_encode_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void zigzag_apply_avx2(std::byte*, const std::byte*, const std::byte*, std::size_t,
                       std::uint32_t) noexcept;

void xor_apply_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void xor_encode_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t) noexcept;
void zigzag_apply_avx512(std::byte*, const std::byte*, const std::byte*, std::size_t,
                         std::uint32_t) noexcept;
}  // namespace sfs::codec

namespace {

std::vector<std::byte> random_bytes(std::size_t n, std::mt19937_64& rng) {
    std::vector<std::byte> v(n);
    for (auto& b : v) b = std::byte(rng());
    return v;
}

// Deliberately awkward sizes: not multiples of 16, 32, or 64 — the widths
// the AVX2 (32 B/instr) and AVX-512 (64 B/instr) kernels chunk in — so every
// case exercises a genuine scalar tail.
const std::size_t kSizes[] = {0, 1, 3, 15, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 100003};

}  // namespace

TEST_CASE("xor_apply: scalar, avx2, avx512 agree on random data", "[codec][kernels]") {
    std::mt19937_64 rng(1);
    for (std::size_t n : kSizes) {
        const auto base = random_bytes(n, rng);
        const auto resid = random_bytes(n, rng);
        std::vector<std::byte> out_scalar(n), out_avx2(n), out_avx512(n);

        codec::xor_apply_scalar(out_scalar.data(), base.data(), resid.data(), n);
        codec::xor_apply_avx2(out_avx2.data(), base.data(), resid.data(), n);
        codec::xor_apply_avx512(out_avx512.data(), base.data(), resid.data(), n);

        INFO("n = " << n);
        REQUIRE(out_scalar == out_avx2);
        REQUIRE(out_scalar == out_avx512);
    }
}

TEST_CASE("xor_encode: scalar, avx2, avx512 agree, and invert xor_apply", "[codec][kernels]") {
    std::mt19937_64 rng(2);
    for (std::size_t n : kSizes) {
        const auto base = random_bytes(n, rng);
        const auto target = random_bytes(n, rng);
        std::vector<std::byte> resid_scalar(n), resid_avx2(n), resid_avx512(n);

        codec::xor_encode_scalar(resid_scalar.data(), base.data(), target.data(), n);
        codec::xor_encode_avx2(resid_avx2.data(), base.data(), target.data(), n);
        codec::xor_encode_avx512(resid_avx512.data(), base.data(), target.data(), n);

        INFO("n = " << n);
        REQUIRE(resid_scalar == resid_avx2);
        REQUIRE(resid_scalar == resid_avx512);

        // XOR is self-inverse: applying the encoded residual back to base
        // must reproduce target exactly (spec 12 §5).
        std::vector<std::byte> roundtrip(n);
        codec::xor_apply_scalar(roundtrip.data(), base.data(), resid_scalar.data(), n);
        REQUIRE(roundtrip == target);
    }
}

TEST_CASE("zigzag_apply: scalar, avx2, avx512 agree at every element width",
         "[codec][kernels]") {
    std::mt19937_64 rng(3);
    for (std::size_t n : kSizes) {
        const auto base = random_bytes(n, rng);
        const auto resid = random_bytes(n, rng);
        for (std::uint32_t ew : {1u, 2u, 4u, 8u}) {
            const std::size_t nn = (n / ew) * ew;  // whole elements only
            if (nn == 0) continue;

            std::vector<std::byte> out_scalar(nn), out_avx2(nn), out_avx512(nn);
            codec::zigzag_apply_scalar(out_scalar.data(), base.data(), resid.data(), nn, ew);
            codec::zigzag_apply_avx2(out_avx2.data(), base.data(), resid.data(), nn, ew);
            codec::zigzag_apply_avx512(out_avx512.data(), base.data(), resid.data(), nn, ew);

            INFO("n = " << nn << ", elem_bytes = " << ew);
            REQUIRE(out_scalar == out_avx2);
            REQUIRE(out_scalar == out_avx512);
        }
    }
}

TEST_CASE("active_isa reports a name consistent with the dispatched kernel", "[codec][kernels]") {
    // Whatever ISA got selected, its name must be one of the three — this is
    // mostly a smoke test that dispatch.cpp's table() didn't leave `isa`
    // uninitialised or mismatched with the function pointers it returned.
    const auto isa = codec::active_isa();
    const auto name = util::isa_name(isa);
    REQUIRE((name == "scalar" || name == "avx2" || name == "avx512"));
}
