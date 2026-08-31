/// write_permutation/read_permutation round-trip, the identity zero-byte
/// encoding, the u16/u32 width boundary at n = 65536, and — the one place a
/// corrupt object could become a memory-safety problem rather than a
/// rejected read (spec 12 §3) — that a malformed permutation is REJECTED
/// before anything could index with it.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>

#include <synapsefs/codec/diff_encoder.hpp>
#include <synapsefs/format/residual_hdr.hpp>

using namespace sfs;

namespace {
std::vector<std::uint32_t> random_permutation(std::uint32_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}
}  // namespace

TEST_CASE("identity permutation writes zero payload bytes", "[codec][permutation]") {
    std::vector<std::uint32_t> identity(50);
    std::iota(identity.begin(), identity.end(), 0);

    std::vector<std::byte> payload;
    auto ref = codec::write_permutation(payload, identity);

    REQUIRE(ref.kind == format::PermKind::Identity);
    REQUIRE(payload.empty());

    // spec 12 §4.2: identity's dependency set is the identical range —
    // read_permutation returns empty rather than materialising [0, n).
    auto read_back = format::read_permutation(ref, payload);
    REQUIRE(read_back.has_value());
    REQUIRE(read_back->empty());
}

TEST_CASE("a permutation that happens to be non-identity round-trips at u16 width",
         "[codec][permutation]") {
    auto perm = random_permutation(1000, 1);
    // Guard against an astronomically unlikely random identity, which would
    // make this test accidentally exercise the identity path instead.
    REQUIRE_FALSE(std::is_sorted(perm.begin(), perm.end()));

    std::vector<std::byte> payload;
    auto ref = codec::write_permutation(payload, perm);

    REQUIRE(ref.kind == format::PermKind::Explicit);
    REQUIRE(ref.width == 2);  // n <= 65536
    REQUIRE(ref.len == perm.size() * 2);
    REQUIRE(payload.size() == ref.len);

    auto read_back = format::read_permutation(ref, payload);
    REQUIRE(read_back.has_value());
    REQUIRE(*read_back == perm);
}

TEST_CASE("a permutation larger than 65536 round-trips at u32 width", "[codec][permutation]") {
    auto perm = random_permutation(70000, 2);

    std::vector<std::byte> payload;
    auto ref = codec::write_permutation(payload, perm);

    REQUIRE(ref.kind == format::PermKind::Explicit);
    REQUIRE(ref.width == 4);
    REQUIRE(ref.len == perm.size() * 4);

    auto read_back = format::read_permutation(ref, payload);
    REQUIRE(read_back.has_value());
    REQUIRE(*read_back == perm);
}

TEST_CASE("read_permutation rejects a duplicate value", "[codec][permutation]") {
    // A permutation of [0, 4) that has 0 twice and never names 3 — not a
    // bijection. Hand-encoded, since write_permutation would never produce
    // this; the point is that read_permutation must catch it anyway, since
    // the artifact bytes are untrusted file content.
    format::PermutationRef ref;
    ref.kind = format::PermKind::Explicit;
    ref.n = 4;
    ref.width = 2;
    ref.off = 0;
    ref.len = 8;

    std::vector<std::byte> payload(8);
    const std::uint16_t values[4] = {0, 1, 0, 2};  // 0 repeated, 3 missing
    std::memcpy(payload.data(), values, sizeof(values));

    auto result = format::read_permutation(ref, payload);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().kind == core::ErrKind::InvalidPermutation);
}

TEST_CASE("read_permutation rejects an out-of-range value", "[codec][permutation]") {
    format::PermutationRef ref;
    ref.kind = format::PermKind::Explicit;
    ref.n = 4;
    ref.width = 2;
    ref.off = 0;
    ref.len = 8;

    std::vector<std::byte> payload(8);
    const std::uint16_t values[4] = {0, 1, 2, 99};  // 99 is out of [0, 4)
    std::memcpy(payload.data(), values, sizeof(values));

    auto result = format::read_permutation(ref, payload);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().kind == core::ErrKind::InvalidPermutation);
}

TEST_CASE("read_permutation rejects a range extending past the payload", "[codec][permutation]") {
    format::PermutationRef ref;
    ref.kind = format::PermKind::Explicit;
    ref.n = 100;
    ref.width = 2;
    ref.off = 0;
    ref.len = 200;  // 100 * 2, but the payload below is shorter

    std::vector<std::byte> payload(50);  // too small
    auto result = format::read_permutation(ref, payload);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().kind == core::ErrKind::MalformedObject);
}
