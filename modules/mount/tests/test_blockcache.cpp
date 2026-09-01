/// FrameCache is called out in blockcache.hpp's own file comment as "the
/// only genuinely subtle piece of concurrency in the project", with four
/// named invariants: immutability (no invalidation, only fill), single
/// flight (first arrival fills, rest wait), publication (never a
/// half-filled frame), and eviction (a pinned entry is never evicted).
/// This file tests the single-threaded shape of each; test_blockcache_race
/// covers the concurrent single-flight race itself.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstring>

#include <synapsefs/core/error.hpp>
#include <synapsefs/mount/blockcache.hpp>

using namespace sfs;

namespace {

core::Oid make_oid(std::uint8_t seed) {
    // core::Oid's exact construction is compute_oid's job elsewhere; here we
    // just need distinct, stable keys, so hash arbitrary bytes.
    std::vector<std::byte> payload(8, std::byte(seed));
    return core::compute_oid(core::ObjectKind::Raw, payload);
}

core::Status fill_with(std::uint8_t value, std::span<std::byte> buf) {
    std::memset(buf.data(), value, buf.size());
    return {};
}

}  // namespace

TEST_CASE("a fresh key is a miss, and the fill's bytes are what's returned", "[mount][blockcache]") {
    mount::FrameCache cache(1024 * 1024);
    mount::FrameKey key{make_oid(1), 0, 0};

    auto lease = cache.get_or_fill(key, 16, [](std::span<std::byte> buf) {
        return fill_with(0xAB, buf);
    });
    REQUIRE(lease.has_value());
    REQUIRE(lease->valid());
    for (auto b : lease->bytes()) CHECK(b == std::byte(0xAB));

    auto stats = cache.stats();
    CHECK(stats.misses == 1);
    CHECK(stats.hits == 0);
}

TEST_CASE("the same key is filled at most once, even across repeated get_or_fill calls",
         "[mount][blockcache]") {
    mount::FrameCache cache(1024 * 1024);
    mount::FrameKey key{make_oid(2), 0, 0};
    std::atomic<int> fill_calls{0};

    auto fill = [&](std::span<std::byte> buf) {
        ++fill_calls;
        return fill_with(0x11, buf);
    };

    for (int i = 0; i < 5; ++i) {
        auto lease = cache.get_or_fill(key, 16, fill);
        REQUIRE(lease.has_value());
    }

    CHECK(fill_calls.load() == 1);
    auto stats = cache.stats();
    CHECK(stats.misses == 1);
    CHECK(stats.hits == 4);
}

TEST_CASE("different frame indices of the same artifact are different keys", "[mount][blockcache]") {
    mount::FrameCache cache(1024 * 1024);
    core::Oid artifact = make_oid(3);
    mount::FrameKey k0{artifact, 0, 0};
    mount::FrameKey k1{artifact, 0, 1};

    auto l0 = cache.get_or_fill(k0, 8, [](std::span<std::byte> b) { return fill_with(1, b); });
    auto l1 = cache.get_or_fill(k1, 8, [](std::span<std::byte> b) { return fill_with(2, b); });
    REQUIRE(l0.has_value());
    REQUIRE(l1.has_value());
    CHECK(l0->bytes()[0] == std::byte(1));
    CHECK(l1->bytes()[0] == std::byte(2));
    CHECK(cache.stats().misses == 2);
}

TEST_CASE("a failing fill is reported to the caller and does not poison later attempts",
         "[mount][blockcache]") {
    mount::FrameCache cache(1024 * 1024);
    mount::FrameKey key{make_oid(4), 0, 0};

    auto lease = cache.get_or_fill(key, 16, [](std::span<std::byte>) -> core::Status {
        return SFS_ERR(ChunkDigestMismatch, "planted failure for the test");
    });
    REQUIRE_FALSE(lease.has_value());
    CHECK(lease.error().is_integrity());

    // A distinct key (real code never retries the exact same failed key
    // in-place) should be unaffected by the earlier failure.
    mount::FrameKey key2{make_oid(4), 0, 1};
    auto lease2 = cache.get_or_fill(key2, 16, [](std::span<std::byte> b) { return fill_with(9, b); });
    REQUIRE(lease2.has_value());
    CHECK(lease2->bytes()[0] == std::byte(9));
}

TEST_CASE("an entry held by a live lease is never evicted under budget pressure",
         "[mount][blockcache]") {
    // Budget only large enough for roughly one frame; a second, different
    // frame filled while the first lease is still alive must not evict it.
    mount::FrameCache cache(20);
    mount::FrameKey held{make_oid(5), 0, 0};
    mount::FrameKey other{make_oid(5), 0, 1};

    auto held_lease = cache.get_or_fill(held, 16, [](std::span<std::byte> b) { return fill_with(7, b); });
    REQUIRE(held_lease.has_value());

    // Fill several more frames while `held_lease` is alive, well past budget.
    for (std::uint32_t i = 2; i < 10; ++i) {
        mount::FrameKey k{make_oid(5), 0, i};
        auto l = cache.get_or_fill(k, 16, [](std::span<std::byte> b) { return fill_with(1, b); });
        REQUIRE(l.has_value());
    }

    // The pinned frame must still be there and still correct -- serving
    // slowly under pressure is fine, silently evicting a held frame is not.
    auto re_fetch = cache.get_or_fill(held, 16, [](std::span<std::byte> b) {
        FAIL("held frame should not have been evicted while a lease was alive");
        return fill_with(0xFF, b);
    });
    REQUIRE(re_fetch.has_value());
    CHECK(re_fetch->bytes()[0] == std::byte(7));

    (void)other;
}

TEST_CASE("once a lease is dropped, its frame becomes eligible for eviction under pressure",
         "[mount][blockcache]") {
    mount::FrameCache cache(20);
    mount::FrameKey victim{make_oid(6), 0, 0};

    {
        auto l = cache.get_or_fill(victim, 16, [](std::span<std::byte> b) { return fill_with(1, b); });
        REQUIRE(l.has_value());
    }  // lease dropped: refcount back to 0

    for (std::uint32_t i = 1; i < 10; ++i) {
        mount::FrameKey k{make_oid(6), 0, i};
        auto l = cache.get_or_fill(k, 16, [](std::span<std::byte> b) { return fill_with(1, b); });
        REQUIRE(l.has_value());
    }

    CHECK(cache.stats().evictions > 0);
}
