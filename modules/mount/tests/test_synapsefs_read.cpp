/// SynapseFs is fs.hpp's whole point: "the read path... with no FUSE in it,
/// so that the whole read path can be tested without mounting anything."
/// This file is that test. It follows docs/testing.md's test ladder for the
/// mount as far as a FUSE-free fixture can:
///   1. file_size()/file_name() back getattr/readdir.
///   2. sequential read of the whole file matches a byte-for-byte checkout.
///   3. random offset/size reads match.
///   4. edge cases: header/buffer boundary, group boundary, 1-byte read,
///      read at EOF, read past EOF.
///   7. concurrent readers see identical, uncorrupted bytes.
/// mmap (5) and safetensors.torch.load_file() (6) need a real FUSE mount and
/// belong in tests/e2e.py instead; scale/RSS (8) needs the real fixtures.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include <synapsefs/mount/fs.hpp>

#include "mount_test_common.hpp"

using namespace sfs;

namespace {

/// A small header plus two groups of different sizes, one of which does not
/// end on a "round" byte count -- so a naive frame-aligned copy that assumes
/// tidy sizes would be caught.
std::unique_ptr<mount::SynapseFs> make_fs(test::BuiltFixture& fx, mount::FsOptions opts = {}) {
    codec::ReadCtx ctx;
    ctx.blocks = &fx.store;
    // history/ctx.manifest get wired by SynapseFs::create's Impl; a
    // NullObjectSource would need to outlive the call, but this fixture has
    // no Delta groups, so history is never dereferenced. Left null is fine
    // for Full-only manifests; codec::read_range only consults it on the
    // Delta branch.
    ctx.history = nullptr;

    auto fs = mount::SynapseFs::create(ctx, fx.manifest, opts);
    REQUIRE(fs.has_value());
    return std::move(*fs);
}

test::BuiltFixture standard_fixture() {
    auto header = test::make_bytes(24, 1);
    std::vector<test::GroupSpec> groups = {
        {"0.weight", test::make_bytes(1000, 11)},
        {"0.bias", test::make_bytes(37, 22)},  // deliberately not frame- or word-aligned
    };
    return test::build_fixture(header, groups);
}

}  // namespace

TEST_CASE("file_size and file_name reflect the manifest", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto fs = make_fs(fx);
    CHECK(fs->file_name() == "model.safetensors");
    CHECK(fs->file_size() == 24 + 1000 + 37);
}

TEST_CASE("sequential whole-file read matches header+groups byte for byte", "[mount][synapsefs]") {
    auto header = test::make_bytes(24, 1);
    std::vector<test::GroupSpec> groups = {
        {"0.weight", test::make_bytes(1000, 11)},
        {"0.bias", test::make_bytes(37, 22)},
    };
    auto fx = test::build_fixture(header, groups);
    auto expected = test::flatten(header, groups);
    auto fs = make_fs(fx);

    std::vector<std::byte> actual(fs->file_size());
    std::size_t got = 0;
    while (got < actual.size()) {
        auto n = fs->read(got, std::span(actual).subspan(got));
        REQUIRE(n.has_value());
        REQUIRE(*n > 0);  // no forward progress would hang a real reader
        got += *n;
    }
    CHECK(actual == expected);
}

TEST_CASE("random offset/size reads match the checkout", "[mount][synapsefs]") {
    auto header = test::make_bytes(24, 1);
    std::vector<test::GroupSpec> groups = {
        {"0.weight", test::make_bytes(1000, 11)},
        {"0.bias", test::make_bytes(37, 22)},
    };
    auto fx = test::build_fixture(header, groups);
    auto expected = test::flatten(header, groups);
    auto fs = make_fs(fx);

    std::mt19937_64 rng(7);
    std::uniform_int_distribution<std::uint64_t> off_dist(0, expected.size() - 1);
    std::uniform_int_distribution<std::uint64_t> len_dist(1, 200);

    for (int i = 0; i < 200; ++i) {
        const std::uint64_t offset = off_dist(rng);
        const std::uint64_t want = std::min<std::uint64_t>(len_dist(rng), expected.size() - offset);

        std::vector<std::byte> buf(want);
        auto n = fs->read(offset, buf);
        REQUIRE(n.has_value());
        REQUIRE(*n == want);
        INFO("offset=" << offset << " want=" << want);
        CHECK(std::equal(buf.begin(), buf.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset)));
    }
}

TEST_CASE("read exactly spanning the header/buffer boundary is correct", "[mount][synapsefs]") {
    auto header = test::make_bytes(24, 1);
    std::vector<test::GroupSpec> groups = {{"0.weight", test::make_bytes(1000, 11)}};
    auto fx = test::build_fixture(header, groups);
    auto expected = test::flatten(header, groups);
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(8);  // [20, 28): last 4 bytes of header, first 4 of the group
    auto n = fs->read(20, buf);
    REQUIRE(n.has_value());
    REQUIRE(*n == 8);
    CHECK(std::equal(buf.begin(), buf.end(), expected.begin() + 20));
}

TEST_CASE("read exactly spanning a tensor/group boundary is correct", "[mount][synapsefs]") {
    auto header = test::make_bytes(0, 1);
    std::vector<test::GroupSpec> groups = {
        {"g0", test::make_bytes(50, 3)},
        {"g1", test::make_bytes(50, 9)},
    };
    auto fx = test::build_fixture(header, groups);
    auto expected = test::flatten(header, groups);
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(10);  // [45, 55): straddles g0/g1
    auto n = fs->read(45, buf);
    REQUIRE(n.has_value());
    REQUIRE(*n == 10);
    CHECK(std::equal(buf.begin(), buf.end(), expected.begin() + 45));
}

TEST_CASE("a single-byte read at the very last byte of the file is correct", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto expected = test::flatten(test::make_bytes(24, 1),
                                  {{"0.weight", test::make_bytes(1000, 11)},
                                   {"0.bias", test::make_bytes(37, 22)}});
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(1);
    auto n = fs->read(fs->file_size() - 1, buf);
    REQUIRE(n.has_value());
    REQUIRE(*n == 1);
    CHECK(buf[0] == expected.back());
}

TEST_CASE("a read exactly at EOF is a short zero-length read, not an error", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(16);
    auto n = fs->read(fs->file_size(), buf);
    REQUIRE(n.has_value());
    CHECK(*n == 0);
}

TEST_CASE("a read past EOF is a short zero-length read, not an error", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(16);
    auto n = fs->read(fs->file_size() + 1000, buf);
    REQUIRE(n.has_value());
    CHECK(*n == 0);
}

TEST_CASE("a read starting before EOF but extending past it is short, not padded or an error",
         "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto expected = test::flatten(test::make_bytes(24, 1),
                                  {{"0.weight", test::make_bytes(1000, 11)},
                                   {"0.bias", test::make_bytes(37, 22)}});
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(50);  // last interval is only 37 bytes' worth of bias tail
    auto n = fs->read(fs->file_size() - 10, buf);
    REQUIRE(n.has_value());
    CHECK(*n == 10);  // short: only 10 real bytes remained
    CHECK(std::equal(buf.begin(), buf.begin() + 10, expected.end() - 10));
}

TEST_CASE("an empty output span reads zero bytes without touching the store", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto fs = make_fs(fx);

    std::vector<std::byte> buf;  // zero-length
    auto n = fs->read(0, buf);
    REQUIRE(n.has_value());
    CHECK(*n == 0);
}

TEST_CASE("cache_stats records hits after a frame has already been filled", "[mount][synapsefs]") {
    auto fx = standard_fixture();
    auto fs = make_fs(fx);

    std::vector<std::byte> buf(4);
    REQUIRE(fs->read(30, buf).has_value());  // first touch: miss + fill
    auto stats1 = fs->cache_stats();
    CHECK(stats1.misses >= 1);

    REQUIRE(fs->read(30, buf).has_value());  // same frame again: hit
    auto stats2 = fs->cache_stats();
    CHECK(stats2.hits >= 1);
    CHECK(stats2.misses == stats1.misses);  // no re-fill of the same frame
}

TEST_CASE("many concurrent readers all see byte-identical data and nothing crashes",
         "[mount][synapsefs][concurrency]") {
    // Rung 7 of the mount test ladder, minus the FUSE daemon: several
    // threads hammering the same SynapseFs (and therefore the same
    // FrameCache) with overlapping random reads must never observe a
    // half-filled frame or corrupt bytes from another reader's copy.
    auto header = test::make_bytes(24, 1);
    std::vector<test::GroupSpec> groups = {
        {"0.weight", test::make_bytes(1000, 11)},
        {"0.bias", test::make_bytes(37, 22)},
    };
    auto fx = test::build_fixture(header, groups);
    auto expected = test::flatten(header, groups);

    mount::FsOptions opts;
    opts.cache_bytes = 4096;  // small on purpose: forces eviction pressure
    auto fs = make_fs(fx, opts);

    constexpr int kThreads = 8;
    constexpr int kItersPerThread = 300;
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, seed = t] {
            std::mt19937_64 rng(1000 + seed);
            std::uniform_int_distribution<std::uint64_t> off_dist(0, expected.size() - 1);
            std::uniform_int_distribution<std::uint64_t> len_dist(1, 64);

            for (int i = 0; i < kItersPerThread && !failed.load(); ++i) {
                const std::uint64_t offset = off_dist(rng);
                const std::uint64_t want =
                    std::min<std::uint64_t>(len_dist(rng), expected.size() - offset);
                std::vector<std::byte> buf(want);
                auto n = fs->read(offset, buf);
                if (!n.has_value() || *n != want ||
                    !std::equal(buf.begin(), buf.end(),
                                expected.begin() + static_cast<std::ptrdiff_t>(offset))) {
                    failed.store(true);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    CHECK_FALSE(failed.load());
}
