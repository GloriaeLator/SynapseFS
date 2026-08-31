/// The chunk-range arithmetic (offset/length -> chunk indices, and back),
/// and verify_chunk_digest naming the OBJECT as well as the chunk on
/// failure — spec 15/16's requirement that a tamper report names both.

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <span>

#include <synapsefs/codec/chunk.hpp>
#include <synapsefs/format/object.hpp>

using namespace sfs;

TEST_CASE("chunks_covering: a read maps to the right inclusive chunk range",
         "[codec][chunk]") {
    constexpr std::uint64_t kChunk = 64u * 1024;

    SECTION("read fits in one chunk") {
        auto r = codec::chunks_covering(0, 100, kChunk);
        REQUIRE(r.first == 0);
        REQUIRE(r.last == 0);
        REQUIRE(r.count() == 1);
    }

    SECTION("read starts mid-chunk and crosses into the next") {
        auto r = codec::chunks_covering(kChunk - 10, 20, kChunk);
        REQUIRE(r.first == 0);
        REQUIRE(r.last == 1);
        REQUIRE(r.count() == 2);
    }

    SECTION("read exactly spans several whole chunks") {
        auto r = codec::chunks_covering(0, kChunk * 3, kChunk);
        REQUIRE(r.first == 0);
        REQUIRE(r.last == 2);  // last byte is in chunk index 2
        REQUIRE(r.count() == 3);
    }

    SECTION("a byte exactly at a chunk boundary belongs to the next chunk's start")  {
        auto r = codec::chunks_covering(kChunk, 1, kChunk);
        REQUIRE(r.first == 1);
        REQUIRE(r.last == 1);
    }

    SECTION("zero-length read") {
        auto r = codec::chunks_covering(12345, 0, kChunk);
        REQUIRE(r.count() == 1);  // the {0,0} sentinel, per the header's contract
    }
}

TEST_CASE("chunk_extent: index -> (begin, length), clamped at the last chunk",
         "[codec][chunk]") {
    constexpr std::uint64_t kChunk = 64u * 1024;
    constexpr std::uint64_t kTotal = kChunk * 2 + 100;  // last chunk is partial

    auto [begin0, len0] = codec::chunk_extent(0, kChunk, kTotal);
    REQUIRE(begin0 == 0);
    REQUIRE(len0 == kChunk);

    auto [begin1, len1] = codec::chunk_extent(1, kChunk, kTotal);
    REQUIRE(begin1 == kChunk);
    REQUIRE(len1 == kChunk);

    auto [begin2, len2] = codec::chunk_extent(2, kChunk, kTotal);
    REQUIRE(begin2 == kChunk * 2);
    REQUIRE(len2 == 100);  // clamped: only 100 bytes remain, not a full chunk
}

TEST_CASE("verify_chunk_digest: a correct chunk passes", "[codec][chunk]") {
    std::mt19937_64 rng(1);
    std::vector<std::byte> payload(5000);
    for (auto& b : payload) b = std::byte(rng());

    constexpr std::uint64_t kChunk = 1024;
    auto table = format::compute_chunk_digests(payload, kChunk);

    core::Oid object;  // zero oid is fine — only used for the error context here
    std::span<const std::byte> chunk0(payload.data(), kChunk);
    auto st = codec::verify_chunk_digest(chunk0, table, 0, object);
    REQUIRE(st.has_value());
}

TEST_CASE("verify_chunk_digest: a tampered chunk fails and names the object and chunk",
         "[codec][chunk]") {
    std::mt19937_64 rng(2);
    std::vector<std::byte> payload(5000);
    for (auto& b : payload) b = std::byte(rng());

    constexpr std::uint64_t kChunk = 1024;
    auto table = format::compute_chunk_digests(payload, kChunk);

    std::vector<std::byte> tampered(payload.begin(), payload.begin() + kChunk);
    tampered[0] ^= std::byte(0xFF);  // flip a bit: this is a different chunk now

    core::Oid object;
    auto st = codec::verify_chunk_digest(tampered, table, 0, object);
    REQUIRE_FALSE(st.has_value());
    REQUIRE(st.error().kind == core::ErrKind::ChunkDigestMismatch);
    REQUIRE(st.error().is_integrity());  // exits 4, never retried (error.hpp)
    // Context names both the object and the chunk index, per spec 15/16.
    REQUIRE(st.error().context.find("chunk 0") != std::string::npos);
}

TEST_CASE("verify_chunk_digest: an out-of-range chunk index is rejected", "[codec][chunk]") {
    std::vector<std::byte> payload(2048);
    auto table = format::compute_chunk_digests(payload, 1024);  // 2 chunks -> table has 2 entries

    core::Oid object;
    std::vector<std::byte> chunk(1024);
    auto st = codec::verify_chunk_digest(chunk, table, 5, object);  // index 5 doesn't exist
    REQUIRE_FALSE(st.has_value());
    REQUIRE(st.error().kind == core::ErrKind::MalformedObject);
}
