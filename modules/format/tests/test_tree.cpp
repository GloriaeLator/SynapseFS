/// Unit tests for format::Tree — the multi-file (sharded) checkpoint object.
///
/// A Tree is addressed by its canonical serialisation, so the tests that
/// matter are not "does it hold the data" but "is there exactly one byte
/// string for a given tree, and does every other byte string get rejected".
/// That is what parse()'s round-trip check buys, and it is only worth
/// anything if the rejections are actually exercised.

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

#include <synapsefs/format/tree.hpp>

using namespace sfs;
using format::Tree;
using format::TreeEntry;

namespace {

core::Oid oid_of(const std::string& hex_seed) {
    // A distinct, valid, non-null oid per seed character.
    std::string hex(core::kOidHexChars, '0');
    for (std::size_t i = 0; i < hex_seed.size() && i < hex.size(); ++i) hex[i] = hex_seed[i];
    auto o = core::Oid::parse("b3:" + hex);
    REQUIRE(o.has_value());
    return *o;
}

std::string as_text(const std::vector<std::byte>& b) {
    return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

std::vector<std::byte> as_bytes(const std::string& s) {
    std::vector<std::byte> b(s.size());
    std::memcpy(b.data(), s.data(), s.size());
    return b;
}

Tree three_shards() {
    auto t = Tree::make({{"model-00002-of-00003.safetensors", oid_of("b2")},
                         {"model-00001-of-00003.safetensors", oid_of("a1")},
                         {"model-00003-of-00003.safetensors", oid_of("c3")}});
    REQUIRE(t.has_value());
    return *t;
}

}  // namespace

TEST_CASE("make sorts entries by name", "[tree]") {
    auto t = three_shards();
    REQUIRE(t.entries.size() == 3);
    CHECK(t.entries[0].name == "model-00001-of-00003.safetensors");
    CHECK(t.entries[1].name == "model-00002-of-00003.safetensors");
    CHECK(t.entries[2].name == "model-00003-of-00003.safetensors");
    CHECK(t.format_version == format::kTreeFormatVersion);
}

TEST_CASE("canonical json is sorted-key, whitespace-free, newline-free", "[tree]") {
    const auto text = as_text(three_shards().to_canonical_json());
    CHECK(text.find(' ') == std::string::npos);
    CHECK(text.find('\n') == std::string::npos);
    // "entries" sorts before "format_version".
    CHECK(text.rfind(R"({"entries":[)", 0) == 0);
    CHECK(text.find(R"("format_version":2)") != std::string::npos);
    // Within an entry, "manifest" sorts before "name".
    CHECK(text.find(R"({"manifest":"b3:a1)") != std::string::npos);
}

TEST_CASE("round-trips through parse", "[tree]") {
    const auto original = three_shards();
    const auto bytes = original.to_canonical_json();

    auto reparsed = Tree::parse(bytes);
    REQUIRE(reparsed.has_value());
    CHECK(reparsed->format_version == original.format_version);
    CHECK(reparsed->entries == original.entries);
    CHECK(reparsed->oid() == original.oid());
}

TEST_CASE("entry order changes the address", "[tree]") {
    // Not a property of Tree so much as the reason make() sorts: the same
    // three shards in a different order must not be a second valid object.
    auto sorted = three_shards();

    Tree unsorted;
    unsorted.entries = {sorted.entries[2], sorted.entries[1], sorted.entries[0]};

    CHECK_FALSE(unsorted.validate().has_value());
    CHECK(unsorted.oid() != sorted.oid());
    // And the unsorted serialisation is not accepted back.
    CHECK_FALSE(Tree::parse(unsorted.to_canonical_json()).has_value());
}

TEST_CASE("oid differs from a manifest with identical payload bytes", "[tree]") {
    // The kind is inside the hashed frame (oid.hpp). Without ObjectKind::Tree
    // a tree payload and a manifest payload would share an address.
    const auto bytes = three_shards().to_canonical_json();
    CHECK(core::compute_oid(core::ObjectKind::Tree, bytes) !=
          core::compute_oid(core::ObjectKind::Manifest, bytes));
}

TEST_CASE("find locates entries by name", "[tree]") {
    const auto t = three_shards();
    const auto* e = t.find("model-00002-of-00003.safetensors");
    REQUIRE(e != nullptr);
    CHECK(e->manifest == oid_of("b2"));
    CHECK(t.find("index.json") == nullptr);
}

TEST_CASE("validate rejects malformed trees", "[tree]") {
    const auto good = oid_of("a1");

    SECTION("empty") {
        Tree t;
        CHECK_FALSE(t.validate().has_value());
    }
    SECTION("wrong format_version") {
        auto t = three_shards();
        t.format_version = 1;
        CHECK_FALSE(t.validate().has_value());
    }
    SECTION("duplicate names") {
        Tree t;
        t.entries = {{"a.safetensors", good}, {"a.safetensors", oid_of("b2")}};
        CHECK_FALSE(t.validate().has_value());
    }
    SECTION("null oid") {
        Tree t;
        t.entries = {{"a.safetensors", core::Oid{}}};
        CHECK_FALSE(t.validate().has_value());
    }
    SECTION("path separator in name") {
        CHECK_FALSE(Tree::make({{"../etc/passwd", good}}).has_value());
        CHECK_FALSE(Tree::make({{"sub/dir.safetensors", good}}).has_value());
        CHECK_FALSE(Tree::make({{"back\\slash", good}}).has_value());
    }
    SECTION("dot names") {
        CHECK_FALSE(Tree::make({{".", good}}).has_value());
        CHECK_FALSE(Tree::make({{"..", good}}).has_value());
    }
    SECTION("empty and control-character names") {
        CHECK_FALSE(Tree::make({{"", good}}).has_value());
        CHECK_FALSE(Tree::make({{std::string("a\0b", 3), good}}).has_value());
        CHECK_FALSE(Tree::make({{"a\nb", good}}).has_value());
    }
    SECTION("a leading dot is a normal file name") {
        CHECK(Tree::make({{".hidden", good}}).has_value());
    }
}

TEST_CASE("parse rejects non-canonical and malformed input", "[tree]") {
    SECTION("whitespace-padded json") {
        CHECK_FALSE(Tree::parse(as_bytes(
            R"({"entries": [{"manifest": "b3:)" + std::string(64, '0').replace(0, 2, "a1") +
            R"(", "name": "a"}], "format_version": 2})")).has_value());
    }
    SECTION("trailing newline") {
        auto bytes = three_shards().to_canonical_json();
        bytes.push_back(std::byte{'\n'});
        CHECK_FALSE(Tree::parse(bytes).has_value());
    }
    SECTION("not json") {
        CHECK_FALSE(Tree::parse(as_bytes("not json at all")).has_value());
    }
    SECTION("entries is not an array") {
        CHECK_FALSE(Tree::parse(as_bytes(R"({"entries":{},"format_version":2})")).has_value());
    }
    SECTION("missing field") {
        CHECK_FALSE(Tree::parse(as_bytes(R"({"format_version":2})")).has_value());
    }
    SECTION("bad oid") {
        CHECK_FALSE(Tree::parse(as_bytes(
            R"({"entries":[{"manifest":"deadbeef","name":"a"}],"format_version":2})")).has_value());
    }
    SECTION("version 1 payload is refused") {
        CHECK_FALSE(Tree::parse(as_bytes(
            R"({"entries":[{"manifest":"b3:)" + std::string(64, '0').replace(0, 2, "a1") +
            R"(","name":"a"}],"format_version":1})")).has_value());
    }
}
