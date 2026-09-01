/// Unit tests for IntervalTable: built once from the manifest's buffer
/// layout (inode.hpp), then must answer find(offset) correctly for every
/// interval boundary. docs/testing.md's mount test ladder calls the
/// header/buffer boundary and tensor-boundary cases "where most bugs live",
/// so those get the most cases here, plus the malformed-manifest paths
/// build() is documented to reject.

#include <catch2/catch_test_macros.hpp>

#include <synapsefs/mount/inode.hpp>

#include "mount_test_common.hpp"

using namespace sfs;

namespace {

format::Manifest make_manifest(std::uint64_t header_len,
                               const std::vector<std::pair<std::string, std::uint64_t>>& groups) {
    format::Manifest m;
    m.file.name = "x.safetensors";
    m.file.header_block = core::Oid{};

    std::uint64_t cursor = header_len;
    for (const auto& [name, nbytes] : groups) {
        format::GroupEntry ge;
        ge.mode = format::GroupMode::Full;
        ge.block = core::Oid{};
        m.groups[name] = ge;

        core::BufferEntry be;
        be.tensor = name;
        be.off = cursor;
        be.nbytes = nbytes;
        be.group = name;
        m.buffer.push_back(be);
        cursor += nbytes;
    }
    m.file.total_bytes = cursor;
    return m;
}

}  // namespace

TEST_CASE("header interval covers [0, header_len)", "[mount][interval_table]") {
    auto m = make_manifest(16, {{"g0", 100}});
    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());

    const auto* at0 = table->find(0);
    REQUIRE(at0 != nullptr);
    CHECK(at0->is_header);
    CHECK(at0->file_offset == 0);
    CHECK(at0->length == 16);

    const auto* at15 = table->find(15);  // last byte of header
    REQUIRE(at15 != nullptr);
    CHECK(at15->is_header);
}

TEST_CASE("first data byte after the header starts a non-header interval", "[mount][interval_table]") {
    auto m = make_manifest(16, {{"g0", 100}});
    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());

    const auto* at16 = table->find(16);  // header/buffer boundary
    REQUIRE(at16 != nullptr);
    CHECK_FALSE(at16->is_header);
    CHECK(at16->file_offset == 16);
    CHECK(at16->group_offset == 0);
    CHECK(table->group_name(at16->group_index) == "g0");
}

TEST_CASE("tensor boundary lands in the next group at group_offset 0", "[mount][interval_table]") {
    auto m = make_manifest(0, {{"g0", 50}, {"g1", 30}});
    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());

    const auto* last_of_g0 = table->find(49);
    REQUIRE(last_of_g0 != nullptr);
    CHECK(table->group_name(last_of_g0->group_index) == "g0");
    CHECK(last_of_g0->group_offset + (49 - last_of_g0->file_offset) == 49);

    const auto* first_of_g1 = table->find(50);
    REQUIRE(first_of_g1 != nullptr);
    CHECK(table->group_name(first_of_g1->group_index) == "g1");
    CHECK(first_of_g1->group_offset == 0);
    CHECK(first_of_g1->file_offset == 50);
}

TEST_CASE("find at exactly EOF returns nullptr", "[mount][interval_table]") {
    auto m = make_manifest(0, {{"g0", 10}});
    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());

    CHECK(table->total_bytes() == 10);
    CHECK(table->find(10) == nullptr);   // one past the last valid byte
    CHECK(table->find(9) != nullptr);
}

TEST_CASE("find far past EOF returns nullptr, not a wraparound match", "[mount][interval_table]") {
    auto m = make_manifest(0, {{"g0", 10}});
    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());
    CHECK(table->find(1'000'000) == nullptr);
}

TEST_CASE("repeated group names share one group index, not duplicated entries", "[mount][interval_table]") {
    // Several buffer entries can legitimately belong to the same permutation
    // group (many tensors sharing one group id); group_cursor must keep
    // advancing across them rather than resetting per-entry.
    format::Manifest m;
    m.file.name = "x.safetensors";
    m.file.header_block = core::Oid{};
    format::GroupEntry ge;
    ge.mode = format::GroupMode::Full;
    ge.block = core::Oid{};
    m.groups["shared"] = ge;

    core::BufferEntry a{"t0", 0, 20, "shared"};
    core::BufferEntry b{"t1", 20, 30, "shared"};
    m.buffer = {a, b};
    m.file.total_bytes = 50;

    auto table = mount::IntervalTable::build(m);
    REQUIRE(table.has_value());

    const auto* iv_a = table->find(0);
    const auto* iv_b = table->find(20);
    REQUIRE(iv_a != nullptr);
    REQUIRE(iv_b != nullptr);
    CHECK(iv_a->group_index == iv_b->group_index);
    CHECK(iv_a->group_offset == 0);
    CHECK(iv_b->group_offset == 20);  // continues, doesn't reset to 0
}

TEST_CASE("a manifest with no buffer entries is rejected", "[mount][interval_table]") {
    format::Manifest m;
    m.file.name = "empty.safetensors";
    m.file.total_bytes = 0;
    auto table = mount::IntervalTable::build(m);
    REQUIRE_FALSE(table.has_value());
}

TEST_CASE("a gap between buffer entries is rejected", "[mount][interval_table]") {
    format::Manifest m;
    m.file.name = "x.safetensors";
    format::GroupEntry ge;
    ge.mode = format::GroupMode::Full;
    ge.block = core::Oid{};
    m.groups["g0"] = ge;

    core::BufferEntry a{"t0", 0, 10, "g0"};
    core::BufferEntry b{"t1", 20, 10, "g0"};  // gap: should be 10, not 20
    m.buffer = {a, b};
    m.file.total_bytes = 30;

    auto table = mount::IntervalTable::build(m);
    REQUIRE_FALSE(table.has_value());
}

TEST_CASE("a buffer entry naming an undeclared group is rejected", "[mount][interval_table]") {
    format::Manifest m;
    m.file.name = "x.safetensors";
    core::BufferEntry a{"t0", 0, 10, "no_such_group"};
    m.buffer = {a};
    m.file.total_bytes = 10;

    auto table = mount::IntervalTable::build(m);
    REQUIRE_FALSE(table.has_value());
}
