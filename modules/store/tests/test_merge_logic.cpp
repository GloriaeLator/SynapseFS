/// Unit tests for the per-group merge decision.
///
/// resolve_group is exposed precisely so this can run without a repository
/// (merge_logic.hpp), which is what makes these tests fast and total: every
/// cell of the base x ours x theirs table is enumerable.

#include <catch2/catch_test_macros.hpp>

#include <synapsefs/store/merge_logic.hpp>

using namespace sfs;
using sfs::store::GroupResolution;
using sfs::store::resolve_group;

namespace {

core::Oid oid_of(std::uint8_t tag) {
    std::array<std::byte, core::kOidBytes> b{};
    b[0] = static_cast<std::byte>(tag);
    return core::Oid(b);
}

format::GroupEntry full(std::uint8_t tag) {
    format::GroupEntry g;
    g.mode = format::GroupMode::Full;
    g.block = oid_of(tag);
    g.chain_depth = 0;
    return g;
}

format::GroupEntry delta(std::uint8_t diff_tag, std::uint8_t base_commit_tag,
                         std::string base_group = "g") {
    format::GroupEntry g;
    g.mode = format::GroupMode::Delta;
    g.diff_block = oid_of(diff_tag);
    g.base = format::DeltaBase{oid_of(base_commit_tag), std::move(base_group)};
    g.chain_depth = 1;
    return g;
}

}  // namespace

TEST_CASE("neither side touched the group", "[merge]") {
    auto b = full(1);
    auto o = full(1);
    auto t = full(1);
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::Unchanged);
}

TEST_CASE("changed on one side only takes that side", "[merge]") {
    auto b = full(1);
    auto unchanged = full(1);
    auto changed = full(2);

    SECTION("ours changed") {
        REQUIRE(resolve_group(&b, &changed, &unchanged) == GroupResolution::TakeOurs);
    }
    SECTION("theirs changed") {
        REQUIRE(resolve_group(&b, &unchanged, &changed) == GroupResolution::TakeTheirs);
    }
}

TEST_CASE("changed on both sides conflicts", "[merge]") {
    auto b = full(1);
    auto o = full(2);
    auto t = full(3);
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::Conflict);
}

TEST_CASE("both sides made the identical change", "[merge]") {
    // Convergent edits are not a conflict: content addressing means identical
    // bytes have identical oids, so there is nothing to choose between.
    auto b = full(1);
    auto o = full(2);
    auto t = full(2);
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::Unchanged);
}

TEST_CASE("group added on one side only", "[merge]") {
    auto added = full(5);
    SECTION("added by us") {
        REQUIRE(resolve_group(nullptr, &added, nullptr) == GroupResolution::TakeOurs);
    }
    SECTION("added by them") {
        REQUIRE(resolve_group(nullptr, nullptr, &added) == GroupResolution::TakeTheirs);
    }
}

TEST_CASE("both sides added the same group name with different content", "[merge]") {
    // No base to compare against, and the two sides disagree. Picking either
    // silently would produce an artifact neither author wrote.
    auto o = full(5);
    auto t = full(6);
    REQUIRE(resolve_group(nullptr, &o, &t) == GroupResolution::Conflict);
}

TEST_CASE("both sides added the same group name with identical content", "[merge]") {
    auto o = full(5);
    auto t = full(5);
    REQUIRE(resolve_group(nullptr, &o, &t) == GroupResolution::Unchanged);
}

TEST_CASE("delete on one side, change on the other, conflicts", "[merge]") {
    auto b = full(1);
    auto changed = full(2);

    SECTION("we deleted, they changed") {
        REQUIRE(resolve_group(&b, nullptr, &changed) == GroupResolution::Conflict);
    }
    SECTION("they deleted, we changed") {
        REQUIRE(resolve_group(&b, &changed, nullptr) == GroupResolution::Conflict);
    }
}

TEST_CASE("a mode change from Full to Delta is a change", "[merge]") {
    auto b = full(1);
    auto o = delta(9, 1);
    auto t = full(1);
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::TakeOurs);
}

TEST_CASE("same residual against a different base is different content", "[merge]") {
    // The bytes of the diff block alone do not identify a delta group: the
    // same residual applied to a different base reconstructs different
    // tensors, so this must not be reported as Unchanged.
    auto b = delta(9, 1);
    auto o = delta(9, 1);
    auto t = delta(9, 2);
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::TakeTheirs);
}

TEST_CASE("same residual against a different base group is different content", "[merge]") {
    auto b = delta(9, 1, "layer0");
    auto o = delta(9, 1, "layer0");
    auto t = delta(9, 1, "layer1");
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::TakeTheirs);
}

TEST_CASE("chain_depth alone is not a content difference", "[merge]") {
    // chain_depth is bookkeeping derived from the base, not content. Two
    // entries that resolve to the same bytes must compare equal even if a
    // rewrite renumbered the chain.
    auto b = full(1);
    auto o = full(1);
    o.chain_depth = 0;
    auto t = full(1);
    t.chain_depth = 3;
    REQUIRE(resolve_group(&b, &o, &t) == GroupResolution::Unchanged);
}
