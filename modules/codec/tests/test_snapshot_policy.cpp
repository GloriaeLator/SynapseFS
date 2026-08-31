/// The four storage-decision rules, spec 12 §7 / ADR 0005, checked in the
/// order decide() applies them: no base, not alignable, chain too deep,
/// delta too large relative to full. Each case isolates exactly one rule by
/// keeping every earlier-checked condition satisfied.

#include <catch2/catch_test_macros.hpp>

#include <synapsefs/codec/snapshot_policy.hpp>

using namespace sfs;

namespace {
core::RepoConfig default_cfg() {
    core::RepoConfig cfg;
    cfg.max_chain_depth = 5;
    cfg.snapshot_alpha = 0.5;
    return cfg;
}
}  // namespace

TEST_CASE("no base -> FullNoBase", "[codec][snapshot_policy]") {
    codec::SnapshotInputs in;
    in.has_base = false;
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::FullNoBase);
}

TEST_CASE("aligner reported not alignable -> FullNotAlignable", "[codec][snapshot_policy]") {
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = false;
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::FullNotAlignable);
}

TEST_CASE("base chain already at the limit -> FullChainTooDeep", "[codec][snapshot_policy]") {
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = true;
    in.base_chain_depth = 5;  // +1 > max_chain_depth (5)
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::FullChainTooDeep);
}

TEST_CASE("chain depth exactly at the boundary is still a Delta", "[codec][snapshot_policy]") {
    // base_chain_depth + 1 == max_chain_depth is allowed, not rejected —
    // only exceeding it is (spec 12 §7's ">", not ">=").
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = true;
    in.base_chain_depth = 4;  // +1 == max_chain_depth (5)
    in.delta_bytes = 10;
    in.full_bytes = 100;
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::Delta);
}

TEST_CASE("delta larger than alpha * full -> FullDeltaTooLarge", "[codec][snapshot_policy]") {
    // The case measured for real on this branch (docs/tradeoffs.md §1.4):
    // unrelated checkpoints compressed to ratio ~1.0, sometimes slightly
    // over — exactly what this rule exists to catch.
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = true;
    in.base_chain_depth = 0;
    in.full_bytes = 1000;
    in.delta_bytes = 501;  // > 0.5 * 1000
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::FullDeltaTooLarge);
}

TEST_CASE("delta at exactly alpha * full is still a Delta", "[codec][snapshot_policy]") {
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = true;
    in.full_bytes = 1000;
    in.delta_bytes = 500;  // == 0.5 * 1000, not >
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::Delta);
}

TEST_CASE("a good delta is stored as Delta", "[codec][snapshot_policy]") {
    // The headline case (docs/tradeoffs.md §1.4): a near-zero residual.
    codec::SnapshotInputs in;
    in.has_base = true;
    in.alignable = true;
    in.base_chain_depth = 2;
    in.full_bytes = 115456;
    in.delta_bytes = 22;
    REQUIRE(codec::decide(in, default_cfg()) == codec::StorageDecision::Delta);
}

TEST_CASE("to_string names every decision", "[codec][snapshot_policy]") {
    REQUIRE(codec::to_string(codec::StorageDecision::Delta) == "delta");
    REQUIRE(codec::to_string(codec::StorageDecision::FullNoBase) == "full_no_base");
    REQUIRE(codec::to_string(codec::StorageDecision::FullNotAlignable) == "full_not_alignable");
    REQUIRE(codec::to_string(codec::StorageDecision::FullChainTooDeep) == "full_chain_too_deep");
    REQUIRE(codec::to_string(codec::StorageDecision::FullDeltaTooLarge) == "full_delta_too_large");
}

TEST_CASE("is_full is true for everything except Delta", "[codec][snapshot_policy]") {
    REQUIRE_FALSE(codec::is_full(codec::StorageDecision::Delta));
    REQUIRE(codec::is_full(codec::StorageDecision::FullNoBase));
    REQUIRE(codec::is_full(codec::StorageDecision::FullNotAlignable));
    REQUIRE(codec::is_full(codec::StorageDecision::FullChainTooDeep));
    REQUIRE(codec::is_full(codec::StorageDecision::FullDeltaTooLarge));
}
