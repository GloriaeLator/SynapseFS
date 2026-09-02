/// The integration test docs/testing.md names as one of the four that must
/// never break: "every single-byte tamper detected AND NAMED." Referenced by
/// tests/CMakeLists.txt since before this file existed (docs/known-gaps.md).
///
/// Builds a real, on-disk repository through TempRepo (apps/sfs/cmd/commit.cpp's
/// own Full-only orchestration, see tests/common/temprepo.cpp), then uses
/// FaultInjectingStore (tests/common/fault_inject.hpp) to corrupt individual
/// bytes of specific objects and checks two things for each case: that
/// store::verify() reports a finding at all, and that the finding NAMES the
/// exact object that was corrupted — a detector that fires without saying
/// which object failed is not something an operator can act on.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include <fault_inject.hpp>
#include <harness.hpp>
#include <temprepo.hpp>

#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/verify.hpp>

using namespace sfs;

namespace {

bool findings_name(const store::VerifyReport& report, const core::Oid& oid) {
    return std::any_of(report.findings.begin(), report.findings.end(),
                       [&](const store::VerifyFinding& f) { return f.object == oid; });
}

/// One tensor is enough: tamper detection is a property of the storage
/// layer, not of alignment, and TempRepo only ever commits Full groups
/// anyway (see temprepo.cpp's header comment).
std::filesystem::path write_one_tensor_checkpoint(const std::filesystem::path& dest) {
    test::SyntheticSpec spec;
    spec.layer_widths = {8, 16, 10};
    (void)test::write_synthetic_checkpoint(dest, spec);
    return dest;
}

}  // namespace

TEST_CASE("tamper: a clean repository verifies with no findings", "[integration][tamper]") {
    test::TempRepo repo;
    auto ckpt = write_one_tensor_checkpoint(repo.root() / "ckpt.safetensors");
    auto commit_oid = repo.commit_checkpoint(ckpt, "root commit");

    std::vector<core::Oid> tips{commit_oid};
    auto report = store::verify(repo.blocks(), repo.commits(), repo.manifests(), repo.refs(), tips,
                                {.full = true});
    REQUIRE(report.has_value());
    CHECK(report->ok());
    CHECK(report->findings.empty());
}

TEST_CASE("tamper: at-rest corruption of a Raw tensor block is detected and named",
         "[integration][tamper]") {
    test::TempRepo repo;
    auto ckpt = write_one_tensor_checkpoint(repo.root() / "ckpt.safetensors");
    auto commit_oid = repo.commit_checkpoint(ckpt, "root commit");

    auto commit = repo.commits().read(commit_oid);
    REQUIRE(commit.has_value());
    auto manifest = repo.manifests().read(commit->manifest);
    REQUIRE(manifest.has_value());
    REQUIRE_FALSE(manifest->groups.empty());

    // Every group is Full (TempRepo never plans deltas), so the first one's
    // block is a real ObjectKind::Raw oid.
    core::Oid raw_oid = *manifest->groups.begin()->second.block;

    test::FaultInjectingStore fault(repo.blocks());
    fault.corrupt_at_rest(raw_oid, /*offset=*/0);

    std::vector<core::Oid> tips{commit_oid};
    auto report =
        store::verify(fault, repo.commits(), repo.manifests(), repo.refs(), tips, {.full = true});
    REQUIRE(report.has_value());
    CHECK_FALSE(report->ok());
    CHECK(findings_name(*report, raw_oid));
}

TEST_CASE("tamper: at-rest corruption of the safetensors Header block is detected and named",
         "[integration][tamper]") {
    test::TempRepo repo;
    auto ckpt = write_one_tensor_checkpoint(repo.root() / "ckpt.safetensors");
    auto commit_oid = repo.commit_checkpoint(ckpt, "root commit");

    auto commit = repo.commits().read(commit_oid);
    REQUIRE(commit.has_value());
    auto manifest = repo.manifests().read(commit->manifest);
    REQUIRE(manifest.has_value());

    core::Oid header_oid = manifest->file.header_block;

    test::FaultInjectingStore fault(repo.blocks());
    fault.corrupt_at_rest(header_oid, /*offset=*/4);  // inside the JSON header, past the length prefix

    std::vector<core::Oid> tips{commit_oid};
    auto report =
        store::verify(fault, repo.commits(), repo.manifests(), repo.refs(), tips, {.full = true});
    REQUIRE(report.has_value());
    CHECK_FALSE(report->ok());
    CHECK(findings_name(*report, header_oid));
}

TEST_CASE("tamper: at-rest corruption of the Manifest object is detected and named",
         "[integration][tamper]") {
    test::TempRepo repo;
    auto ckpt = write_one_tensor_checkpoint(repo.root() / "ckpt.safetensors");
    auto commit_oid = repo.commit_checkpoint(ckpt, "root commit");

    auto commit = repo.commits().read(commit_oid);
    REQUIRE(commit.has_value());
    core::Oid manifest_oid = commit->manifest;

    // Route commit/manifest reads through the SAME FaultInjectingStore too,
    // so verify()'s manifest-parsing step (docs/spec — check 3) sees the
    // corruption, not just the raw-block check (check 5).
    test::FaultInjectingStore fault(repo.blocks());
    fault.corrupt_at_rest(manifest_oid, /*offset=*/2);  // inside the canonical JSON body

    store::CommitStore   fault_commits(fault, repo.refs());
    store::ManifestStore fault_manifests(fault, fault_commits);

    std::vector<core::Oid> tips{commit_oid};
    auto report =
        store::verify(fault, fault_commits, fault_manifests, repo.refs(), tips, {.full = true});
    REQUIRE(report.has_value());
    CHECK_FALSE(report->ok());
    CHECK(findings_name(*report, manifest_oid));
}

TEST_CASE("tamper: corrupt_on_read changes get()'s bytes without failing the call itself",
         "[integration][tamper]") {
    // Distinguishes the two injection modes fault_inject.hpp documents:
    // corrupt_at_rest is caught by the store's own re-verification (the
    // three cases above); corrupt_on_read models a bit flip AFTER that
    // verification already passed, so get() itself must still succeed with
    // silently-wrong bytes -- it is downstream code's job (a manifest
    // witness hash, e.g.) to catch this one, not the block store's.
    test::TempRepo repo;
    auto ckpt = write_one_tensor_checkpoint(repo.root() / "ckpt.safetensors");
    auto commit_oid = repo.commit_checkpoint(ckpt, "root commit");

    auto commit = repo.commits().read(commit_oid);
    REQUIRE(commit.has_value());
    auto manifest = repo.manifests().read(commit->manifest);
    REQUIRE(manifest.has_value());
    core::Oid raw_oid = *manifest->groups.begin()->second.block;

    auto clean = repo.blocks().get(raw_oid, core::ObjectKind::Raw);
    REQUIRE(clean.has_value());

    test::FaultInjectingStore fault(repo.blocks());
    fault.corrupt_on_read(raw_oid, /*offset=*/0);

    auto corrupted = fault.get(raw_oid, core::ObjectKind::Raw);
    REQUIRE(corrupted.has_value());  // the call itself must still succeed
    REQUIRE(corrupted->size() == clean->size());
    CHECK((*corrupted)[0] != (*clean)[0]);  // exactly the flipped byte differs
    CHECK(std::equal(clean->begin() + 1, clean->end(), corrupted->begin() + 1));
}

TEST_CASE("crash_matrix building block: CrashingStore before-write leaves no object at all",
         "[integration][tamper]") {
    test::TempRepo repo;
    test::CrashingStore crashing(repo.blocks(), /*fail_on_nth_put=*/1,
                                 test::CrashingStore::When::BeforeWrite);

    std::vector<std::byte> payload{std::byte{1}, std::byte{2}, std::byte{3}};
    auto result = crashing.put(core::ObjectKind::Raw, payload);
    CHECK_FALSE(result.has_value());

    // A real store never saw the payload, so its would-be oid was never
    // written -- verified by asking the real (unwrapped) store directly.
    core::Oid would_be_oid = core::compute_oid(core::ObjectKind::Raw, payload);
    auto exists = repo.blocks().contains(would_be_oid);
    REQUIRE(exists.has_value());
    CHECK_FALSE(*exists);
}

TEST_CASE("crash_matrix building block: CrashingStore after-write leaves a valid, unreferenced "
         "object",
         "[integration][tamper]") {
    test::TempRepo repo;
    test::CrashingStore crashing(repo.blocks(), /*fail_on_nth_put=*/1,
                                 test::CrashingStore::When::AfterWriteBeforeRename);

    std::vector<std::byte> payload{std::byte{4}, std::byte{5}, std::byte{6}};
    auto result = crashing.put(core::ObjectKind::Raw, payload);
    CHECK_FALSE(result.has_value());  // the CALLER never learns it succeeded...

    // ...but the object is genuinely, durably there -- exactly the "wasted
    // disk, not a broken repository" outcome docs/known-gaps.md documents
    // for a crash after write. verify_block on the real store must pass.
    core::Oid oid = core::compute_oid(core::ObjectKind::Raw, payload);
    auto exists = repo.blocks().contains(oid);
    REQUIRE(exists.has_value());
    CHECK(*exists);
    auto vst = repo.blocks().verify_block(oid, core::ObjectKind::Raw);
    CHECK(vst.has_value());
}
