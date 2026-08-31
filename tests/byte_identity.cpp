/// The property docs/testing.md names as one of the four that must never
/// break: sha256(reconstruct(A, diff(A,B))) == sha256(B).
///
/// This exercises the align <-> codec glue (store::plan_commit_groups),
/// including the ONE case modules/codec/tests/test_byte_identity.cpp
/// structurally cannot: a tensor with TWO simultaneously-permuted axes
/// (dim 0 from its own group, dim 1 from a DIFFERENT, already-resolved
/// group) — the ColumnPermutingSource adapter's reason for existing.
///
/// align::Matcher itself needs Torch to link (sparse_match.cpp); this test
/// never calls it. It builds an align::MatchReport BY HAND instead — the
/// same "precomputed permutation, consumed as plain data" approach used
/// throughout the codec branch — since plan_commit_groups only consumes
/// MatchReport's plain-data types, never Matcher.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <random>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/topology.hpp>
#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/commit_planner.hpp>

using namespace sfs;

namespace {

// ---- fakes, same shape as modules/codec/tests/test_byte_identity.cpp's ----

struct TensorData {
    core::TensorMeta meta;
    std::vector<std::byte> bytes;
};

class FakeTensorSource : public core::ITensorSource {
public:
    void add(const std::string& name, std::vector<std::uint64_t> shape, core::DType dtype,
            std::vector<std::byte> bytes) {
        core::TensorMeta m;
        m.shape_owner = name;
        m.shape = std::move(shape);
        m.dtype = dtype;
        m.nbytes = bytes.size();
        tensors_[name] = TensorData{m, std::move(bytes)};
    }
    std::span<const std::byte> header_bytes() const override { return {}; }
    std::span<const core::BufferEntry> buffer_layout() const override {
        // Built once, lazily, so the returned span stays valid.
        if (layout_.empty()) {
            for (const auto& [name, td] : tensors_) {
                core::BufferEntry e;
                e.tensor = name;
                e.nbytes = td.bytes.size();
                e.group = name;
                layout_.push_back(e);
            }
        }
        return layout_;
    }
    const core::TensorMeta* meta(std::string_view name) const override {
        auto it = tensors_.find(std::string(name));
        return it == tensors_.end() ? nullptr : &it->second.meta;
    }
    std::uint64_t total_bytes() const override {
        std::uint64_t n = 0;
        for (const auto& [k, v] : tensors_) n += v.bytes.size();
        return n;
    }
    core::Result<std::size_t> read_units(std::string_view name, std::uint64_t first,
                                         std::uint64_t count, std::span<std::byte> out) override {
        auto it = tensors_.find(std::string(name));
        if (it == tensors_.end()) return SFS_ERR(ObjectNotFound, "no such tensor", std::string(name));
        auto ub = it->second.meta.unit_bytes(0);
        if (!ub) return std::unexpected(ub.error());
        const std::uint64_t off = first * (*ub);
        const std::uint64_t n = count * (*ub);
        std::memcpy(out.data(), it->second.bytes.data() + off, n);
        return n;
    }

private:
    std::map<std::string, TensorData> tensors_;
    mutable std::vector<core::BufferEntry> layout_;
};

class FakeBlockStore : public core::IBlockStore {
public:
    core::Result<core::Oid> put(core::ObjectKind kind, std::span<const std::byte> payload) override {
        auto oid = core::compute_oid(kind, payload);
        objs_[oid] = std::vector<std::byte>(payload.begin(), payload.end());
        return oid;
    }
    core::Result<std::vector<std::byte>> get(const core::Oid& oid, core::ObjectKind) override {
        auto it = objs_.find(oid);
        if (it == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object", oid.to_string());
        return it->second;
    }
    core::Result<std::size_t> read_range(const core::Oid& oid, core::ObjectKind,
                                         std::uint64_t offset, std::span<std::byte> out) override {
        auto it = objs_.find(oid);
        if (it == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object", oid.to_string());
        const auto& bytes = it->second;
        const std::size_t n = std::min<std::size_t>(out.size(), bytes.size() - offset);
        std::memcpy(out.data(), bytes.data() + offset, n);
        return n;
    }
    core::Status verify_block(const core::Oid&, core::ObjectKind) override { return {}; }
    core::Result<bool> contains(const core::Oid& oid) const override { return objs_.count(oid) > 0; }
    core::Result<std::uint64_t> size_of(const core::Oid& oid) const override {
        return objs_.at(oid).size();
    }
    core::Result<core::ObjectKind> kind_of(const core::Oid&) const override {
        return core::ObjectKind::Raw;
    }

private:
    std::map<core::Oid, std::vector<std::byte>> objs_;
};

class FakeObjectSource : public core::IObjectSource {
public:
    void add(const core::Oid& commit, const format::Manifest* m) { manifests_[commit] = m; }
    core::Result<const format::Manifest*> manifest_for(const core::Oid& commit) override {
        auto it = manifests_.find(commit);
        if (it == manifests_.end()) return SFS_ERR(ObjectNotFound, "no such commit");
        return it->second;
    }
    core::Result<bool> is_ancestor(const core::Oid&, const core::Oid&) override { return true; }

private:
    std::map<core::Oid, const format::Manifest*> manifests_;
};

std::vector<std::byte> random_bytes(std::size_t n, std::mt19937_64& rng) {
    std::vector<std::byte> v(n);
    for (auto& b : v) b = std::byte(rng());
    return v;
}

std::vector<std::uint32_t> random_perm(std::uint32_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), std::mt19937_64(seed));
    return p;
}

// Row-gather: out[i] = src[perm[i]] over `rows` of `cols`-wide F16 elements.
std::vector<std::byte> gather_rows(const std::vector<std::byte>& src, std::uint32_t rows,
                                   std::uint32_t cols, const std::vector<std::uint32_t>& perm) {
    std::vector<std::byte> out(src.size());
    const std::uint64_t row_bytes = std::uint64_t(cols) * 2;
    for (std::uint32_t i = 0; i < rows; ++i)
        std::memcpy(out.data() + i * row_bytes, src.data() + perm[i] * row_bytes, row_bytes);
    return out;
}

// Column-gather: out[:, j] = src[:, perm[j]], for `rows` of `cols`-wide F16
// elements — the other half of a two-axis permutation.
std::vector<std::byte> gather_cols(const std::vector<std::byte>& src, std::uint32_t rows,
                                   std::uint32_t cols, const std::vector<std::uint32_t>& perm) {
    std::vector<std::byte> out(src.size());
    for (std::uint32_t i = 0; i < rows; ++i)
        for (std::uint32_t j = 0; j < cols; ++j)
            std::memcpy(out.data() + (i * cols + j) * 2, src.data() + (i * cols + perm[j]) * 2, 2);
    return out;
}

}  // namespace

TEST_CASE("plan_commit_groups + read_range: byte-exact end to end, including a "
         "two-axis-permuted tensor",
         "[integration][byte_identity]") {
    // A 3-layer MLP: 0 (in=48 -> g0=64), 2 (g0=64 -> g2=32), 4 (g2=32 -> out=10,
    // pinned). "2.weight" has BOTH axes non-pinned — dim0 owns g2, dim1
    // depends on g0 — the case that needs ColumnPermutingSource.
    constexpr std::uint32_t kIn = 48, kG0 = 64, kG2 = 32, kOut = 10;
    std::mt19937_64 rng(123);

    auto w0_base = random_bytes(std::uint64_t(kG0) * kIn * 2, rng);
    auto b0_base = random_bytes(std::uint64_t(kG0) * 2, rng);
    auto w2_base = random_bytes(std::uint64_t(kG2) * kG0 * 2, rng);
    auto b2_base = random_bytes(std::uint64_t(kG2) * 2, rng);
    auto w4_base = random_bytes(std::uint64_t(kOut) * kG2 * 2, rng);
    auto b4_base = random_bytes(std::uint64_t(kOut) * 2, rng);

    const auto perm_g0 = random_perm(kG0, 1);
    const auto perm_g2 = random_perm(kG2, 2);

    // Target = base with every group's permutation applied, consistently,
    // everywhere that group appears — the "function-identical, byte-different"
    // construction spec 12 exists to collapse back down.
    auto w0_target = gather_rows(w0_base, kG0, kIn, perm_g0);
    auto b0_target = gather_rows(b0_base, kG0, 1, perm_g0);
    // 2.weight: rows by g2 (its own group), THEN columns by g0 (the
    // previous layer's group) — the two-axis case.
    auto w2_target = gather_cols(gather_rows(w2_base, kG2, kG0, perm_g2), kG2, kG0, perm_g0);
    auto b2_target = gather_rows(b2_base, kG2, 1, perm_g2);
    // 4.weight: dim0 (out) is pinned, dim1 (g2) is not — columns move,
    // rows don't.
    auto w4_target = gather_cols(w4_base, kOut, kG2, perm_g2);
    auto b4_target = b4_base;  // dim0 pinned (out): unchanged

    FakeTensorSource base_src, target_src;
    base_src.add("0.weight", {kG0, kIn}, core::DType::F16, w0_base);
    base_src.add("0.bias", {kG0}, core::DType::F16, b0_base);
    base_src.add("2.weight", {kG2, kG0}, core::DType::F16, w2_base);
    base_src.add("2.bias", {kG2}, core::DType::F16, b2_base);
    base_src.add("4.weight", {kOut, kG2}, core::DType::F16, w4_base);
    base_src.add("4.bias", {kOut}, core::DType::F16, b4_base);

    target_src.add("0.weight", {kG0, kIn}, core::DType::F16, w0_target);
    target_src.add("0.bias", {kG0}, core::DType::F16, b0_target);
    target_src.add("2.weight", {kG2, kG0}, core::DType::F16, w2_target);
    target_src.add("2.bias", {kG2}, core::DType::F16, b2_target);
    target_src.add("4.weight", {kOut, kG2}, core::DType::F16, w4_target);
    target_src.add("4.bias", {kOut}, core::DType::F16, b4_target);

    core::Topology topo;
    topo.groups["in"] = {kIn, true};
    topo.groups["g0"] = {kG0, false};
    topo.groups["g2"] = {kG2, false};
    topo.groups["out"] = {kOut, true};
    topo.tensors["0.weight"] = {{{0, "g0", 1}, {1, "in", 1}}};
    topo.tensors["0.bias"] = {{{0, "g0", 1}}};
    topo.tensors["2.weight"] = {{{0, "g2", 1}, {1, "g0", 1}}};
    topo.tensors["2.bias"] = {{{0, "g2", 1}}};
    topo.tensors["4.weight"] = {{{0, "out", 1}, {1, "g2", 1}}};
    topo.tensors["4.bias"] = {{{0, "out", 1}}};

    // Built by hand, not by align::Matcher::run() (needs Torch) — but the
    // SAME plain-data type plan_commit_groups actually consumes.
    align::MatchReport report;
    align::GroupMatch gm_in;
    gm_in.group = "in";
    gm_in.identity = true;
    gm_in.alignable = true;
    report.groups["in"] = gm_in;

    align::GroupMatch gm_out = gm_in;
    gm_out.group = "out";
    report.groups["out"] = gm_out;

    align::GroupMatch gm_g0;
    gm_g0.group = "g0";
    gm_g0.permutation = perm_g0;
    gm_g0.identity = false;
    gm_g0.alignable = true;
    gm_g0.cost_raw = -500.0;
    gm_g0.cost_normalized = 0.02;
    report.groups["g0"] = gm_g0;

    align::GroupMatch gm_g2 = gm_g0;
    gm_g2.group = "g2";
    gm_g2.permutation = perm_g2;
    report.groups["g2"] = gm_g2;

    // First commit: everything Full (no parent) — apps/sfs/cmd/commit.cpp's
    // actual behaviour for a root commit, alignment or not.
    FakeBlockStore blocks;
    format::Manifest manifest_a;
    core::Oid commit_a{};  // placeholder id, only used as a map key here
    std::unordered_map<std::string, store::ParentTensorInfo> parent_info;
    for (const auto& entry : base_src.buffer_layout()) {
        std::vector<std::byte> buf(entry.nbytes);
        const auto* m = base_src.meta(entry.tensor);
        REQUIRE(base_src.read_units(entry.tensor, 0, m->shape[0], buf).has_value());
        auto oid = blocks.put(core::ObjectKind::Raw, buf);
        REQUIRE(oid.has_value());

        format::GroupEntry g;
        g.mode = format::GroupMode::Full;
        g.block = *oid;
        g.chain_depth = 0;
        manifest_a.groups[entry.tensor] = g;
        parent_info[entry.tensor] = store::ParentTensorInfo{commit_a, 0};
    }

    // Second commit: the actual thing under test.
    core::RepoConfig cfg;  // defaults: max_chain_depth=5, snapshot_alpha=0.5
    auto entries_b =
        store::plan_commit_groups(base_src, target_src, topo, report, parent_info, blocks, cfg);
    REQUIRE(entries_b.has_value());

    // The two group-owning tensors with a clean permutation must have been
    // planned as Delta (well under alpha for a pure permutation).
    CHECK(entries_b->at("0.weight").mode == format::GroupMode::Delta);
    CHECK(entries_b->at("2.weight").mode == format::GroupMode::Delta);
    // "4.weight"/"4.bias" own no non-identity group of their own (their
    // dim-0 is pinned "out") — a real, current limitation: a tensor whose
    // OUTPUT axis is pinned gets no residual even if an input axis moved,
    // since residuals are only ever computed for the OWNING group. Full is
    // the correct fallback here, not a missed Delta.
    CHECK(entries_b->at("4.weight").mode == format::GroupMode::Full);
    CHECK(entries_b->at("4.bias").mode == format::GroupMode::Full);

    format::Manifest manifest_b;
    for (const auto& [name, entry] : *entries_b) manifest_b.groups[name] = entry;
    FakeObjectSource history;
    history.add(commit_a, &manifest_a);

    codec::ReadCtx ctx;
    ctx.blocks = &blocks;
    ctx.manifest = &manifest_b;
    ctx.history = &history;
    ctx.max_depth = cfg.max_chain_depth;
    ctx.topology = &topo;  // needed for "2.weight"'s secondary-axis lookup

    const std::pair<std::string, const std::vector<std::byte>*> expected[] = {
        {"0.weight", &w0_target}, {"0.bias", &b0_target}, {"2.weight", &w2_target},
        {"2.bias", &b2_target},   {"4.weight", &w4_target}, {"4.bias", &b4_target},
    };
    for (const auto& [name, exp] : expected) {
        std::vector<std::byte> actual(exp->size());
        auto n = codec::read_range(ctx, name, 0, actual);
        INFO("tensor = " << name << (n ? "" : (" error: " + n.error().to_string())));
        REQUIRE(n.has_value());
        REQUIRE(*n == exp->size());
        REQUIRE(actual == *exp);
    }
}

TEST_CASE("plan_commit_groups: an unaligned/too-large delta falls back to Full",
         "[integration][byte_identity]") {
    // Base and target share nothing (independent random data) — the
    // "unrelated checkpoints" case docs/tradeoffs.md §1.4 measured for real
    // (ratio > 1). snapshot_policy must reject the Delta and this function
    // must actually honour that, not just compute decide() and ignore it.
    constexpr std::uint32_t kG0 = 32;
    std::mt19937_64 rng(9);
    auto base_bytes = random_bytes(std::uint64_t(kG0) * 2, rng);
    auto target_bytes = random_bytes(std::uint64_t(kG0) * 2, rng);  // independent, not a permutation

    FakeTensorSource base_src, target_src;
    base_src.add("0.bias", {kG0}, core::DType::F16, base_bytes);
    target_src.add("0.bias", {kG0}, core::DType::F16, target_bytes);

    core::Topology topo;
    topo.groups["g0"] = {kG0, false};
    topo.tensors["0.bias"] = {{{0, "g0", 1}}};

    align::MatchReport report;
    align::GroupMatch gm;
    gm.group = "g0";
    gm.permutation = random_perm(kG0, 3);  // align still reports SOME permutation
    gm.identity = false;
    gm.alignable = true;
    report.groups["g0"] = gm;

    FakeBlockStore blocks;
    auto base_oid = blocks.put(core::ObjectKind::Raw, base_bytes);
    REQUIRE(base_oid.has_value());
    std::unordered_map<std::string, store::ParentTensorInfo> parent_info;
    parent_info["0.bias"] = store::ParentTensorInfo{core::Oid{}, 0};

    core::RepoConfig cfg;
    auto entries = store::plan_commit_groups(base_src, target_src, topo, report, parent_info,
                                             blocks, cfg);
    REQUIRE(entries.has_value());
    CHECK(entries->at("0.bias").mode == format::GroupMode::Full);
}

TEST_CASE("plan_commit_groups: a secondary dependency that can't be recorded forces "
         "Full, and cascades",
         "[integration][byte_identity]") {
    // Same 3-layer shape as the main test, but "0.weight"/"0.bias" (g0's
    // owners) have NO parent info -- forcing g0 to FullNoBase even though
    // align found a perfectly good permutation for it. "2.weight" (g2)
    // secondarily depends on g0, so it must cascade to Full too: g0's
    // permutation would otherwise be computed, used to build 2.weight's
    // residual, and never recorded anywhere a reader could find it.
    constexpr std::uint32_t kIn = 48, kG0 = 64, kG2 = 32, kOut = 10;
    std::mt19937_64 rng(321);

    auto w0_base = random_bytes(std::uint64_t(kG0) * kIn * 2, rng);
    auto b0_base = random_bytes(std::uint64_t(kG0) * 2, rng);
    auto w2_base = random_bytes(std::uint64_t(kG2) * kG0 * 2, rng);
    auto b2_base = random_bytes(std::uint64_t(kG2) * 2, rng);

    const auto perm_g0 = random_perm(kG0, 11);
    const auto perm_g2 = random_perm(kG2, 12);

    auto w0_target = gather_rows(w0_base, kG0, kIn, perm_g0);
    auto b0_target = gather_rows(b0_base, kG0, 1, perm_g0);
    auto w2_target = gather_cols(gather_rows(w2_base, kG2, kG0, perm_g2), kG2, kG0, perm_g0);
    auto b2_target = gather_rows(b2_base, kG2, 1, perm_g2);

    FakeTensorSource base_src, target_src;
    base_src.add("0.weight", {kG0, kIn}, core::DType::F16, w0_base);
    base_src.add("0.bias", {kG0}, core::DType::F16, b0_base);
    base_src.add("2.weight", {kG2, kG0}, core::DType::F16, w2_base);
    base_src.add("2.bias", {kG2}, core::DType::F16, b2_base);
    target_src.add("0.weight", {kG0, kIn}, core::DType::F16, w0_target);
    target_src.add("0.bias", {kG0}, core::DType::F16, b0_target);
    target_src.add("2.weight", {kG2, kG0}, core::DType::F16, w2_target);
    target_src.add("2.bias", {kG2}, core::DType::F16, b2_target);

    core::Topology topo;
    topo.groups["in"] = {kIn, true};
    topo.groups["g0"] = {kG0, false};
    topo.groups["g2"] = {kG2, false};
    topo.tensors["0.weight"] = {{{0, "g0", 1}, {1, "in", 1}}};
    topo.tensors["0.bias"] = {{{0, "g0", 1}}};
    topo.tensors["2.weight"] = {{{0, "g2", 1}, {1, "g0", 1}}};
    topo.tensors["2.bias"] = {{{0, "g2", 1}}};

    align::MatchReport report;
    align::GroupMatch gm_g0;
    gm_g0.group = "g0";
    gm_g0.permutation = perm_g0;
    gm_g0.identity = false;
    gm_g0.alignable = true;
    report.groups["g0"] = gm_g0;

    align::GroupMatch gm_g2 = gm_g0;
    gm_g2.group = "g2";
    gm_g2.permutation = perm_g2;
    report.groups["g2"] = gm_g2;

    FakeBlockStore blocks;
    // Deliberately NO parent_info at all: every group is has_base=false,
    // forcing FullNoBase regardless of alignment quality.
    std::unordered_map<std::string, store::ParentTensorInfo> parent_info;

    core::RepoConfig cfg;
    auto entries = store::plan_commit_groups(base_src, target_src, topo, report, parent_info,
                                             blocks, cfg);
    REQUIRE(entries.has_value());

    CHECK(entries->at("0.weight").mode == format::GroupMode::Full);
    CHECK(entries->at("0.bias").mode == format::GroupMode::Full);
    // The cascade: g2 had a perfectly good permutation and would otherwise
    // have been Delta, but its secondary dependency (g0) isn't Delta here,
    // so it must be forced to Full too.
    CHECK(entries->at("2.weight").mode == format::GroupMode::Full);
    CHECK(entries->at("2.bias").mode == format::GroupMode::Full);

    // And the Full bytes stored must still be byte-exact target bytes —
    // the whole point of the fallback.
    format::Manifest manifest;
    for (const auto& [name, entry] : *entries) manifest.groups[name] = entry;
    codec::ReadCtx ctx;
    ctx.blocks = &blocks;
    ctx.manifest = &manifest;

    std::vector<std::byte> actual(w2_target.size());
    auto r = codec::read_range(ctx, "2.weight", 0, actual);
    REQUIRE(r.has_value());
    REQUIRE(actual == w2_target);
}
