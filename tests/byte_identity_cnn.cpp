/// Regression test for a real bug: the secondary-axis (multi-axis
/// permutation) machinery was only ever built and tested against rank-2
/// (Linear) tensors. A rank-4 conv weight ([out_c, in_c, kh, kw]) whose
/// in-channel axis (dim 1, NOT the last dimension) depends on a different,
/// non-pinned group needs each in-channel's whole [kh, kw] block moved
/// together — not per-scalar reordering. This was caught by actually
/// running a two-conv-layer CNN through the pipeline, which failed with
/// "ColumnPermutingSource: short read from inner source" before the fix in
/// commit_planner.cpp / reconstruct.cpp (trailing-dimension awareness).
///
/// Real align::topology_parser runs here (needs no Torch — only the sparse
/// large-group matching path does); align::Matcher itself is still avoided,
/// same as tests/byte_identity.cpp.

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <map>
#include <numeric>
#include <random>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>
#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/commit_planner.hpp>

using namespace sfs;

namespace {

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
    core::Result<std::uint64_t> size_of(const core::Oid& oid) const override { return objs_.at(oid).size(); }
    core::Result<core::ObjectKind> kind_of(const core::Oid&) const override { return core::ObjectKind::Raw; }

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

std::vector<std::byte> gather_rows_blocked(const std::vector<std::byte>& src, std::uint32_t rows,
                                           std::uint32_t row_bytes,
                                           const std::vector<std::uint32_t>& perm) {
    std::vector<std::byte> out(src.size());
    for (std::uint32_t i = 0; i < rows; ++i)
        std::memcpy(out.data() + i * row_bytes, src.data() + perm[i] * row_bytes, row_bytes);
    return out;
}

}  // namespace

TEST_CASE("two-conv CNN: a rank-4 conv weight with a non-pinned secondary axis "
         "round-trips byte-exact",
         "[integration][byte_identity][cnn]") {
    // conv0(3->kG0,k3,pad1) -> bn0 -> relu -> conv1(kG0->kG1,k3,pad1) -> bn1
    // -> relu -> maxpool(2) -> flatten -> linear(96->5). nn.Sequential
    // indices: 0=conv0 1=bn0 2=relu 3=conv1 4=bn1 5=relu 6=maxpool 7=flatten
    // 8=linear. "3.weight" (shape [kG1, kG0, 3, 3]) is the case: dim 0 owns
    // its own group, dim 1 (in-channels) depends on conv0's group -- and
    // dim 1 is NOT the last dimension (kh, kw follow it).
    constexpr std::uint32_t kIn = 3, kG0 = 64, kG1 = 96, kOut = 5, kK = 3;
    constexpr std::uint32_t kFlat = kG1 * 4 * 4;  // after maxpool(2) on 8x8 -> 4x4

    FakeTensorSource meta_src;  // shapes only, for topology parsing
    meta_src.add("0.weight", {kG0, kIn, kK, kK}, core::DType::F16,
                std::vector<std::byte>(std::uint64_t(kG0) * kIn * kK * kK * 2));
    meta_src.add("0.bias", {kG0}, core::DType::F16, std::vector<std::byte>(kG0 * 2));
    meta_src.add("1.weight", {kG0}, core::DType::F16, std::vector<std::byte>(kG0 * 2));
    meta_src.add("1.bias", {kG0}, core::DType::F16, std::vector<std::byte>(kG0 * 2));
    meta_src.add("1.running_mean", {kG0}, core::DType::F16, std::vector<std::byte>(kG0 * 2));
    meta_src.add("1.running_var", {kG0}, core::DType::F16, std::vector<std::byte>(kG0 * 2));
    meta_src.add("3.weight", {kG1, kG0, kK, kK}, core::DType::F16,
                std::vector<std::byte>(std::uint64_t(kG1) * kG0 * kK * kK * 2));
    meta_src.add("3.bias", {kG1}, core::DType::F16, std::vector<std::byte>(kG1 * 2));
    meta_src.add("4.weight", {kG1}, core::DType::F16, std::vector<std::byte>(kG1 * 2));
    meta_src.add("4.bias", {kG1}, core::DType::F16, std::vector<std::byte>(kG1 * 2));
    meta_src.add("4.running_mean", {kG1}, core::DType::F16, std::vector<std::byte>(kG1 * 2));
    meta_src.add("4.running_var", {kG1}, core::DType::F16, std::vector<std::byte>(kG1 * 2));
    meta_src.add("8.weight", {kOut, kFlat}, core::DType::F16,
                std::vector<std::byte>(std::uint64_t(kOut) * kFlat * 2));
    meta_src.add("8.bias", {kOut}, core::DType::F16, std::vector<std::byte>(kOut * 2));

    const char* config_json = R"({
      "input_shape": [3, 8, 8],
      "layers": [
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "batchnorm2d"},
        {"type": "relu"},
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "batchnorm2d"},
        {"type": "relu"},
        {"type": "maxpool2d", "kernel_size": 2},
        {"type": "flatten"},
        {"type": "linear"}
      ]
    })";
    std::span<const std::byte> cfg_bytes(reinterpret_cast<const std::byte*>(config_json),
                                        strlen(config_json));

    auto topo_r = align::parse_topology(meta_src, cfg_bytes, align::ParseOptions{});
    REQUIRE(topo_r.has_value());
    auto topo = *topo_r;

    // Real topology_parser assigns arbitrary group names; find which is which.
    std::string conv0_group, conv1_group;
    for (const auto& ax : topo.tensors.at("0.weight").axes)
        if (ax.dim == 0) conv0_group = ax.group;
    for (const auto& ax : topo.tensors.at("3.weight").axes)
        if (ax.dim == 0) conv1_group = ax.group;
    REQUIRE_FALSE(conv0_group.empty());
    REQUIRE_FALSE(conv1_group.empty());
    // Confirms the union-find actually chained conv1's input axis to conv0's
    // output group across the intervening BatchNorm/ReLU layers.
    bool conv1_depends_on_conv0 = false;
    for (const auto& ax : topo.tensors.at("3.weight").axes)
        if (ax.dim != 0 && ax.group == conv0_group) conv1_depends_on_conv0 = true;
    REQUIRE(conv1_depends_on_conv0);

    std::mt19937_64 rng(42);
    auto w0_base = random_bytes(std::uint64_t(kG0) * kIn * kK * kK * 2, rng);
    auto b0_base = random_bytes(std::uint64_t(kG0) * 2, rng);
    auto w3_base = random_bytes(std::uint64_t(kG1) * kG0 * kK * kK * 2, rng);
    auto b3_base = random_bytes(std::uint64_t(kG1) * 2, rng);
    auto bn0_w_base = random_bytes(kG0 * 2, rng), bn0_b_base = random_bytes(kG0 * 2, rng);
    auto bn0_rm_base = random_bytes(kG0 * 2, rng), bn0_rv_base = random_bytes(kG0 * 2, rng);
    auto bn1_w_base = random_bytes(kG1 * 2, rng), bn1_b_base = random_bytes(kG1 * 2, rng);
    auto bn1_rm_base = random_bytes(kG1 * 2, rng), bn1_rv_base = random_bytes(kG1 * 2, rng);

    const auto perm_g0 = random_perm(kG0, 1);
    const auto perm_g1 = random_perm(kG1, 2);

    auto w0_target = gather_rows_blocked(w0_base, kG0, kIn * kK * kK * 2, perm_g0);
    auto b0_target = gather_rows_blocked(b0_base, kG0, 2, perm_g0);

    // 3.weight ground truth: out[i,j,:,:] = base[perm_g1[i], perm_g0[j], :, :]
    // -- each in-channel's WHOLE [kh,kw] block moves together, not scalars.
    const std::uint32_t block_bytes = kK * kK * 2;
    const std::uint32_t row_bytes_3 = kG0 * block_bytes;
    std::vector<std::byte> w3_target(w3_base.size());
    for (std::uint32_t i = 0; i < kG1; ++i) {
        for (std::uint32_t j = 0; j < kG0; ++j) {
            const std::byte* src = w3_base.data() + perm_g1[i] * row_bytes_3 + perm_g0[j] * block_bytes;
            std::byte* dst = w3_target.data() + i * row_bytes_3 + j * block_bytes;
            std::memcpy(dst, src, block_bytes);
        }
    }
    auto b3_target = gather_rows_blocked(b3_base, kG1, 2, perm_g1);

    FakeTensorSource base_src, target_src;
    base_src.add("0.weight", {kG0, kIn, kK, kK}, core::DType::F16, w0_base);
    base_src.add("0.bias", {kG0}, core::DType::F16, b0_base);
    base_src.add("1.weight", {kG0}, core::DType::F16, bn0_w_base);
    base_src.add("1.bias", {kG0}, core::DType::F16, bn0_b_base);
    base_src.add("1.running_mean", {kG0}, core::DType::F16, bn0_rm_base);
    base_src.add("1.running_var", {kG0}, core::DType::F16, bn0_rv_base);
    base_src.add("3.weight", {kG1, kG0, kK, kK}, core::DType::F16, w3_base);
    base_src.add("3.bias", {kG1}, core::DType::F16, b3_base);
    base_src.add("4.weight", {kG1}, core::DType::F16, bn1_w_base);
    base_src.add("4.bias", {kG1}, core::DType::F16, bn1_b_base);
    base_src.add("4.running_mean", {kG1}, core::DType::F16, bn1_rm_base);
    base_src.add("4.running_var", {kG1}, core::DType::F16, bn1_rv_base);

    target_src.add("0.weight", {kG0, kIn, kK, kK}, core::DType::F16, w0_target);
    target_src.add("0.bias", {kG0}, core::DType::F16, b0_target);
    target_src.add("1.weight", {kG0}, core::DType::F16, gather_rows_blocked(bn0_w_base, kG0, 2, perm_g0));
    target_src.add("1.bias", {kG0}, core::DType::F16, gather_rows_blocked(bn0_b_base, kG0, 2, perm_g0));
    target_src.add("1.running_mean", {kG0}, core::DType::F16,
                   gather_rows_blocked(bn0_rm_base, kG0, 2, perm_g0));
    target_src.add("1.running_var", {kG0}, core::DType::F16,
                   gather_rows_blocked(bn0_rv_base, kG0, 2, perm_g0));
    target_src.add("3.weight", {kG1, kG0, kK, kK}, core::DType::F16, w3_target);
    target_src.add("3.bias", {kG1}, core::DType::F16, b3_target);
    target_src.add("4.weight", {kG1}, core::DType::F16, gather_rows_blocked(bn1_w_base, kG1, 2, perm_g1));
    target_src.add("4.bias", {kG1}, core::DType::F16, gather_rows_blocked(bn1_b_base, kG1, 2, perm_g1));
    target_src.add("4.running_mean", {kG1}, core::DType::F16,
                   gather_rows_blocked(bn1_rm_base, kG1, 2, perm_g1));
    target_src.add("4.running_var", {kG1}, core::DType::F16,
                   gather_rows_blocked(bn1_rv_base, kG1, 2, perm_g1));

    align::MatchReport report;
    align::GroupMatch gm0;
    gm0.group = conv0_group;
    gm0.permutation = perm_g0;
    gm0.identity = false;
    gm0.alignable = true;
    report.groups[conv0_group] = gm0;

    align::GroupMatch gm1 = gm0;
    gm1.group = conv1_group;
    gm1.permutation = perm_g1;
    report.groups[conv1_group] = gm1;

    FakeBlockStore blocks;
    format::Manifest manifest_a;
    core::Oid commit_a{};
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

    core::RepoConfig cfg;
    auto entries = store::plan_commit_groups(base_src, target_src, topo, report, parent_info,
                                             blocks, cfg);
    REQUIRE(entries.has_value());
    // The whole point: "3.weight" must be a real Delta, not a Full fallback
    // hiding a broken multi-axis path.
    CHECK(entries->at("3.weight").mode == format::GroupMode::Delta);

    format::Manifest manifest_b;
    for (const auto& [name, e] : *entries) manifest_b.groups[name] = e;
    FakeObjectSource history;
    history.add(commit_a, &manifest_a);

    codec::ReadCtx ctx;
    ctx.blocks = &blocks;
    ctx.manifest = &manifest_b;
    ctx.history = &history;
    ctx.topology = &topo;
    ctx.max_depth = cfg.max_chain_depth;

    std::vector<std::byte> actual(w3_target.size());
    auto r = codec::read_range(ctx, "3.weight", 0, actual);
    REQUIRE(r.has_value());
    REQUIRE(*r == w3_target.size());
    REQUIRE(actual == w3_target);
}
