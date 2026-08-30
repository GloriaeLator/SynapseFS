/// Plants a known permutation on a 2-linear-layer MLP's hidden units and
/// checks Matcher::run() recovers it exactly, per docs/alignment_algorithm.md's
/// "how we know it works" table. Runtime-verified (not just compiled) during
/// development: with poorly-chosen fixture data (every row pointing the same
/// direction after L2 normalisation) this test caught a real sign-convention
/// bug in confidence::assess for the default NegInnerProduct metric, since
/// fixed there.
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/core/tensor.hpp>
#include <synapsefs/core/topology.hpp>

using namespace sfs;

namespace {

class FakeSource final : public core::ITensorSource {
public:
    void add(const std::string& name, std::vector<std::uint64_t> shape, std::vector<float> data) {
        core::TensorMeta meta;
        meta.shape_owner = name;
        meta.shape = std::move(shape);
        meta.dtype = core::DType::F32;
        meta.data_off = 0;
        meta.nbytes = data.size() * sizeof(float);
        metas_[name] = meta;
        data_[name] = std::move(data);
    }

    std::span<const std::byte> header_bytes() const override { return {}; }
    std::span<const core::BufferEntry> buffer_layout() const override { return {}; }
    const core::TensorMeta* meta(std::string_view name) const override {
        auto it = metas_.find(std::string(name));
        return it == metas_.end() ? nullptr : &it->second;
    }
    std::uint64_t total_bytes() const override { return 0; }

    core::Result<std::size_t> read_units(std::string_view name, std::uint64_t first, std::uint64_t count,
                                         std::span<std::byte> out) override {
        const auto& d = data_.at(std::string(name));
        const auto& m = metas_.at(std::string(name));
        const std::uint64_t row_elems = m.elem_count() / m.shape[0];
        const std::size_t n_bytes = count * row_elems * sizeof(float);
        std::memcpy(out.data(), d.data() + first * row_elems, n_bytes);
        return n_bytes;
    }

private:
    std::map<std::string, core::TensorMeta> metas_;
    std::map<std::string, std::vector<float>> data_;
};

}  // namespace

TEST_CASE("Matcher recovers a planted permutation on a two-layer MLP", "[align][matcher]") {
    // Hidden layer: 4 units. Input: 3. Output (classifier): 2, pinned.
    const std::uint32_t hidden = 4, in = 3, out = 2;

    // Each hidden unit gets a genuinely distinct direction (mixed signs, not
    // a shared monotonic trend) so identity is clearly a bad match and the
    // planted permutation is clearly the best one -- rows that all point the
    // same way after L2 normalisation can't discriminate between units.
    std::vector<float> w1_base(hidden * in), b1_base(hidden), w2_base(out * hidden), b2_base(out);
    for (std::uint32_t i = 0; i < hidden; ++i) {
        for (std::uint32_t k = 0; k < in; ++k) {
            const float sign = ((i + k) % 2 == 0) ? 1.0F : -1.0F;
            w1_base[i * in + k] = sign * static_cast<float>((i + 1) * 3 + k * 7 % 5);
        }
        b1_base[i] = (i % 2 == 0 ? 1.0F : -1.0F) * static_cast<float>(i + 1) * 1.7F;
    }
    for (std::uint32_t o = 0; o < out; ++o) {
        for (std::uint32_t i = 0; i < hidden; ++i) {
            const float sign = ((o + i) % 2 == 0) ? -1.0F : 1.0F;
            w2_base[o * hidden + i] = sign * static_cast<float>((o + 1) * 5 + i * 3 % 7);
        }
    }
    for (std::uint32_t i = 0; i < b2_base.size(); ++i) b2_base[i] = static_cast<float>(i + 1) * 0.71F;

    // Ground-truth permutation: target's hidden unit i contains base unit
    // perm[i]'s data verbatim -- a gather, matching lap.cpp's convention
    // (assignment[target_row] = base_col it best matches).
    const std::vector<std::uint32_t> perm = {2, 0, 3, 1};

    std::vector<float> w1_tgt(hidden * in), b1_tgt(hidden), w2_tgt(out * hidden);
    for (std::uint32_t i = 0; i < hidden; ++i) {
        for (std::uint32_t k = 0; k < in; ++k) w1_tgt[i * in + k] = w1_base[perm[i] * in + k];
        b1_tgt[i] = b1_base[perm[i]];
        for (std::uint32_t o = 0; o < out; ++o) w2_tgt[o * hidden + i] = w2_base[o * hidden + perm[i]];
    }
    std::vector<float> b2_tgt = b2_base;  // pinned group, untouched

    FakeSource base, target;
    base.add("w1", {hidden, in}, w1_base);
    base.add("b1", {hidden}, b1_base);
    base.add("w2", {out, hidden}, w2_base);
    base.add("b2", {out}, b2_base);
    target.add("w1", {hidden, in}, w1_tgt);
    target.add("b1", {hidden}, b1_tgt);
    target.add("w2", {out, hidden}, w2_tgt);
    target.add("b2", {out}, b2_tgt);

    core::Topology topo;
    topo.groups["g0"] = core::PermGroup{hidden, /*pinned=*/false};
    topo.groups["g1"] = core::PermGroup{out, /*pinned=*/true};
    topo.tensors["w1"].axes.push_back(core::AxisBinding{0, "g0", 1});
    topo.tensors["b1"].axes.push_back(core::AxisBinding{0, "g0", 1});
    topo.tensors["w2"].axes.push_back(core::AxisBinding{0, "g1", 1});
    topo.tensors["w2"].axes.push_back(core::AxisBinding{1, "g0", 1});
    topo.tensors["b2"].axes.push_back(core::AxisBinding{0, "g1", 1});

    align::Matcher matcher(base, target, topo);
    auto result = matcher.run();
    REQUIRE(result.has_value());

    const auto& g0 = result->groups.at("g0");
    CHECK_FALSE(g0.identity);
    REQUIRE(g0.alignable);
    REQUIRE(g0.permutation.size() == hidden);
    for (std::uint32_t i = 0; i < hidden; ++i) {
        CHECK(g0.permutation[i] == perm[i]);
    }
    CHECK(g0.cost_normalized < 0.85);

    const auto& g1 = result->groups.at("g1");
    CHECK(g1.identity);  // pinned: identity is the only legal permutation
}
