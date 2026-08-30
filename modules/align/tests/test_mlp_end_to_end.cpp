/// Runs the matcher against REAL .safetensors files (not a hand-built
/// core::Topology or in-memory fake) and checks recovery against a planted
/// ground truth, exercising topology_parser.cpp's real config.json path and
/// SimpleStSource's real safetensors header parsing together.
///
/// This is also the regression test for a real bug: matcher.cpp's outgoing-
/// evidence branch (a tensor's own dim-0 rows) never reordered a row's OTHER
/// axis (e.g. a linear layer's input columns) by that axis's already-solved
/// permutation, only incoming evidence did. A single-hidden-layer fixture
/// can't catch this (its one non-pinned group's other axis is the pinned
/// input, which is already in a canonical order), so it takes two hidden
/// layers in sequence -- exactly this fixture -- to expose it.
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>

#include "simple_st_source.hpp"

using namespace sfs;

namespace {

std::filesystem::path fixture_dir() {
    // tests run from the build tree; the fixtures live alongside this
    // source file in the repo, found via this TU's own __FILE__.
    return std::filesystem::path(__FILE__).parent_path() / "fixtures" / "mlp_end_to_end";
}

std::vector<std::uint32_t> load_perm(const nlohmann::json& gt, const std::string& key) {
    std::vector<std::uint32_t> out;
    for (const auto& v : gt.at(key)) out.push_back(v.get<std::uint32_t>());
    return out;
}

std::string dim0_group(const core::Topology& topo, const std::string& tensor) {
    for (const auto& b : topo.tensors.at(tensor).axes) {
        if (b.dim == 0) return b.group;
    }
    FAIL("no dim-0 binding for " << tensor);
    return {};
}

}  // namespace

TEST_CASE("Matcher recovers planted permutations on a real two-hidden-layer MLP checkpoint",
         "[align][matcher][e2e]") {
    const auto dir = fixture_dir();

    auto base = align::tools::SimpleStSource::open(dir / "base.safetensors");
    REQUIRE(base.has_value());
    auto target = align::tools::SimpleStSource::open(dir / "target.safetensors");
    REQUIRE(target.has_value());

    auto topo = align::parse_topology_file(*base, dir / "config.json");
    REQUIRE(topo.has_value());
    CHECK(topo->tensors.size() == 6);  // 0.{weight,bias}, 2.{weight,bias}, 4.{weight,bias}

    align::Matcher matcher(*base, *target, *topo);
    auto report = matcher.run();
    REQUIRE(report.has_value());

    std::ifstream gt_file(dir / "ground_truth.json");
    REQUIRE(gt_file.is_open());
    nlohmann::json gt;
    gt_file >> gt;

    // "0.weight" dim 0 and "2.weight" dim 0 name the groups directly --
    // avoids depending on the parser's internal "gN" numbering.
    const std::string g_layer0 = dim0_group(*topo, "0.weight");
    const std::string g_layer2 = dim0_group(*topo, "2.weight");

    const auto& m0 = report->groups.at(g_layer0);
    REQUIRE_FALSE(m0.identity);
    REQUIRE(m0.alignable);
    CHECK(m0.permutation == load_perm(gt, "layer0_output_perm"));

    const auto& m2 = report->groups.at(g_layer2);
    REQUIRE_FALSE(m2.identity);
    REQUIRE(m2.alignable);
    CHECK(m2.permutation == load_perm(gt, "layer2_output_perm"));
}
