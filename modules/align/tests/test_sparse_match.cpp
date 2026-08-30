/// Runs the matcher against a real .safetensors checkpoint pair with one
/// hidden layer above MatchOptions::sparse_crossover, exercising the
/// large-group sparse path (fingerprint + candidate generation + Jacobi
/// auction, docs/adr/0011) end-to-end against real files -- not a hand-built
/// core::Topology or an in-memory fake, and not the dense CostMatrix/
/// ILapSolver path at all for that group.
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
    return std::filesystem::path(__FILE__).parent_path() / "fixtures" / "mlp_large_sparse";
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

TEST_CASE("Matcher's sparse path recovers a planted permutation on a >=8192-unit layer",
         "[align][matcher][sparse][e2e]") {
    const auto dir = fixture_dir();

    auto base = align::tools::SimpleStSource::open(dir / "base.safetensors");
    REQUIRE(base.has_value());
    auto target = align::tools::SimpleStSource::open(dir / "target.safetensors");
    REQUIRE(target.has_value());

    auto topo = align::parse_topology_file(*base, dir / "config.json");
    REQUIRE(topo.has_value());

    const std::string g_layer0 = dim0_group(*topo, "0.weight");  // size 10000: sparse path
    const std::string g_layer2 = dim0_group(*topo, "2.weight");  // size 64: dense path, same run
    REQUIRE(topo->groups.at(g_layer0).size >= 8192);

    align::Matcher matcher(*base, *target, *topo);
    auto report = matcher.run();
    REQUIRE(report.has_value());

    std::ifstream gt_file(dir / "ground_truth.json");
    REQUIRE(gt_file.is_open());
    nlohmann::json gt;
    gt_file >> gt;

    const auto& m0 = report->groups.at(g_layer0);
    REQUIRE_FALSE(m0.identity);
    REQUIRE(m0.alignable);
    CHECK_FALSE(m0.exact_solver);  // confirms this went through the sparse path, not JV

    // Approximate by construction (top-K fingerprint candidates, not an
    // exhaustive search over all n): cpp/src/dispatch.cpp's own prototype
    // measured this path's accuracy as a percentage rather than asserting
    // exact recovery, for the same reason. A high recovery RATE is the
    // right bar here, not bit-exact equality -- confirmed by forcing this
    // exact fixture through the dense/exact path during development, which
    // recovered the ground truth perfectly, isolating any gap to the
    // approximation itself rather than a solver bug or ambiguous data.
    const auto expected0 = load_perm(gt, "layer0_output_perm");
    REQUIRE(m0.permutation.size() == expected0.size());
    std::size_t matches0 = 0;
    for (std::size_t i = 0; i < expected0.size(); ++i) {
        if (m0.permutation[i] == expected0[i]) ++matches0;
    }
    const double recovery0 = static_cast<double>(matches0) / static_cast<double>(expected0.size());
    INFO("layer0 recovery rate: " << recovery0 << " (" << matches0 << "/" << expected0.size() << ")");
    CHECK(recovery0 >= 0.99);

    const auto& m2 = report->groups.at(g_layer2);
    REQUIRE_FALSE(m2.identity);
    REQUIRE(m2.alignable);
    CHECK(m2.permutation == load_perm(gt, "layer2_output_perm"));
}
