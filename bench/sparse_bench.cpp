/// bench/sparse_bench.cpp — docs/benchmarks.md §1's sparse-path scaling
/// table: the one method in align::Matcher's routing with no benchmark data
/// anywhere else in this repo.
///
/// align::Matcher routes any group of size n >= sparse_crossover (8192,
/// default) to match_group_sparse (fingerprint + Jacobi auction,
/// modules/align/src/sparse_match.cpp) instead of the dense LAP path
/// bench/lap_bench.cpp measures. Unlike the LAP solvers, match_group_sparse
/// takes real tensor sources, not a plain cost matrix, so this can't be
/// synthesized as cheaply as lap_bench.cpp's random matrices -- it runs
/// against real (if synthetic) checkpoints from bench/scripts/gen_sparse_scale.py,
/// one per size, each a two-hidden-layer MLP whose FIRST hidden layer is the
/// size under test (always >= 8192, so it always takes the sparse path) and
/// whose second is a fixed, tiny 8 -- present only so the fixture's shape
/// matches what align::Matcher would see in practice, not because this
/// benchmark times it.
///
/// Calls Matcher::match_group() directly on just the first hidden layer's
/// group, timed by hand -- MatchReport::wall_seconds covers a whole run()
/// (every group), which would blend the sparse group's cost with the
/// second, trivially-fast dense group's.
///
/// Usage:
///   sparse_bench [--dir fixtures/out/sparse_scale] [--sizes 8192,16384,32768,65536] [--json]
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>

#include "simple_st_source.hpp"

using namespace sfs;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string dim0_group(const core::Topology& topo, const std::string& tensor) {
    auto it = topo.tensors.find(tensor);
    if (it == topo.tensors.end()) return {};
    for (const auto& b : it->second.axes) {
        if (b.dim == 0) return b.group;
    }
    return {};
}

struct SizeResult {
    std::uint32_t n = 0;
    bool ok = false;
    std::string error;
    double wall_seconds = 0.0;
    bool identity = false;
    bool alignable = false;
    bool exact_solver = false;
    std::size_t correct = 0;
    std::size_t total = 0;
};

SizeResult run_size(const std::string& dir, std::uint32_t n) {
    SizeResult r;
    r.n = n;

    auto base = align::tools::SimpleStSource::open(dir + "/base.safetensors");
    if (!base) { r.error = "open base: " + base.error().to_string(); return r; }
    auto target = align::tools::SimpleStSource::open(dir + "/target.safetensors");
    if (!target) { r.error = "open target: " + target.error().to_string(); return r; }

    auto topo = align::parse_topology_file(*base, dir + "/config.json");
    if (!topo) { r.error = "parse topology: " + topo.error().to_string(); return r; }

    const std::string group = dim0_group(*topo, "0.weight");
    if (group.empty()) { r.error = "no dim-0 binding for 0.weight"; return r; }

    align::Matcher matcher(*base, *target, *topo);
    const auto t0 = std::chrono::steady_clock::now();
    auto gm = matcher.match_group(group);
    const auto t1 = std::chrono::steady_clock::now();
    if (!gm) { r.error = "match_group: " + gm.error().to_string(); return r; }

    r.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
    r.identity     = gm->identity;
    r.alignable    = gm->alignable;
    r.exact_solver = gm->exact_solver;

    std::ifstream gt_file(dir + "/ground_truth.json");
    if (gt_file) {
        json gt;
        gt_file >> gt;
        std::vector<std::uint32_t> expected;
        for (const auto& v : gt.at("layer0_output_perm")) expected.push_back(v.get<std::uint32_t>());
        std::vector<std::uint32_t> actual;
        if (!gm->identity && gm->permutation.size() == expected.size()) {
            actual = gm->permutation;
        } else {
            actual.resize(expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = static_cast<std::uint32_t>(i);
        }
        r.total = expected.size();
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (actual[i] == expected[i]) ++r.correct;
        }
    }

    r.ok = true;
    return r;
}

std::vector<std::uint32_t> parse_sizes(const std::string& csv) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
    }
    return out;
}

void print_table(const std::vector<SizeResult>& results) {
    std::cout << "| n | Wall-clock | Identity | Alignable | Exact solver | Accuracy |\n";
    std::cout << "|---|---|---|---|---|---|\n";
    for (const auto& r : results) {
        std::cout << "| " << r.n << " | ";
        if (!r.ok) {
            std::cout << "ERROR: " << r.error << " |\n";
            continue;
        }
        std::cout << (r.wall_seconds * 1000.0) << " ms | "
                  << (r.identity ? "true" : "false") << " | "
                  << (r.alignable ? "true" : "false") << " | "
                  << (r.exact_solver ? "true" : "false") << " | ";
        if (r.total > 0) {
            std::cout << (100.0 * r.correct / r.total) << "% (" << r.correct << "/" << r.total << ")";
        } else {
            std::cout << "n/a";
        }
        std::cout << " |\n";
    }
}

json to_json(const std::vector<SizeResult>& results) {
    json arr = json::array();
    for (const auto& r : results) {
        json j = {{"n", r.n}, {"ok", r.ok}};
        if (!r.ok) {
            j["error"] = r.error;
            arr.push_back(j);
            continue;
        }
        j["wall_seconds"] = r.wall_seconds;
        j["identity"]     = r.identity;
        j["alignable"]    = r.alignable;
        j["exact_solver"] = r.exact_solver;
        if (r.total > 0) {
            j["accuracy"]               = static_cast<double>(r.correct) / static_cast<double>(r.total);
            j["accuracy_units_correct"] = r.correct;
            j["accuracy_units_total"]   = r.total;
        } else {
            j["accuracy"] = nullptr;
        }
        arr.push_back(j);
    }
    return arr;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "fixtures/out/sparse_scale";
    std::vector<std::uint32_t> sizes = {8192, 16384, 32768, 65536};
    bool as_json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") as_json = true;
        else if (arg == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (arg == "--sizes" && i + 1 < argc) sizes = parse_sizes(argv[++i]);
    }

    std::vector<SizeResult> results;
    for (const auto n : sizes) {
        const std::string size_dir = dir + "/n" + std::to_string(n);
        if (!fs::exists(size_dir + "/base.safetensors")) {
            std::cerr << "skipping n=" << n << ": " << size_dir
                     << " not found (run bench/scripts/gen_sparse_scale.py first)\n";
            continue;
        }
        results.push_back(run_size(size_dir, n));
    }

    if (results.empty()) {
        std::cerr << "no sparse_scale fixtures found under " << dir << "\n";
        return 1;
    }

    if (as_json) {
        std::cout << to_json(results).dump(2) << "\n";
    } else {
        print_table(results);
    }

    for (const auto& r : results) {
        if (!r.ok) return 1;
    }
    return 0;
}
