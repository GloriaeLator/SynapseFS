/// bench/align_time.cpp — docs/benchmarks.md §1's wall-clock/accuracy table.
///
/// Runs align::Matcher against a real base/target checkpoint pair under a
/// real config.json topology, and reports what MatchReport already computes
/// (sweeps, wall_seconds, peak_bytes) plus, when a ground-truth permutation
/// file is given, per-unit recovery accuracy against the planted
/// permutation -- the same "<layer>_output_perm" convention
/// modules/align/tests/fixtures/generate.py writes and
/// modules/align/tools/align_demo.cpp already checks (reused here, turned
/// into a fraction instead of a pass/fail).
///
/// Usage:
///   align_time [--json]
///     Runs the fixed set of fixtures docs/benchmarks.md §1 reports:
///     modules/align/tests/fixtures/{mlp_end_to_end,mlp_large_sparse} (both
///     have planted ground truth; the latter exercises the sparse-path
///     crossover at layer0's size 8192) plus fixtures/out/mlp's real
///     fine-tune pair (no ground truth -- accuracy is n/a there by
///     definition, per docs/benchmarks.md §1's own note).
///   align_time --base B --target T --topology C [--ground-truth G]
///              [--label L] [--json]
///     Runs one ad-hoc fixture instead of the fixed set.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>

#include "simple_st_source.hpp"

using namespace sfs;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct FixtureSpec {
    std::string label;
    std::string base, target, topology;
    std::optional<std::string> ground_truth;
};

struct FixtureResult {
    std::string label;
    bool ok = false;
    std::string error;
    std::uint64_t params = 0;
    std::size_t   groups = 0;
    std::uint32_t sweeps = 0;
    double        wall_seconds = 0.0;
    std::uint64_t peak_bytes = 0;
    std::optional<double> accuracy;  // fraction of units correctly recovered
    std::size_t accuracy_correct = 0;
    std::size_t accuracy_total = 0;
};

std::string dim0_group(const core::Topology& topo, const std::string& tensor) {
    auto it = topo.tensors.find(tensor);
    if (it == topo.tensors.end()) return {};
    for (const auto& b : it->second.axes) {
        if (b.dim == 0) return b.group;
    }
    return {};
}

// GroupMatch::permutation is empty for an identity match (matcher.hpp's own
// convention) -- expand that to a real [0..size) array so it lines up
// index-for-index against a planted ground-truth permutation.
std::vector<std::uint32_t> materialize(const align::GroupMatch& gm, std::size_t size) {
    if (!gm.identity && gm.permutation.size() == size) return gm.permutation;
    std::vector<std::uint32_t> id(size);
    for (std::size_t i = 0; i < size; ++i) id[i] = static_cast<std::uint32_t>(i);
    return id;
}

// Same "<layer>_output_perm" -> "<layer>.weight" -> dim-0 group convention
// modules/align/tools/align_demo.cpp's check_ground_truth() uses, but
// tallying per-unit correctness across every checked group instead of a
// single pass/fail verdict.
std::pair<std::size_t, std::size_t> score_ground_truth(const core::Topology& topo,
                                                        const align::MatchReport& report,
                                                        const std::string& path) {
    std::ifstream in(path);
    if (!in) return {0, 0};
    json gt;
    in >> gt;

    std::size_t correct = 0, total = 0;
    for (auto it = gt.begin(); it != gt.end(); ++it) {
        const std::string& key = it.key();
        const std::string suffix = "_output_perm";
        if (key.size() <= suffix.size() ||
           key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        std::string layer_idx;
        for (char c : key) {
            if (c >= '0' && c <= '9') layer_idx += c;
            else if (!layer_idx.empty()) break;
        }
        const std::string tensor_name = layer_idx + ".weight";
        const std::string group = dim0_group(topo, tensor_name);
        if (group.empty()) continue;
        auto git = report.groups.find(group);
        if (git == report.groups.end()) continue;

        std::vector<std::uint32_t> expected;
        for (const auto& v : it.value()) expected.push_back(v.get<std::uint32_t>());

        const auto actual = materialize(git->second, expected.size());
        if (actual.size() != expected.size()) {
            total += expected.size();  // size mismatch: none of it counts as recovered
            continue;
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            ++total;
            if (actual[i] == expected[i]) ++correct;
        }
    }
    return {correct, total};
}

FixtureResult run_fixture(const FixtureSpec& spec) {
    FixtureResult r;
    r.label = spec.label;

    auto base = align::tools::SimpleStSource::open(spec.base);
    if (!base) { r.error = "open base: " + base.error().to_string(); return r; }
    auto target = align::tools::SimpleStSource::open(spec.target);
    if (!target) { r.error = "open target: " + target.error().to_string(); return r; }

    // Total parameter count: every tensor the checkpoint actually declares,
    // not just the ones topology.tensors binds an axis for (e.g. a pinned
    // tensor with no permutable axis still counts toward "Params").
    for (const auto& entry : base->buffer_layout()) {
        if (const auto* m = base->meta(entry.tensor)) r.params += m->elem_count();
    }

    auto topo = align::parse_topology_file(*base, spec.topology);
    if (!topo) { r.error = "parse topology: " + topo.error().to_string(); return r; }
    r.groups = topo->groups.size();

    align::Matcher matcher(*base, *target, *topo);
    auto report = matcher.run();
    if (!report) { r.error = "matcher.run(): " + report.error().to_string(); return r; }

    r.sweeps       = report->sweeps;
    r.wall_seconds = report->wall_seconds;
    r.peak_bytes   = report->peak_bytes;

    if (spec.ground_truth) {
        auto [correct, total] = score_ground_truth(*topo, *report, *spec.ground_truth);
        r.accuracy_correct = correct;
        r.accuracy_total   = total;
        if (total > 0) r.accuracy = static_cast<double>(correct) / static_cast<double>(total);
    }

    r.ok = true;
    return r;
}

std::vector<FixtureSpec> default_fixtures() {
    return {
        {"MLP, dense path (planted perm)",
        "modules/align/tests/fixtures/mlp_end_to_end/base.safetensors",
        "modules/align/tests/fixtures/mlp_end_to_end/target.safetensors",
        "modules/align/tests/fixtures/mlp_end_to_end/config.json",
        "modules/align/tests/fixtures/mlp_end_to_end/ground_truth.json"},
        {"MLP, sparse+dense mix (planted perm, layer0 size 8192)",
        "modules/align/tests/fixtures/mlp_large_sparse/base.safetensors",
        "modules/align/tests/fixtures/mlp_large_sparse/target.safetensors",
        "modules/align/tests/fixtures/mlp_large_sparse/config.json",
        "modules/align/tests/fixtures/mlp_large_sparse/ground_truth.json"},
        {"MLP fine-tune, mid (~919k params)",
        "fixtures/out/mlp_step0.safetensors",
        "fixtures/out/mlp_step1.safetensors",
        "fixtures/out/mlp_layers_config.json",
        std::nullopt},
    };
}

std::string human_bytes(std::uint64_t b) {
    char buf[64];
    if (b >= (1ull << 30))      std::snprintf(buf, sizeof buf, "%.2f GiB", b / double(1ull << 30));
    else if (b >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.2f MiB", b / double(1ull << 20));
    else                        std::snprintf(buf, sizeof buf, "%.2f KiB", b / double(1ull << 10));
    return buf;
}

void print_table(const std::vector<FixtureResult>& results) {
    std::cout << "| Fixture | Params | Groups | Sweeps | Wall-clock | Peak RSS | Accuracy |\n";
    std::cout << "|---|---|---|---|---|---|---|\n";
    for (const auto& r : results) {
        std::cout << "| " << r.label << " | ";
        if (!r.ok) {
            std::cout << "ERROR: " << r.error << " |\n";
            continue;
        }
        std::cout << r.params << " | " << r.groups << " | " << r.sweeps << " | "
                  << (r.wall_seconds * 1000.0) << " ms | " << human_bytes(r.peak_bytes) << " | ";
        if (r.accuracy) {
            std::cout << (100.0 * *r.accuracy) << "% (" << r.accuracy_correct << "/"
                      << r.accuracy_total << ") |\n";
        } else {
            std::cout << "n/a |\n";
        }
    }
}

json to_json(const std::vector<FixtureResult>& results) {
    json arr = json::array();
    for (const auto& r : results) {
        json j = {{"fixture", r.label}, {"ok", r.ok}};
        if (!r.ok) {
            j["error"] = r.error;
            arr.push_back(j);
            continue;
        }
        j["params"]       = r.params;
        j["groups"]       = r.groups;
        j["sweeps"]       = r.sweeps;
        j["wall_seconds"] = r.wall_seconds;
        j["peak_bytes"]   = r.peak_bytes;
        if (r.accuracy) {
            j["accuracy"]               = *r.accuracy;
            j["accuracy_units_correct"] = r.accuracy_correct;
            j["accuracy_units_total"]   = r.accuracy_total;
        } else {
            j["accuracy"] = nullptr;
        }
        arr.push_back(j);
    }
    return arr;
}

// ------------------------------------------------------------------ CLI

struct Args {
    bool json = false;
    std::optional<std::string> base, target, topology, ground_truth, label;
};

std::optional<std::string> take_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) return std::nullopt;
    return std::string(argv[++i]);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") a.json = true;
        else if (arg == "--base") a.base = take_value(argc, argv, i);
        else if (arg == "--target") a.target = take_value(argc, argv, i);
        else if (arg == "--topology") a.topology = take_value(argc, argv, i);
        else if (arg == "--ground-truth") a.ground_truth = take_value(argc, argv, i);
        else if (arg == "--label") a.label = take_value(argc, argv, i);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::vector<FixtureSpec> specs;
    if (args.base && args.target && args.topology) {
        specs.push_back({args.label.value_or("fixture"), *args.base, *args.target, *args.topology,
                        args.ground_truth});
    } else {
        for (auto& spec : default_fixtures()) {
            if (fs::exists(spec.base) && fs::exists(spec.target) && fs::exists(spec.topology)) {
                specs.push_back(std::move(spec));
            } else {
                std::cerr << "skipping " << spec.label << ": fixture files not found (expected "
                         << spec.base << ")\n";
            }
        }
        if (specs.empty()) {
            std::cerr << "no fixtures found -- run "
                        "modules/align/tests/fixtures/generate.py and fixtures/gen_mlp.py first\n";
            return 1;
        }
    }

    std::vector<FixtureResult> results;
    for (const auto& spec : specs) results.push_back(run_fixture(spec));

    if (args.json) {
        std::cout << to_json(results).dump(2) << "\n";
    } else {
        print_table(results);
    }

    for (const auto& r : results) {
        if (!r.ok) return 1;
    }
    return 0;
}
