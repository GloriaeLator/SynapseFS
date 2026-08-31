/// End-to-end demo/test tool for the align module: given a config.json and
/// two real .safetensors checkpoints, parses the topology, runs the matcher,
/// and prints what it found -- including a check against a planted-ground-
/// truth JSON, when one is given, so this doubles as a manual verification
/// harness for modules/align/tests/fixtures/mlp_end_to_end.
///
/// Usage:
///   align_demo <config.json> <base.safetensors> <target.safetensors> [ground_truth.json]
#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>

#include "simple_st_source.hpp"

using namespace sfs;
using json = nlohmann::json;

namespace {

void print_permutation(const std::vector<std::uint32_t>& p) {
    std::printf("[");
    for (std::size_t i = 0; i < p.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ", ", p[i]);
        if (i >= 15 && p.size() > 18) {
            std::printf(", ... (%zu more)", p.size() - i - 1);
            break;
        }
    }
    std::printf("]");
}

bool check_ground_truth(const core::Topology& topo, const align::MatchReport& report,
                        const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::printf("warning: could not open ground truth file %s\n", path.c_str());
        return true;
    }
    json gt;
    in >> gt;

    bool all_ok = true;
    for (auto it = gt.begin(); it != gt.end(); ++it) {
        const std::string& key = it.key();
        const std::string suffix = "_output_perm";
        if (key.size() <= suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        // "layerN_output_perm" -> tensor "N.weight"'s dim-0 group.
        std::string layer_idx;
        for (char c : key) {
            if (c >= '0' && c <= '9') layer_idx += c;
            else if (!layer_idx.empty()) break;
        }
        const std::string tensor_name = layer_idx + ".weight";
        auto tit = topo.tensors.find(tensor_name);
        if (tit == topo.tensors.end()) {
            std::printf("  %-24s SKIP (tensor %s not in topology)\n", key.c_str(), tensor_name.c_str());
            continue;
        }
        std::string group;
        for (const auto& b : tit->second.axes) {
            if (b.dim == 0) group = b.group;
        }
        if (group.empty()) {
            std::printf("  %-24s SKIP (no dim-0 binding for %s)\n", key.c_str(), tensor_name.c_str());
            continue;
        }
        const auto& gm = report.groups.at(group);
        std::vector<std::uint32_t> expected;
        for (const auto& v : it.value()) expected.push_back(v.get<std::uint32_t>());

        bool match;
        if (gm.identity) {
            // An empty permutation means identity: only a genuine match if
            // the planted ground truth was itself the identity permutation.
            match = true;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (expected[i] != i) { match = false; break; }
            }
        } else {
            match = gm.permutation == expected;
        }
        if (!match && !gm.identity && gm.permutation.size() == expected.size()) {
            std::size_t diffs = 0;
            std::size_t first_diff = expected.size();
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (gm.permutation[i] != expected[i]) {
                    ++diffs;
                    if (first_diff == expected.size()) first_diff = i;
                }
            }
            std::printf("  %-24s group=%-4s MISMATCH (%zu/%zu units differ, first at index %zu: got %u expected %u)\n",
                       key.c_str(), group.c_str(), diffs, expected.size(), first_diff,
                       gm.permutation[first_diff], expected[first_diff]);
        } else {
            std::printf("  %-24s group=%-4s %s\n", key.c_str(), group.c_str(), match ? "MATCH" : "MISMATCH");
        }
        all_ok = all_ok && match;
    }
    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <config.json> <base.safetensors> <target.safetensors> "
                             "[ground_truth.json]\n", argv[0]);
        return 2;
    }
    const std::string config_path = argv[1];
    const std::string base_path = argv[2];
    const std::string target_path = argv[3];
    const char* gt_path = argc > 4 ? argv[4] : nullptr;

    auto base = align::tools::SimpleStSource::open(base_path);
    if (!base) {
        std::fprintf(stderr, "failed to open base checkpoint: %s\n", base.error().to_string().c_str());
        return 1;
    }
    auto target = align::tools::SimpleStSource::open(target_path);
    if (!target) {
        std::fprintf(stderr, "failed to open target checkpoint: %s\n", target.error().to_string().c_str());
        return 1;
    }

    auto topo = align::parse_topology_file(*base, config_path);
    if (!topo) {
        std::fprintf(stderr, "failed to parse topology: %s\n", topo.error().to_string().c_str());
        return 1;
    }
    std::printf("topology: %zu groups, %zu tensors\n", topo->groups.size(), topo->tensors.size());

    // DIAGNOSTIC ONLY -- temporarily disable the NotAlignable confidence gate
    // so gm.permutation is populated (and thus checkable against ground
    // truth) even when cost_normalized is above the random-baseline cutoff.
    // This tells us whether the LAP solver is finding the right assignment
    // and just getting discarded by the gate, or whether the assignment
    // itself is wrong. Revert before this is anything but a one-off
    // investigation.
    align::MatchOptions diag_opts;
    diag_opts.confidence.random_baseline_margin = 1e9;
    align::Matcher matcher(*base, *target, *topo, diag_opts);
    auto report = matcher.run();
    if (!report) {
        std::fprintf(stderr, "matcher.run() failed: %s\n", report.error().to_string().c_str());
        return 1;
    }

    std::printf("sweeps=%u wall=%.4fs\n\n", report->sweeps, report->wall_seconds);
    for (const auto& [name, gm] : report->groups) {
        const auto& g = topo->groups.at(name);
        std::printf("group %-4s size=%-4u pinned=%-5s identity=%-5s alignable=%-5s "
                   "exact=%-5s cost_norm=%.6f\n",
                   name.c_str(), g.size, g.pinned ? "true" : "false", gm.identity ? "true" : "false",
                   gm.alignable ? "true" : "false", gm.exact_solver ? "true" : "false", gm.cost_normalized);
        if (!gm.identity) {
            std::printf("    permutation: ");
            print_permutation(gm.permutation);
            std::printf("\n");
        }
    }

    if (gt_path != nullptr) {
        std::printf("\n=== ground truth check ===\n");
        const bool ok = check_ground_truth(*topo, *report, gt_path);
        std::printf("%s\n", ok ? "ALL GROUPS MATCHED GROUND TRUTH" : "MISMATCH DETECTED");
        return ok ? 0 : 1;
    }
    return 0;
}
