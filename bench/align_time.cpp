/// bench/align_time.cpp — docs/benchmarks.md §1's wall-clock/accuracy table,
/// and this project's own graded number for alignment.
///
/// Runs `align::Matcher::run()` end to end on real fixture checkpoint pairs,
/// auto-discovered the same way bench/residual_codec.cpp discovers its own
/// pairs: `<prefix>_step0.safetensors` + `<prefix>_step1.safetensors` under
/// --fixtures-dir (default fixtures/out) is a fine-tune pair,
/// `<prefix>_step0.safetensors` + `<prefix>_permuted.safetensors` is a
/// permutation-only pair. A pair only runs if a matching "layers"-shaped
/// config.json (SPEC 13 §2) can be found for it — `<prefix>_layers_config.json`
/// inside --fixtures-dir, which fixtures/gen_mlp.py and fixtures/gen_cnn.py
/// both write directly; other prefixes are reported as skipped rather than
/// fed an empty config, which would just hand every tensor its own pinned
/// singleton group and silently "measure" a no-op.
///
/// For a permutation-only pair, the recovered permutation is also checked
/// against permute.py's own `*_permuted.permutation.json` — that pair's
/// correct answer is known exactly, so this doubles as a regression check
/// the way modules/align/tools/align_demo.cpp's ground-truth mode does.
/// The real parser's group names ("g0", "g1", ...) don't correspond to
/// permute.py's own hand-picked names for the same axes -- both are just
/// sequential/arbitrary labels, so comparing by name directly checks
/// unrelated groups by coincidence (confirmed: an earlier version of this
/// file did exactly that, and reported spurious mismatches on *pinned*
/// groups that permute.py never even touches, while never actually
/// checking either non-pinned group at all). `bridge_group_name()` below
/// fixes this the same way bench/residual_codec.cpp's own version does:
/// match by (tensor, dim), the one thing both schemas describe identically,
/// via the `<prefix>_topology.json` sidecar next to each fixture.
///
/// LAP solver timing lives in bench/lap_bench.cpp, not here: that tool sweeps
/// up to the sparse crossover (8192) with an exact-JV cutoff matching
/// align::MatchOptions::lap_crossover's own default, which this file
/// previously duplicated with a different (smaller, less-justified) default
/// before the two were reconciled — one LAP benchmark, not two disagreeing
/// ones. The sparse fingerprint+auction path (n >= sparse_crossover) has its
/// own dedicated benchmark too: bench/sparse_bench.cpp.
///
/// Real .safetensors files are read through align::tools::SimpleStSource
/// (modules/align/tools/simple_st_source.hpp), the same mmap-backed
/// core::ITensorSource align's own end-to-end tests and align_demo.cpp use —
/// not stio::StSource, whose read_units() only supports axis-0 reads today,
/// which doesn't cover every axis align's cost-feature builder needs to
/// slice along (e.g. a Linear weight's input axis is dim 1).
///
/// Deliberately not wired through apps/sfs's CLI, for the reason
/// bench/verify_time.cpp already gives for itself: this needs only
/// core/align (plus align's own tools/ library to open a real file), not the
/// full `sfs` binary.
///
/// Deliberately one of the few bench targets that DOES link synapsefs::align
/// (and therefore Torch) — see bench/CMakeLists.txt. Every other bench
/// target includes align's headers without linking the library, since none
/// of them call compiled align:: code; this one exists only to measure that
/// compiled code, so the trade reverses for it.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/align/topology_parser.hpp>
#include <synapsefs/core/topology.hpp>

#include <simple_st_source.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace sfs;
using clk = std::chrono::steady_clock;

namespace {

// ------------------------------------------------------------------- I/O

json read_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    json j;
    f >> j;
    return j;
}

// ------------------------------------------------------- fixture discovery

struct DiscoveredPair {
    std::string name;   // e.g. "tiny_mlp_finetune"
    std::string kind;    // "finetune" | "permuted_only" | "manual"
    fs::path base;
    fs::path target;
    std::optional<fs::path> config;
    std::optional<fs::path> ground_truth;   // only ever set for "permuted_only"
    std::optional<fs::path> old_topology;   // needed to bridge ground_truth's group names (see below)
};

// `<prefix>_layers_config.json` inside --fixtures-dir: the real
// align::topology_parser "layers" schema (SPEC 13 §2), which
// fixtures/gen_mlp.py and fixtures/gen_cnn.py both write next to every
// checkpoint they generate. NOT `<prefix>_topology.json` (the older
// perm_groups/tensors sidecar permute.py and bench/residual_codec.cpp
// consume directly) — that shape predates topology_parser.cpp and this
// bench needs the real parser's own input format.
std::optional<fs::path> resolve_config(const fs::path& fixtures_dir, const std::string& prefix) {
    if (auto candidate = fixtures_dir / (prefix + "_layers_config.json"); fs::exists(candidate)) {
        return candidate;
    }
    return std::nullopt;
}

std::vector<DiscoveredPair> discover_pairs(const fs::path& dir) {
    std::vector<DiscoveredPair> out;
    if (!fs::exists(dir)) return out;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().filename().string();
        static constexpr std::string_view kSuffix = "_step0.safetensors";
        if (fname.size() <= kSuffix.size() ||
            fname.compare(fname.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
            continue;
        }
        const std::string prefix = fname.substr(0, fname.size() - kSuffix.size());

        const auto step0 = dir / (prefix + "_step0.safetensors");
        const auto step1 = dir / (prefix + "_step1.safetensors");
        const auto permuted = dir / (prefix + "_permuted.safetensors");
        const auto perm_json = dir / (prefix + "_permuted.permutation.json");
        const auto old_topo = dir / (prefix + "_topology.json");
        const auto config = resolve_config(dir, prefix);

        if (fs::exists(step1)) {
            out.push_back({prefix + "_finetune", "finetune", step0, step1, config,
                           std::nullopt, std::nullopt});
        }
        if (fs::exists(permuted)) {
            out.push_back({prefix + "_permuted_only", "permuted_only", step0, permuted, config,
                           fs::exists(perm_json) ? std::optional(perm_json) : std::nullopt,
                           fs::exists(old_topo) ? std::optional(old_topo) : std::nullopt});
        }
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    return out;
}

// ------------------------------------------------------ ground truth check

// permute.py's sidecar: {"source": "...", "groups": {group_name: [perm...]}}
// — one entry per NON-pinned group (pinned groups are never permuted, so
// they're simply absent, not present-as-identity).
struct GroundTruth {
    std::unordered_map<std::string, std::vector<std::uint32_t>> groups;
};

GroundTruth load_ground_truth(const fs::path& path) {
    GroundTruth gt;
    const json j = read_json(path);
    // Not every ground_truth.json in this repo uses this schema -- align's
    // own test fixtures (modules/align/tests/fixtures/generate.py) use a
    // different "<layer>_output_perm" convention instead (see
    // modules/align/tools/align_demo.cpp). Fail with a clear message rather
    // than an uncaught nlohmann::json exception when handed one of those.
    if (!j.contains("groups")) {
        throw std::runtime_error(path.string() +
                                 ": no top-level \"groups\" key -- this isn't permute.py's "
                                 "ground-truth schema (see load_ground_truth's comment)");
    }
    for (auto it = j.at("groups").begin(); it != j.at("groups").end(); ++it) {
        std::vector<std::uint32_t> perm;
        perm.reserve(it.value().size());
        for (const auto& v : it.value()) perm.push_back(v.get<std::uint32_t>());
        gt.groups.emplace(it.key(), std::move(perm));
    }
    return gt;
}

// align::topology_parser auto-assigns its own group names ("g0", "g1", ...
// in discovery order), which do NOT correspond to permute.py's own
// hand-picked names for the same axes (its topology_sidecar() names them
// "in"/"g0"/"g2"/"out", independently of what the real parser calls them) --
// both are just sequential labels, so "g0" on one side colliding with "g0"
// on the other is coincidence, not identity. Comparing gm.group directly
// against ground_truth's keys therefore checks arbitrary, unrelated groups
// on any fixture where the two labelings don't happen to agree, exactly the
// bug bench/residual_codec.cpp's own bridge_group_name() exists to avoid
// (see that file's top comment). Same fix here: bridge by (tensor, dim) --
// the one thing both schemas describe identically -- not by label.
std::optional<std::string> bridge_group_name(const json& old_topology, const std::string& tensor,
                                             std::uint32_t dim) {
    const auto tit = old_topology.at("tensors").find(tensor);
    if (tit == old_topology.at("tensors").end()) return std::nullopt;
    for (const auto& axis : tit.value().at("axes")) {
        if (axis.at("dim").get<std::uint32_t>() == dim) return axis.at("group").get<std::string>();
    }
    return std::nullopt;
}

// Any (tensor, dim) binding for this real group will do -- bridge_group_name
// only needs one axis that names it, and every axis bound to the same real
// group bridges to the same old-schema name by construction.
std::optional<std::pair<std::string, std::uint32_t>> any_binding_for_group(
    const core::Topology& topo, const std::string& group) {
    for (const auto& [tensor, axes] : topo.tensors) {
        for (const auto& axis : axes.axes) {
            if (axis.group == group) return std::make_pair(tensor, axis.dim);
        }
    }
    return std::nullopt;
}

// nullopt: this group has no planted ground truth to check against (it was
// pinned, so permute.py never touched it, or no old-schema topology was
// available to bridge names through). true/false: whether the recovered
// permutation matches exactly.
std::optional<bool> check_ground_truth(const core::Topology& topo, const align::GroupMatch& gm,
                                       const GroundTruth& gt, const json* old_topology) {
    if (old_topology == nullptr) return std::nullopt;
    const auto binding = any_binding_for_group(topo, gm.group);
    if (!binding) return std::nullopt;
    const auto old_name = bridge_group_name(*old_topology, binding->first, binding->second);
    if (!old_name) return std::nullopt;  // pinned on the old side too: never permuted, nothing to check

    const auto it = gt.groups.find(*old_name);
    if (it == gt.groups.end()) return std::nullopt;
    const auto& expected = it->second;
    if (gm.identity) {
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] != i) return false;
        }
        return true;
    }
    if (gm.permutation.size() != expected.size()) return false;
    return gm.permutation == expected;
}

// ---------------------------------------------------------- pair benchmark

json run_pair(const DiscoveredPair& p, const align::MatchOptions& opts, bool as_json) {
    json j = {{"name", p.name}, {"kind", p.kind},
             {"base", p.base.string()}, {"target", p.target.string()}};

    if (!p.config) {
        j["skipped_reason"] = "no matching topology config.json found for this fixture prefix";
        if (!as_json) {
            std::printf("%-24s SKIPPED (no <prefix>_layers_config.json found next to the "
                       "checkpoint)\n", p.name.c_str());
        }
        return j;
    }
    j["config"] = p.config->string();

    auto base = align::tools::SimpleStSource::open(p.base);
    if (!base) { j["error"] = "open base: " + base.error().to_string(); return j; }
    auto target = align::tools::SimpleStSource::open(p.target);
    if (!target) { j["error"] = "open target: " + target.error().to_string(); return j; }

    auto topo = align::parse_topology_file(*base, *p.config);
    if (!topo) { j["error"] = "topology: " + topo.error().to_string(); return j; }
    j["topology_groups"] = topo->groups.size();

    align::Matcher matcher(*base, *target, *topo, opts);
    const auto t0 = clk::now();
    auto report = matcher.run();
    const double wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (!report) { j["error"] = "matcher: " + report.error().to_string(); return j; }

    std::optional<GroundTruth> gt;
    std::optional<json> old_topology;
    try {
        if (p.ground_truth) gt = load_ground_truth(*p.ground_truth);
        if (p.old_topology) old_topology = read_json(*p.old_topology);
    } catch (const std::exception& e) {
        // A malformed or wrong-schema ground-truth/topology sidecar
        // shouldn't crash the whole run -- report it and skip just the
        // ground-truth check for this pair, same as every other failure
        // mode in this function.
        j["error"] = std::string("ground truth: ") + e.what();
        return j;
    }

    // MatchReport::groups is an unordered_map; sort names for stable,
    // diffable output.
    std::vector<std::string> names;
    names.reserve(report->groups.size());
    for (const auto& [name, gm] : report->groups) names.push_back(name);
    std::sort(names.begin(), names.end());

    json groups_j = json::array();
    std::vector<std::string> regressions;
    for (const auto& name : names) {
        const auto& gm = report->groups.at(name);
        const auto& pg = topo->groups.at(name);
        json gj = {
            {"name", name}, {"size", pg.size}, {"pinned", pg.pinned},
            {"identity", gm.identity}, {"alignable", gm.alignable},
            {"exact_solver", gm.exact_solver},
            {"cost_raw", gm.cost_raw}, {"cost_normalized", gm.cost_normalized},
        };
        std::optional<bool> gt_ok;
        if (gt) {
            gt_ok = check_ground_truth(*topo, gm, *gt, old_topology ? &*old_topology : nullptr);
            if (gt_ok) {
                gj["ground_truth_match"] = *gt_ok;
                if (!*gt_ok) regressions.push_back(name);
            }
        }
        groups_j.push_back(gj);
        if (!as_json) {
            std::printf("  group %-6s size=%-5u pinned=%-5s identity=%-5s alignable=%-5s "
                       "exact=%-5s cost_norm=%.6f%s\n",
                       name.c_str(), pg.size, pg.pinned ? "true" : "false",
                       gm.identity ? "true" : "false", gm.alignable ? "true" : "false",
                       gm.exact_solver ? "true" : "false", gm.cost_normalized,
                       (gt_ok && !*gt_ok) ? "  ** GROUND TRUTH MISMATCH **" : "");
        }
    }

    j["wall_seconds"] = wall_s;
    j["sweeps"] = report->sweeps;
    j["peak_bytes"] = report->peak_bytes;
    j["groups"] = groups_j;
    j["ground_truth_checked"] = gt.has_value() && old_topology.has_value();
    j["regressions"] = regressions;
    j["ok"] = regressions.empty();

    if (!as_json) {
        std::printf("%-24s wall=%.4fs sweeps=%u groups=%zu%s\n\n", p.name.c_str(), wall_s,
                   report->sweeps, names.size(),
                   regressions.empty() ? "" : "  ** REGRESSIONS PRESENT **");
    }
    return j;
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixtures_dir = "fixtures/out";
    std::optional<std::string> pair_base, pair_target, single_config, single_gt;
    bool as_json = false;

    auto next = [&](int& i) -> std::string {
        return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fixtures-dir") fixtures_dir = next(i);
        else if (arg == "--pair") {
            const std::string v = next(i);
            const auto comma = v.find(',');
            if (comma == std::string::npos) {
                std::fprintf(stderr, "--pair wants base.safetensors,target.safetensors\n");
                return 2;
            }
            pair_base = v.substr(0, comma);
            pair_target = v.substr(comma + 1);
        }
        else if (arg == "--config") single_config = next(i);
        else if (arg == "--ground-truth") single_gt = next(i);
        else if (arg == "--json") as_json = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: align_time [--fixtures-dir DIR]\n"
                "                  [--pair base.safetensors,target.safetensors --config C.json "
                "[--ground-truth G.json]]\n"
                "                  [--json]\n"
                "no args: auto-discovers fixture pairs under fixtures/out "
                "(bench/scripts/run_all.sh's calling convention). LAP solver timing is "
                "bench/lap_bench.cpp; the sparse path is bench/sparse_bench.cpp.\n");
            return 0;
        }
        else {
            std::fprintf(stderr, "unrecognized argument: %s (see --help)\n", arg.c_str());
            return 2;
        }
    }

    // Library defaults throughout: this benchmarks the shipped configuration,
    // not a tuning knob. Override MatchOptions in code, not via flags here,
    // if a deliberate crossover/budget experiment is needed.
    align::MatchOptions opts;

    json pairs_j = json::array();
    if (pair_base && pair_target) {
        if (!single_config) {
            std::fprintf(stderr, "--pair requires --config\n");
            return 2;
        }
        DiscoveredPair p{fs::path(*pair_base).stem().string() + "_vs_" +
                            fs::path(*pair_target).stem().string(),
                        "manual", fs::path(*pair_base), fs::path(*pair_target),
                        fs::path(*single_config),
                        single_gt ? std::optional(fs::path(*single_gt)) : std::nullopt};
        pairs_j.push_back(run_pair(p, opts, as_json));
    } else {
        const auto discovered = discover_pairs(fixtures_dir);
        if (discovered.empty() && !as_json) {
            std::printf("no fixture pairs found under %s (run `make fixtures-small` first)\n",
                       fixtures_dir.c_str());
        }
        for (const auto& p : discovered) pairs_j.push_back(run_pair(p, opts, as_json));
    }

    bool any_problem = false;
    for (const auto& pj : pairs_j) {
        if (pj.contains("error")) any_problem = true;
        if (pj.contains("ok") && pj.at("ok") == false) any_problem = true;
    }

    if (as_json) {
        const json out = {{"pairs", pairs_j}, {"ok", !any_problem}};
        std::printf("%s\n", out.dump(2).c_str());
    } else {
        std::printf("\n%s\n", any_problem ? "align_time: PROBLEMS FOUND (see above)" : "align_time: OK");
    }
    return any_problem ? 1 : 0;
}
