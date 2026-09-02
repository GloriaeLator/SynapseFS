/// bench/align_time.cpp — the alignment module's own graded number.
/// Referenced by bench/scripts/run_all.sh (`align_time --json`) since before
/// this file existed; see docs/known-gaps.md for that history.
///
/// Two independent measurements, both run by default so `align_time --json`
/// with no other flags — run_all.sh's exact calling convention — always
/// produces something:
///
///   1. End-to-end `align::Matcher::run()` on real fixture checkpoint pairs,
///      auto-discovered the same way bench/residual_codec.cpp discovers its
///      own pairs: `<prefix>_step0.safetensors` + `<prefix>_step1.safetensors`
///      under --fixtures-dir (default fixtures/out) is a fine-tune pair,
///      `<prefix>_step0.safetensors` + `<prefix>_permuted.safetensors` is a
///      permutation-only pair. A pair only runs if a matching "layers"-shaped
///      config.json (SPEC 13 §2) can be found for it — today that means
///      fixtures/mlp_layers_config.json, the one real one in the tree, for
///      any `mlp`/`tiny_mlp`-prefixed pair (see resolve_config below); other
///      prefixes are reported as skipped rather than fed an empty config,
///      which would just hand every tensor its own pinned singleton group
///      and silently "measure" a no-op. For a permutation-only pair, the
///      recovered permutation is also checked against permute.py's own
///      `*_permuted.permutation.json` — that pair's correct answer is known
///      exactly, so this doubles as a regression check the way
///      modules/align/tools/align_demo.cpp's ground-truth mode does.
///
///   2. A synthetic LAP solver sweep: `align::make_jv_solver()` (exact)
///      against `align::make_greedy_solver()` (heuristic) on random dense
///      n x n cost matrices at a range of n. This is the number
///      modules/align/include/synapsefs/align/lap.hpp's own header comment
///      asks for by name: "we used greedy above n = X, and it costs 0.Y%
///      accuracy for a Z-times speedup is an answer, 'greedy is faster' is
///      not." Costs are uniform random with no planted structure — a
///      solver-quality stress test, not a stand-in for real alignment cost
///      matrices (see cost.hpp), so treat the reported cost ratio as a
///      worst-case bound on greedy's loss, not a prediction. Exact JV only
///      runs up to --lap-max-exact (default 1024): it is worst-case O(n^3),
///      and this tool's default invocation has to stay fast enough to run on
///      every bench/scripts/run_all.sh pass, not just in a deliberate
///      research session — raise the cap explicitly for a deeper sweep.
///
/// Real .safetensors files are read through align::tools::SimpleStSource
/// (modules/align/tools/simple_st_source.hpp), the same mmap-backed
/// core::ITensorSource align's own end-to-end tests and align_demo.cpp use —
/// not stio::StSource. That is deliberate, not an oversight: stio::StSource
/// ::read_units only supports axis-0 reads today (docs/known-gaps.md), and
/// align's cost-feature builder needs "incoming" slices along whichever axis
/// a group binds to, which is not always axis 0 (e.g. a Linear weight's
/// input axis is dim 1). SimpleStSource is align's own long-standing
/// workaround for exactly this gap; this bench follows the module's own
/// test suite rather than reintroducing a mismatch stio hasn't closed yet.
///
/// Deliberately not wired through apps/sfs's CLI, for the reason
/// bench/verify_time.cpp already gives for itself: this needs only
/// core/align (plus the align module's own tools/ library to open a real
/// file), not the full `sfs` binary.
///
/// Deliberately the one bench target that DOES link synapsefs::align (and
/// therefore Torch) — see bench/CMakeLists.txt. Every other bench target
/// includes align's headers without linking the library, specifically
/// because none of them call compiled align:: code; this one exists only to
/// measure that compiled code, so the trade reverses for it alone.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/align/lap.hpp>
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
};

// The one convention this bench introduces: a per-fixture sidecar living
// beside the checkpoints themselves, `<prefix>_layers_config.json` inside
// --fixtures-dir. No current generator writes one (gen_mlp.py/gen_resnet.py
// both only emit the perm_groups-shaped *_topology.json sidecar, which is
// align's OUTPUT shape, not its config.json INPUT shape — see
// docs/spec/13-topology-config.md), so today this only ever resolves via the
// fallback below. It's here so a future gen_resnet.py-style script has an
// obvious place to put a real config and this bench picks it up with no
// changes.
std::optional<fs::path> resolve_config(const fs::path& fixtures_dir, const std::string& prefix) {
    if (auto candidate = fixtures_dir / (prefix + "_layers_config.json"); fs::exists(candidate)) {
        return candidate;
    }
    // Today's one real "layers"-shaped config.json (SPEC 13 §2) lives at
    // fixtures/mlp_layers_config.json, one level above --fixtures-dir
    // (default fixtures/out), and covers both `mlp` and `tiny_mlp`: gen_mlp
    // .py builds the identical Linear-ReLU-Linear-ReLU-Linear layer chain
    // for both, only the widths differ, and widths are read from the
    // checkpoint itself, not the config.
    if (prefix == "mlp" || prefix == "tiny_mlp") {
        if (auto shared = fixtures_dir / ".." / "mlp_layers_config.json"; fs::exists(shared)) {
            return shared;
        }
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
        const auto config = resolve_config(dir, prefix);

        if (fs::exists(step1)) {
            out.push_back({prefix + "_finetune", "finetune", step0, step1, config, std::nullopt});
        }
        if (fs::exists(permuted)) {
            out.push_back({prefix + "_permuted_only", "permuted_only", step0, permuted, config,
                           fs::exists(perm_json) ? std::optional(perm_json) : std::nullopt});
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
    for (auto it = j.at("groups").begin(); it != j.at("groups").end(); ++it) {
        std::vector<std::uint32_t> perm;
        perm.reserve(it.value().size());
        for (const auto& v : it.value()) perm.push_back(v.get<std::uint32_t>());
        gt.groups.emplace(it.key(), std::move(perm));
    }
    return gt;
}

// nullopt: this group has no planted ground truth to check against (it was
// pinned, so permute.py never touched it). true/false: whether the
// recovered permutation matches exactly.
std::optional<bool> check_ground_truth(const align::GroupMatch& gm, const GroundTruth& gt) {
    const auto it = gt.groups.find(gm.group);
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
            std::printf("%-24s SKIPPED (no topology config -- see resolve_config() / "
                       "fixtures/mlp_layers_config.json)\n", p.name.c_str());
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
    if (p.ground_truth) gt = load_ground_truth(*p.ground_truth);

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
            gt_ok = check_ground_truth(gm, *gt);
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
    j["ground_truth_checked"] = gt.has_value();
    j["regressions"] = regressions;
    j["ok"] = regressions.empty();

    if (!as_json) {
        std::printf("%-24s wall=%.4fs sweeps=%u groups=%zu%s\n\n", p.name.c_str(), wall_s,
                   report->sweeps, names.size(),
                   regressions.empty() ? "" : "  ** REGRESSIONS PRESENT **");
    }
    return j;
}

// --------------------------------------------------------------- LAP sweep

std::vector<float> random_cost_matrix(std::uint32_t n, std::uint64_t seed) {
    std::vector<float> m(static_cast<std::size_t>(n) * n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : m) v = dist(rng);
    return m;
}

json run_lap_sweep(const std::vector<std::uint32_t>& sizes, std::uint32_t max_exact, int trials,
                   bool as_json) {
    auto jv = align::make_jv_solver();
    auto greedy = align::make_greedy_solver();

    json out = json::array();
    for (const std::uint32_t n : sizes) {
        double jv_ms_total = 0.0, jv_cost_total = 0.0;
        double greedy_ms_total = 0.0, greedy_cost_total = 0.0;
        std::uint32_t greedy_iters = 0;
        int jv_runs = 0;

        for (int t = 0; t < trials; ++t) {
            const auto cost = random_cost_matrix(n, 1000ull * n + static_cast<std::uint64_t>(t));

            const auto t0 = clk::now();
            auto gr = greedy->solve(cost, n);
            greedy_ms_total += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (!gr) {
                std::fprintf(stderr, "greedy solver failed at n=%u: %s\n", n,
                            gr.error().to_string().c_str());
                continue;
            }
            greedy_cost_total += gr->cost_raw;
            greedy_iters = gr->iterations;

            if (n <= max_exact) {
                const auto t1 = clk::now();
                auto jr = jv->solve(cost, n);
                jv_ms_total += std::chrono::duration<double, std::milli>(clk::now() - t1).count();
                if (!jr) {
                    std::fprintf(stderr, "jv solver failed at n=%u: %s\n", n,
                                jr.error().to_string().c_str());
                    continue;
                }
                jv_cost_total += jr->cost_raw;
                ++jv_runs;
            }
        }

        const double greedy_ms = greedy_ms_total / trials;
        const double greedy_cost = greedy_cost_total / trials;

        json pj = {{"n", n}, {"trials", trials},
                  {"greedy_ms", greedy_ms}, {"greedy_cost_raw", greedy_cost},
                  {"greedy_iterations", greedy_iters}};

        if (jv_runs > 0) {
            const double jv_ms = jv_ms_total / jv_runs;
            const double jv_cost = jv_cost_total / jv_runs;
            pj["jv_ms"] = jv_ms;
            pj["jv_cost_raw"] = jv_cost;
            // Random, unstructured costs make this a worst-case bound on
            // greedy's loss, not a forecast for real alignment matrices
            // (see cost.hpp) -- reported alongside both raw costs so nobody
            // over-reads the ratio alone.
            pj["greedy_cost_over_jv_cost"] = greedy_cost / jv_cost;
            if (!as_json) {
                std::printf("n=%-6u greedy=%9.3f ms (cost=%.2f)   jv=%9.3f ms (cost=%.2f)   "
                           "greedy/jv cost ratio=%.4f\n",
                           n, greedy_ms, greedy_cost, jv_ms, jv_cost, greedy_cost / jv_cost);
            }
        } else if (!as_json) {
            std::printf("n=%-6u greedy=%9.3f ms (cost=%.2f)   jv=      (skipped: n > "
                       "--lap-max-exact=%u)\n",
                       n, greedy_ms, greedy_cost, max_exact);
        }
        out.push_back(pj);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixtures_dir = "fixtures/out";
    std::optional<std::string> pair_base, pair_target, single_config, single_gt;
    bool as_json = false;
    bool do_pairs = true;
    bool do_lap = true;
    std::vector<std::uint32_t> lap_sizes = {64, 256, 1024, 4096};
    std::uint32_t lap_max_exact = 1024;
    int lap_trials = 1;

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
        else if (arg == "--lap-sizes") {
            lap_sizes.clear();
            std::stringstream ss(next(i));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                lap_sizes.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
            }
        }
        else if (arg == "--lap-max-exact") lap_max_exact = static_cast<std::uint32_t>(std::stoul(next(i)));
        else if (arg == "--lap-trials") lap_trials = std::max(1, std::atoi(next(i).c_str()));
        else if (arg == "--no-pairs") do_pairs = false;
        else if (arg == "--no-lap") do_lap = false;
        else if (arg == "--json") as_json = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: align_time [--fixtures-dir DIR]\n"
                "                  [--pair base.safetensors,target.safetensors --config C.json "
                "[--ground-truth G.json]]\n"
                "                  [--lap-sizes 64,256,1024,4096] [--lap-max-exact 1024] "
                "[--lap-trials 1]\n"
                "                  [--no-pairs] [--no-lap] [--json]\n"
                "no args: auto-discovers fixture pairs under fixtures/out and runs the default "
                "LAP sweep\n"
                "(bench/scripts/run_all.sh's calling convention).\n");
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
    } else if (do_pairs) {
        const auto discovered = discover_pairs(fixtures_dir);
        if (discovered.empty() && !as_json) {
            std::printf("no fixture pairs found under %s (run `make fixtures-small` first) -- "
                       "running the LAP sweep only.\n\n", fixtures_dir.c_str());
        }
        for (const auto& p : discovered) pairs_j.push_back(run_pair(p, opts, as_json));
    }

    json lap_j = json::array();
    if (do_lap) {
        if (!as_json) std::printf("=== LAP solver sweep (synthetic random cost matrices) ===\n");
        lap_j = run_lap_sweep(lap_sizes, lap_max_exact, lap_trials, as_json);
    }

    bool any_problem = false;
    for (const auto& pj : pairs_j) {
        if (pj.contains("error")) any_problem = true;
        if (pj.contains("ok") && pj.at("ok") == false) any_problem = true;
    }

    if (as_json) {
        const json out = {{"pairs", pairs_j}, {"lap_sweep", lap_j}, {"ok", !any_problem}};
        std::printf("%s\n", out.dump(2).c_str());
    } else {
        std::printf("\n%s\n", any_problem ? "align_time: PROBLEMS FOUND (see above)" : "align_time: OK");
    }
    return any_problem ? 1 : 0;
}
