/// bench/lap_bench.cpp — docs/benchmarks.md §1's LAP solver crossover table.
///
/// A checkpoint-independent benchmark: the LAP solvers (modules/align/src/
/// lap.cpp) never see a checkpoint, they just minimize an n x n cost matrix
/// (align::ILapSolver), so this generates synthetic random cost matrices
/// directly rather than routing through a fixture. C++, not Python (as
/// lap.hpp's own comment names this file) -- there are no LAP-solver Python
/// bindings in this repo, so a standalone benchmark against the real
/// align::make_jv_solver()/make_greedy_solver() is the only way to measure
/// this honestly, against the actual compiled solver code, not a
/// reimplementation.
///
/// Exact JV is only run up to `--exact-cutoff` (default 4096, matching
/// align::MatchOptions::lap_crossover's own default): JV is O(n^3) worst
/// case, and align::Matcher itself never calls the dense solvers at all
/// above MatchOptions::sparse_crossover (8192) -- it switches to a
/// completely different sparse fingerprint+auction algorithm there
/// (docs/adr/0011), specifically because a dense n x n matrix stops fitting
/// memory at that scale. So there is no real-system scenario where JV runs
/// above the crossover, and no reason to pay an O(n^3) benchmark run to
/// prove that; greedy is still measured up to 8192 to show it staying
/// usable exactly where the dense path's own ceiling is.
///
/// Accuracy cost is greedy's assignment cost compared against JV's (the
/// actual optimum for that matrix, not an assumed one) at every size where
/// JV ran; above the cutoff there is no exact reference to compare against,
/// so it is reported as not available rather than guessed.
///
/// Usage:
///   lap_bench [--sizes 512,1024,2048,4096,8192] [--exact-cutoff 4096]
///            [--seed 42] [--json]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/align/lap.hpp>

using namespace sfs;
using json = nlohmann::json;

namespace {

std::vector<float> random_cost_matrix(std::uint32_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> cost(static_cast<std::size_t>(n) * n);
    for (auto& c : cost) c = dist(rng);
    return cost;
}

struct SolveOutcome {
    bool   ran = false;
    double seconds = 0.0;
    double cost_raw = 0.0;
    bool   exact = false;
    std::string error;
};

SolveOutcome time_solve(align::ILapSolver& solver, std::span<const float> cost, std::uint32_t n) {
    SolveOutcome out;
    const auto t0 = std::chrono::steady_clock::now();
    auto result = solver.solve(cost, n);
    const auto t1 = std::chrono::steady_clock::now();
    if (!result) {
        out.error = result.error().to_string();
        return out;
    }
    out.ran = true;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.cost_raw = result->cost_raw;
    out.exact = result->exact;
    return out;
}

struct Row {
    std::uint32_t n = 0;
    SolveOutcome jv, greedy;
    std::optional<double> accuracy_cost_pct;  // (greedy - jv) / jv * 100, only when jv ran
};

Row run_size(std::uint32_t n, std::uint32_t exact_cutoff, std::uint64_t seed) {
    Row row;
    row.n = n;
    const auto cost = random_cost_matrix(n, seed);

    auto greedy_solver = align::make_greedy_solver();
    row.greedy = time_solve(*greedy_solver, cost, n);

    if (n <= exact_cutoff) {
        auto jv_solver = align::make_jv_solver();
        row.jv = time_solve(*jv_solver, cost, n);
        if (row.jv.ran && row.greedy.ran && row.jv.cost_raw > 0.0) {
            row.accuracy_cost_pct = 100.0 * (row.greedy.cost_raw - row.jv.cost_raw) / row.jv.cost_raw;
        }
    }
    return row;
}

void print_table(const std::vector<Row>& rows, std::uint32_t exact_cutoff) {
    std::cout << "| n | Exact JV | Greedy + 2-swap | Accuracy cost |\n";
    std::cout << "|---|---|---|---|\n";
    for (const auto& r : rows) {
        std::cout << "| " << r.n << " | ";
        if (r.jv.ran) {
            std::cout << (r.jv.seconds * 1000.0) << " ms";
        } else if (r.n > exact_cutoff) {
            std::cout << "skipped (> " << exact_cutoff << " crossover)";
        } else {
            std::cout << "ERROR: " << r.jv.error;
        }
        std::cout << " | ";
        if (r.greedy.ran) {
            std::cout << (r.greedy.seconds * 1000.0) << " ms";
        } else {
            std::cout << "ERROR: " << r.greedy.error;
        }
        std::cout << " | ";
        if (r.accuracy_cost_pct) {
            std::cout << (*r.accuracy_cost_pct) << "%";
        } else {
            std::cout << "n/a (no exact reference at this size)";
        }
        std::cout << " |\n";
    }
}

json to_json(const std::vector<Row>& rows) {
    json arr = json::array();
    for (const auto& r : rows) {
        json j = {{"n", r.n}};
        j["jv_ran"] = r.jv.ran;
        if (r.jv.ran) {
            j["jv_seconds"] = r.jv.seconds;
            j["jv_cost"]    = r.jv.cost_raw;
        }
        j["greedy_ran"] = r.greedy.ran;
        if (r.greedy.ran) {
            j["greedy_seconds"] = r.greedy.seconds;
            j["greedy_cost"]    = r.greedy.cost_raw;
        }
        j["accuracy_cost_pct"] = r.accuracy_cost_pct ? json(*r.accuracy_cost_pct) : json(nullptr);
        arr.push_back(j);
    }
    return arr;
}

// ------------------------------------------------------------------ CLI

struct Args {
    std::vector<std::uint32_t> sizes = {512, 1024, 2048, 4096, 8192};
    std::uint32_t exact_cutoff = 4096;  // matches align::MatchOptions::lap_crossover's default
    std::uint64_t seed = 42;
    bool json = false;
};

std::vector<std::uint32_t> parse_sizes(const std::string& csv) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") a.json = true;
        else if (arg == "--sizes" && i + 1 < argc) a.sizes = parse_sizes(argv[++i]);
        else if (arg == "--exact-cutoff" && i + 1 < argc) a.exact_cutoff = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--seed" && i + 1 < argc) a.seed = std::stoull(argv[++i]);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    std::vector<Row> rows;
    for (const auto n : args.sizes) {
        rows.push_back(run_size(n, args.exact_cutoff, args.seed));
    }

    if (args.json) {
        std::cout << to_json(rows).dump(2) << "\n";
    } else {
        print_table(rows, args.exact_cutoff);
    }
    return 0;
}
