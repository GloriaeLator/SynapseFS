#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/journal.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>
#include <synapsefs/store/verify.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_verify(const std::string& revspec, bool full, bool repair) {
    auto paths = core::RepoPaths::discover(std::filesystem::current_path());
    if (!paths) {
        std::cerr << "error: not a synapsefs repository\n";
        return ExitCode::NotARepository;
    }

    auto cfg = core::RepoConfig::load(paths->root);
    if (!cfg) {
        std::cerr << "error: " << cfg.error().to_string() << "\n";
        return exit_code_for(cfg.error());
    }

    auto blocks = store::BlockStore::open(*paths, *cfg);
    if (!blocks) {
        std::cerr << "error: " << blocks.error().to_string() << "\n";
        return exit_code_for(blocks.error());
    }

    store::RefStore refs(*paths);

    // verify.hpp §(0): repair happens BEFORE verify() is constructed/called —
    // a torn journal record must be resolved first, and verify() itself is
    // given no path to the journal on purpose (it must work standalone).
    if (repair) {
        store::Journal journal(paths->journal());
        if (auto st = journal.recover(refs); !st) {
            std::cerr << "error: repair failed: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
    }

    store::CommitStore commits(**blocks, refs);
    store::ManifestStore manifests(**blocks, commits);

    std::vector<core::Oid> tips;
    if (revspec.empty()) {
        auto heads = refs.list_heads();
        if (!heads) {
            std::cerr << "error: " << heads.error().to_string() << "\n";
            return exit_code_for(heads.error());
        }
        for (const auto& h : *heads) tips.push_back(h.target);
        if (tips.empty()) {
            std::cout << "nothing to verify: repository has no branches\n";
            return ExitCode::Ok;
        }
    } else {
        auto r = refs.rev_parse(revspec);
        if (!r) {
            std::cerr << "error: " << r.error().to_string() << "\n";
            return exit_code_for(r.error());
        }
        tips.push_back(*r);
    }

    store::VerifyOptions opts;
    opts.full = full;
    opts.repair = repair;

    auto report = store::verify(**blocks, commits, manifests, refs, tips, opts);
    if (!report) {
        std::cerr << "error: " << report.error().to_string() << "\n";
        return exit_code_for(report.error());
    }

    std::cout << "commits walked:  " << report->commits_walked << "\n";
    std::cout << "objects checked: " << report->objects_checked << "\n";

    if (report->ok()) {
        std::cout << "verify: OK\n";
        return ExitCode::Ok;
    }

    for (const auto& f : report->findings) {
        std::cerr << "INTEGRITY: " << core::to_string(f.kind) << " " << f.object.to_string();
        if (f.group) std::cerr << " group=" << *f.group;
        if (f.chunk_index) std::cerr << " chunk=" << *f.chunk_index;
        std::cerr << ": " << f.detail << "\n";
    }
    std::cerr << "verify: FAILED (" << report->findings.size() << " finding(s))\n";
    return ExitCode::Integrity;
}

}  // namespace

void register_verify(CLI::App& app, int& exit_code) {
    static std::string revspec;
    static bool full = false;
    static bool repair = false;
    auto* c = app.add_subcommand("verify", "Verify repository integrity and lineage");
    c->add_option("revision", revspec, "Commit/branch to verify (default: all branch heads)");
    c->add_flag("--full", full, "Re-hash every chunk of every reachable object");
    c->add_flag("--repair", repair, "Replay or roll back a crashed journal before verifying");
    c->callback([&exit_code] { exit_code = run_verify(revspec, full, repair); });
}

}  // namespace sfs::app::cmd
