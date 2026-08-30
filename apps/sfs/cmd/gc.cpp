/// `sfs gc` — sweep crashed temp writes and, with --prune, remove objects no
/// ref reaches. Refuses while a mount daemon is attached (gc.hpp).

#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/gc.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_gc(bool prune, bool pack, bool dry_run) {
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
    store::CommitStore commits(**blocks, refs);
    store::ManifestStore manifests(**blocks, commits);

    store::GcOptions opts;
    opts.prune = prune;
    opts.pack = pack;
    opts.dry_run = dry_run;

    // gc() takes the exclusive lock itself, so nothing is acquired here.
    auto report = store::gc(**blocks, commits, manifests, refs, *paths, opts);
    if (!report) {
        std::cerr << "error: " << report.error().to_string() << "\n";
        return exit_code_for(report.error());
    }

    if (dry_run) std::cout << "(dry run — nothing was removed)\n";
    std::cout << "temp files removed: " << report->temp_files_removed << "\n";
    if (prune) {
        std::cout << "objects scanned:    " << report->objects_scanned << "\n";
        std::cout << "objects pruned:     " << report->objects_pruned << "\n";
    }
    std::cout << "bytes reclaimed:    " << report->bytes_reclaimed << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_gc(CLI::App& app, int& exit_code) {
    static bool prune = false;
    static bool pack = false;
    static bool dry_run = false;
    auto* c = app.add_subcommand("gc", "Clean up temp files and unreachable objects");
    c->add_flag("--prune", prune, "Delete objects no ref reaches");
    c->add_flag("--pack", pack, "Rewrite reachable loose objects into a packfile");
    c->add_flag("-n,--dry-run", dry_run, "Report what would be removed, remove nothing");
    c->callback([&exit_code] { exit_code = run_gc(prune, pack, dry_run); });
}

}  // namespace sfs::app::cmd
