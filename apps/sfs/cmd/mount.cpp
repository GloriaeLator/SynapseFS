/// `sfs mount <revision> <mountpoint>` — FUSE read-only view of one commit's
/// checkpoint. Guarded by SFS_WITH_MOUNT so this translation unit is empty
/// (and pulls in no mount:: symbols) when SFS_BUILD_MOUNT is OFF, matching
/// main.cpp's own conditional registration of this subcommand.
#ifdef SFS_WITH_MOUNT

#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/mount/daemon.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"
#include "../header_only_source.hpp"

namespace sfs::app::cmd {

namespace {

int run_mount(const std::string& revision, const std::string& mountpoint, bool foreground,
             bool debug) {
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

    // These must outlive the daemon (ReadCtx holds raw pointers into them),
    // so they are function-local and the whole function blocks in
    // daemon->run() until unmount — exactly the scope the pointers need.
    auto blocks = store::BlockStore::open(*paths, *cfg);
    if (!blocks) {
        std::cerr << "error: " << blocks.error().to_string() << "\n";
        return exit_code_for(blocks.error());
    }
    store::RefStore refs(*paths);
    store::CommitStore commits(**blocks, refs);
    store::ManifestStore manifests(**blocks, commits);

    auto target = refs.rev_parse(revision);
    if (!target) {
        std::cerr << "error: " << target.error().to_string() << "\n";
        return exit_code_for(target.error());
    }

    auto commit = commits.read(*target);
    if (!commit) {
        std::cerr << "error: " << commit.error().to_string() << "\n";
        return exit_code_for(commit.error());
    }

    auto manifest = manifests.read(commit->manifest);
    if (!manifest) {
        std::cerr << "error: " << manifest.error().to_string() << "\n";
        return exit_code_for(manifest.error());
    }

    mount::FsOptions fs_opts;
    fs_opts.cache_bytes = cfg->cache_bytes;

    // Function-local like blocks/manifest above (ReadCtx holds a raw pointer
    // into it, and SynapseFs::create copies the ReadCtx as-is — see
    // fs.cpp's Impl constructor). Best-effort; see header_only_source.hpp
    // for why a parse failure here is not fatal.
    auto topology = app::load_commit_topology(**blocks, *commit, *manifest);

    codec::ReadCtx ctx;
    ctx.blocks = blocks->get();
    ctx.manifest = &*manifest;
    ctx.history = &manifests;
    ctx.max_depth = cfg->max_chain_depth;
    ctx.topology = topology ? &*topology : nullptr;

    auto fs = mount::SynapseFs::create(ctx, *manifest, fs_opts);
    if (!fs) {
        std::cerr << "error: " << fs.error().to_string() << "\n";
        return exit_code_for(fs.error());
    }

    mount::DaemonOptions dopts;
    dopts.mountpoint = mountpoint;
    dopts.foreground = foreground;
    dopts.debug = debug;
    dopts.fs = fs_opts;

    auto daemon = mount::Daemon::start(std::move(*fs), dopts);
    if (!daemon) {
        std::cerr << "error: " << daemon.error().to_string() << "\n";
        return exit_code_for(daemon.error());
    }

    if (auto st = (*daemon)->register_with_repo(*paths); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    std::cout << "Mounted " << target->abbrev() << " at " << mountpoint << " (read-only)\n";

    if (auto st = (*daemon)->run(); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }
    return ExitCode::Ok;
}

}  // namespace

void register_mount(CLI::App& app, int& exit_code) {
    static std::string revision;
    static std::string mountpoint;
    static bool foreground = false;
    static bool debug = false;
    auto* c = app.add_subcommand("mount", "Mount a commit's checkpoint read-only via FUSE");
    c->add_option("revision", revision, "Commit oid, abbreviation, or branch name")->required();
    c->add_option("mountpoint", mountpoint, "Directory to mount at")->required();
    c->add_flag("-f,--foreground", foreground, "Run in the foreground (required under sanitizers)");
    c->add_flag("--debug", debug, "Enable libfuse debug logging");
    c->callback([&exit_code] { exit_code = run_mount(revision, mountpoint, foreground, debug); });
}

}  // namespace sfs::app::cmd

#endif  // SFS_WITH_MOUNT
