/// `sfs checkout <revision|branch>` — reconstructs a stored checkpoint into
/// the working tree, or switches the active branch.
///
/// Pre-2.23 git semantics per docs/spec/15-cli-contract.md and refs.hpp:
/// `checkout <branch-name>` switches HEAD to that branch (no restore of
/// files unless -o/--output is also given, in which case it also writes the
/// branch's tip checkpoint). `checkout <revision> -o <path>` reconstructs
/// that commit's checkpoint without moving HEAD (a detached-style restore).

#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/stio/st_writer.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_checkout(const std::string& revision, const std::string& output) {
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

    // Does `revision` name a branch? If so this is a branch switch
    // (pre-2.23: checkout <branch-name>). Otherwise treat it as a revision
    // to resolve (oid, abbreviation, or "HEAD").
    bool is_branch = false;
    {
        auto heads = refs.list_heads();
        if (heads) {
            for (const auto& h : *heads) {
                if (h.name == "refs/heads/" + revision || h.name == revision) {
                    is_branch = true;
                    break;
                }
            }
        }
    }

    core::Oid target;
    if (is_branch) {
        auto tip = refs.resolve("refs/heads/" + revision);
        if (!tip) {
            std::cerr << "error: " << tip.error().to_string() << "\n";
            return exit_code_for(tip.error());
        }
        target = *tip;
        if (auto st = refs.set_head_symbolic("refs/heads/" + revision); !st) {
            std::cerr << "error: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
    } else {
        auto r = refs.rev_parse(revision);
        if (!r) {
            std::cerr << "error: " << r.error().to_string() << "\n";
            return exit_code_for(r.error());
        }
        target = *r;
        if (auto st = refs.set_head_detached(target); !st) {
            std::cerr << "error: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
    }

    auto commit = commits.read(target);
    if (!commit) {
        std::cerr << "error: " << commit.error().to_string() << "\n";
        return exit_code_for(commit.error());
    }

    auto manifest = manifests.read(commit->manifest);
    if (!manifest) {
        std::cerr << "error: " << manifest.error().to_string() << "\n";
        return exit_code_for(manifest.error());
    }

    // Bare branch switch with no -o: just move HEAD/refs, like `git checkout
    // <branch>` with no working-tree files tracked outside .synapsefs. Most
    // users will want -o to actually materialise the checkpoint.
    if (output.empty()) {
        std::cout << (is_branch ? "Switched to branch '" : "HEAD is now at ")
                  << (is_branch ? revision : target.abbrev())
                  << (is_branch ? "'\n" : "\n");
        return ExitCode::Ok;
    }

    std::filesystem::path dest(output);

    codec::ReadCtx ctx;
    ctx.blocks = blocks->get();
    ctx.manifest = &*manifest;
    ctx.history = &manifests;
    ctx.max_depth = cfg->max_chain_depth;

    auto writer = stio::StWriter::create(dest, *manifest);
    if (!writer) {
        std::cerr << "error: " << writer.error().to_string() << "\n";
        return exit_code_for(writer.error());
    }

    auto st = codec::reconstruct_file(ctx, [&](std::span<const std::byte> chunk) {
        return writer->append(chunk);
    });
    if (!st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    if (auto fst = writer->finish(); !fst) {
        std::cerr << "error: " << fst.error().to_string() << "\n";
        return exit_code_for(fst.error());
    }

    std::cout << "Checked out " << target.abbrev() << " -> " << dest.string() << " ("
              << writer->written() << " bytes)\n";
    return ExitCode::Ok;
}

}  // namespace

void register_checkout(CLI::App& app, int& exit_code) {
    static std::string revision;
    static std::string output;
    auto* c = app.add_subcommand("checkout", "Switch branch, or restore a checkpoint version");
    c->add_option("revision", revision, "Branch name, commit oid, or abbreviation")->required();
    c->add_option("-o,--output", output,
                  "Write the reconstructed .safetensors file here");
    c->callback([&exit_code] { exit_code = run_checkout(revision, output); });
}

}  // namespace sfs::app::cmd
