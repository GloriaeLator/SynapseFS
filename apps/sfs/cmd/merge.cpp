/// `sfs merge <branch>` — three-way or fast-forward merge over the commit DAG.
///
/// A merge touches a branch ref AND HEAD, which is one of exactly two
/// operations journal.hpp says cannot be made atomic by rename alone. So the
/// intent record is written first and removed last; a crash in between is
/// recoverable by `sfs verify --repair`.
///
/// We never average weights. Per group: changed on one side takes that side,
/// changed on both is a conflict that refuses and writes nothing unless
/// --ours/--theirs says which to take. docs/tradeoffs.md §2.2.

#include <CLI/CLI.hpp>

#include <cstdlib>
#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/journal.hpp>
#include <synapsefs/store/lockfile.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/merge_logic.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_merge(const std::string& branch, bool take_ours, bool take_theirs,
              const std::string& message) {
    if (take_ours && take_theirs) {
        std::cerr << "error: --ours and --theirs are mutually exclusive\n";
        return ExitCode::Usage;
    }

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

    auto lock = store::RepoLock::acquire(paths->lock(), store::LockMode::Exclusive);
    if (!lock) {
        std::cerr << "error: " << lock.error().to_string() << "\n";
        return exit_code_for(lock.error());
    }

    auto blocks = store::BlockStore::open(*paths, *cfg);
    if (!blocks) {
        std::cerr << "error: " << blocks.error().to_string() << "\n";
        return exit_code_for(blocks.error());
    }

    store::RefStore refs(*paths);
    store::CommitStore commits(**blocks, refs);
    store::ManifestStore manifests(**blocks, commits);

    auto head = refs.read_head();
    if (!head) {
        std::cerr << "error: " << head.error().to_string() << "\n";
        return exit_code_for(head.error());
    }
    if (head->is_detached()) {
        std::cerr << "error: HEAD is detached; check out a branch before merging\n";
        return ExitCode::Failure;
    }
    const std::string ref_name = *head->symbolic;

    auto ours = refs.resolve(ref_name);
    if (!ours) {
        std::cerr << "error: current branch has no commits to merge into\n";
        return ExitCode::Failure;
    }

    auto theirs = refs.rev_parse(branch);
    if (!theirs) {
        std::cerr << "error: " << theirs.error().to_string() << "\n";
        return exit_code_for(theirs.error());
    }

    store::MergeStrategy strategy = store::MergeStrategy::Refuse;
    if (take_ours) strategy = store::MergeStrategy::Ours;
    if (take_theirs) strategy = store::MergeStrategy::Theirs;

    auto plan = store::plan_merge(commits, manifests, *ours, *theirs, strategy);
    if (!plan) {
        std::cerr << "error: " << plan.error().to_string() << "\n";
        return exit_code_for(plan.error());
    }

    if (!plan->conflicts.empty()) {
        std::cerr << "CONFLICT: " << plan->conflicts.size()
                  << " group(s) changed on both sides:\n";
        for (const auto& c : plan->conflicts) {
            std::cerr << "  " << c.group << "  ours=" << c.our_block.abbrev()
                      << " theirs=" << c.their_block.abbrev() << "\n";
        }
        std::cerr << "Nothing was written. Re-run with --ours or --theirs to resolve.\n";
        return ExitCode::Conflict;
    }

    // Fast-forward: no new commit object, just a ref move. HEAD is symbolic
    // and already points at ref_name, so only the ref changes and no journal
    // record is needed.
    if (plan->fast_forward) {
        if (!plan->new_commit) {
            std::cerr << "error: fast-forward produced no target commit\n";
            return ExitCode::Failure;
        }
        if (*plan->new_commit == *ours) {
            std::cout << "Already up to date.\n";
            return ExitCode::Ok;
        }
        if (auto st = refs.update(ref_name, *ours, *plan->new_commit); !st) {
            std::cerr << "error: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
        std::cout << "Fast-forward " << ours->abbrev() << " -> "
                  << plan->new_commit->abbrev() << "\n";
        return ExitCode::Ok;
    }

    if (!plan->merged) {
        std::cerr << "error: merge produced no manifest and no conflicts\n";
        return ExitCode::Failure;
    }

    auto manifest_oid = manifests.write(*plan->merged);
    if (!manifest_oid) {
        std::cerr << "error: " << manifest_oid.error().to_string() << "\n";
        return exit_code_for(manifest_oid.error());
    }

    auto our_commit = commits.read(*ours);
    if (!our_commit) {
        std::cerr << "error: " << our_commit.error().to_string() << "\n";
        return exit_code_for(our_commit.error());
    }

    format::Commit merge_commit;
    merge_commit.parents = {*ours, *theirs};
    merge_commit.manifest = *manifest_oid;
    merge_commit.topology = our_commit->topology;
    merge_commit.timestamp = format::now_timestamp();
    const char* env_author = std::getenv("USER");
    merge_commit.author = env_author ? env_author : "unknown";
    merge_commit.message =
        message.empty() ? ("Merge " + branch + " into " + ref_name) : message;

    // Write the object BEFORE the journal record: an unreferenced commit
    // object is wasted disk, never a broken repository (commit_store.hpp).
    // The journal covers only the ref+HEAD pair that follows.
    auto merge_oid = commits.write(merge_commit);
    if (!merge_oid) {
        std::cerr << "error: " << merge_oid.error().to_string() << "\n";
        return exit_code_for(merge_oid.error());
    }

    store::Journal journal(paths->journal());
    store::JournalRecord rec;
    rec.op = store::JournalOp::Merge;
    rec.timestamp = merge_commit.timestamp;
    rec.ref_name = ref_name;
    rec.ref_old = *ours;
    rec.ref_new = *merge_oid;

    auto seq = journal.begin(rec);
    if (!seq) {
        std::cerr << "error: " << seq.error().to_string() << "\n";
        return exit_code_for(seq.error());
    }

    if (auto st = refs.update(ref_name, *ours, *merge_oid); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    if (auto st = journal.commit(*seq); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    std::cout << "Merge made by three-way strategy.\n";
    std::cout << "[" << ref_name << " " << merge_oid->abbrev() << "] "
              << merge_commit.message << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_merge(CLI::App& app, int& exit_code) {
    static std::string branch;
    static bool take_ours = false;
    static bool take_theirs = false;
    static std::string message;
    auto* c = app.add_subcommand("merge", "Merge another branch into the current one");
    c->add_option("branch", branch, "Branch or commit to merge in")->required();
    c->add_flag("--ours", take_ours, "Resolve every conflict by keeping our side");
    c->add_flag("--theirs", take_theirs, "Resolve every conflict by taking their side");
    c->add_option("-m,--message", message, "Merge commit message");
    c->callback([&exit_code] { exit_code = run_merge(branch, take_ours, take_theirs, message); });
}

}  // namespace sfs::app::cmd
