#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/dag.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_log(const std::string& revspec, int max_count) {
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

    core::Oid start;
    if (revspec.empty()) {
        auto head = refs.read_head();
        if (!head) {
            std::cerr << "error: " << head.error().to_string() << "\n";
            return exit_code_for(head.error());
        }
        if (head->is_detached()) {
            start = *head->detached;
        } else {
            auto tip = refs.resolve(*head->symbolic);
            if (!tip) {
                std::cerr << "fatal: your current branch has no commits yet\n";
                return ExitCode::Failure;
            }
            start = *tip;
        }
    } else {
        auto r = refs.rev_parse(revspec);
        if (!r) {
            std::cerr << "error: " << r.error().to_string() << "\n";
            return exit_code_for(r.error());
        }
        start = *r;
    }

    int printed = 0;
    std::array<core::Oid, 1> tips{start};
    auto st = store::walk_commits(commits, tips,
                                  [&](const core::Oid& oid, const format::Commit& c) {
                                      if (max_count >= 0 && printed >= max_count) return false;
                                      std::cout << "commit " << oid.to_string() << "\n";
                                      if (c.is_merge()) {
                                          std::cout << "Merge:";
                                          for (const auto& p : c.parents)
                                              std::cout << " " << p.abbrev();
                                          std::cout << "\n";
                                      }
                                      std::cout << "Author: " << c.author << "\n";
                                      std::cout << "Date:   " << c.timestamp << "\n\n";
                                      std::cout << "    " << c.message << "\n\n";
                                      ++printed;
                                      return true;
                                  });
    if (!st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }
    if (printed == 0) {
        std::cout << "fatal: your current branch has no commits yet\n";
    }
    return ExitCode::Ok;
}

}  // namespace

void register_log(CLI::App& app, int& exit_code) {
    static std::string revspec;
    static int max_count = -1;
    auto* c = app.add_subcommand("log", "Show commit history");
    c->add_option("revision", revspec, "Commit/branch to start from (default: HEAD)");
    c->add_option("-n,--max-count", max_count, "Limit the number of commits shown");
    c->callback([&exit_code] { exit_code = run_log(revspec, max_count); });
}

}  // namespace sfs::app::cmd
