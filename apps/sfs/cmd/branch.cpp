/// `sfs branch` — create or list branches. Never switches; that is
/// `checkout <branch-name>`, pre-2.23 git semantics (refs.hpp).

#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_branch(const std::string& name, const std::string& start_point, bool delete_flag,
               bool force) {
    auto paths = core::RepoPaths::discover(std::filesystem::current_path());
    if (!paths) {
        std::cerr << "error: not a synapsefs repository\n";
        return ExitCode::NotARepository;
    }

    store::RefStore refs(*paths);

    if (delete_flag) {
        if (name.empty()) {
            std::cerr << "error: --delete requires a branch name\n";
            return ExitCode::Usage;
        }
        if (auto st = refs.delete_branch(name, force); !st) {
            std::cerr << "error: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
        std::cout << "Deleted branch " << name << "\n";
        return ExitCode::Ok;
    }

    if (name.empty()) {
        // List: mark the branch HEAD points at, like `git branch`.
        auto heads = refs.list_heads();
        if (!heads) {
            std::cerr << "error: " << heads.error().to_string() << "\n";
            return exit_code_for(heads.error());
        }
        auto head = refs.read_head();
        std::optional<std::string> current =
            (head && !head->is_detached()) ? head->symbolic : std::nullopt;

        for (const auto& h : *heads) {
            bool is_current = current && *current == h.name;
            std::string short_name = h.name.starts_with("refs/heads/")
                                         ? h.name.substr(11)
                                         : h.name;
            std::cout << (is_current ? "* " : "  ") << short_name << "  " << h.target.abbrev()
                      << "\n";
        }
        return ExitCode::Ok;
    }

    core::Oid at;
    if (start_point.empty()) {
        auto head = refs.read_head();
        if (!head) {
            std::cerr << "error: " << head.error().to_string() << "\n";
            return exit_code_for(head.error());
        }
        core::Result<core::Oid> tip = head->is_detached()
            ? core::Result<core::Oid>(*head->detached)
            : refs.resolve(*head->symbolic);
        if (!tip) {
            std::cerr << "error: HEAD has no commits yet; give a start point\n";
            return ExitCode::Failure;
        }
        at = *tip;
    } else {
        auto r = refs.rev_parse(start_point);
        if (!r) {
            std::cerr << "error: " << r.error().to_string() << "\n";
            return exit_code_for(r.error());
        }
        at = *r;
    }

    if (auto st = refs.create_branch(name, at); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }
    std::cout << "Created branch " << name << " at " << at.abbrev() << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_branch(CLI::App& app, int& exit_code) {
    static std::string name;
    static std::string start_point;
    static bool delete_flag = false;
    static bool force = false;
    auto* c = app.add_subcommand("branch", "Create or list branches");
    c->add_option("name", name, "Branch to create (omit to list all branches)");
    c->add_option("start-point", start_point, "Commit to branch from (default: HEAD)");
    c->add_flag("-d,--delete", delete_flag, "Delete the named branch");
    c->add_flag("-f,--force", force, "Force delete even if unreachable from another ref");
    c->callback([&exit_code] { exit_code = run_branch(name, start_point, delete_flag, force); });
}

}  // namespace sfs::app::cmd
