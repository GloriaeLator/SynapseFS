#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_init(const std::string& dir) {
    std::filesystem::path root = dir.empty() ? std::filesystem::current_path()
                                             : std::filesystem::path(dir);
    core::RepoPaths paths{root};

    std::error_code ec;
    if (std::filesystem::exists(paths.dot(), ec)) {
        std::cerr << "error: " << paths.dot().string() << " already exists\n";
        return ExitCode::Failure;
    }

    std::filesystem::create_directories(paths.objects(), ec);
    std::filesystem::create_directories(paths.tmp(), ec);
    std::filesystem::create_directories(paths.refs_heads(), ec);
    std::filesystem::create_directories(paths.journal(), ec);
    if (ec) {
        std::cerr << "error: cannot create repository layout: " << ec.message() << "\n";
        return ExitCode::Failure;
    }

    core::RepoConfig cfg;  // defaults per repo_config.hpp
    if (auto st = cfg.save(root); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    store::RefStore refs(paths);
    if (auto st = refs.set_head_symbolic("refs/heads/main"); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    std::cout << "Initialized empty synapsefs repository in " << paths.dot().string() << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_init(CLI::App& app, int& exit_code) {
    static std::string dir;
    auto* c = app.add_subcommand("init", "Create a new repository");
    c->add_option("directory", dir, "Where to initialize (default: current directory)");
    c->callback([&exit_code] { exit_code = run_init(dir); });
}

}  // namespace sfs::app::cmd
