/// Subcommands architected in store::/net:: but not ported into this build
/// yet. Registering them here — instead of leaving them unrecognised by
/// CLI11 — means `sfs merge` names what's missing and exits 7
/// (ExitCode::NotImplemented) instead of CLI11 reporting an unknown
/// subcommand. See apps/sfs/CMakeLists.txt: an honest exit 7 beats silently
/// succeeding.

#include <CLI/CLI.hpp>

#include <iostream>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_not_implemented(const std::string& name) {
    std::cerr << "sfs " << name << ": not implemented in this build\n";
    return ExitCode::NotImplemented;
}

}  // namespace

void register_unimplemented(CLI::App& app, int& exit_code) {
    static const char* kNames[] = {"push", "pull", "serve"};
    for (const char* name : kNames) {
        auto* c = app.add_subcommand(name, std::string("(not implemented) ") + name);
        c->allow_extras();
        c->callback([&exit_code, name] { exit_code = run_not_implemented(name); });
    }
}

}  // namespace sfs::app::cmd
