/// The single `sfs` CLI binary. Subcommand implementations live in cmd/*.cpp;
/// this file only builds the CLI11 app and dispatches. docs/spec/15-cli-contract.md.

#include <CLI/CLI.hpp>

#include <iostream>

#include "exitcode.hpp"

namespace sfs::app::cmd {
// Each subcommand registers itself and returns the exit code to use when it
// runs. Declared here, defined in cmd/<name>.cpp — one file per command so
// this file never grows past dispatch.
void register_init(CLI::App& app, int& exit_code);
void register_commit(CLI::App& app, int& exit_code);
void register_checkout(CLI::App& app, int& exit_code);
void register_log(CLI::App& app, int& exit_code);
void register_verify(CLI::App& app, int& exit_code);
void register_branch(CLI::App& app, int& exit_code);
void register_merge(CLI::App& app, int& exit_code);
void register_gc(CLI::App& app, int& exit_code);
void register_serve(CLI::App& app, int& exit_code);
void register_push(CLI::App& app, int& exit_code);
void register_pull(CLI::App& app, int& exit_code);
#ifdef SFS_WITH_MOUNT
void register_mount(CLI::App& app, int& exit_code);
void register_unmount(CLI::App& app, int& exit_code);
#endif
}  // namespace sfs::app::cmd

int main(int argc, char** argv) {
    CLI::App app{"synapsefs — content-addressed checkpoint storage", "sfs"};
    app.require_subcommand(1);

    int exit_code = sfs::app::ExitCode::Ok;

    sfs::app::cmd::register_init(app, exit_code);
    sfs::app::cmd::register_commit(app, exit_code);
    sfs::app::cmd::register_checkout(app, exit_code);
    sfs::app::cmd::register_log(app, exit_code);
    sfs::app::cmd::register_verify(app, exit_code);
    sfs::app::cmd::register_branch(app, exit_code);
    sfs::app::cmd::register_merge(app, exit_code);
    sfs::app::cmd::register_gc(app, exit_code);
    sfs::app::cmd::register_serve(app,exit_code);
    sfs::app::cmd::register_push(app,exit_code);
    sfs::app::cmd::register_pull(app,exit_code);
#ifdef SFS_WITH_MOUNT
    sfs::app::cmd::register_mount(app, exit_code);
    sfs::app::cmd::register_unmount(app, exit_code);
#endif

    CLI11_PARSE(app, argc, argv);

    return exit_code;
}
