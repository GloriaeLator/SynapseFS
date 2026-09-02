#include <CLI/CLI.hpp>

#include <iostream>
#include <synapsefs/net/synapse_sync.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {
    namespace
    {
        int run_serve(int port){
            net::serve(port);
            return ExitCode::Ok;
        }
    } // namespace run_serves
    
    void register_serve(CLI::App& app, int& exit_code){
    // Previously had no ->default_val(), so an unspecified -p left `port`
    // zero-initialised and bound an OS-assigned ephemeral port -- while the
    // help text here claimed 8002 and RepoConfig::listen documents 9418
    // (docs/known-gaps.md's "sfs serve default port" row). 9418 matches
    // RepoConfig::listen's documented default.
    static int port = 9418;
    auto* c = app.add_subcommand("serve","Start your serve to allow push & pull requests from other.");
    c->add_option("-p,--port",port, "Specify PORT number")->default_val(9418);
    c->callback([&exit_code] { exit_code = run_serve(port);});
}
} //namespace sfs::app::cmd

