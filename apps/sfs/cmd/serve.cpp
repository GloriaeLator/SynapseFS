#include <CLI/CLI.hpp>

#include <iostream>
#include <synapsefs/net/repository.hpp>
#include <synapsefs/net/network.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {
    namespace run_branch
    {
        int run_serve(int port){
            net::Repository target = std::filesystem::current_path();
            net::PeerServer(target, <uint16_t>port).serve_forever();
            return ExitCode::Ok;
        }
    } // namespace run_branch
    
    void register_serve(CLI::App& app, int& exitcode){
    static int port;
    auto* c = app.add_subcommand("serve","Start your serve to allow push & pull requests from other.");
    c->add_option("-p,--port",port, "Specify PORT number, By Default 8002");
    c->callback([&exitcode] {exit_code = run_serve(port)})
}
} //namespace sfs::app::cmd

