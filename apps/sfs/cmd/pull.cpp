#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/net/synapse_sync.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_pull(const std::string remote_url){
            net::pull(remote_url);
            return ExitCode::Ok;
        }
    }

    void register_pull(CLI::App& app, int& exit_code){
        static std::string branch;
        static std::string remote_url;
        auto* c = app.add_subcommand("pull","Pull commit history to other user.");
        c->add_option("remote_url",remote_url,"http://ip:port Of the User to PULL From.")->required();
        c->callback([&exit_code] {exit_code = run_pull(remote_url);});
    }
}