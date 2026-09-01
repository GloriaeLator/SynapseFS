#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/net/synapse_sync.hpp>
#include <synapsefs/core/repo_config.hpp>
#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_push(const std::string remote_url){
            auto paths = core::RepoPaths::discover(std::filesystem::current_path());
    if (!paths) {
        std::cerr << "error: not a synapsefs repository\n";
        return ExitCode::NotARepository;
    }
            net::push(remote_url);
            return ExitCode::Ok;
        }
    }

    void register_push(CLI::App& app, int& exit_code){
        static std::string remote_url;
        auto* c = app.add_subcommand("push","Push commit history to other user.");
        c->add_option("remote_url",remote_url,"http://ip:port Of the User to PUSH to.")->required();
        c->callback([&exit_code] {exit_code = run_push(remote_url);});
    }
}