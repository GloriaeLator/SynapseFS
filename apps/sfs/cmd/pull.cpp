#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/net/repository.hpp>
#include <synapsefs/net/network.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_pull(const std::string branch , const std::string remote_url){
            net::Repository target = net::Repository(std::filesystem::current_path());
            net::pull_branch(target,remote_url,branch);
            return ExitCode::Ok;
        }
    }

    void register_pull(CLI::App& app, int& exit_code){
        static std::string branch;
        static std::string remote_url;
        auto* c = app.add_subcommand("pull","Pull commit history to other user.");
        c->add_option("branch",branch,"Select Branch to Pull")->required();
        c->add_option("remote_url",remote_url,"http://ip:port Of the User to PULL From.")->required();
        c->callback([&exit_code] {exit_code = run_pull(branch,remote_url);});
    }
}