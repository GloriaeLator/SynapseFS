#include <CLI/CLI.h>

#include <iostream>

#include <synapsefs/net/repository.hpp>
#include <synapsefs/net/network.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_push(const std::string branch , const std::string remote_url){
            net::Repository target = std::filesystem::current_path();
            net::push_branch(target,remote_url,branch);
            return ExitCode::Ok;
        }
    }

    void register_push(CLI::App& app, int& exit_code){
        static std::string branch;
        static std::string remote_url;
        auto c* = app.add_subcommand("push","Push commit history to other user.");
        c->add_option("branch",branch,"Select Branch to Push")->required();
        c->add_option("remote_url",remote_url,"http://ip:port Of the User to PUSH to.")->required();
        c->callback([&exit_code] {exit_code = run_push(branch,remote_url)})
    }
}