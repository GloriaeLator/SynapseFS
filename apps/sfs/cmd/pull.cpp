#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/net/synapse_sync.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_pull(const std::string remote_url){
            // net::pull()'s return value used to be discarded here
            // (docs/known-gaps.md's "push/pull exit codes" row). Pull is the
            // direction that DOES know its own integrity outcome --
            // receiver_sync_loop now verifies every received object's
            // payload against its claimed address (synapse_sync.cpp) -- so a
            // corrupted or interrupted transfer now genuinely fails the
            // command instead of reporting Ok with a half-synced repo.
            if (!net::pull(remote_url)) {
                std::cerr << "error: pull failed (could not connect, the transfer was "
                             "interrupted, or a received object failed its integrity check."
                             "see stderr above for which file)\n";
                return ExitCode::Network;
            }
            return ExitCode::Ok;
        }
    }

    void register_pull(CLI::App& app, int& exit_code){
        static std::string branch;
        static std::string remote_url;
        auto* c = app.add_subcommand("pull","Pull commit history to other user.");
        c->add_option("remote_url",remote_url,
                      "ip:port of the remote to PULL from (e.g. 127.0.0.1:9418)"
                      )
            ->required();
        c->callback([&exit_code] {exit_code = run_pull(remote_url);});
    }
}