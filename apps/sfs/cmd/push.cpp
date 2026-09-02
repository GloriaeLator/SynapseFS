#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/net/synapse_sync.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd{
    namespace{
        int run_push(const std::string remote_url){
            // net::push()'s return value used to be discarded here, so a
            // failed transfer was invisible at the exit-code level
            // (docs/known-gaps.md's "push/pull exit codes" row). It's still
            // an incomplete signal -- push has no ack channel back from the
            // receiver's own integrity check, see synapse_sync.cpp's
            // push()/serve() comments -- but a connection failure or a local
            // socket error now at least surfaces as ExitCode::Network
            // instead of a silent Ok.
            if (!net::push(remote_url)) {
                std::cerr << "error: push failed (could not connect, or the transfer was "
                             "interrupted)\n";
                return ExitCode::Network;
            }
            // net::push() returns a plain bool -- no object count to report,
            // unlike pull's per-object "Successfully synced" lines (those
            // come from the RECEIVER's own write path in synapse_sync.cpp,
            // which during a push is the remote server, not this terminal).
            // Still worth a visible confirmation instead of silent success.
            std::cout << "Pushed to " << remote_url << "\n";
            return ExitCode::Ok;
        }
    }

    void register_push(CLI::App& app, int& exit_code){
        static std::string remote_url;
        auto* c = app.add_subcommand("push","Push commit history to other user.");
        c->add_option("remote_url",remote_url,
                      "ip:port of the remote to PUSH to (e.g. 127.0.0.1:9418)"
                      )
            ->required();
        c->callback([&exit_code] {exit_code = run_push(remote_url);});
    }
}