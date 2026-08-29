#pragma once
/// \file server.hpp
/// `sfs serve`. Default 127.0.0.1:9418, overridable by --listen and by
/// .synapsefs/config. The address, port and config MUST be documented in the
/// README — it is a listed deliverable.
///
/// One thread per connection. Takes LOCK_SH while streaming, so `gc` cannot
/// remove an object mid-transfer.
///
/// No authentication, no TLS. See docs/threat_model.md for why that is the
/// right scope here and exactly what it costs.

#include <atomic>
#include <memory>
#include <string>

#include <synapsefs/net/session.hpp>

namespace sfs::net {

struct ServerOptions {
    std::string   listen = "127.0.0.1:9418";
    bool          read_only = false;
    std::uint32_t max_connections = 16;
};

class Server {
public:
    [[nodiscard]] static core::Result<std::unique_ptr<Server>> bind(SessionDeps,
                                                                    ServerOptions);
    ~Server();

    /// Blocks. Returns when stop() is called or the listener fails.
    [[nodiscard]] core::Status run();
    void stop() noexcept;

    /// The bound address, after binding to port 0 in tests.
    [[nodiscard]] std::string address() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Server();
};

}  // namespace sfs::net
