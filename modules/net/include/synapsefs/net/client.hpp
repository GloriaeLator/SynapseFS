#pragma once
/// \file client.hpp
/// TCP transport and the push/pull entry points.

#include <chrono>
#include <memory>
#include <string>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/net/session.hpp>

namespace sfs::net {

class TcpTransport final : public core::ITransport {
public:
    [[nodiscard]] static core::Result<std::unique_ptr<TcpTransport>> connect(
        std::string_view host_port, std::chrono::milliseconds timeout);
    [[nodiscard]] static std::unique_ptr<TcpTransport> adopt(int fd);
    ~TcpTransport() override;

    [[nodiscard]] core::Status send(FrameType, std::span<const std::byte>) override;
    [[nodiscard]] core::Result<core::WireFrame> recv(std::chrono::milliseconds) override;
    void close() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    TcpTransport();
};

/// In-process transport, so test_havewant.cpp runs the REAL negotiation with no
/// sockets and no ports.
[[nodiscard]] std::pair<std::unique_ptr<core::ITransport>, std::unique_ptr<core::ITransport>>
make_pipe_transport();

/// Drops the connection after `cut_after_bytes`. Deterministic, which is what
/// makes tests/sync_interrupt.cpp a test rather than a race.
[[nodiscard]] std::unique_ptr<core::ITransport> make_flaky_transport(
    std::unique_ptr<core::ITransport> inner, std::uint64_t cut_after_bytes);

[[nodiscard]] core::Result<SessionStats> push(SessionDeps, std::string_view url,
                                              std::string_view ref_name, bool force);
[[nodiscard]] core::Result<SessionStats> pull(SessionDeps, std::string_view url,
                                              std::string_view ref_name);

}  // namespace sfs::net
