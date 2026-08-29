#pragma once
/// \file session.hpp
/// One push or pull, driven over an ITransport.
///
/// Ordering rule, on the wire exactly as on disk: OBJECTS BEFORE REFS. The
/// REF_UPDATE frame is sent only after every object it depends on has been
/// acknowledged, so a crash mid-push leaves the receiver with unreferenced
/// objects — garbage, not corruption.
///
/// Every received block goes through the FULL verify path, not the fast one:
/// this is untrusted input.

#include <memory>
#include <unordered_set>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/net/havewant.hpp>
#include <synapsefs/net/protocol.hpp>
#include <synapsefs/store/manifest_store.hpp>

namespace sfs::net {

struct SessionStats {
    std::uint64_t blocks_sent = 0;
    std::uint64_t blocks_received = 0;
    std::uint64_t bytes_transferred = 0;
    std::uint64_t blocks_skipped = 0;   ///< already present — the number that shows
                                        ///< differential transfer is real
    std::uint32_t negotiation_rounds = 0;
};

struct SessionDeps {
    core::IBlockStore*    blocks = nullptr;
    store::CommitStore*   commits = nullptr;
    store::ManifestStore* manifests = nullptr;
    store::RefStore*      refs = nullptr;
    core::RepoPaths       paths;
};

class Session {
public:
    Session(core::ITransport&, SessionDeps);
    ~Session();

    [[nodiscard]] core::Status push(std::string_view ref_name, bool force);
    [[nodiscard]] core::Status pull(std::string_view ref_name);

    /// Server side: handle frames until DONE or the connection drops.
    [[nodiscard]] core::Status serve_once(bool read_only);

    [[nodiscard]] const SessionStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfs::net
