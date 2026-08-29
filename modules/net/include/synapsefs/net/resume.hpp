#pragma once
/// \file resume.hpp
/// Resuming an interrupted sync.
///
/// THERE IS NO TRANSFER JOURNAL, and that is a design decision rather than an
/// omission. Content addressing already provides everything a journal would
/// record:
///
///   * fully received objects are in objects/ and are self-identifying
///   * partially received objects are in .synapsefs/incoming/<oid>.part with
///     their chunk digests from BLOCK_HDR; on resume we verify the chunks we
///     hold, discard from the first bad one, and ask for the remainder
///   * the want set is RECOMPUTED from current state — anything that arrived is
///     now a HAVE and is not requested again
///
/// Derived state cannot disagree with reality the way recorded state can. The
/// Python prototype had a resume module doing this bookkeeping, and deleting it
/// removed a whole class of "the journal says we have it but we don't" bugs.
///
/// What remains is thin, and correctness does not depend on it.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::net {

/// Lets a resumed connection skip re-negotiation when nothing changed. An
/// optimisation, not a correctness mechanism.
struct ResumeToken {
    std::uint64_t session_nonce = 0;
    std::string   ref_name;
    core::Oid     target;

    [[nodiscard]] std::string encode() const;
    [[nodiscard]] static core::Result<ResumeToken> decode(std::string_view);
};

struct PartialBlock {
    core::Oid              oid;
    std::uint64_t          verified_bytes = 0;   ///< prefix that passed its chunk digests
    std::uint64_t          total_bytes = 0;
    std::vector<std::byte> chunk_digests;
};

/// Scan .synapsefs/incoming/, verify each partial's chunks, and report how far
/// each one can be trusted. Anything after the first bad chunk is discarded.
[[nodiscard]] core::Result<std::vector<PartialBlock>> scan_incoming(
    const std::filesystem::path& incoming_dir);

/// Remove partials for objects that are now complete, and anything older than
/// this process's start time.
[[nodiscard]] core::Status prune_incoming(const std::filesystem::path& incoming_dir);

}  // namespace sfs::net
