#pragma once
/// \file verify.hpp
/// `sfs verify`. Must work STANDALONE — no checkout, no mount. That is a PS
/// requirement and it is 10% of the grade on its own.
///
/// Checks, in order:
///   1. every ref resolves to a commit that exists
///   2. each commit hashes to its id, and canonical re-serialisation reproduces it
///   3. manifest and topology exist and are well-formed
///   4. buffer layout: no gaps, no overlaps, total == file.total_bytes
///   5. every referenced block exists and matches its address
///   6. the ANCESTOR INVARIANT for every delta group
///   7. chain_depth consistency down every chain
///   8. --full: every chunk of every reachable object
///
/// Exit code 4 on any integrity failure, naming the object AND the chunk.
/// "Something is wrong" is a fail.

#include <cstdint>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/store/dag.hpp>
#include <synapsefs/store/manifest_store.hpp>

namespace sfs::store {

struct VerifyOptions {
    bool full   = false;   ///< re-hash every chunk of every reachable object
    bool repair = false;   ///< replay or roll back a crashed journal first
    bool stop_on_first_error = false;
};

struct VerifyFinding {
    core::ErrKind kind{};
    Oid           object;
    std::string   detail;
    std::optional<std::uint32_t> chunk_index;   ///< set when a chunk digest failed
    std::optional<std::string>   group;
};

struct VerifyReport {
    std::uint64_t commits_walked = 0;
    std::uint64_t objects_checked = 0;
    std::uint64_t bytes_hashed    = 0;
    std::vector<VerifyFinding> findings;

    [[nodiscard]] bool ok() const noexcept { return findings.empty(); }
};

[[nodiscard]] Result<VerifyReport> verify(core::IBlockStore&, CommitStore&, ManifestStore&,
                                          RefStore&, std::span<const Oid> tips,
                                          const VerifyOptions& = {});

/// The invariant that makes push correct: every delta's base commit must be an
/// ancestor of the commit containing it. Without it a peer can accept a
/// manifest whose base lives on a branch it does not have, pass every
/// per-object hash check, and still be unreadable.
[[nodiscard]] Status check_ancestor_invariant(CommitStore&, const Oid& commit,
                                              const format::Manifest&);

}  // namespace sfs::store
