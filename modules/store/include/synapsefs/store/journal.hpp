#pragma once
/// \file journal.hpp
/// Intent records for the two operations that touch more than one ref-like
/// file and therefore cannot be made atomic by rename alone:
///   merge     - a branch ref plus HEAD
///   gc --pack - add a packfile, remove loose objects
///
/// Everything else uses atomic rename and needs no journal at all
/// (docs/adr/0007-crash-safety-journal-vs-rename.md). Keeping the journal this
/// small is the point: it is the only part of the system with recoverable
/// states, so there should be as few of them as possible.
///
/// On open, a leftover record means a previous process died mid-operation.
/// Replay if the record is complete and its effects are idempotent; otherwise
/// REFUSE and tell the user to run `sfs verify --repair`. The PS explicitly
/// allows refusing, and refusing beats guessing.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::store {

using core::Oid;
using core::Result;
using core::Status;

enum class JournalOp : std::uint8_t { Merge, Pack };

struct JournalRecord {
    std::uint32_t format_version = 1;
    JournalOp     op{};
    std::uint64_t seq = 0;
    std::string   timestamp;

    /// Merge: the ref, its expected old value, its new value, and the HEAD
    /// update that must accompany it.
    std::string           ref_name;
    std::optional<Oid>    ref_old;
    std::optional<Oid>    ref_new;

    /// Pack: the new pack file and the loose objects it subsumes.
    std::string           pack_name;
    std::vector<Oid>      subsumed;

    /// Framed and digested, so a torn record is DETECTED rather than replayed.
    [[nodiscard]] std::vector<std::byte> encode() const;
    [[nodiscard]] static Result<JournalRecord> decode(std::span<const std::byte>);
};

class Journal {
public:
    explicit Journal(std::filesystem::path dir);

    /// Written atomically BEFORE the first mutation.
    [[nodiscard]] Result<std::uint64_t> begin(const JournalRecord&);
    /// Removed after the last mutation.
    [[nodiscard]] Status commit(std::uint64_t seq);

    [[nodiscard]] Result<std::vector<JournalRecord>> pending() const;

    /// Replay or roll back. Returns ErrKind::JournalTorn if a record cannot be
    /// decoded — the caller must then refuse, not improvise.
    [[nodiscard]] Status recover(class RefStore&);

private:
    std::filesystem::path dir_;
};

}  // namespace sfs::store
