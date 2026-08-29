#pragma once
/// \file merge_logic.hpp
/// Three-way merge over tensor groups.
///
/// Fast-forward when one tip is an ancestor of the other. Otherwise, per group,
/// against the merge base:
///     changed on one side  -> take that side
///     changed on both      -> CONFLICT, refuse, require --ours / --theirs
///
/// We never average weights. As model-soup research that is a defensible idea;
/// as version control it is not — a VCS that silently produces an artifact
/// neither author wrote is broken. This is a likely Q&A question and the
/// tempting answer is the wrong one. docs/tradeoffs.md §2.2.

#include <optional>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/dag.hpp>

namespace sfs::store {

enum class MergeStrategy : std::uint8_t { Refuse, Ours, Theirs };

enum class GroupResolution : std::uint8_t {
    Unchanged, TakeOurs, TakeTheirs, Conflict,
};

struct GroupConflict {
    std::string group;
    Oid         base_block;
    Oid         our_block;
    Oid         their_block;
};

struct MergeResult {
    bool                      fast_forward = false;
    std::optional<Oid>        new_commit;
    std::optional<format::Manifest> merged;      ///< unset when conflicted
    std::vector<GroupConflict>      conflicts;
};

/// Decide the merge without writing anything. `merge` in the CLI writes only
/// after this returns clean, so a conflicted merge leaves the repository
/// untouched.
[[nodiscard]] Result<MergeResult> plan_merge(CommitStore&, ManifestStore&,
                                             const Oid& ours, const Oid& theirs,
                                             MergeStrategy);

/// Per-group decision against the base. Exposed for test_merge_logic.cpp, which
/// exercises it without a repository.
[[nodiscard]] GroupResolution resolve_group(const format::GroupEntry* base,
                                            const format::GroupEntry* ours,
                                            const format::GroupEntry* theirs) noexcept;

}  // namespace sfs::store
