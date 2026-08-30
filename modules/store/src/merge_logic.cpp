#include <synapsefs/store/merge_logic.hpp>

#include <synapsefs/store/manifest_store.hpp>

namespace sfs::store {

namespace {

/// Two group entries are "the same content" when they resolve to the same
/// bytes. For a Full group that is its block oid; for a Delta group it is the
/// diff block AND its base, since the same residual against a different base
/// is different content. Comparing the whole entry would also work but would
/// call chain_depth differences a change, which they are not.
bool same_content(const format::GroupEntry& a, const format::GroupEntry& b) noexcept {
    if (a.mode != b.mode) return false;
    if (a.mode == format::GroupMode::Full) return a.block == b.block;

    if (a.diff_block != b.diff_block) return false;
    if (a.base.has_value() != b.base.has_value()) return false;
    if (!a.base) return true;
    return a.base->commit == b.base->commit && a.base->group == b.base->group;
}

/// The block a conflict report should name for a side. Full groups have
/// `block`, Delta groups have `diff_block`; a group with neither has already
/// been rejected by Manifest::validate().
Oid reportable_block(const format::GroupEntry* g) noexcept {
    if (!g) return Oid{};
    if (g->block) return *g->block;
    if (g->diff_block) return *g->diff_block;
    return Oid{};
}

}  // namespace

GroupResolution resolve_group(const format::GroupEntry* base, const format::GroupEntry* ours,
                              const format::GroupEntry* theirs) noexcept {
    // Added on one side only, or added identically on both. A group that
    // exists on neither side cannot reach here (the caller iterates the union
    // of both sides' group names).
    if (!ours && !theirs) return GroupResolution::Unchanged;
    if (!ours) return base ? GroupResolution::Conflict   // deleted by us, changed by them
                           : GroupResolution::TakeTheirs;  // added by them only
    if (!theirs) return base ? GroupResolution::Conflict   // changed by us, deleted by them
                             : GroupResolution::TakeOurs;   // added by us only

    if (same_content(*ours, *theirs)) return GroupResolution::Unchanged;

    // Both sides present and differing. Without a base, both sides added the
    // same group name with different content: that is a conflict, not a
    // silent pick.
    if (!base) return GroupResolution::Conflict;

    bool ours_changed = !same_content(*base, *ours);
    bool theirs_changed = !same_content(*base, *theirs);

    if (ours_changed && theirs_changed) return GroupResolution::Conflict;
    if (theirs_changed) return GroupResolution::TakeTheirs;
    if (ours_changed) return GroupResolution::TakeOurs;
    return GroupResolution::Unchanged;
}

Result<MergeResult> plan_merge(CommitStore& commits, ManifestStore& manifests, const Oid& ours,
                               const Oid& theirs, MergeStrategy strategy) {
    MergeResult result;

    if (ours == theirs) {
        // Already the same commit: nothing to do, and reporting it as a
        // fast-forward to itself is the least surprising answer.
        result.fast_forward = true;
        result.new_commit = ours;
        return result;
    }

    // Fast-forward: if ours is an ancestor of theirs, theirs already contains
    // our history and the merge is a ref move. The reverse (theirs is an
    // ancestor of ours) is a no-op for the same reason.
    auto ours_is_ancestor = is_ancestor(commits, ours, theirs);
    if (!ours_is_ancestor) return std::unexpected(ours_is_ancestor.error());
    if (*ours_is_ancestor) {
        result.fast_forward = true;
        result.new_commit = theirs;
        return result;
    }

    auto theirs_is_ancestor = is_ancestor(commits, theirs, ours);
    if (!theirs_is_ancestor) return std::unexpected(theirs_is_ancestor.error());
    if (*theirs_is_ancestor) {
        result.fast_forward = true;
        result.new_commit = ours;
        return result;
    }

    auto base_oid = merge_base(commits, ours, theirs);
    if (!base_oid) return std::unexpected(base_oid.error());
    if (!*base_oid) {
        // Unrelated histories. merge_base reports this rather than guessing
        // (dag.hpp), and so do we.
        return SFS_ERR(MergeConflict, "refusing to merge unrelated histories");
    }

    auto our_commit = commits.read(ours);
    if (!our_commit) return std::unexpected(our_commit.error());
    auto their_commit = commits.read(theirs);
    if (!their_commit) return std::unexpected(their_commit.error());
    auto base_commit = commits.read(**base_oid);
    if (!base_commit) return std::unexpected(base_commit.error());

    auto our_manifest = manifests.read(our_commit->manifest);
    if (!our_manifest) return std::unexpected(our_manifest.error());
    auto their_manifest = manifests.read(their_commit->manifest);
    if (!their_manifest) return std::unexpected(their_manifest.error());
    auto base_manifest = manifests.read(base_commit->manifest);
    if (!base_manifest) return std::unexpected(base_manifest.error());

    // The merged file is ours structurally — name, header block, buffer
    // layout. Merging two checkpoints with different tensor sets or a
    // different buffer layout is not a group-level merge at all, so we
    // refuse rather than produce a file whose layout matches neither parent.
    if (our_manifest->file.header_block != their_manifest->file.header_block) {
        return SFS_ERR(MergeConflict,
                       "checkpoint headers differ; the two branches are not the same file shape");
    }

    format::Manifest merged = *our_manifest;
    merged.groups.clear();

    // Union of group names across both sides, so a group added on either
    // side is considered.
    std::vector<std::string> names;
    names.reserve(our_manifest->groups.size() + their_manifest->groups.size());
    for (const auto& [name, _] : our_manifest->groups) names.push_back(name);
    for (const auto& [name, _] : their_manifest->groups) {
        if (!our_manifest->groups.count(name)) names.push_back(name);
    }

    for (const auto& name : names) {
        const auto* b = base_manifest->find_group(name);
        const auto* o = our_manifest->find_group(name);
        const auto* t = their_manifest->find_group(name);

        GroupResolution decision = resolve_group(b, o, t);

        if (decision == GroupResolution::Conflict) {
            switch (strategy) {
                case MergeStrategy::Ours:
                    decision = o ? GroupResolution::TakeOurs : GroupResolution::Unchanged;
                    break;
                case MergeStrategy::Theirs:
                    decision = t ? GroupResolution::TakeTheirs : GroupResolution::Unchanged;
                    break;
                case MergeStrategy::Refuse:
                    result.conflicts.push_back(GroupConflict{
                        name, reportable_block(b), reportable_block(o), reportable_block(t)});
                    continue;
            }
        }

        switch (decision) {
            case GroupResolution::TakeTheirs:
                if (t) merged.groups[name] = *t;
                break;
            case GroupResolution::TakeOurs:
            case GroupResolution::Unchanged:
                if (o) merged.groups[name] = *o;
                else if (t) merged.groups[name] = *t;
                break;
            case GroupResolution::Conflict:
                break;  // handled above
        }
    }

    // Conflicted: report and write nothing. plan_merge's contract is that a
    // conflicted merge leaves the repository untouched, and returning no
    // manifest is how the caller is stopped from writing one.
    if (!result.conflicts.empty()) return result;

    // The merged manifest must still describe a valid file: every buffer
    // entry's group has to exist. Taking one side per group can only break
    // that if a group was deleted on one side, which resolve_group already
    // treats as a conflict — validate() is the check that this held.
    if (auto st = merged.validate(); !st) return std::unexpected(st.error());

    result.merged = std::move(merged);
    return result;
}

}  // namespace sfs::store
