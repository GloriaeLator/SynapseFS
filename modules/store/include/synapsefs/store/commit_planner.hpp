#pragma once
/// \file commit_planner.hpp
/// The align <-> codec glue: turns one align::MatchReport into per-tensor
/// storage decisions, writing whatever objects each decision needs.
///
/// This is NOT apps/sfs/cmd/commit.cpp's writer path — it has no access to a
/// repository's config file, refs, or lockfile. It IS what that command's
/// own comment refers to: "this command's manifest-building logic does not
/// need to change when [alignment] lands — only how each group's
/// block/base gets filled in." This file is that "how".
///
/// Six things this does, in order:
///   1. A group align reports as identity, or that is pinned in the
///      topology, or that align says is not alignable, is never diffed —
///      it and every tensor bound to it fall through to Full storage.
///   2. Storage is decided by codec::snapshot_policy::decide(), using the
///      ACTUAL encoded artifact size — encode first, decide after
///      (spec 12 §7: "Rule 4 is free — the writer already holds both byte
///      strings"), not assumed to always be Delta.
///   3. One encode_group() call per permutation group, covering every
///      tensor bound to it (spec 13's conv+BatchNorm example; the golden
///      fixture tests/golden/diff_artifact.json holds two tensors) — but
///      format::Manifest.groups stays keyed one entry PER TENSOR
///      (apps/sfs/cmd/commit.cpp's existing convention), several sharing
///      one diff_block.
///   4. A tensor whose secondary (non-dim-0) axis is bound to a DIFFERENT,
///      already-resolved group has its base rows column-permuted on the
///      fly — see ColumnPermutingSource in the .cpp — so encode_group's
///      existing row-gather logic never has to know a second axis moved.
///   5. format::AlignmentInfo is filled straight from GroupMatch's cost
///      fields, no translation needed.
///   6. `report` must already hold EVERY group's final permutation (i.e.
///      come from a completed align::Matcher::run()) — this function makes
///      no alignment decisions of its own, and needs no particular
///      processing order across groups because of that.

#include <string>
#include <unordered_map>

#include <synapsefs/align/matcher.hpp>
#include <synapsefs/codec/diff_encoder.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/core/topology.hpp>
#include <synapsefs/format/manifest.hpp>

namespace sfs::store {

/// What the planner needs about a tensor's PARENT (base commit) entry, to
/// fill in a Delta's DeltaBase and chain_depth. Absent for a tensor with no
/// parent (first commit, or a tensor new to this commit) — in which case
/// SnapshotInputs::has_base is false and the tensor is stored Full,
/// regardless of what align found for its group.
struct ParentTensorInfo {
    core::Oid     parent_commit;
    std::uint32_t chain_depth = 0;
};

/// Decide storage for every tensor `target` names, and WRITE whichever
/// objects each decision needs (Raw blocks for Full, Diff artifacts for
/// Delta) into `blocks`. Returns one GroupEntry per tensor, ready to fold
/// into a format::Manifest alongside BufferEntry construction from
/// `target.buffer_layout()` — exactly what apps/sfs/cmd/commit.cpp already
/// does for the Full-only case today, just no longer hardcoded to Full.
[[nodiscard]] core::Result<std::unordered_map<std::string, format::GroupEntry>>
plan_commit_groups(core::ITensorSource& base, core::ITensorSource& target,
                   const core::Topology& topology, const align::MatchReport& report,
                   const std::unordered_map<std::string, ParentTensorInfo>& parent_info,
                   core::IBlockStore& blocks, const core::RepoConfig& cfg,
                   codec::EncodeOptions encode_opts = {});

}  // namespace sfs::store
