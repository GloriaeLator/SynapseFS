#pragma once
/// \file dag.hpp
/// Commit DAG walks: ancestry, merge bases, and reachability.
///
/// Reachability is the definition of "live" for gc (docs/spec/11 §6) and the
/// definition of "what to send" for push (docs/spec/14 §3.2). It is one
/// function so that those two can never disagree.

#include <functional>
#include <unordered_set>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/store/commit_store.hpp>

namespace sfs::store {

/// Walk commits from `tips` in reverse-chronological-ish order (topological,
/// ties broken by timestamp). `visit` returning false stops that branch.
[[nodiscard]] Status walk_commits(CommitStore&, std::span<const Oid> tips,
                                  const std::function<bool(const Oid&,
                                                           const format::Commit&)>& visit);

[[nodiscard]] Result<bool> is_ancestor(CommitStore&, const Oid& maybe_ancestor,
                                       const Oid& of);

/// Lowest common ancestor. Empty result means unrelated histories, which
/// `merge` reports rather than guessing at.
[[nodiscard]] Result<std::optional<Oid>> merge_base(CommitStore&, const Oid& a, const Oid& b);

/// Every object reachable from `tips`: commits, their manifests and topologies,
/// each manifest's header block and every group's block or diff_block.
/// This is the ONE definition used by both gc and push.
[[nodiscard]] Result<std::unordered_set<Oid>> reachable_objects(
    CommitStore&, class ManifestStore&, std::span<const Oid> tips);

/// Commit ids walked back from `tips` in exponentially widening strides
/// (1, 2, 4, 8, … generations), capped at `limit`. This is the HAVE set for
/// have/want negotiation: the common case is two peers a handful of commits
/// apart, and that costs one round trip. docs/spec/14 §3.2.
[[nodiscard]] Result<std::vector<Oid>> have_probe(CommitStore&, std::span<const Oid> tips,
                                                  std::size_t limit, std::uint32_t round);

}  // namespace sfs::store
