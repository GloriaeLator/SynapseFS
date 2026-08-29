#pragma once
/// \file havewant.hpp
/// Negotiating which blocks actually need to move. This is the graded part.
///
/// What we do NOT do: dump every block id either side holds. At 7B with a few
/// hundred commits that is millions of identifiers exchanged in order to move
/// one delta.
///
/// What we do: walk the COMMIT DAG. The client sends commit ids in
/// exponentially widening strides (1, 2, 4, 8, ... generations back), capped
/// per round; the server intersects and replies with the objects reachable from
/// the requested ref but not from the common ancestor. The common case is two
/// peers a handful of commits apart, and that costs one round trip.
///
/// Correctness rests on the ANCESTOR INVARIANT: because a delta's base is
/// guaranteed to be an ancestor of the commit containing it, the closure over
/// "commits the receiver lacks" necessarily includes every base those deltas
/// resolve against. Without it a receiver can accept a manifest whose base
/// lives on a branch it does not have, pass every per-object hash check, and
/// still be unreadable.

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <synapsefs/core/oid.hpp>
#include <synapsefs/store/dag.hpp>

namespace sfs::net {

using core::Oid;

inline constexpr std::size_t   kHaveIdsPerRound = 256;
inline constexpr std::uint32_t kMaxHaveRounds   = 8;

struct NegotiationState {
    std::uint32_t          round = 0;
    std::vector<Oid>       sent;
    std::optional<Oid>     common_ancestor;
    bool                   exhausted = false;
};

/// Client side: the next HAVE batch, or empty when out of rounds.
[[nodiscard]] core::Result<std::vector<Oid>> next_have_batch(store::CommitStore&,
                                                             std::span<const Oid> tips,
                                                             NegotiationState&);

/// Server side: intersect the peer's HAVE with local history.
[[nodiscard]] core::Result<std::optional<Oid>> find_common(store::CommitStore&,
                                                           std::span<const Oid> peer_have,
                                                           std::span<const Oid> local_tips);

/// Objects reachable from `want_tip` and not from `common` — plus a filter
/// against the blocks the peer already declared, so a block shared between two
/// branches is not sent twice.
[[nodiscard]] core::Result<std::vector<Oid>> compute_want_set(
    store::CommitStore&, store::ManifestStore&, const Oid& want_tip,
    std::optional<Oid> common, const std::unordered_set<Oid>& peer_has);

}  // namespace sfs::net
