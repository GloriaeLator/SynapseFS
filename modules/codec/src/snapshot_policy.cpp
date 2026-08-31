#include <synapsefs/codec/snapshot_policy.hpp>

namespace sfs::codec {

std::string_view to_string(StorageDecision d) noexcept {
    switch (d) {
        case StorageDecision::Delta:             return "delta";
        case StorageDecision::FullNoBase:        return "full_no_base";
        case StorageDecision::FullNotAlignable:  return "full_not_alignable";
        case StorageDecision::FullChainTooDeep:  return "full_chain_too_deep";
        case StorageDecision::FullDeltaTooLarge: return "full_delta_too_large";
    }
    return "?";
}

StorageDecision decide(const SnapshotInputs& in, const core::RepoConfig& cfg) noexcept {
    // Four conditions, checked in the order spec 12 §7 lists them — cheapest
    // and most certain first, so a group that clearly can't be a delta
    // (no base at all) never pays for the alpha comparison below it.
    if (!in.has_base) return StorageDecision::FullNoBase;
    if (!in.alignable) return StorageDecision::FullNotAlignable;
    if (in.base_chain_depth + 1 > cfg.max_chain_depth) return StorageDecision::FullChainTooDeep;

    // Free to evaluate: the writer already holds both byte strings at this
    // point. Exists because XOR/zigzag of two UNRELATED fp16 tensors is
    // high-entropy noise that compresses to LARGER than its input — measured
    // directly on this branch (docs/tradeoffs.md §1.4: unrelated-checkpoints
    // ratio 1.0001) — so without this bound a badly aligned group can make
    // the repository grow faster than storing full copies.
    const double delta = static_cast<double>(in.delta_bytes);
    const double full = static_cast<double>(in.full_bytes);
    if (delta > cfg.snapshot_alpha * full) return StorageDecision::FullDeltaTooLarge;

    return StorageDecision::Delta;
}

}  // namespace sfs::codec
