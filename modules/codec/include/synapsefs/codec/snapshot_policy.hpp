#pragma once
/// \file snapshot_policy.hpp
/// When to store a group full instead of as a delta.
///
/// Four conditions, all cheap at the point they are needed:
///   1. no base (first commit, or a new group)
///   2. the aligner reported alignable = false
///   3. base.chain_depth + 1 > max_chain_depth   -> bounds read LATENCY
///   4. len(delta) > alpha * len(full)           -> bounds SPACE
///
/// Rule 4 is free: the writer is already holding both byte strings. It exists
/// because the XOR of two UNRELATED fp16 tensors is high-entropy noise that
/// compresses to LARGER than its input, so an unbounded design can make a
/// repository grow faster than storing full copies.

#include <cstdint>

#include <synapsefs/core/repo_config.hpp>

namespace sfs::codec {

enum class StorageDecision : std::uint8_t {
    Delta,
    FullNoBase,
    FullNotAlignable,
    FullChainTooDeep,
    FullDeltaTooLarge,
};

[[nodiscard]] std::string_view to_string(StorageDecision) noexcept;
[[nodiscard]] constexpr bool is_full(StorageDecision d) noexcept {
    return d != StorageDecision::Delta;
}

struct SnapshotInputs {
    bool          has_base = false;
    bool          alignable = true;
    std::uint32_t base_chain_depth = 0;
    std::uint64_t delta_bytes = 0;
    std::uint64_t full_bytes  = 0;
};

[[nodiscard]] StorageDecision decide(const SnapshotInputs&, const core::RepoConfig&) noexcept;

}  // namespace sfs::codec
