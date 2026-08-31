#include <synapsefs/store/commit_planner.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

#include <synapsefs/codec/permute.hpp>
#include <synapsefs/codec/snapshot_policy.hpp>
#include <synapsefs/core/dtype.hpp>

namespace sfs::store {

using core::Result;

namespace {

// Step 4: wraps a real base ITensorSource, reordering ONE tensor's
// non-dim-0 axis by an already-resolved OTHER group's permutation before
// handing rows back — so encode_group's own row-gather logic (which only
// ever calls read_units for whole dim-0 rows) never has to know a second
// axis moved. Row and column permutation commute
// (base[perm_row[i], perm_col[j]] doesn't care which is applied first), so
// this composes with encode_group's existing gather with no changes there.
class ColumnPermutingSource : public core::ITensorSource {
public:
    ColumnPermutingSource(core::ITensorSource& inner, std::string tensor,
                         std::vector<std::uint32_t> expanded_col_perm, std::uint32_t elem_bytes)
        : inner_(inner), tensor_(std::move(tensor)), col_perm_(std::move(expanded_col_perm)),
          elem_bytes_(elem_bytes) {}

    std::span<const std::byte> header_bytes() const override { return inner_.header_bytes(); }
    std::span<const core::BufferEntry> buffer_layout() const override {
        return inner_.buffer_layout();
    }
    const core::TensorMeta* meta(std::string_view name) const override { return inner_.meta(name); }
    std::uint64_t total_bytes() const override { return inner_.total_bytes(); }

    Result<std::size_t> read_units(std::string_view name, std::uint64_t first, std::uint64_t count,
                                   std::span<std::byte> out) override {
        if (name != tensor_) return inner_.read_units(name, first, count, out);

        auto n = inner_.read_units(name, first, count, out);
        if (!n) return n;

        const auto row_bytes = static_cast<std::uint64_t>(col_perm_.size()) * elem_bytes_;
        if (*n != count * row_bytes) {
            return SFS_ERR(Internal, "ColumnPermutingSource: short read from inner source",
                          tensor_);
        }

        std::vector<std::byte> scratch(row_bytes);
        for (std::uint64_t r = 0; r < count; ++r) {
            std::byte* row = out.data() + r * row_bytes;
            codec::permute_units(scratch, std::span<const std::byte>(row, row_bytes), col_perm_,
                                elem_bytes_);
            std::memcpy(row, scratch.data(), row_bytes);
        }
        return n;
    }

private:
    core::ITensorSource& inner_;
    std::string tensor_;
    std::vector<std::uint32_t> col_perm_;
    std::uint32_t elem_bytes_;
};

// Step 1: a group counts as "moved" only if align found a genuine,
// non-identity, alignable permutation for it. Pinned (topology) or absent
// (no evidence gathered) both mean the same thing here: nothing to diff,
// and — for a SECONDARY dependency specifically — nothing that needs its
// permutation recorded anywhere to be recoverable on read.
bool group_is_effectively_identity(const core::Topology& topo, const align::MatchReport& report,
                                   const std::string& group) {
    const auto* g = topo.find_group(group);
    if (g != nullptr && g->pinned) return true;
    auto it = report.groups.find(group);
    if (it == report.groups.end()) return true;
    return it->second.identity || !it->second.alignable;
}

// Every tensor whose dim-0 axis is bound to `group` — the same selection
// encode_group itself makes internally, needed here too for chain-depth
// bookkeeping and for writing one GroupEntry per member once storage is
// decided.
std::vector<std::string> tensors_owning_group(const core::Topology& topo,
                                              const std::string& group) {
    std::vector<std::string> out;
    for (const auto& [tensor, axes] : topo.tensors) {
        for (const auto& ax : axes.axes) {
            if (ax.dim == 0 && ax.group == group) { out.push_back(tensor); break; }
        }
    }
    return out;
}

// Every OTHER, non-identity group that any of `members` depends on via a
// secondary (non-dim-0) axis. A group with no such dependency (the common
// case) returns empty.
std::set<std::string> secondary_dependencies(const core::Topology& topo,
                                             const align::MatchReport& report,
                                             const std::vector<std::string>& members) {
    std::set<std::string> deps;
    for (const auto& tensor : members) {
        for (const auto& ax : topo.tensors.at(tensor).axes) {
            if (ax.dim != 0 && !group_is_effectively_identity(topo, report, ax.group)) {
                deps.insert(ax.group);
            }
        }
    }
    return deps;
}

// Everything computed for one candidate (non-identity, alignable) group
// before the cross-group dependency check — encoding happens exactly once
// per group regardless of how the dependency pass resolves, per spec 12
// §7's "the writer already holds both byte strings" ordering (step 2).
struct GroupPlan {
    std::vector<std::string> members;
    codec::EncodeResult encoded;
    bool has_base = false;
    std::uint32_t max_base_depth = 0;
    codec::StorageDecision decision = codec::StorageDecision::FullNoBase;
    std::set<std::string> secondary_deps;
};

}  // namespace

Result<std::unordered_map<std::string, format::GroupEntry>> plan_commit_groups(
    core::ITensorSource& base, core::ITensorSource& target, const core::Topology& topology,
    const align::MatchReport& report,
    const std::unordered_map<std::string, ParentTensorInfo>& parent_info,
    core::IBlockStore& blocks, const core::RepoConfig& cfg, codec::EncodeOptions encode_opts) {
    // ---- Phase A: encode every non-identity, alignable group once, and
    // record its OWN (dependency-blind) storage decision. ----
    std::unordered_map<std::string, GroupPlan> plans;

    for (const auto& [group_name, perm_group] : topology.groups) {
        if (group_is_effectively_identity(topology, report, group_name)) continue;

        const auto& gm = report.groups.at(group_name);
        auto members = tensors_owning_group(topology, group_name);
        if (members.empty()) continue;  // no tensor actually owns this group: nothing to encode

        // Step 4: chain a ColumnPermutingSource per member tensor that has
        // a secondary axis bound to a DIFFERENT, non-identity group. Kept
        // alive only for this encode_group call; each is name-scoped, so
        // chaining them in any order is safe.
        std::vector<std::unique_ptr<ColumnPermutingSource>> adapters;
        core::ITensorSource* base_for_encode = &base;
        for (const auto& tensor : members) {
            const core::AxisBinding* secondary = nullptr;
            for (const auto& ax : topology.tensors.at(tensor).axes) {
                if (ax.dim != 0) { secondary = &ax; break; }
            }
            if (secondary == nullptr) continue;
            if (group_is_effectively_identity(topology, report, secondary->group)) continue;

            const auto& other_gm = report.groups.at(secondary->group);
            const auto* meta = target.meta(tensor);
            if (meta == nullptr)
                return SFS_ERR(TensorNotInBufferLayout, "tensor missing from target", tensor);

            auto expanded = codec::expand(other_gm.permutation, secondary->block);
            adapters.push_back(std::make_unique<ColumnPermutingSource>(
                *base_for_encode, tensor, std::move(expanded), core::dtype_size(meta->dtype)));
            base_for_encode = adapters.back().get();
        }

        format::AlignmentInfo align_info;  // step 5: straight passthrough
        align_info.method = "weight_matching_lap";
        align_info.cost_raw = gm.cost_raw;
        align_info.cost_normalized = gm.cost_normalized;

        auto encoded = codec::encode_group(*base_for_encode, target, topology, group_name,
                                          gm.permutation, gm.alignable, align_info, encode_opts);
        if (!encoded) return std::unexpected(encoded.error());

        bool has_base = false;
        std::uint32_t max_base_depth = 0;
        for (const auto& tensor : members) {
            auto pit = parent_info.find(tensor);
            if (pit == parent_info.end()) continue;
            has_base = true;
            max_base_depth = std::max(max_base_depth, pit->second.chain_depth);
        }

        codec::SnapshotInputs inputs;
        inputs.has_base = has_base;
        inputs.alignable = gm.alignable;
        inputs.base_chain_depth = max_base_depth;
        inputs.delta_bytes = encoded->artifact.size();
        inputs.full_bytes = encoded->full_bytes;

        GroupPlan plan;
        plan.members = members;
        plan.encoded = std::move(*encoded);
        plan.has_base = has_base;
        plan.max_base_depth = max_base_depth;
        plan.decision = codec::decide(inputs, cfg);
        plan.secondary_deps = secondary_dependencies(topology, report, members);
        plans.emplace(group_name, std::move(plan));
    }

    // ---- Phase B: fixed-point dependency downgrade. A group can only stay
    // Delta if every non-identity group it secondarily depends on is ALSO
    // Delta — otherwise that dependency's permutation is unrecoverable on
    // read (docs/spec/12 §6's secondary-axis lookup needs it to exist). A
    // downgrade can cascade (rare in practice — sequential networks don't
    // chain secondary dependencies deeply — but handled generally). ----
    std::unordered_map<std::string, bool> is_delta;
    for (const auto& [name, plan] : plans) is_delta[name] = (plan.decision == codec::StorageDecision::Delta);

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& [name, plan] : plans) {
            if (!is_delta[name]) continue;
            for (const auto& dep : plan.secondary_deps) {
                // A dependency on a group that was never even a Delta
                // candidate (not in `plans` at all, e.g. identity/pinned)
                // is already satisfied by construction — only a REAL
                // candidate that got decided/downgraded to Full is a
                // problem.
                if (plans.contains(dep) && !is_delta.at(dep)) {
                    is_delta[name] = false;
                    changed = true;
                    break;
                }
            }
        }
    }

    // ---- Phase C: write the survivors. ----
    std::unordered_map<std::string, format::GroupEntry> entries;
    for (const auto& [group_name, plan] : plans) {
        if (!is_delta.at(group_name)) continue;  // falls through to the Full pass below

        auto diff_oid = blocks.put(core::ObjectKind::Diff, plan.encoded.artifact);
        if (!diff_oid) return std::unexpected(diff_oid.error());

        for (const auto& tensor : plan.members) {
            format::GroupEntry e;
            e.mode = format::GroupMode::Delta;
            e.base = format::DeltaBase{parent_info.at(tensor).parent_commit, tensor};
            e.diff_block = *diff_oid;
            e.chain_depth = parent_info.at(tensor).chain_depth + 1;
            entries[tensor] = e;
        }
    }

    // Every tensor not planned as Delta above — identity/pinned groups, any
    // group decide() rejected on its own terms, and any group the
    // dependency pass downgraded — gets stored Full, reading straight from
    // `target`. A Full group's bytes ARE the reconstructed file's bytes
    // verbatim (reconstruct.cpp), never derived from `base`.
    for (const auto& entry : target.buffer_layout()) {
        if (entries.contains(entry.tensor)) continue;

        const auto* meta = target.meta(entry.tensor);
        if (meta == nullptr)
            return SFS_ERR(TensorNotInBufferLayout, "buffer entry missing from target",
                          entry.tensor);

        std::vector<std::byte> buf(entry.nbytes);
        auto n = target.read_units(entry.tensor, 0, meta->shape.empty() ? 1 : meta->shape[0], buf);
        if (!n) return std::unexpected(n.error());
        if (*n != buf.size())
            return SFS_ERR(Internal, "short read storing Full tensor", entry.tensor);

        auto oid = blocks.put(core::ObjectKind::Raw, buf);
        if (!oid) return std::unexpected(oid.error());

        format::GroupEntry e;
        e.mode = format::GroupMode::Full;
        e.block = *oid;
        e.chain_depth = 0;
        entries[entry.tensor] = e;
    }

    return entries;
}

}  // namespace sfs::store
