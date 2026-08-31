#include <synapsefs/codec/reconstruct.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include <synapsefs/codec/compress.hpp>
#include <synapsefs/codec/permute.hpp>
#include <synapsefs/codec/residual_codec.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/format/residual_hdr.hpp>

namespace sfs::codec {

using core::ErrKind;
using core::ObjectKind;
using core::Result;
using core::Status;

namespace {

// Gather the base bytes a Delta frame depends on: [unit_begin, unit_end) in
// TARGET order, resolved through the base group's OWN read_range — which
// recurses transparently whether the base itself is Full or another Delta.
// `expanded_perm` empty means Identity (spec 12 §4.2: the dependency set is
// the identical range, trivially aligned — no gather needed).
Result<std::vector<std::byte>> gather_base_bytes(const ReadCtx& base_ctx,
                                                 std::string_view base_group_name,
                                                 std::span<const std::uint32_t> expanded_perm,
                                                 std::uint64_t unit_begin, std::uint64_t unit_end,
                                                 std::uint64_t unit_bytes);

Result<std::size_t> read_range_impl(const ReadCtx& ctx, std::string_view group_name,
                                    std::uint64_t offset, std::span<std::byte> out);

Result<std::vector<std::byte>> gather_base_bytes(const ReadCtx& base_ctx,
                                                 std::string_view base_group_name,
                                                 std::span<const std::uint32_t> expanded_perm,
                                                 std::uint64_t unit_begin, std::uint64_t unit_end,
                                                 std::uint64_t unit_bytes) {
    const std::uint64_t count = unit_end - unit_begin;
    std::vector<std::byte> out(count * unit_bytes);

    if (expanded_perm.empty()) {
        auto n = read_range_impl(base_ctx, base_group_name, unit_begin * unit_bytes, out);
        if (!n) return std::unexpected(n.error());
        if (*n != out.size())
            return SFS_ERR(MalformedObject, "short read reconstructing delta base",
                           std::string(base_group_name));
        return out;
    }

    // Permutation destroys locality, so the dependency set is scattered on
    // the base side; dependency_runs collapses it into a few contiguous
    // reads (spec 12 §4.2), and the scratch map below re-orders them back
    // into target order — the read-side mirror of diff_encoder.cpp's
    // gather_base_units on the write side.
    const auto runs = dependency_runs(expanded_perm, unit_begin, count);
    std::vector<std::byte> scratch;
    std::unordered_map<std::uint32_t, std::uint64_t> unit_to_scratch_off;
    for (const auto& run : runs) {
        const std::uint64_t off = scratch.size();
        scratch.resize(off + run.count * unit_bytes);
        auto n = read_range_impl(base_ctx, base_group_name, run.first * unit_bytes,
                                 std::span(scratch.data() + off, run.count * unit_bytes));
        if (!n) return std::unexpected(n.error());
        if (*n != run.count * unit_bytes)
            return SFS_ERR(MalformedObject, "short read reconstructing delta base",
                           std::string(base_group_name));
        for (std::uint64_t k = 0; k < run.count; ++k)
            unit_to_scratch_off[static_cast<std::uint32_t>(run.first + k)] = off + k * unit_bytes;
    }

    for (std::uint64_t i = 0; i < count; ++i) {
        const auto it = unit_to_scratch_off.find(expanded_perm[unit_begin + i]);
        if (it == unit_to_scratch_off.end())
            return SFS_ERR(Internal, "gather_base_bytes: dependency run missing a unit",
                           std::string(base_group_name));
        std::memcpy(out.data() + i * unit_bytes, scratch.data() + it->second, unit_bytes);
    }
    return out;
}

// A tensor's SECONDARY (non-dim-0) axis, if it has one and it isn't
// pinned/identity, is bound to a DIFFERENT permutation group than the one
// this diff artifact's own `permutation` field describes. That group's
// permutation isn't stored here — it lives in whichever OTHER tensor owns
// that group's dim-0 axis, in ITS OWN diff artifact, from this SAME commit
// (`ctx.manifest`, not `base_ctx`). This is a single same-commit object
// fetch, not a walk through commit history — independent of chain_depth
// entirely, unlike the primary base recursion above.
//
// Returns an empty vector for "no secondary axis, or it's pinned/identity —
// nothing to do", the same convention gather_base_bytes uses for Identity.
Result<std::vector<std::uint32_t>> resolve_secondary_permutation(const ReadCtx& ctx,
                                                                 const format::TensorDiff& tdiff) {
    if (ctx.topology == nullptr) return std::vector<std::uint32_t>{};

    const auto tit = ctx.topology->tensors.find(tdiff.shape_name);
    if (tit == ctx.topology->tensors.end()) return std::vector<std::uint32_t>{};

    const core::AxisBinding* secondary = nullptr;
    for (const auto& ax : tit->second.axes) {
        if (ax.dim != 0) { secondary = &ax; break; }
    }
    if (secondary == nullptr) return std::vector<std::uint32_t>{};  // single-axis tensor

    const auto* pg = ctx.topology->find_group(secondary->group);
    if (pg != nullptr && pg->pinned) return std::vector<std::uint32_t>{};  // identity

    // Whichever tensor owns `secondary->group` via ITS OWN dim-0 axis is
    // where that group's permutation, for this commit, is recorded —
    // commit_planner.cpp's naming: the group's "owning" tensor.
    std::string owner_name;
    for (const auto& [name, axes] : ctx.topology->tensors) {
        for (const auto& ax : axes.axes) {
            if (ax.dim == 0 && ax.group == secondary->group) { owner_name = name; break; }
        }
        if (!owner_name.empty()) break;
    }
    if (owner_name.empty())
        return SFS_ERR(Internal, "no tensor owns secondary axis group", secondary->group);

    if (ctx.manifest == nullptr)
        return SFS_ERR(Internal, "ReadCtx missing manifest for secondary axis lookup");
    const auto* owner_group = ctx.manifest->find_group(owner_name);
    if (owner_group == nullptr)
        return SFS_ERR(MalformedObject, "secondary axis owner missing from manifest", owner_name);

    if (owner_group->mode != format::GroupMode::Delta) {
        // Full here means the owning group's real permutation was never
        // persisted — commit_planner.cpp's write-side invariant guarantees
        // this only happens when that permutation IS identity. Trusting
        // that invariant rather than erroring is deliberate: it's the same
        // invariant that made this tensor eligible for Delta at all.
        return std::vector<std::uint32_t>{};
    }
    if (!owner_group->diff_block)
        return SFS_ERR(MalformedObject, "secondary axis owner missing diff_block", owner_name);

    auto owner_bytes = ctx.blocks->get(*owner_group->diff_block, ObjectKind::Diff);
    if (!owner_bytes) return std::unexpected(owner_bytes.error());
    auto owner_artifact = format::parse_diff_artifact(*owner_bytes);
    if (!owner_artifact) return std::unexpected(owner_artifact.error());

    if (owner_artifact->header.permutation.kind == format::PermKind::Identity)
        return std::vector<std::uint32_t>{};

    auto owner_perm = format::read_permutation(owner_artifact->header.permutation,
                                              owner_artifact->payload);
    if (!owner_perm) return std::unexpected(owner_perm.error());

    return expand(*owner_perm, secondary->block);
}

// One tensor's contribution to a Delta read: the frame loop from spec 12 §6,
// clipped to [local_offset, local_offset + out.size()) within THIS tensor's
// own byte range, writing into the corresponding slice of `out`.
//
// `ctx` is the ORIGINAL (target-commit) context — needed for
// resolve_secondary_permutation's same-commit lookup — while `base_ctx` is
// where the primary base recursion actually reads from (the parent commit's
// manifest, or deeper).
Result<std::size_t> read_tensor_range(const ReadCtx& ctx, const ReadCtx& base_ctx,
                                      const format::DiffHeader& header,
                                      std::span<const std::byte> payload,
                                      const format::TensorDiff& tdiff,
                                      std::string_view base_group_name, std::uint64_t local_offset,
                                      std::span<std::byte> out) {
    if (tdiff.frames.empty())
        return SFS_ERR(MalformedObject, "diff tensor has no frames", tdiff.shape_name);

    const std::uint64_t axis_length = tdiff.frames.back().unit_end;
    std::uint64_t elems = 1;
    for (auto d : tdiff.shape) elems *= d;
    const std::uint64_t tensor_bytes = elems * core::dtype_size(tdiff.dtype);
    if (axis_length == 0 || tensor_bytes % axis_length != 0)
        return SFS_ERR(MalformedObject, "tensor bytes do not divide by unit count",
                       tdiff.shape_name);
    const std::uint64_t unit_bytes = tensor_bytes / axis_length;

    // Permutation is stored at GROUP granularity; expand to this tensor's
    // own axis granularity via its block factor, derived from
    // axis_length / permutation.n rather than stored redundantly per tensor
    // (spec 12 §2). Identity needs none of this — expanded_perm stays empty.
    std::vector<std::uint32_t> expanded_perm;
    if (header.permutation.kind != format::PermKind::Identity) {
        auto group_perm = format::read_permutation(header.permutation, payload);
        if (!group_perm) return std::unexpected(group_perm.error());
        if (header.permutation.n == 0 || axis_length % header.permutation.n != 0)
            return SFS_ERR(BlockFactorMismatch, "axis length does not divide by permutation n",
                           tdiff.shape_name);
        const auto block = static_cast<std::uint32_t>(axis_length / header.permutation.n);
        expanded_perm = expand(*group_perm, block);
    }

    // Resolved once per tensor, not per frame — it doesn't vary by frame.
    auto secondary_perm = resolve_secondary_permutation(ctx, tdiff);
    if (!secondary_perm) return std::unexpected(secondary_perm.error());
    const std::uint32_t elem_bytes = core::dtype_size(tdiff.dtype);
    if (!secondary_perm->empty() && unit_bytes != secondary_perm->size() * elem_bytes) {
        return SFS_ERR(BlockFactorMismatch, "secondary axis length does not match row width",
                       tdiff.shape_name);
    }

    std::size_t written = 0;
    const std::uint64_t want_end = local_offset + out.size();
    for (const auto& frame : tdiff.frames) {
        const std::uint64_t frame_begin = frame.unit_begin * unit_bytes;
        const std::uint64_t frame_end = frame.unit_end * unit_bytes;
        if (frame_end <= local_offset || frame_begin >= want_end) continue;  // no overlap

        if (frame.off + frame.len > payload.size())
            return SFS_ERR(MalformedObject, "frame extends past payload", tdiff.shape_name);

        auto base_bytes = gather_base_bytes(base_ctx, base_group_name, expanded_perm,
                                           frame.unit_begin, frame.unit_end, unit_bytes);
        if (!base_bytes) return std::unexpected(base_bytes.error());

        // Column-permute each row by the secondary group's permutation, on
        // top of the row-gather gather_base_bytes just did. Row-gather and
        // column-gather commute (result[i,j] = base[perm_row[i],
        // perm_col[j]] either way), so doing this AFTER the primary gather
        // rather than folded into it — the way diff_encoder.cpp's WRITE-side
        // ColumnPermutingSource does it BEFORE — produces the identical
        // bytes.
        if (!secondary_perm->empty()) {
            const std::uint64_t row_count = frame.unit_end - frame.unit_begin;
            std::vector<std::byte> reordered(base_bytes->size());
            std::vector<std::byte> scratch(unit_bytes);
            for (std::uint64_t r = 0; r < row_count; ++r) {
                std::span<const std::byte> row(base_bytes->data() + r * unit_bytes, unit_bytes);
                permute_units(scratch, row, *secondary_perm, elem_bytes);
                std::memcpy(reordered.data() + r * unit_bytes, scratch.data(), unit_bytes);
            }
            *base_bytes = std::move(reordered);
        }

        const std::uint64_t frame_bytes_raw = frame_end - frame_begin;
        std::vector<std::byte> decompressed(frame_bytes_raw);
        auto dn = decompress_frame(payload.subspan(frame.off, frame.len), decompressed);
        if (!dn) return std::unexpected(dn.error());
        if (*dn != frame_bytes_raw)
            return SFS_ERR(MalformedObject, "frame decompressed to unexpected size",
                           tdiff.shape_name);

        std::vector<std::byte> reconstructed(frame_bytes_raw);
        auto st = apply_residual(tdiff.residual, tdiff.transform, tdiff.dtype, *base_bytes,
                                decompressed, reconstructed);
        if (!st) return std::unexpected(st.error());

        // Digest covers the RECONSTRUCTED TARGET BYTES, checked on every
        // hop: a corrupted base block, a corrupted residual, or a wrong
        // permutation all fail the same check, at a cost proportional to
        // what was actually read (spec 12 §4/§6).
        if (core::digest(reconstructed) != frame.digest) {
            return SFS_ERR(FrameDigestMismatch, "frame digest mismatch",
                           tdiff.shape_name + " unit [" + std::to_string(frame.unit_begin) + "," +
                               std::to_string(frame.unit_end) + ")");
        }

        const std::uint64_t lo = std::max(frame_begin, local_offset);
        const std::uint64_t hi = std::min(frame_end, want_end);
        std::memcpy(out.data() + (lo - local_offset), reconstructed.data() + (lo - frame_begin),
                   hi - lo);
        written += hi - lo;
    }

    return written;
}

// The one recursive case: reconstructing a Delta group by fetching the base
// commit's manifest via ctx.history, recursing read_range on the base's
// group PER FRAME, applying the residual, and digest-checking the result.
// spec 12 §6.
//
// `group.diff_block` may be SHARED across several tensors' manifest entries:
// a diff artifact legitimately holds several tensors that share one
// permutation group (spec 13's own example: a conv's weight/bias and the
// following BatchNorm's scale all share one dim-0 group; the golden fixture
// tests/golden/diff_artifact.json holds two). `group_name` here is THIS
// manifest entry's own tensor name (format::Manifest.groups is keyed one
// entry per tensor, apps/sfs/cmd/commit.cpp) — so the right TensorDiff is
// found BY NAME inside the (possibly shared) artifact, and `offset` is local
// to just that one tensor's own bytes, exactly like the Full-mode branch
// treats `offset` as local to `group.block`'s own bytes. No cross-tensor
// byte-offset concatenation is needed or implied.
Result<std::size_t> read_delta_range(const ReadCtx& ctx, const format::GroupEntry& group,
                                     std::string_view group_name, std::uint64_t offset,
                                     std::span<std::byte> out) {
    if (!group.base || !group.diff_block)
        return SFS_ERR(MalformedObject, "Delta group missing base or diff_block",
                       std::string(group_name));
    if (group.chain_depth > ctx.max_depth)
        return SFS_ERR(ChainTooDeep, "chain depth exceeds max_depth", std::string(group_name));
    if (ctx.history == nullptr)
        return SFS_ERR(Internal, "ReadCtx missing history for delta base");

    auto diff_bytes = ctx.blocks->get(*group.diff_block, ObjectKind::Diff);
    if (!diff_bytes) return std::unexpected(diff_bytes.error());

    auto artifact = format::parse_diff_artifact(*diff_bytes);
    if (!artifact) return std::unexpected(artifact.error());
    const auto& header = artifact->header;
    const auto payload = artifact->payload;

    const format::TensorDiff* tdiff = nullptr;
    for (const auto& t : header.tensors) {
        if (t.shape_name == group_name) { tdiff = &t; break; }
    }
    if (tdiff == nullptr)
        return SFS_ERR(MalformedObject, "diff artifact has no entry for this tensor",
                       std::string(group_name));

    auto base_manifest = ctx.history->manifest_for(group.base->commit);
    if (!base_manifest) return std::unexpected(base_manifest.error());
    ReadCtx base_ctx = ctx;
    base_ctx.manifest = *base_manifest;

    return read_tensor_range(ctx, base_ctx, header, payload, *tdiff, group.base->group, offset,
                            out);
}

Result<std::size_t> read_range_impl(const ReadCtx& ctx, std::string_view group_name,
                                    std::uint64_t offset, std::span<std::byte> out) {
    if (!ctx.blocks || !ctx.manifest)
        return SFS_ERR(Internal, "ReadCtx missing blocks or manifest");

    const auto* group = ctx.manifest->find_group(group_name);
    if (!group) return SFS_ERR(MalformedObject, "unknown group", std::string(group_name));

    if (group->mode == format::GroupMode::Delta)
        return read_delta_range(ctx, *group, group_name, offset, out);

    if (!group->block)
        return SFS_ERR(MalformedObject, "Full group missing block", std::string(group_name));

    // Full group: the group's bytes ARE the raw object's payload bytes,
    // verbatim. read_range on the block store already verifies exactly the
    // chunks this read touches (format::verify_chunk), so a single call here
    // gives both the copy and the integrity check the interface promises.
    return ctx.blocks->read_range(*group->block, ObjectKind::Raw, offset, out);
}

}  // namespace

Result<std::size_t> read_range(const ReadCtx& ctx, std::string_view group_name,
                               std::uint64_t offset, std::span<std::byte> out) {
    return read_range_impl(ctx, group_name, offset, out);
}

Status reconstruct_file(const ReadCtx& ctx,
                        const std::function<Status(std::span<const std::byte>)>& sink) {
    if (!ctx.manifest) return SFS_ERR(Internal, "ReadCtx missing manifest");

    // Header block first (verbatim, stored under ObjectKind::Header), then
    // every buffer entry's bytes via its owning group, strictly in buffer
    // order — the same order stio::StWriter requires from its caller.
    auto header = ctx.blocks->get(ctx.manifest->file.header_block, ObjectKind::Header);
    if (!header) return std::unexpected(header.error());
    if (auto st = sink(*header); !st) return st;

    constexpr std::size_t kStreamChunk = 4u << 20;  // 4 MiB: bounded peak memory
    std::vector<std::byte> buf(kStreamChunk);

    for (const auto& entry : ctx.manifest->buffer) {
        const auto* group = ctx.manifest->find_group(entry.group);
        if (!group)
            return SFS_ERR(MalformedObject, "buffer entry references unknown group", entry.group);

        std::uint64_t done = 0;
        while (done < entry.nbytes) {
            std::size_t want =
                static_cast<std::size_t>(std::min<std::uint64_t>(kStreamChunk, entry.nbytes - done));
            auto n = read_range(ctx, entry.group, done, std::span(buf.data(), want));
            if (!n) return std::unexpected(n.error());
            if (*n != want)
                return SFS_ERR(MalformedObject, "short read reconstructing group", entry.group);
            if (auto st = sink(std::span<const std::byte>(buf.data(), want)); !st) return st;
            done += want;
        }
    }
    return {};
}

Result<std::uint32_t> chain_depth(const ReadCtx& ctx, std::string_view group_name) {
    if (!ctx.manifest) return SFS_ERR(Internal, "ReadCtx missing manifest");
    const auto* group = ctx.manifest->find_group(group_name);
    if (!group)
        return SFS_ERR(MalformedObject, "unknown group", std::string(group_name));
    return group->chain_depth;
}

}  // namespace sfs::codec
