#include <synapsefs/codec/reconstruct.hpp>

namespace sfs::codec {

using core::ErrKind;
using core::ObjectKind;
using core::Result;
using core::Status;

Result<std::size_t> read_range(const ReadCtx& ctx, std::string_view group_name,
                               std::uint64_t offset, std::span<std::byte> out) {
    if (!ctx.blocks || !ctx.manifest)
        return SFS_ERR(Internal, "ReadCtx missing blocks or manifest");

    const auto* group = ctx.manifest->find_group(group_name);
    if (!group)
        return SFS_ERR(MalformedObject, "unknown group", std::string(group_name));

    if (group->mode == format::GroupMode::Delta) {
        // The one recursive case this port does not implement: reconstructing
        // a Delta group means fetching the base commit's manifest via
        // ctx.history, recursing read_range on the base's group, and applying
        // the residual at ctx.blocks->get(*group->diff_block). Nothing in
        // this build's write path (store::CommitStore / apps/sfs `commit`)
        // ever produces a Delta group, so this path is unreachable in
        // practice; it returns NotImplemented rather than a wrong answer.
        return SFS_ERR(NotImplemented,
                       "delta-group reconstruction is not implemented in this build",
                       std::string(group_name));
    }

    if (!group->block)
        return SFS_ERR(MalformedObject, "Full group missing block", std::string(group_name));

    // Full group: the group's bytes ARE the raw object's payload bytes,
    // verbatim. read_range on the block store already verifies exactly the
    // chunks this read touches (format::verify_chunk), so a single call here
    // gives both the copy and the integrity check the interface promises.
    return ctx.blocks->read_range(*group->block, ObjectKind::Raw, offset, out);
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
