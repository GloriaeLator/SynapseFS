/// \file fs.cpp
/// The read-only filesystem view of one commit, with no FUSE in it (fuse_ll.cpp
/// is a thin adapter over this, per fs.hpp's file comment).
///
/// read() is exactly the three steps promised in docs/spec/16-consistency.md
/// §3.2: binary search the interval table, one or more read_range-shaped
/// calls, then a copy.
///
/// --- A note on codec::ReadCtx::cache ---
/// codec/reconstruct.hpp forward-declares its own `sfs::codec::FrameCache`
/// ("blockcache lives in mount/, but read_range takes one") and gives
/// ReadCtx an optional `codec::FrameCache* cache`. That is a different type
/// from `sfs::mount::FrameCache` defined in blockcache.hpp -- two classes
/// named FrameCache in different namespaces are not the same type, and
/// nothing in the headers unifies them. Rather than guess at an unwritten
/// adapter, this file leaves `ReadCtx::cache` unset (nullptr) on every call
/// and implements the single-flight LRU described in SPEC 16 §5 at the mount
/// layer instead: SynapseFs owns a mount::FrameCache and calls
/// codec::read_range() from inside FrameCache::get_or_fill's `fill`
/// callback, once per cache-aligned frame. That satisfies the same
/// observable contract (bounded resident decompressed bytes, single-flight
/// fill, no pre-materialisation) without depending on a cross-module type
/// that the headers declare but never define.

#include <synapsefs/mount/fs.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/mount/stats.hpp>

namespace sfs::mount {

namespace {
// Mount-layer cache granularity. Independent of whatever internal chunking
// codec/residual_codec uses for delta chains (see file header note) -- this
// is purely the unit the LRU evicts and single-flights on. Matches
// core::RepoConfig's documented default frame_bytes (128 KiB) so that peak
// RSS behaves the way docs/spec/16-consistency.md §4 describes:
//   cache_bytes + frame_bytes * max_chain_depth * concurrent_readers
inline constexpr std::uint64_t kFrameBytes = 128ull * 1024;
}  // namespace

struct SynapseFs::Impl {
    codec::ReadCtx  ctx;         // blocks/history point into the caller's repo
    format::Manifest manifest;   // owned copy: read() must outlive any transient view
    IntervalTable    table;
    FrameCache       cache;
    FsOptions        opts;

    std::string   file_name;
    std::uint64_t file_size = 0;
    core::Oid     header_block;
    core::Oid     artifact_oid;  // manifest.oid(); disambiguates cache keys per commit

    Impl(codec::ReadCtx c, format::Manifest m, IntervalTable t, FsOptions o)
        : ctx(c),
          manifest(std::move(m)),
          table(std::move(t)),
          cache(o.cache_bytes),
          opts(o),
          file_name(manifest.file.name),
          file_size(manifest.file.total_bytes),
          header_block(manifest.file.header_block),
          artifact_oid(manifest.oid()) {
        // ctx.manifest must point at OUR copy, not the caller's (which may
        // not outlive us) -- codec::read_range dereferences it on every call.
        ctx.manifest = &manifest;
    }
};

SynapseFs::SynapseFs() = default;
SynapseFs::~SynapseFs() = default;

core::Result<std::unique_ptr<SynapseFs>> SynapseFs::create(codec::ReadCtx ctx,
                                                            const format::Manifest& manifest,
                                                            FsOptions opts) {
    SFS_TRY_VOID(manifest.validate());
    auto table = SFS_TRY(IntervalTable::build(manifest));

    // codec::read_range needs a live IObjectSource/IBlockStore even though
    // this call only touches the manifest and interval math; both are
    // supplied by the caller (the daemon wires them from the repo it opened)
    // and are not optional for a mount that will actually serve reads.
    if (ctx.blocks == nullptr) {
        return std::unexpected(
            core::make_error(core::ErrKind::Internal, "SynapseFs::create: null block store"));
    }

    auto fs = std::unique_ptr<SynapseFs>(new SynapseFs());
    fs->impl_ = std::make_unique<Impl>(ctx, manifest, std::move(table), opts);
    return fs;
}

std::string_view SynapseFs::file_name() const noexcept { return impl_->file_name; }
std::uint64_t    SynapseFs::file_size() const noexcept { return impl_->file_size; }

const IntervalTable& SynapseFs::intervals() const noexcept { return impl_->table; }

FrameCache::Stats SynapseFs::cache_stats() const noexcept { return impl_->cache.stats(); }

core::Result<std::size_t> SynapseFs::read(std::uint64_t offset, std::span<std::byte> out) {
    Impl& im = *impl_;
    global_stats().reads.fetch_add(1, std::memory_order_relaxed);

    if (offset >= im.file_size || out.empty()) {
        return std::size_t{0};  // EOF: a short (zero) read, not an error
    }

    std::size_t   total_copied = 0;
    std::uint64_t cur          = offset;
    std::span<std::byte> remaining = out;

    while (!remaining.empty() && cur < im.file_size) {
        const Interval* iv = im.table.find(cur);
        if (iv == nullptr) break;  // past EOF; loop condition already guards this

        const std::uint64_t within           = cur - iv->file_offset;
        const std::uint64_t avail_in_interval = iv->length - within;

        if (iv->is_header) {
            const std::uint64_t want = std::min<std::uint64_t>(remaining.size(), avail_in_interval);
            auto n = im.ctx.blocks->read_range(im.header_block, core::ObjectKind::Header, within,
                                                remaining.first(want));
            if (!n) return std::unexpected(n.error());
            if (*n == 0) break;  // defensive: object shorter than the manifest claims
            cur += *n;
            remaining     = remaining.subspan(*n);
            total_copied += *n;
            continue;
        }

        // Tensor-group bytes: served one cache-aligned frame at a time so
        // concurrent faults on the same frame single-flight through
        // FrameCache rather than each re-decompressing it.
        const std::string_view group_name = im.table.group_name(iv->group_index);
        const std::uint64_t    group_off  = iv->group_offset + within;
        const std::uint64_t    frame_idx  = group_off / kFrameBytes;
        const std::uint64_t    frame_start = frame_idx * kFrameBytes;
        const std::uint64_t    offset_in_frame = group_off - frame_start;
        const std::uint64_t    avail_in_frame  = kFrameBytes - offset_in_frame;

        const std::uint64_t want =
            std::min({remaining.size(), avail_in_interval, avail_in_frame});

        // Captured by value except ctx/group_name, which outlive this call
        // (ctx points into im, group_name into im.table's group list).
        const codec::ReadCtx& ctx_ref = im.ctx;
        auto fill = [&ctx_ref, group_name, frame_start](std::span<std::byte> buf) -> core::Status {
            auto n = codec::read_range(ctx_ref, group_name, frame_start, buf);
            if (!n) return std::unexpected(n.error());
            global_stats().frames_decompressed.fetch_add(1, std::memory_order_relaxed);
            return {};
        };

        FrameKey key{im.artifact_oid, iv->group_index, static_cast<std::uint32_t>(frame_idx)};
        auto lease = im.cache.get_or_fill(key, kFrameBytes, fill);
        if (!lease) {
            if (lease.error().is_integrity()) {
                global_stats().digest_failures.fetch_add(1, std::memory_order_relaxed);
            }
            return std::unexpected(lease.error());
        }

        auto bytes = lease->bytes();
        // Invariant: `want` never exceeds avail_in_interval, and the
        // interval table's lengths sum to exactly the group's real content
        // (built from the manifest's buffer layout) -- so [offset_in_frame,
        // offset_in_frame + want) was genuinely written by `fill`, even if
        // this is the tail frame of a group shorter than kFrameBytes.
        std::memcpy(remaining.data(), bytes.data() + offset_in_frame, want);

        cur += want;
        remaining     = remaining.subspan(want);
        total_copied += want;
    }

    global_stats().bytes_served.fetch_add(total_copied, std::memory_order_relaxed);
    return total_copied;
}

}  // namespace sfs::mount
