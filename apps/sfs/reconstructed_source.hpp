#pragma once
/// \file reconstructed_source.hpp
/// A core::ITensorSource backed by a REAL, already-committed checkpoint,
/// reconstructed on demand through codec::read_range.
///
/// align::Matcher and store::plan_commit_groups both need a `base`
/// ITensorSource for the PARENT commit — the actual bytes, not just shapes
/// (unlike header_only_source.hpp's HeaderOnlyTensorSource, which exists
/// purely so align::parse_topology has shapes to validate against without
/// needing data). There is no raw file to open for a historical commit; its
/// bytes exist only as (possibly delta-chained) objects in the block store.
/// This adapter makes an arbitrary historical commit look like an
/// ITensorSource by streaming it through the one reconstructor
/// (reconstruct.hpp), the same way stio::StSource makes a real file look
/// like one — `sfs commit`'s alignment step and `sfs checkout`/the mount
/// read path end up sharing that single reconstruction implementation
/// either way.

#include <string>
#include <utility>
#include <vector>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/st_header.hpp>

namespace sfs::app {

class ReconstructedTensorSource : public core::ITensorSource {
public:
    /// `ctx` is copied (it is a small bag of pointers, same convention as
    /// every other codec::ReadCtx use in this app) but everything IT points
    /// to — blocks, manifest, history, the parent's own topology — must
    /// outlive this object.
    ReconstructedTensorSource(codec::ReadCtx ctx, format::StHeader header)
        : ctx_(ctx), header_(std::move(header)), layout_(header_.buffer_layout()) {}

    // Never used by align::Matcher or store::plan_commit_groups (neither
    // regenerates a file), same as header_only_source.hpp's convention.
    std::span<const std::byte> header_bytes() const override { return {}; }
    std::span<const core::BufferEntry> buffer_layout() const override { return layout_; }
    const core::TensorMeta* meta(std::string_view name) const override {
        auto it = header_.tensors.find(std::string(name));
        return it == header_.tensors.end() ? nullptr : &it->second;
    }
    std::uint64_t total_bytes() const override {
        std::uint64_t n = header_.header_extent;
        for (const auto& e : layout_) n += e.nbytes;
        return n;
    }

    core::Result<std::size_t> read_units(std::string_view name, std::uint64_t first,
                                         std::uint64_t /*count*/,
                                         std::span<std::byte> out) override {
        const auto* m = meta(name);
        if (m == nullptr) {
            return SFS_ERR(ObjectNotFound, "no such tensor in the parent commit",
                          std::string(name));
        }
        auto ub = m->unit_bytes(0);
        if (!ub) return std::unexpected(ub.error());
        const std::uint64_t byte_off = first * (*ub);
        // codec::read_range reads by GROUP, and commit_planner.cpp's own
        // convention (format::Manifest::groups keyed one entry PER TENSOR,
        // spec 13's shared-artifact case notwithstanding) is that a tensor's
        // own name IS the key to look it up by — the unit/byte conversion
        // above is the only translation this seam needs to do.
        return codec::read_range(ctx_, name, byte_off, out);
    }

private:
    codec::ReadCtx ctx_;
    format::StHeader header_;
    std::vector<core::BufferEntry> layout_;
};

}  // namespace sfs::app
