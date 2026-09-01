#pragma once
/// \file commit_tensor_source.hpp
/// A core::ITensorSource over a COMMITTED checkpoint's real tensor bytes —
/// the missing piece for wiring alignment into `sfs commit`.
///
/// header_only_source.hpp's HeaderOnlyTensorSource deliberately cannot read
/// bytes (it exists to answer shape/dtype queries without reconstructing
/// anything). align::Matcher and store::plan_commit_groups need the
/// opposite: a `base` that DOES return real tensor bytes for the PARENT
/// commit, so the new checkpoint can be weight-matched and diffed against
/// it. This is that source.
///
/// Shapes/dtypes still come for free from the parent's stored header block
/// (same trick as HeaderOnlyTensorSource — no reconstruction needed just to
/// know a tensor's shape). Actual bytes come from codec::read_range against
/// the parent's own ReadCtx, which already knows how to walk a Delta chain
/// if the parent itself stored some tensors as deltas — callers of this
/// class never need to know or care.

#include <string>
#include <utility>
#include <vector>

#include <synapsefs/codec/reconstruct.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/st_header.hpp>

namespace sfs::app {

class CommitTensorSource final : public core::ITensorSource {
public:
    /// `ctx` is copied (it's a handful of raw pointers), but everything it
    /// points to — blocks, the parent's manifest, the manifest store used
    /// as `history`, the parent's topology — must outlive this object. The
    /// caller (run_commit) keeps all of those function-local for exactly
    /// this reason, same discipline mount.cpp and checkout.cpp already use
    /// for their own ReadCtx.
    CommitTensorSource(format::StHeader header, const codec::ReadCtx& ctx)
        : header_(std::move(header)), layout_(header_.buffer_layout()), ctx_(ctx) {}

    // Never used to write anything out — this source only ever feeds the
    // aligner and the diff encoder's base-side reads, neither of which
    // reproduces a header. Matches HeaderOnlyTensorSource's own choice here.
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
                                         std::uint64_t count, std::span<std::byte> out) override {
        const core::TensorMeta* m = meta(name);
        if (m == nullptr) {
            return SFS_ERR(TensorNotInBufferLayout, "tensor not in base commit", std::string(name));
        }

        // Rows along axis 0 — the same convention stio::StSource::read_units
        // documents, and the one every ITensorSource in this project agrees
        // on: "unit" always means a whole dim-0 row unless a caller layers
        // its own axis-aware adapter on top (see commit_planner.cpp's
        // ColumnPermutingSource, which does exactly that to US when a
        // tensor's secondary axis needs it).
        auto ub = m->unit_bytes(0);
        if (!ub) return std::unexpected(ub.error());
        const std::uint64_t stride = *ub;

        const std::uint64_t offset = first * stride;
        const auto want = static_cast<std::size_t>(count * stride);
        if (out.size() < want) {
            return SFS_ERR(Internal, "output buffer too small for requested units",
                          std::string(name));
        }

        // read_range is scoped to just THIS group's own reconstructed bytes
        // (Full or Delta, chain-walked transparently by codec) — offset is
        // relative to the tensor's own start, never the file's data section.
        return codec::read_range(ctx_, name, offset, out.subspan(0, want));
    }

private:
    format::StHeader               header_;
    std::vector<core::BufferEntry> layout_;
    codec::ReadCtx                 ctx_;
};

}  // namespace sfs::app
