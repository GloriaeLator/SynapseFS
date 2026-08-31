#pragma once
/// \file header_only_source.hpp
/// A core::ITensorSource backed only by a parsed safetensors HEADER, no data
/// section.
///
/// align::parse_topology needs an ITensorSource to validate shapes against —
/// but it only ever calls .meta()/.buffer_layout(), never .read_units()
/// (topology_parser.cpp's walk_layers reads shapes, not bytes). checkout and
/// mount need a core::Topology to populate codec::ReadCtx::topology, and the
/// only way to get one is to parse it — but parsing needs shapes, and
/// getting real bytes would mean reconstructing the checkpoint first, which
/// is what ReadCtx is FOR. This breaks that cycle: the safetensors header
/// (verbatim bytes at manifest.file.header_block) gives every tensor's
/// shape/dtype for free, no reconstruction needed.

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <synapsefs/align/topology_parser.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/format/st_header.hpp>

namespace sfs::app {

class HeaderOnlyTensorSource : public core::ITensorSource {
public:
    explicit HeaderOnlyTensorSource(format::StHeader header)
        : header_(std::move(header)), layout_(header_.buffer_layout()) {}

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
    // Never called by align::parse_topology; present only to satisfy the
    // interface. Any real caller trying to read tensor bytes through this
    // is a bug — this source structurally cannot have them.
    core::Result<std::size_t> read_units(std::string_view name, std::uint64_t, std::uint64_t,
                                         std::span<std::byte>) override {
        return SFS_ERR(NotImplemented, "HeaderOnlyTensorSource has no tensor data",
                       std::string(name));
    }

private:
    format::StHeader header_;
    std::vector<core::BufferEntry> layout_;
};

/// Best-effort: parses `commit`'s topology object using `manifest`'s header
/// block (via HeaderOnlyTensorSource) to answer parse_topology's shape
/// queries. Returns std::nullopt — deliberately not an error — for a
/// missing, placeholder ("{}"), or malformed topology.
///
/// commit.cpp currently always stores "{}" until alignment is wired into
/// it, and align::parse_topology treats a non-empty-but-"layers"-less
/// config as a hard error, not "no topology" — so failing to parse must
/// NOT fail checkout/mount, or every commit made by today's commit.cpp
/// would stop being checkable out at all. This is also the correct
/// invariant, not just a workaround: a tensor can only need a REAL
/// topology at read time if one was successfully parsed and validated when
/// it was WRITTEN (store::plan_commit_groups requires a core::Topology to
/// run at all) — so a parse failure here correctly implies no tensor in
/// this commit could possibly need codec::ReadCtx::topology.
inline std::optional<core::Topology> load_commit_topology(core::IBlockStore& blocks,
                                                          const format::Commit& commit,
                                                          const format::Manifest& manifest) {
    auto header_bytes = blocks.get(manifest.file.header_block, core::ObjectKind::Header);
    if (!header_bytes) return std::nullopt;
    auto st_header = format::parse_st_header(*header_bytes);
    if (!st_header) return std::nullopt;

    auto topo_bytes = blocks.get(commit.topology, core::ObjectKind::Topology);
    if (!topo_bytes) return std::nullopt;

    HeaderOnlyTensorSource src(std::move(*st_header));
    auto topo = align::parse_topology(src, *topo_bytes, align::ParseOptions{});
    if (!topo) return std::nullopt;
    return std::move(*topo);
}

}  // namespace sfs::app
