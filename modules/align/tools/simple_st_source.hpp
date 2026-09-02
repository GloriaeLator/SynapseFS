#pragma once
/// \file simple_st_source.hpp
/// A minimal, mmap-backed core::ITensorSource over a real .safetensors file,
/// for exercising the align module against actual checkpoint bytes without
/// loading the whole file into the process's heap.
///
/// This is NOT stio's implementation (module A2's ownership per
/// docs/ownership.md) -- it exists so align can be demonstrated and tested
/// end-to-end on real files before stio's real reader lands. It is, however,
/// genuinely lazy: mmap(MAP_PRIVATE) means the kernel pages tensor bytes in
/// from the file/page cache only as read_units() actually touches them, and
/// can evict clean pages under memory pressure -- unlike an earlier version
/// of this file, which read the whole file into a std::vector<std::byte> up
/// front (an instant multi-GB allocation on a large checkpoint, and the
/// direct cause of an OOM kill on a ~2B-parameter model; see
/// docs/adr/0012-libtorch-for-large-groups.md).
///
/// Linux-only (POSIX mmap), matching the rest of the project's stated scope.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <synapsefs/core/interfaces.hpp>

namespace sfs::align::tools {

class SimpleStSource final : public core::ITensorSource {
public:
    [[nodiscard]] static core::Result<SimpleStSource> open(const std::filesystem::path& path);

    ~SimpleStSource() override;
    SimpleStSource(SimpleStSource&&) noexcept;
    SimpleStSource& operator=(SimpleStSource&&) noexcept;
    SimpleStSource(const SimpleStSource&) = delete;
    SimpleStSource& operator=(const SimpleStSource&) = delete;

    std::span<const std::byte> header_bytes() const override;
    std::span<const core::BufferEntry> buffer_layout() const override;
    const core::TensorMeta* meta(std::string_view name) const override;
    std::uint64_t total_bytes() const override;
    core::Result<std::size_t> read_units(std::string_view name, std::uint64_t first, std::uint64_t count,
                                         std::span<std::byte> out) override;

    /// Zero-copy view over the mapped file, for callers (the sparse
    /// fingerprint/auction path) that want to wrap a range directly with
    /// torch::from_blob instead of paying read_units's copy-into-caller-
    /// buffer contract. The mapping outlives any view taken from it as long
    /// as this SimpleStSource is alive, exactly like read_units's guarantees.
    [[nodiscard]] const std::byte* mapped_data() const noexcept { return data_; }

private:
    SimpleStSource() = default;

    const std::byte* data_ = nullptr;  ///< mmap base; nullptr when moved-from
    std::uint64_t size_ = 0;
    std::uint64_t data_start_ = 0;
    std::unordered_map<std::string, core::TensorMeta> metas_;
    std::vector<core::BufferEntry> layout_;
};

}  // namespace sfs::align::tools
