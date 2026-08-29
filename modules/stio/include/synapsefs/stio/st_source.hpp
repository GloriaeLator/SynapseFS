#pragma once
/// \file st_source.hpp
/// Lazy reader for a .safetensors file. Implements core::ITensorSource.
///
/// "Lazy" is not an optimisation here. Fixtures reach ~7B parameters in fp16
/// under a 16 GB cap, and an OOM at fixture size fails the metric even if it
/// passes locally. This is the ONLY way align and codec read weights, on every
/// fixture size including the 1M-parameter one, so that there is no untested
/// second path. docs/adr/0008-out-of-core-streaming.md

#include <filesystem>
#include <memory>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/st_header.hpp>
#include <synapsefs/util/mmap.hpp>

namespace sfs::stio {

using core::Result;

struct StSourceOptions {
    /// Map the data section instead of pread-ing it. Good for a checkout
    /// (sequential); NOT used by the aligner, whose access pattern is permuted
    /// and scattered, and whose peak RSS must be ours to control.
    bool use_mmap = false;
};

class StSource final : public core::ITensorSource {
public:
    using Options = StSourceOptions;

    [[nodiscard]] static Result<std::unique_ptr<StSource>> open(
        const std::filesystem::path&, StSourceOptions = {});
    ~StSource() override;

    [[nodiscard]] std::span<const std::byte> header_bytes() const override;
    [[nodiscard]] std::span<const core::BufferEntry> buffer_layout() const override;
    [[nodiscard]] const core::TensorMeta* meta(std::string_view) const override;
    [[nodiscard]] std::uint64_t total_bytes() const override;
    [[nodiscard]] Result<std::size_t> read_units(std::string_view name, std::uint64_t first,
                                                 std::uint64_t count,
                                                 std::span<std::byte> out) override;

    /// Raw ranged read into the data section, for the commit path when it is
    /// storing a group as `full`.
    [[nodiscard]] Result<std::size_t> read_raw(std::uint64_t data_offset,
                                               std::span<std::byte> out);

    [[nodiscard]] const format::StHeader& header() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    StSource();
};

}  // namespace sfs::stio
