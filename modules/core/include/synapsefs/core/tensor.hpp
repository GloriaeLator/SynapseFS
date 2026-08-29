#pragma once
/// \file tensor.hpp
/// Non-owning views over tensor bytes, and the buffer-layout entry that the
/// manifest stores.
///
/// There is no owning Tensor type here on purpose. Owning one invites loading a
/// whole tensor, and the out-of-core requirement (docs/adr/0008) says the
/// streaming path is the only path.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <synapsefs/core/dtype.hpp>

namespace sfs::core {

/// One entry of the manifest's buffer layout: a tensor's extent in the file's
/// data section. Every tensor in the file appears, whether or not the topology
/// models it. docs/spec/10-object-model.md §4.2.
struct BufferEntry {
    std::string   tensor;
    std::uint64_t off    = 0;   ///< from the start of the data section
    std::uint64_t nbytes = 0;
    std::string   group;        ///< permutation group id, or a singleton
};

/// Shape plus dtype, as declared in the safetensors header.
struct TensorMeta {
    std::string                shape_owner;   ///< name, kept for diagnostics
    std::vector<std::uint64_t> shape;
    DType                      dtype = DType::F16;
    std::uint64_t              data_off = 0;
    std::uint64_t              nbytes   = 0;

    [[nodiscard]] std::uint64_t elem_count() const noexcept;
    /// Bytes per output unit along `dim` — nbytes / shape[dim]. Must divide
    /// exactly; a non-integer result means the topology is wrong for this
    /// tensor and the aligner reports NotAlignable rather than guessing.
    [[nodiscard]] Result<std::uint64_t> unit_bytes(std::uint32_t dim) const;
};

/// A read-only window onto tensor bytes already in memory. Never owns.
struct TensorView {
    std::span<const std::byte> bytes;
    const TensorMeta*          meta = nullptr;

    [[nodiscard]] std::span<const std::byte> unit(std::uint64_t index,
                                                  std::uint64_t unit_bytes) const noexcept {
        return bytes.subspan(index * unit_bytes, unit_bytes);
    }
};

/// A contiguous run of output units. The reader turns a scattered dependency
/// set p[a:b] into runs so that a frame costs a few pread calls, not one per
/// unit. docs/spec/12-residual-format.md §4.2.
struct UnitRun {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
};

/// Collapse a set of unit indices into ascending runs.
[[nodiscard]] std::vector<UnitRun> to_runs(std::span<const std::uint32_t> indices);

}  // namespace sfs::core
