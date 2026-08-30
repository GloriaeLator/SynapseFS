#include <synapsefs/core/tensor.hpp>

#include <algorithm>

namespace sfs::core {

std::uint64_t TensorMeta::elem_count() const noexcept {
    std::uint64_t n = 1;
    for (auto d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

Result<std::uint64_t> TensorMeta::unit_bytes(std::uint32_t dim) const {
    if (dim >= shape.size())
        return SFS_ERR(ShapeMismatch, "axis out of range for tensor", shape_owner);
    std::uint64_t units = shape[dim];
    if (units == 0)
        return SFS_ERR(ShapeMismatch, "zero-length axis", shape_owner);
    if (nbytes % units != 0)
        return SFS_ERR(ShapeMismatch, "tensor bytes do not divide evenly by axis length",
                       shape_owner);
    return nbytes / units;
}

std::vector<UnitRun> to_runs(std::span<const std::uint32_t> indices) {
    std::vector<UnitRun> runs;
    if (indices.empty()) return runs;

    std::uint64_t start = indices[0];
    std::uint64_t count = 1;
    for (std::size_t i = 1; i < indices.size(); ++i) {
        if (static_cast<std::uint64_t>(indices[i]) == start + count) {
            ++count;
        } else {
            runs.push_back({start, count});
            start = indices[i];
            count = 1;
        }
    }
    runs.push_back({start, count});
    return runs;
}

}  // namespace sfs::core
