#pragma once
/// Internal to modules/align/src -- not installed, not part of the public
/// include/ surface. Shared by matcher.cpp and the sparse large-group path
/// (fingerprint.cpp, sparse_match.cpp) so "read N rows through
/// core::ITensorSource and widen to float32" has exactly one implementation.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <synapsefs/align/fp16.hpp>
#include <synapsefs/core/dtype.hpp>
#include <synapsefs/core/interfaces.hpp>

namespace sfs::align::detail {

/// Reads `count` raw dim-0 rows of `tensor` starting at `first` and widens
/// them to float, `elems_per_row` floats each. `first`/`count` are RAW
/// axis-0 indices (not group units) -- the caller is responsible for any
/// block expansion; this is the one place fp16/bf16 -> float widening
/// happens, and only for computing a cost, never for reconstruction
/// (fp16.hpp's own docstring).
inline core::Result<std::vector<float>> read_rows_as_float(core::ITensorSource& src, const std::string& tensor,
                                                            std::uint64_t first, std::uint64_t count,
                                                            std::uint64_t elems_per_row, core::DType dtype) {
    const std::uint32_t elem_size = core::dtype_size(dtype);
    std::vector<std::byte> raw(count * elems_per_row * elem_size);
    const std::size_t got = SFS_TRY(src.read_units(tensor, first, count, std::span(raw)));
    if (got != raw.size()) {
        // Every caller here requests rows within bounds already checked by
        // Topology::validate; a short read means the source is truncated or
        // corrupt, not end-of-object, so it is an error rather than a partial
        // result to tolerate.
        return SFS_ERR(Internal, "short read building alignment features", tensor);
    }

    std::vector<float> out(count * elems_per_row);
    if (dtype == core::DType::F32) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            float f = 0.0F;
            std::memcpy(&f, raw.data() + i * 4, 4);
            out[i] = f;
        }
    } else if (dtype == core::DType::F16 || dtype == core::DType::BF16) {
        std::vector<std::uint16_t> bits(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::memcpy(&bits[i], raw.data() + i * 2, 2);
        }
        if (dtype == core::DType::F16) {
            fp16_to_float_n(bits.data(), out.data(), out.size());
        } else {
            bf16_to_float_n(bits.data(), out.data(), out.size());
        }
    } else {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = core::to_float(dtype, raw.data() + i * elem_size);
        }
    }
    return out;
}

}  // namespace sfs::align::detail
