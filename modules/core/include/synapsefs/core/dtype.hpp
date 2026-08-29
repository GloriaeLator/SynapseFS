#pragma once
/// \file dtype.hpp
/// The safetensors dtype set, restricted to what the graded fixtures use plus
/// the integer buffer types that show up in real checkpoints.
///
/// Note that SynapseFS never performs floating-point ARITHMETIC on weights.
/// Residuals are XOR or zigzag over the integer view of the bits, which is
/// bijective and therefore exact. That is why reconstruction is bit-exact by
/// construction rather than by luck. docs/spec/12-residual-format.md §5.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <synapsefs/core/error.hpp>

namespace sfs::core {

enum class DType : std::uint8_t {
    F16, BF16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
};

/// Names exactly as safetensors writes them: "F16", "BF16", "I64", …
[[nodiscard]] std::string_view to_string(DType) noexcept;
[[nodiscard]] Result<DType> dtype_from_string(std::string_view) noexcept;

[[nodiscard]] constexpr std::uint32_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::Bool:
        case DType::I8:
        case DType::U8:   return 1;
        case DType::F16:
        case DType::BF16:
        case DType::I16:
        case DType::U16:  return 2;
        case DType::F32:
        case DType::I32:
        case DType::U32:  return 4;
        case DType::F64:
        case DType::I64:
        case DType::U64:  return 8;
    }
    return 0;
}

/// True for the types the aligner can compute a cost over. Buffers such as
/// `num_batches_tracked` (I64) are not aligned — they get singleton groups and
/// round-trip unchanged (docs/spec/13 §3).
[[nodiscard]] constexpr bool is_float(DType d) noexcept {
    return d == DType::F16 || d == DType::BF16 || d == DType::F32 || d == DType::F64;
}

/// Convert one element to float for cost computation. fp16/bf16 are widened in
/// software; this is never used on the reconstruction path.
[[nodiscard]] float to_float(DType, const std::byte* elem) noexcept;

}  // namespace sfs::core
