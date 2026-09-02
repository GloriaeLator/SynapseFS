#include <synapsefs/core/dtype.hpp>

#include <cstring>

namespace sfs::core {

std::string_view to_string(DType d) noexcept {
    switch (d) {
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        case DType::F32:  return "F32";
        case DType::F64:  return "F64";
        case DType::I8:   return "I8";
        case DType::I16:  return "I16";
        case DType::I32:  return "I32";
        case DType::I64:  return "I64";
        case DType::U8:   return "U8";
        case DType::U16:  return "U16";
        case DType::U32:  return "U32";
        case DType::U64:  return "U64";
        case DType::Bool: return "BOOL";
    }
    return "?";
}

Result<DType> dtype_from_string(std::string_view s) noexcept {
    if (s == "F16")  return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "F32")  return DType::F32;
    if (s == "F64")  return DType::F64;
    if (s == "I8")   return DType::I8;
    if (s == "I16")  return DType::I16;
    if (s == "I32")  return DType::I32;
    if (s == "I64")  return DType::I64;
    if (s == "U8")   return DType::U8;
    if (s == "U16")  return DType::U16;
    if (s == "U32")  return DType::U32;
    if (s == "U64")  return DType::U64;
    if (s == "BOOL") return DType::Bool;
    return SFS_ERR(UnsupportedDType, "unknown safetensors dtype", std::string(s));
}

namespace {

float f16_to_f32(std::uint16_t h) noexcept {
    // h & 0x8000u already has type uint32_t (usual arithmetic conversions
    // promote h's uint16_t to the wider unsigned operand): no cast needed.
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal: normalise.
            int e = -1;
            do { mant <<= 1; ++e; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16_to_f32(std::uint16_t h) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

}  // namespace

float to_float(DType d, const std::byte* elem) noexcept {
    switch (d) {
        case DType::F16: {
            std::uint16_t v;
            std::memcpy(&v, elem, 2);
            return f16_to_f32(v);
        }
        case DType::BF16: {
            std::uint16_t v;
            std::memcpy(&v, elem, 2);
            return bf16_to_f32(v);
        }
        case DType::F32: {
            float v;
            std::memcpy(&v, elem, 4);
            return v;
        }
        case DType::F64: {
            double v;
            std::memcpy(&v, elem, 8);
            return static_cast<float>(v);
        }
        default:
            return 0.0f;  // not a cost-bearing type; callers gate on is_float()
    }
}

}  // namespace sfs::core
