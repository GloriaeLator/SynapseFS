#include <synapsefs/codec/residual_codec.hpp>

#include <cstring>
#include <vector>

#include <synapsefs/codec/compress.hpp>

namespace sfs::codec {

using core::DType;
using core::Status;
using format::ResidualKind;
using format::Transform;

Status apply_residual(ResidualKind kind, Transform transform, DType dtype,
                      std::span<const std::byte> base, std::span<const std::byte> residual,
                      std::span<std::byte> out) {
    if (out.size() != residual.size())
        return SFS_ERR(Internal, "apply_residual: out and residual frame sizes differ");

    const std::uint32_t elem_bytes = core::dtype_size(dtype);
    if (elem_bytes == 0) return SFS_ERR(UnsupportedDType, "apply_residual: unknown dtype");

    // The transform (byteplane/bitshuffle) was applied to whichever bytes
    // are stored in the frame, BEFORE compression, on the write side — so it
    // is undone first, before any residual kind gets to see the bytes.
    // spec 12 §5.
    std::vector<std::byte> untransformed;
    std::span<const std::byte> payload = residual;
    if (transform != Transform::None) {
        untransformed.resize(residual.size());
        if (transform == Transform::BytePlane)
            byteplane_join(residual, untransformed, elem_bytes);
        else
            bitunshuffle(residual, untransformed, elem_bytes);
        payload = untransformed;
    }

    switch (kind) {
        case ResidualKind::Raw:
            // Frame bytes ARE the target bytes; no base is read. spec 12 §5.
            if (out.data() != payload.data())
                std::memcpy(out.data(), payload.data(), payload.size());
            return {};

        case ResidualKind::XorAfterPermute:
            if (base.size() != payload.size())
                return SFS_ERR(MalformedObject, "apply_residual: base/residual size mismatch");
            xor_apply_dispatch()(out.data(), base.data(), payload.data(), out.size());
            return {};

        case ResidualKind::ZigzagAfterPermute:
            if (base.size() != payload.size())
                return SFS_ERR(MalformedObject, "apply_residual: base/residual size mismatch");
            if (payload.size() % elem_bytes != 0)
                return SFS_ERR(MalformedObject,
                               "apply_residual: residual size not a multiple of element size");
            zigzag_apply_dispatch()(out.data(), base.data(), payload.data(), out.size(),
                                   elem_bytes);
            return {};
    }

    return SFS_ERR(Internal, "apply_residual: unknown residual kind");
}

}  // namespace sfs::codec
