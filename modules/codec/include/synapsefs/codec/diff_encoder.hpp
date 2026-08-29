#pragma once
/// \file diff_encoder.hpp
/// Building a diff artifact: permutation, framed residuals, digests.
///
/// The writer MUST compute each frame digest from the bytes it intends the
/// READER to produce. In debug builds it re-reads them back through
/// reconstruct.hpp to prove it; a writer that digests its own in-memory target
/// is not testing reconstruction, and tests/byte_identity.cpp is what stops
/// that being merged.

#include <cstdint>
#include <span>
#include <vector>

#include <synapsefs/codec/compress.hpp>
#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/topology.hpp>
#include <synapsefs/format/residual_hdr.hpp>

namespace sfs::codec {

struct EncodeOptions {
    format::ResidualKind residual  = format::ResidualKind::XorAfterPermute;
    format::Transform    transform = format::Transform::None;
    format::Codec        codec     = format::Codec::Zstd;
    CompressOptions      compress;
    /// Target frame size, rounded to whole output units. Baked into the
    /// artifact; readers take it from the header, never from config.
    std::uint64_t frame_bytes = 128u * 1024;
    /// Re-read every frame through the reader path before emitting. On in
    /// debug and in CI; the round trip is the only thing that actually proves
    /// byte-exactness.
    bool verify_round_trip = false;
};

struct EncodeResult {
    std::vector<std::byte> artifact;      ///< [8-byte len][JSON][payload]
    std::uint64_t          payload_bytes = 0;
    std::uint64_t          full_bytes    = 0;   ///< for the snapshot-ratio test
    double                 ratio = 0.0;
};

/// Encode one permutation group. `base` and `target` are streamed, never fully
/// resident.
[[nodiscard]] core::Result<EncodeResult> encode_group(
    core::ITensorSource& base, core::ITensorSource& target, const core::Topology&,
    std::string_view group, std::span<const std::uint32_t> permutation,
    bool alignable, const format::AlignmentInfo&, const EncodeOptions& = {});

/// Serialise a permutation into the artifact payload: u16 when n <= 65536, u32
/// otherwise, and ZERO bytes when it is the identity — which is the common case
/// for a fine-tune and is worth its own encoding.
[[nodiscard]] format::PermutationRef write_permutation(std::vector<std::byte>& payload,
                                                       std::span<const std::uint32_t> perm);

}  // namespace sfs::codec
