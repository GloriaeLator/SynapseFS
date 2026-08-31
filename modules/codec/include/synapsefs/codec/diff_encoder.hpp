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
    // Winner of the six-candidate experiment (docs/tradeoffs.md §1.4,
    // ADR 0005): zigzag(b-a) + no transform beat XOR and both byte-level
    // transforms on BOTH ratio and decompress throughput on the fine-tune
    // pair, the only one of the three measured pairs with real signal to
    // discriminate on.
    format::ResidualKind residual  = format::ResidualKind::ZigzagAfterPermute;
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

/// Encode one manifest group. `group` IS the tensor's name — a manifest group
/// is always exactly one tensor (apps/sfs/cmd/commit.cpp: "every tensor is
/// stored as its own singleton Full group"), never several. `permutation` is
/// the array for whichever permutation group that tensor's OWN dim-0 (output)
/// axis is bound to in `topology` — the caller resolves which permutation
/// group that is and looks up its array; this function only needs the result.
///
/// A tensor's OTHER axes (e.g. a hidden layer's weight matrix also has an
/// input axis bound to the PREVIOUS layer's permutation group) are the
/// caller's concern: reconstructing one correctly needs `base`'s bytes
/// pre-permuted along that axis (e.g. via an input-permuting ITensorSource
/// adapter) before calling this function. `base` and `target` are streamed,
/// never fully resident.
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
