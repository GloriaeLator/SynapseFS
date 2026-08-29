#pragma once
/// \file residual_hdr.hpp
/// The diff artifact's JSON header and frame index.
/// docs/spec/12-residual-format.md.
///
/// Layout: [8-byte LE header_len][JSON header][payload]
/// All `off` values in the header are relative to the START OF THE PAYLOAD.
/// The framing mirrors .safetensors deliberately, so the same reader primitive
/// serves both.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <synapsefs/core/dtype.hpp>
#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::format {

using core::DType;
using core::Result;
using core::Status;

inline constexpr std::string_view kDiffMagic = "SYNDIFF";

enum class ResidualKind : std::uint8_t {
    Raw,                 ///< frame bytes ARE the target bytes; no base read
    XorAfterPermute,     ///< target[i] = base[p[i]] ^ residual[i]
    ZigzagAfterPermute,  ///< target = base + zigzag_decode(residual)
};

enum class Transform : std::uint8_t { None, BytePlane, Bitshuffle };
enum class Codec     : std::uint8_t { None, Zstd };

enum class PermKind : std::uint8_t {
    Identity,   ///< ZERO payload bytes. The common case for a fine-tune.
    Explicit,
};

struct PermutationRef {
    PermKind      kind = PermKind::Identity;
    std::uint32_t n     = 0;
    std::uint8_t  width = 2;    ///< 2 (u16) if n <= 65536, else 4
    std::uint64_t off   = 0;    ///< payload-relative
    std::uint64_t len   = 0;
};

/// One independently decompressible zstd frame covering a contiguous run of
/// output units.
struct FrameIndexEntry {
    std::uint64_t unit_begin = 0;
    std::uint64_t unit_end   = 0;   ///< half-open
    std::uint64_t off        = 0;   ///< payload-relative
    std::uint64_t len        = 0;   ///< compressed length
    /// BLAKE3 of the RECONSTRUCTED TARGET BYTES for these units — after
    /// decompression and after the residual is applied. That is what makes
    /// tamper detection survive the reconstruction path.
    std::array<std::byte, core::kOidBytes> digest{};
};

struct TensorDiff {
    std::string                shape_name;
    std::vector<std::uint64_t> shape;
    DType                      dtype = DType::F16;
    ResidualKind               residual = ResidualKind::XorAfterPermute;
    Transform                  transform = Transform::None;
    std::vector<FrameIndexEntry> frames;

    /// Frames must tile [0, group_size) exactly: begin at 0, contiguous, and
    /// end at group_size.
    [[nodiscard]] Status validate_tiling(std::uint64_t group_size) const;
    /// Index of the first frame covering `unit`, or npos.
    [[nodiscard]] std::size_t frame_for_unit(std::uint64_t unit) const noexcept;
};

struct AlignmentInfo {
    std::string method = "weight_matching_lap";
    double cost_raw = 0.0;
    double cost_normalized = 0.0;
};

struct DiffHeader {
    std::uint32_t   format_version = 1;
    std::string     group;
    Codec           codec = Codec::Zstd;
    PermutationRef  permutation;
    std::vector<TensorDiff> tensors;
    /// false means the aligner found no meaningful correspondence. The
    /// containing manifest entry must then be Full.
    bool            alignable = true;
    AlignmentInfo   alignment;

    [[nodiscard]] std::vector<std::byte> to_canonical_json() const;
    /// Magic mismatch is a hard error, never a fallback.
    [[nodiscard]] static Result<DiffHeader> parse(std::span<const std::byte> json);
};

/// Split an artifact into its header JSON and its payload span.
struct DiffArtifactView {
    DiffHeader                 header;
    std::span<const std::byte> payload;
};

[[nodiscard]] Result<DiffArtifactView> parse_diff_artifact(std::span<const std::byte>);

/// Read the permutation named by `ref` out of `payload`, validating that it is
/// a bijection on [0, n). MUST be called before the values index anything.
[[nodiscard]] Result<std::vector<std::uint32_t>> read_permutation(
    const PermutationRef&, std::span<const std::byte> payload);

}  // namespace sfs::format
