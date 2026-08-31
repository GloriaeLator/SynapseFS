#include <synapsefs/codec/diff_encoder.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include <synapsefs/codec/permute.hpp>
#include <synapsefs/codec/residual_codec.hpp>
#include <synapsefs/core/endian.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::codec {

using core::Result;
using core::Status;

namespace {

// ---------------------------------------------------------- zigzag encode
//
// Not exposed by residual_codec.hpp: only *_apply is ISA-dispatched (only
// the read path is ADR 0011's hot loop; encode happens once per commit).
// Mirrors residual_scalar.cpp's decode formula in reverse — the standard
// zigzag bit trick, generalised from a fixed 64-bit width to whatever the
// dtype's element width is. (Same helper as bench/residual_codec.cpp's,
// duplicated intentionally rather than shared: it's three lines, and giving
// it a shared home would mean adding a private-detail header for one
// function used in exactly two places, neither of which is on a hot path.)
template <typename U>
void zigzag_encode_width(std::byte* dst, const std::byte* base, const std::byte* target,
                         std::size_t count) noexcept {
    constexpr U kTop = U(1) << (sizeof(U) * 8 - 1);
    for (std::size_t i = 0; i < count; ++i) {
        U b = 0, t = 0;
        std::memcpy(&b, base + i * sizeof(U), sizeof(U));
        std::memcpy(&t, target + i * sizeof(U), sizeof(U));
        const U delta_bits = static_cast<U>(t - b);  // wraparound: bits of (t - b)
        const U sign_mask = (delta_bits & kTop) ? static_cast<U>(~U(0)) : U(0);
        const U z = static_cast<U>((delta_bits << 1) ^ sign_mask);
        std::memcpy(dst + i * sizeof(U), &z, sizeof(U));
    }
}

void zigzag_encode(std::byte* dst, const std::byte* base, const std::byte* target,
                   std::size_t n, std::uint32_t elem_bytes) noexcept {
    switch (elem_bytes) {
        case 1: zigzag_encode_width<std::uint8_t>(dst, base, target, n / 1); return;
        case 2: zigzag_encode_width<std::uint16_t>(dst, base, target, n / 2); return;
        case 4: zigzag_encode_width<std::uint32_t>(dst, base, target, n / 4); return;
        case 8: zigzag_encode_width<std::uint64_t>(dst, base, target, n / 8); return;
        default: return;
    }
}

// --------------------------------------------------------- gather helper

// Read the base units named by expanded_perm[first, first+count) — scattered
// by the permutation — into `out`, in TARGET order. Reads each contiguous
// run of base units once (codec::dependency_runs), then gathers from a
// scratch buffer into final order, rather than one read per unit.
Result<void> gather_base_units(core::ITensorSource& base, std::string_view name,
                               std::span<const std::uint32_t> expanded_perm, std::uint64_t first,
                               std::uint64_t count, std::uint64_t unit_bytes,
                               std::span<std::byte> out) {
    const auto runs = dependency_runs(expanded_perm, first, count);

    std::vector<std::byte> scratch;
    std::unordered_map<std::uint32_t, std::uint64_t> unit_to_scratch_off;
    for (const auto& run : runs) {
        const std::uint64_t off = scratch.size();
        scratch.resize(off + run.count * unit_bytes);
        auto n = base.read_units(name, run.first, run.count,
                                 std::span(scratch.data() + off, run.count * unit_bytes));
        if (!n) return std::unexpected(n.error());
        for (std::uint64_t k = 0; k < run.count; ++k)
            unit_to_scratch_off[static_cast<std::uint32_t>(run.first + k)] = off + k * unit_bytes;
    }

    for (std::uint64_t i = 0; i < count; ++i) {
        const auto it = unit_to_scratch_off.find(expanded_perm[first + i]);
        if (it == unit_to_scratch_off.end())
            return SFS_ERR(Internal, "gather_base_units: dependency run missing a unit",
                          std::string(name));
        std::memcpy(out.data() + i * unit_bytes, scratch.data() + it->second, unit_bytes);
    }
    return {};
}

}  // namespace

format::PermutationRef write_permutation(std::vector<std::byte>& payload,
                                         std::span<const std::uint32_t> perm) {
    format::PermutationRef ref;
    ref.n = static_cast<std::uint32_t>(perm.size());

    // Identity is worth its own zero-byte encoding: the common case for a
    // fine-tune, and it removes both the storage and the per-hop parse
    // (spec 12 §3).
    bool is_identity = true;
    for (std::uint32_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != i) { is_identity = false; break; }
    }
    if (is_identity) {
        ref.kind = format::PermKind::Identity;
        return ref;
    }

    ref.kind = format::PermKind::Explicit;
    ref.width = ref.n <= 65536 ? 2 : 4;
    ref.off = payload.size();
    ref.len = static_cast<std::uint64_t>(ref.n) * ref.width;

    const std::size_t base = payload.size();
    payload.resize(base + ref.len);
    for (std::uint32_t i = 0; i < ref.n; ++i) {
        if (ref.width == 2)
            core::store_le<std::uint16_t>(payload.data() + base + i * 2,
                                          static_cast<std::uint16_t>(perm[i]));
        else
            core::store_le<std::uint32_t>(payload.data() + base + i * 4, perm[i]);
    }
    return ref;
}

Result<EncodeResult> encode_group(core::ITensorSource& base, core::ITensorSource& target,
                                  const core::Topology& topology, std::string_view group,
                                  std::span<const std::uint32_t> permutation, bool alignable,
                                  const format::AlignmentInfo& alignment_info,
                                  const EncodeOptions& opts) {
    format::DiffHeader header;
    header.format_version = 1;
    header.group = std::string(group);
    header.codec = opts.codec;
    header.alignable = alignable;
    header.alignment = alignment_info;

    std::vector<std::byte> payload;
    header.permutation = write_permutation(payload, permutation);

    std::uint64_t full_bytes = 0;

    // `group` is a PERMUTATION group name (topology.groups' key, e.g. "g0"),
    // not a tensor name: spec 13's own worked example has "0.weight",
    // "0.bias" AND "1.weight" (a BatchNorm scale riding on a preceding
    // conv's channel permutation) all sharing one dim-0 group, and the
    // golden fixture (tests/golden/diff_artifact.json) confirms one artifact
    // legitimately holds several tensors. Every tensor whose dim-0 (output)
    // axis is bound to this permutation group goes into this one artifact.
    //
    // format::Manifest.groups stays keyed one entry PER TENSOR regardless
    // (apps/sfs/cmd/commit.cpp) — several tensors' GroupEntry can share the
    // same diff_block Oid produced here; the reader finds which one it
    // wants inside the shared artifact BY NAME (TensorDiff.shape_name), not
    // by any cross-tensor byte-offset concatenation. See reconstruct.cpp.
    //
    // Deliberately dim-0-only, not "any matching axis": a tensor can also
    // have a non-pinned INPUT axis bound to this same group (a hidden
    // layer's weight matrix has dim0 -> its own output group AND
    // dim1 -> the previous layer's output group). That input-side axis is a
    // column, not a row — permute_units/gather_base_units's unit-sized
    // memcpy only makes sense for the row-contiguous dim-0 case. Correctly
    // reconstructing a tensor with a permuted INPUT axis needs the caller
    // (whatever orchestrates all of a commit's groups together, outside
    // codec/) to compose multiple encode_group calls' permutations; this
    // function, called once per permutation group, only ever touches dim 0.
    for (const auto& [name, axes] : topology.tensors) {
        const core::AxisBinding* binding = nullptr;
        for (const auto& axis : axes.axes) {
            if (axis.dim == 0 && axis.group == group) { binding = &axis; break; }
        }
        if (binding == nullptr) continue;

        const auto* meta = target.meta(name);
        if (meta == nullptr) return SFS_ERR(TensorNotInBufferLayout, "tensor not in target", name);

        auto unit_bytes_r = meta->unit_bytes(binding->dim);
        if (!unit_bytes_r) return std::unexpected(unit_bytes_r.error());
        const std::uint64_t unit_bytes = *unit_bytes_r;

        const auto expanded_perm = expand(permutation, binding->block);
        const std::uint64_t total_units = expanded_perm.size();
        if (meta->shape[binding->dim] != total_units)
            return SFS_ERR(BlockFactorMismatch, "expanded permutation size != axis length", name);

        // Frame size targets frame_bytes, rounded to a whole number of
        // units; a single unit larger than the target still gets its own
        // frame (spec 12 §4).
        const std::uint64_t frame_units = std::max<std::uint64_t>(1, opts.frame_bytes / std::max<std::uint64_t>(1, unit_bytes));

        format::TensorDiff tdiff;
        tdiff.shape_name = name;
        tdiff.shape = meta->shape;
        tdiff.dtype = meta->dtype;
        tdiff.residual = opts.residual;
        tdiff.transform = opts.transform;

        for (std::uint64_t a = 0; a < total_units; a += frame_units) {
            const std::uint64_t b = std::min(total_units, a + frame_units);
            const std::uint64_t frame_bytes_raw = (b - a) * unit_bytes;

            std::vector<std::byte> target_bytes(frame_bytes_raw);
            auto tr = target.read_units(name, a, b - a, target_bytes);
            if (!tr) return std::unexpected(tr.error());
            full_bytes += *tr;

            std::vector<std::byte> resid(frame_bytes_raw);
            if (opts.residual == format::ResidualKind::Raw) {
                resid = target_bytes;
            } else {
                std::vector<std::byte> base_gathered(frame_bytes_raw);
                auto g = gather_base_units(base, name, expanded_perm, a, b - a, unit_bytes,
                                           base_gathered);
                if (!g) return std::unexpected(g.error());

                if (opts.residual == format::ResidualKind::XorAfterPermute) {
                    xor_encode_dispatch()(resid.data(), base_gathered.data(), target_bytes.data(),
                                         frame_bytes_raw);
                } else {
                    zigzag_encode(resid.data(), base_gathered.data(), target_bytes.data(),
                                 frame_bytes_raw, core::dtype_size(meta->dtype));
                }

                if (opts.verify_round_trip) {
                    // The writer MUST compute each digest from the bytes it
                    // intends the reader to produce, proven by re-reading
                    // them through the reader path (spec 12 §8) — this is
                    // that check, done inline rather than trusting that
                    // encode and apply_residual are exact inverses.
                    std::vector<std::byte> check(frame_bytes_raw);
                    auto st = apply_residual(opts.residual, format::Transform::None,
                                            meta->dtype, base_gathered, resid, check);
                    if (!st) return std::unexpected(st.error());
                    if (check != target_bytes)
                        return SFS_ERR(Internal,
                                       "encode/apply_residual round trip mismatch", name);
                }
            }

            std::vector<std::byte> transformed(resid.size());
            if (opts.transform == format::Transform::None) {
                transformed = resid;
            } else if (opts.transform == format::Transform::BytePlane) {
                byteplane_split(resid, transformed, core::dtype_size(meta->dtype));
            } else {
                bitshuffle(resid, transformed, core::dtype_size(meta->dtype));
            }

            auto comp = compress_frame(transformed, opts.compress);
            if (!comp) return std::unexpected(comp.error());

            format::FrameIndexEntry entry;
            entry.unit_begin = a;
            entry.unit_end = b;
            entry.off = payload.size();
            entry.len = comp->size();
            // Digest covers the RECONSTRUCTED TARGET BYTES, so tamper
            // detection survives the reconstruction path (spec 12 §4): this
            // is exactly what the reader will reproduce given a correct
            // base, residual and transform, verified above when
            // verify_round_trip is set.
            entry.digest = core::digest(target_bytes);

            payload.insert(payload.end(), comp->begin(), comp->end());
            tdiff.frames.push_back(entry);
        }

        if (auto st = tdiff.validate_tiling(total_units); !st)
            return std::unexpected(st.error());
        header.tensors.push_back(std::move(tdiff));
    }

    EncodeResult result;
    const auto header_json = header.to_canonical_json();

    result.artifact.resize(8 + header_json.size() + payload.size());
    core::store_le<std::uint64_t>(result.artifact.data(), header_json.size());
    std::memcpy(result.artifact.data() + 8, header_json.data(), header_json.size());
    std::memcpy(result.artifact.data() + 8 + header_json.size(), payload.data(), payload.size());

    result.payload_bytes = payload.size();
    result.full_bytes = full_bytes;
    result.ratio = full_bytes > 0
                      ? static_cast<double>(result.artifact.size()) / static_cast<double>(full_bytes)
                      : 0.0;
    return result;
}

}  // namespace sfs::codec
