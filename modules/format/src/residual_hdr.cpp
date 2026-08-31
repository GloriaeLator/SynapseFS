#include <synapsefs/format/residual_hdr.hpp>

#include <cstring>

#include <nlohmann/json.hpp>

#include <synapsefs/core/endian.hpp>
#include <synapsefs/core/topology.hpp>  // core::is_valid_permutation

// No implementation existed anywhere in the tree for this header — both
// diff_encoder.cpp (write side) and reconstruct.cpp's Delta branch (read
// side) need it directly to build/parse a real diff artifact, so it's
// written here rather than left as a second gap alongside align/.
//
// JSON shape matches tests/golden/diff_artifact.json and spec 12 §1/§3/§4/§5
// exactly — this is the wire format, not an internal convenience, so field
// names and enum strings are load-bearing.

namespace sfs::format {

using json = nlohmann::json;

namespace {

std::string_view residual_kind_to_string(ResidualKind k) noexcept {
    switch (k) {
        case ResidualKind::Raw:                return "raw";
        case ResidualKind::XorAfterPermute:     return "xor_after_permute";
        case ResidualKind::ZigzagAfterPermute:  return "zigzag_after_permute";
    }
    return "?";
}

Result<ResidualKind> residual_kind_from_string(std::string_view s) {
    if (s == "raw") return ResidualKind::Raw;
    if (s == "xor_after_permute") return ResidualKind::XorAfterPermute;
    if (s == "zigzag_after_permute") return ResidualKind::ZigzagAfterPermute;
    return SFS_ERR(MalformedObject, "unknown residual kind", std::string(s));
}

std::string_view transform_to_string(Transform t) noexcept {
    switch (t) {
        case Transform::None:       return "none";
        case Transform::BytePlane:  return "byteplane";
        case Transform::Bitshuffle: return "bitshuffle";
    }
    return "?";
}

Result<Transform> transform_from_string(std::string_view s) {
    if (s == "none") return Transform::None;
    if (s == "byteplane") return Transform::BytePlane;
    if (s == "bitshuffle") return Transform::Bitshuffle;
    return SFS_ERR(MalformedObject, "unknown transform", std::string(s));
}

std::string_view codec_to_string(Codec c) noexcept {
    switch (c) {
        case Codec::None: return "none";
        case Codec::Zstd: return "zstd";
    }
    return "?";
}

Result<Codec> codec_from_string(std::string_view s) {
    if (s == "none") return Codec::None;
    if (s == "zstd") return Codec::Zstd;
    return SFS_ERR(MalformedObject, "unknown codec", std::string(s));
}

std::string hex_encode(std::span<const std::byte> bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (auto b : bytes) {
        const auto v = std::to_integer<unsigned>(b);
        out.push_back(kHex[v >> 4]);
        out.push_back(kHex[v & 0xFu]);
    }
    return out;
}

Result<unsigned> hex_nibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
    return SFS_ERR(MalformedObject, "invalid hex digit in digest");
}

Result<std::array<std::byte, core::kOidBytes>> hex_decode_digest(const std::string& s) {
    if (s.size() != core::kOidBytes * 2)
        return SFS_ERR(MalformedObject, "digest has wrong length", s);
    std::array<std::byte, core::kOidBytes> out{};
    for (std::size_t i = 0; i < core::kOidBytes; ++i) {
        auto hi = hex_nibble(s[i * 2]);
        if (!hi) return std::unexpected(hi.error());
        auto lo = hex_nibble(s[i * 2 + 1]);
        if (!lo) return std::unexpected(lo.error());
        out[i] = static_cast<std::byte>((*hi << 4) | *lo);
    }
    return out;
}

json permutation_ref_to_json(const PermutationRef& p) {
    if (p.kind == PermKind::Identity) return json{{"kind", "identity"}};
    return json{{"kind", "explicit"},
               {"n", p.n},
               {"dtype", p.width == 2 ? "u16" : "u32"},
               {"off", p.off},
               {"len", p.len}};
}

Result<PermutationRef> permutation_ref_from_json(const json& j) {
    PermutationRef ref;
    const auto kind = j.at("kind").get<std::string>();
    if (kind == "identity") {
        ref.kind = PermKind::Identity;
        return ref;
    }
    if (kind != "explicit")
        return SFS_ERR(MalformedObject, "unknown permutation kind", kind);

    ref.kind = PermKind::Explicit;
    ref.n = j.at("n").get<std::uint32_t>();
    const auto dt = j.at("dtype").get<std::string>();
    if (dt == "u16") ref.width = 2;
    else if (dt == "u32") ref.width = 4;
    else return SFS_ERR(MalformedObject, "unknown permutation dtype", dt);
    ref.off = j.at("off").get<std::uint64_t>();
    ref.len = j.at("len").get<std::uint64_t>();
    return ref;
}

}  // namespace

Status TensorDiff::validate_tiling(std::uint64_t group_size) const {
    if (frames.empty())
        return group_size == 0 ? Status{}
                               : SFS_ERR(MalformedObject, "no frames for a non-empty group",
                                        shape_name);
    if (frames.front().unit_begin != 0)
        return SFS_ERR(MalformedObject, "frames do not start at unit 0", shape_name);
    for (std::size_t i = 0; i + 1 < frames.size(); ++i) {
        if (frames[i].unit_end != frames[i + 1].unit_begin)
            return SFS_ERR(MalformedObject, "frame gap or overlap", shape_name);
    }
    if (frames.back().unit_end != group_size)
        return SFS_ERR(MalformedObject, "frames do not cover the whole group", shape_name);
    return {};
}

std::size_t TensorDiff::frame_for_unit(std::uint64_t unit) const noexcept {
    for (std::size_t i = 0; i < frames.size(); ++i)
        if (unit >= frames[i].unit_begin && unit < frames[i].unit_end) return i;
    return static_cast<std::size_t>(-1);
}

std::vector<std::byte> DiffHeader::to_canonical_json() const {
    json tensors_json = json::array();
    for (const auto& t : tensors) {
        json frames_json = json::array();
        for (const auto& f : t.frames) {
            frames_json.push_back({{"units", {f.unit_begin, f.unit_end}},
                                   {"off", f.off},
                                   {"len", f.len},
                                   {"digest", hex_encode(f.digest)}});
        }
        tensors_json.push_back({{"name", t.shape_name},
                                {"shape", t.shape},
                                {"dtype", std::string(core::to_string(t.dtype))},
                                {"residual", std::string(residual_kind_to_string(t.residual))},
                                {"transform", std::string(transform_to_string(t.transform))},
                                {"frames", frames_json}});
    }

    const json j = {
        {"magic", std::string(kDiffMagic)},
        {"format_version", format_version},
        {"group", group},
        {"codec", std::string(codec_to_string(codec))},
        {"permutation", permutation_ref_to_json(permutation)},
        {"tensors", tensors_json},
        {"alignable", alignable},
        {"alignment", {{"method", alignment.method},
                       {"cost_raw", alignment.cost_raw},
                       {"cost_normalized", alignment.cost_normalized}}},
    };

    const std::string dumped = j.dump();
    std::vector<std::byte> out(dumped.size());
    std::memcpy(out.data(), dumped.data(), dumped.size());
    return out;
}

Result<DiffHeader> DiffHeader::parse(std::span<const std::byte> bytes) {
    json j;
    try {
        j = json::parse(
            std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("diff header JSON parse failed: ") + e.what());
    }
    if (!j.is_object()) return SFS_ERR(MalformedObject, "diff header is not a JSON object");

    // Magic mismatch is a hard error, never a fallback (this header's own
    // doc comment, spec 12 §1).
    const auto magic = j.value("magic", std::string());
    if (magic != kDiffMagic) return SFS_ERR(MalformedObject, "bad diff artifact magic", magic);

    DiffHeader h;
    h.format_version = j.at("format_version").get<std::uint32_t>();
    h.group = j.at("group").get<std::string>();

    auto codec_r = codec_from_string(j.at("codec").get<std::string>());
    if (!codec_r) return std::unexpected(codec_r.error());
    h.codec = *codec_r;

    auto perm_r = permutation_ref_from_json(j.at("permutation"));
    if (!perm_r) return std::unexpected(perm_r.error());
    h.permutation = *perm_r;

    for (const auto& tj : j.at("tensors")) {
        TensorDiff t;
        t.shape_name = tj.at("name").get<std::string>();
        for (const auto& d : tj.at("shape")) t.shape.push_back(d.get<std::uint64_t>());

        auto dt = core::dtype_from_string(tj.at("dtype").get<std::string>());
        if (!dt) return std::unexpected(dt.error());
        t.dtype = *dt;

        auto res_r = residual_kind_from_string(tj.at("residual").get<std::string>());
        if (!res_r) return std::unexpected(res_r.error());
        t.residual = *res_r;

        auto tr_r = transform_from_string(tj.at("transform").get<std::string>());
        if (!tr_r) return std::unexpected(tr_r.error());
        t.transform = *tr_r;

        for (const auto& fj : tj.at("frames")) {
            const auto& units = fj.at("units");
            if (!units.is_array() || units.size() != 2)
                return SFS_ERR(MalformedObject, "malformed frame units", t.shape_name);

            FrameIndexEntry f;
            f.unit_begin = units[0].get<std::uint64_t>();
            f.unit_end = units[1].get<std::uint64_t>();
            f.off = fj.at("off").get<std::uint64_t>();
            f.len = fj.at("len").get<std::uint64_t>();

            auto dig = hex_decode_digest(fj.at("digest").get<std::string>());
            if (!dig) return std::unexpected(dig.error());
            f.digest = *dig;
            t.frames.push_back(std::move(f));
        }
        h.tensors.push_back(std::move(t));
    }

    h.alignable = j.value("alignable", true);
    if (j.contains("alignment")) {
        const auto& aj = j.at("alignment");
        h.alignment.method = aj.value("method", std::string("weight_matching_lap"));
        h.alignment.cost_raw = aj.value("cost_raw", 0.0);
        h.alignment.cost_normalized = aj.value("cost_normalized", 0.0);
    }

    return h;
}

Result<DiffArtifactView> parse_diff_artifact(std::span<const std::byte> bytes) {
    if (bytes.size() < 8)
        return SFS_ERR(MalformedObject, "diff artifact truncated (no header length)");
    const std::uint64_t header_len = core::load_le<std::uint64_t>(bytes.data());
    if (header_len == 0 || bytes.size() < 8 + header_len)
        return SFS_ERR(MalformedObject, "diff artifact truncated or implausible header length");

    auto hdr = DiffHeader::parse(bytes.subspan(8, header_len));
    if (!hdr) return std::unexpected(hdr.error());

    DiffArtifactView view;
    view.header = std::move(*hdr);
    view.payload = bytes.subspan(8 + header_len);
    return view;
}

Result<std::vector<std::uint32_t>> read_permutation(const PermutationRef& ref,
                                                    std::span<const std::byte> payload) {
    if (ref.kind == PermKind::Identity) {
        // Zero payload bytes, and the header encodes no N for it (spec 12
        // §3/§4.2): callers special-case Identity as "same range on both
        // sides", trivially aligned, rather than expect a materialised
        // array here.
        return std::vector<std::uint32_t>{};
    }

    if (ref.len != static_cast<std::uint64_t>(ref.n) * ref.width)
        return SFS_ERR(MalformedObject, "permutation length does not match n * width");
    if (ref.off + ref.len > payload.size())
        return SFS_ERR(MalformedObject, "permutation extends past payload");

    std::vector<std::uint32_t> out(ref.n);
    const std::byte* p = payload.data() + ref.off;
    for (std::uint32_t i = 0; i < ref.n; ++i) {
        out[i] = ref.width == 2 ? core::load_le<std::uint16_t>(p + i * 2)
                                : core::load_le<std::uint32_t>(p + i * 4);
    }

    // MUST be validated before any of these values index anything: spec 12
    // §3 — a malformed permutation otherwise drives an out-of-bounds read
    // from file contents, the one place a corrupt object could become a
    // memory-safety problem rather than a rejected read.
    if (!core::is_valid_permutation(out, ref.n))
        return SFS_ERR(InvalidPermutation, "permutation is not a bijection on [0, n)");

    return out;
}

}  // namespace sfs::format
