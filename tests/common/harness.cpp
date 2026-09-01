/// \file harness.cpp
/// Implements tests/common/harness.hpp.
///
/// write_synthetic_checkpoint() builds a real .safetensors file byte-by-byte
/// in C++ (8-byte LE header length, JSON header, concatenated tensor data —
/// docs/storage_format.md §2 / format/st_header.hpp), rather than shelling
/// out to Python: the whole point (see harness.hpp's own file comment) is
/// that `ctest --preset unit` runs with no Python and no downloads. The
/// architecture it generates mirrors fixtures/gen_mlp.py and
/// fixtures/gen_resnet.py exactly — same nn.Sequential-index tensor naming,
/// same "biases/running-stats before weights" awkward insertion order — so a
/// test using this harness is exercising the identical shape those Python
/// fixtures exercise, just without needing them on disk.
///
/// SyntheticSpec.layer_widths, decoded:
///   with_conv == false: an MLP. widths.size()-1 Linear layers, named
///     "0.weight"/"0.bias", "2.weight"/"2.bias", ... (ReLU at the odd
///     indices, contributing no tensors — matches gen_mlp.py).
///   with_conv == true: widths must supply at least 4 entries, read as
///     [in_channels, conv0_out, conv1_out, linear_out]; extra entries are
///     ignored. Fixed spatial=8, kernel=3, maxpool(2) — the same
///     conv/bn/relu/conv/bn/relu/maxpool/flatten/linear shape as
///     gen_resnet.py, indices 0/1/3/4/8 (2/5/6/7 are ReLU/pool/flatten and
///     own no tensors). with_batchnorm toggles whether "1.*"/"4.*" exist.
///
/// A caller that needs a core::Topology matching a given SyntheticSpec builds
/// one directly from these same names/shapes (see
/// fixtures/gen_mlp.py::topology_sidecar / gen_resnet.py::topology_sidecar
/// for the reference shape) — write_synthetic_checkpoint does not return one
/// itself, since SyntheticSpec's naming is deterministic and the header
/// above documents it exactly.

#include "harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <synapsefs/core/dtype.hpp>
#include <synapsefs/core/tensor.hpp>
#include <synapsefs/format/st_header.hpp>
#include <synapsefs/stio/st_writer.hpp>

namespace sfs::test {

namespace {

using json = nlohmann::ordered_json;  // preserves insertion order verbatim

// ---- float16 encode (core/src/dtype.cpp only ever decodes; fixture
// generation is the one place that needs the other direction) ----

std::uint16_t f32_to_f16(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;

    if (exp <= 0) {
        // Flush to zero (subnormal fp16 is not worth the extra branch for a
        // test fixture generator).
        return static_cast<std::uint16_t>(sign);
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);  // inf
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (mant >> 13));
}

std::uint16_t f32_to_bf16(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

/// The write-direction counterpart to core::to_float — encodes `v` as
/// dtype `d` at `dst`. Only the float dtypes matter here; perturb() leaves
/// every other dtype's bytes untouched rather than calling this.
void write_float(core::DType d, std::byte* dst, float v) noexcept {
    switch (d) {
        case core::DType::F16: {
            std::uint16_t h = f32_to_f16(v);
            std::memcpy(dst, &h, 2);
            return;
        }
        case core::DType::BF16: {
            std::uint16_t h = f32_to_bf16(v);
            std::memcpy(dst, &h, 2);
            return;
        }
        case core::DType::F32: {
            std::memcpy(dst, &v, 4);
            return;
        }
        case core::DType::F64: {
            double d64 = v;
            std::memcpy(dst, &d64, 8);
            return;
        }
        default:
            return;  // not reached: callers gate on core::is_float()
    }
}

struct GenTensor {
    std::string                name;
    std::vector<std::uint64_t> shape;
    core::DType                dtype = core::DType::F16;
    std::vector<std::byte>     bytes;
};

std::uint64_t elem_count(const std::vector<std::uint64_t>& shape) {
    std::uint64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

/// Fill `t.bytes` with fp16 samples: He-ish scale for weights, a small
/// nonzero constant offset for bias-shaped tensors (a zero vector stays a
/// zero vector under any permutation, which would make a planted-permutation
/// test trivially byte-identical on the bias tensors and prove nothing —
/// same reasoning fixtures/gen_mlp.py documents).
void fill_normal(GenTensor& t, std::mt19937_64& rng, double scale, double bias_mean) {
    const std::uint64_t n = elem_count(t.shape);
    t.bytes.resize(n * core::dtype_size(t.dtype));
    std::normal_distribution<double> dist(0.0, 1.0);
    std::byte* out = t.bytes.data();
    const std::uint32_t es = core::dtype_size(t.dtype);
    for (std::uint64_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(dist(rng) * scale + bias_mean);
        write_float(t.dtype, out + i * es, v);
    }
}

GenTensor make_linear_weight(std::string name, std::uint64_t out_dim, std::uint64_t in_dim,
                             std::mt19937_64& rng) {
    GenTensor t;
    t.name = std::move(name);
    t.shape = {out_dim, in_dim};
    const double scale = std::sqrt(2.0 / static_cast<double>(in_dim));
    fill_normal(t, rng, scale, 0.0);
    return t;
}

GenTensor make_bias(std::string name, std::uint64_t dim, std::mt19937_64& rng) {
    GenTensor t;
    t.name = std::move(name);
    t.shape = {dim};
    fill_normal(t, rng, 0.01, 0.0);  // small, nonzero
    return t;
}

GenTensor make_conv_weight(std::string name, std::uint64_t out_c, std::uint64_t in_c,
                           std::uint64_t k, std::mt19937_64& rng) {
    GenTensor t;
    t.name = std::move(name);
    t.shape = {out_c, in_c, k, k};
    const double scale = std::sqrt(2.0 / static_cast<double>(in_c * k * k));
    fill_normal(t, rng, scale, 0.0);
    return t;
}

/// BatchNorm's four buffers: weight starts near 1 (not 0 — an all-ones
/// vector under permutation is still trivially "correct" either way, but
/// starting near the real BN init keeps this fixture representative), the
/// rest as gen_resnet.py's batchnorm2d().
void make_batchnorm(std::uint32_t idx, std::uint64_t c, std::mt19937_64& rng,
                    std::vector<GenTensor>& weights, std::vector<GenTensor>& others) {
    GenTensor w;
    w.name = std::to_string(idx) + ".weight";
    w.shape = {c};
    fill_normal(w, rng, 0.02, 1.0);
    weights.push_back(std::move(w));

    GenTensor b;
    b.name = std::to_string(idx) + ".bias";
    b.shape = {c};
    fill_normal(b, rng, 0.01, 0.0);
    weights.push_back(std::move(b));

    GenTensor rm;
    rm.name = std::to_string(idx) + ".running_mean";
    rm.shape = {c};
    fill_normal(rm, rng, 0.05, 0.0);
    others.push_back(std::move(rm));

    GenTensor rv;
    rv.name = std::to_string(idx) + ".running_var";
    rv.shape = {c};
    // abs(normal) + 0.5: a running variance is never negative.
    const std::uint64_t n = c;
    rv.bytes.resize(n * core::dtype_size(rv.dtype));
    std::normal_distribution<double> dist(0.0, 1.0);
    for (std::uint64_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(std::abs(dist(rng)) + 0.5);
        write_float(rv.dtype, rv.bytes.data() + i * core::dtype_size(rv.dtype), v);
    }
    others.push_back(std::move(rv));
}

void build_mlp(const SyntheticSpec& spec, std::mt19937_64& rng, std::vector<GenTensor>& weights,
               std::vector<GenTensor>& others) {
    if (spec.layer_widths.size() < 2)
        throw std::invalid_argument("SyntheticSpec.layer_widths needs >= 2 entries for an MLP");
    const std::size_t n_layers = spec.layer_widths.size() - 1;
    for (std::size_t i = 0; i < n_layers; ++i) {
        const std::uint32_t idx = static_cast<std::uint32_t>(2 * i);
        const auto in_dim = spec.layer_widths[i];
        const auto out_dim = spec.layer_widths[i + 1];
        weights.push_back(make_linear_weight(std::to_string(idx) + ".weight", out_dim, in_dim, rng));
        others.push_back(make_bias(std::to_string(idx) + ".bias", out_dim, rng));
    }
}

void build_conv(const SyntheticSpec& spec, std::mt19937_64& rng,
                std::vector<GenTensor>& weights, std::vector<GenTensor>& others) {
    if (spec.layer_widths.size() < 4)
        throw std::invalid_argument(
            "SyntheticSpec.layer_widths needs >= 4 entries (in_c, g0, g1, out) for a conv net");
    constexpr std::uint64_t kSpatial = 8, kKernel = 3;
    const auto in_c = spec.layer_widths[0];
    const auto g0 = spec.layer_widths[1];
    const auto g1 = spec.layer_widths[2];
    const auto out_dim = spec.layer_widths[3];

    weights.push_back(make_conv_weight("0.weight", g0, in_c, kKernel, rng));
    others.push_back(make_bias("0.bias", g0, rng));
    if (spec.with_batchnorm) make_batchnorm(1, g0, rng, weights, others);

    weights.push_back(make_conv_weight("3.weight", g1, g0, kKernel, rng));
    others.push_back(make_bias("3.bias", g1, rng));
    if (spec.with_batchnorm) make_batchnorm(4, g1, rng, weights, others);

    const std::uint64_t pooled = kSpatial / 2;
    const std::uint64_t flat = g1 * pooled * pooled;
    weights.push_back(make_linear_weight("8.weight", out_dim, flat, rng));
    others.push_back(make_bias("8.bias", out_dim, rng));
}

/// Serialise `tensors` (already in final data order) to a real .safetensors
/// byte buffer: [8-byte LE header length][JSON header][concatenated data].
std::vector<std::byte> serialize(const std::vector<const GenTensor*>& order, bool with_metadata,
                                 std::string_view arch) {
    json header;
    if (with_metadata) {
        header["__metadata__"] = {{"format", "synapsefs-fixture"}, {"arch", std::string(arch)}};
    }

    std::uint64_t cursor = 0;
    for (const auto* t : order) {
        json shape_arr = json::array();
        for (auto s : t->shape) shape_arr.push_back(s);
        header[t->name] = {{"dtype", std::string(core::to_string(t->dtype))},
                           {"shape", shape_arr},
                           {"data_offsets", json::array({cursor, cursor + t->bytes.size()})}};
        cursor += t->bytes.size();
    }

    const std::string json_str = header.dump();
    const std::uint64_t json_len = json_str.size();

    std::vector<std::byte> out;
    out.resize(8 + json_len + cursor);
    std::memcpy(out.data(), &json_len, 8);  // host is little-endian (x86_64/AArch64 targets only)
    std::memcpy(out.data() + 8, json_str.data(), json_len);

    std::uint64_t off = 8 + json_len;
    for (const auto* t : order) {
        std::memcpy(out.data() + off, t->bytes.data(), t->bytes.size());
        off += t->bytes.size();
    }
    return out;
}

}  // namespace

std::string write_synthetic_checkpoint(const std::filesystem::path& dest,
                                       const SyntheticSpec& spec) {
    std::mt19937_64 rng(spec.seed);

    // `weights` and `others` (bias / running-stat tensors) are kept as
    // separate lists specifically so the awkward-order path below can
    // interleave them the way a real writer's dict insertion order would
    // (see fixtures/gen_mlp.py / gen_resnet.py's own tensors={} literals).
    std::vector<GenTensor> weights, others;
    if (spec.with_conv) {
        build_conv(spec, rng, weights, others);
    } else {
        build_mlp(spec, rng, weights, others);
    }

    std::vector<const GenTensor*> order;
    order.reserve(weights.size() + others.size());
    if (spec.awkward_header) {
        for (const auto& t : others) order.push_back(&t);
        for (const auto& t : weights) order.push_back(&t);
    } else {
        std::vector<const GenTensor*> all;
        for (const auto& t : weights) all.push_back(&t);
        for (const auto& t : others) all.push_back(&t);
        std::sort(all.begin(), all.end(),
                 [](const GenTensor* a, const GenTensor* b) { return a->name < b->name; });
        order = std::move(all);
    }

    const auto bytes = serialize(order, spec.awkward_header, spec.with_conv ? "resnet" : "mlp");

    std::error_code mkdir_ec;
    std::filesystem::create_directories(dest.parent_path(), mkdir_ec);
    std::ofstream f(dest, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("write_synthetic_checkpoint: cannot open " + dest.string());
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    f.close();

    stio::Sha256Stream sha;
    sha.update(bytes);
    return sha.finish_hex();
}

namespace {

std::vector<std::byte> read_whole_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    const auto size = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(size));
    f.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

/// dst[o*axis_len + i] = src[o*axis_len + expanded_perm[i]] for every outer
/// index o, treating the tensor as row-major over `shape` and moving whole
/// trailing-dimension units at once. Reduces to a plain row-gather when
/// dim==0 (outer==1... no: outer=1 only when dim==0 AND shape has no leading
/// axes, which dim==0 always satisfies) and generalises the same way
/// modules/store/src/commit_planner.cpp's ColumnPermutingSource and
/// modules/codec/src/reconstruct.cpp's resolve_secondary_permutation had to,
/// for a secondary axis that is not the last dimension (e.g. a conv weight's
/// in-channel axis, dim 1 of [out_c, in_c, kh, kw]).
void permute_along_axis(std::byte* dst, const std::byte* src,
                        const std::vector<std::uint64_t>& shape, std::uint32_t dim,
                        const std::vector<std::uint32_t>& expanded_perm,
                        std::uint32_t dtype_size) {
    std::uint64_t outer = 1;
    for (std::uint32_t d = 0; d < dim; ++d) outer *= shape[d];
    const std::uint64_t axis_len = shape[dim];
    std::uint64_t inner = 1;
    for (std::uint32_t d = dim + 1; d < shape.size(); ++d) inner *= shape[d];
    const std::uint64_t unit_bytes = inner * dtype_size;

    for (std::uint64_t o = 0; o < outer; ++o) {
        for (std::uint64_t i = 0; i < axis_len; ++i) {
            const std::uint64_t src_idx = o * axis_len + expanded_perm[i];
            const std::uint64_t dst_idx = o * axis_len + i;
            std::memcpy(dst + dst_idx * unit_bytes, src + src_idx * unit_bytes, unit_bytes);
        }
    }
}

}  // namespace

std::vector<std::uint32_t> plant_permutation(const std::filesystem::path& src,
                                             const std::filesystem::path& dest,
                                             const core::Topology& topology,
                                             std::string_view group, std::uint64_t seed) {
    const auto in_bytes = read_whole_file(src);
    auto hdr = format::parse_st_header(in_bytes);
    if (!hdr) throw std::runtime_error("plant_permutation: " + hdr.error().to_string());

    const auto* pg = topology.find_group(group);
    if (pg == nullptr)
        throw std::invalid_argument("plant_permutation: no such group '" + std::string(group) + "'");

    std::vector<std::uint32_t> perm(pg->size);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937_64(seed));

    // Start as an exact copy — the header (verbatim, per format/st_header.hpp's
    // whole reason for existing) and every tensor NOT bound to `group` are
    // untouched.
    std::vector<std::byte> out_bytes = in_bytes;

    for (const auto& [name, axes] : topology.tensors) {
        auto th = hdr->tensors.find(name);
        if (th == hdr->tensors.end()) continue;  // tensor named in topology but not in this file

        for (const auto& axis : axes.axes) {
            if (axis.group != group) continue;
            const auto& shape = th->second.shape;
            if (axis.dim >= shape.size())
                throw std::runtime_error("plant_permutation: axis.dim out of range for " + name);
            if (shape[axis.dim] != static_cast<std::uint64_t>(pg->size) * axis.block) {
                throw std::runtime_error("plant_permutation: shape[dim] != group.size * block for " +
                                         name);
            }
            const auto expanded = core::expand_permutation(perm, axis.block);
            std::byte* dst_region = out_bytes.data() + th->second.data_off;
            const std::byte* src_region = in_bytes.data() + th->second.data_off;
            permute_along_axis(dst_region, src_region, shape, axis.dim, expanded,
                               core::dtype_size(th->second.dtype));
        }
    }

    std::error_code mkdir_ec;
    std::filesystem::create_directories(dest.parent_path(), mkdir_ec);
    std::ofstream f(dest, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("plant_permutation: cannot open " + dest.string());
    f.write(reinterpret_cast<const char*>(out_bytes.data()),
           static_cast<std::streamsize>(out_bytes.size()));
    return perm;
}

void perturb(const std::filesystem::path& src, const std::filesystem::path& dest, float scale,
            std::uint64_t seed) {
    const auto in_bytes = read_whole_file(src);
    auto hdr = format::parse_st_header(in_bytes);
    if (!hdr) throw std::runtime_error("perturb: " + hdr.error().to_string());

    std::vector<std::byte> out_bytes = in_bytes;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (const auto& [name, t] : hdr->tensors) {
        if (!core::is_float(t.dtype)) continue;  // e.g. num_batches_tracked (I64): unchanged
        const std::uint32_t es = core::dtype_size(t.dtype);
        const std::uint64_t n = t.nbytes / es;
        std::byte* region = out_bytes.data() + t.data_off;
        for (std::uint64_t i = 0; i < n; ++i) {
            const float orig = core::to_float(t.dtype, region + i * es);
            const float noisy = orig + static_cast<float>(dist(rng) * scale);
            write_float(t.dtype, region + i * es, noisy);
        }
    }

    std::error_code mkdir_ec;
    std::filesystem::create_directories(dest.parent_path(), mkdir_ec);
    std::ofstream f(dest, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("perturb: cannot open " + dest.string());
    f.write(reinterpret_cast<const char*>(out_bytes.data()),
           static_cast<std::streamsize>(out_bytes.size()));
}

ByteDiff compare_files(const std::filesystem::path& a, const std::filesystem::path& b) {
    const auto ba = read_whole_file(a);
    const auto bb = read_whole_file(b);

    ByteDiff d;
    d.size_a = ba.size();
    d.size_b = bb.size();
    const std::uint64_t common = std::min(ba.size(), bb.size());
    std::uint64_t i = 0;
    for (; i < common; ++i) {
        if (ba[i] != bb[i]) {
            d.identical = false;
            d.first_difference = i;
            return d;
        }
    }
    if (ba.size() != bb.size()) {
        d.identical = false;
        d.first_difference = common;
    }
    return d;
}

std::string sha256_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("sha256_file: cannot open " + p.string());
    stio::Sha256Stream sha;
    std::array<char, 1 << 16> chunk{};
    while (f) {
        f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto got = f.gcount();
        if (got > 0) {
            sha.update(std::as_bytes(std::span(chunk.data(), static_cast<std::size_t>(got))));
        }
    }
    return sha.finish_hex();
}

}  // namespace sfs::test
