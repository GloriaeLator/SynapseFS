/// bench/residual_codec.cpp — docs/benchmarks.md §2.
///
/// Two independent things measured here:
///   1. --pair <a>,<b> [--topology T --permutation P]: residual ratio for one
///      checkpoint pair, and the six-candidate {xor,zigzag} x
///      {none,byteplane,bitshuffle} experiment ADR 0005 asks for, on the SAME
///      bytes. With no --pair, auto-discovers whatever fixture pairs exist
///      under --fixtures-dir (default fixtures/out) so `residual_codec --json`
///      alone (bench/scripts/run_all.sh's calling convention) still produces
///      something.
///   2. --kernel-only: raw GB/s of the dispatched XOR/zigzag kernels,
///      independent of zstd or file I/O, honouring SFS_FORCE_ISA.
///
/// Output matches what bench/scripts/residual_ratio.py already expects: a
/// top-level "candidates" array of {residual, transform, ratio,
/// decompress_mb_s, compress_mb_s}.
///
/// align/ has no implementation yet (see branch notes), so this reads the
/// permutation as plain data from fixtures/permute.py's *_topology.json and
/// *_permuted.permutation.json sidecars rather than calling into it — the
/// same "precomputed/synthetic permutation" approach used throughout this
/// branch.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/codec/compress.hpp>
#include <synapsefs/codec/residual_codec.hpp>
#include <synapsefs/core/dtype.hpp>
#include <synapsefs/format/st_header.hpp>
#include <synapsefs/util/cpuid.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace sfs;

namespace {

// ---------------------------------------------------------------- file I/O

std::vector<std::byte> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    f.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::byte> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
    return buf;
}

json read_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    json j;
    f >> j;
    return j;
}

struct Checkpoint {
    std::vector<std::byte> bytes;
    format::StHeader header;

    [[nodiscard]] std::span<const std::byte> tensor(const std::string& name) const {
        const auto it = header.tensors.find(name);
        if (it == header.tensors.end()) return {};
        return std::span<const std::byte>(bytes).subspan(it->second.data_off, it->second.nbytes);
    }
};

Checkpoint load_checkpoint(const fs::path& p) {
    Checkpoint c;
    c.bytes = read_file(p);
    auto h = format::parse_st_header(c.bytes);
    if (!h) throw std::runtime_error("parsing " + p.string() + ": " + h.error().to_string());
    c.header = std::move(*h);
    return c;
}

// ---------------------------------------------------- permutation (offline)

// expand(p, block)[i*block+k] = p[i]*block + k. spec 13 §1 / spec 12 §2. Same
// formula as codec::permute::expand and fixtures/permute.py's
// expand_permutation — not calling into codec/ here since this predates
// permute.cpp in the roadmap and the logic is three lines.
std::vector<std::uint32_t> expand_permutation(const std::vector<std::uint32_t>& perm,
                                              std::uint32_t block) {
    if (block == 1) return perm;
    std::vector<std::uint32_t> out(perm.size() * block);
    for (std::size_t i = 0; i < perm.size(); ++i)
        for (std::uint32_t k = 0; k < block; ++k) out[i * block + k] = perm[i] * block + k;
    return out;
}

// Product of every dimension of `shape` AFTER `dim` — the size, in elements,
// of the block that moves together with each index along `dim`. 1 when dim
// is the last axis (every 1-D/2-D case this file originally shipped with);
// kh*kw for a conv weight's in-channel axis (dim 1 of [out_c, in_c, kh, kw]),
// which is NOT the last dimension. Same fix as commit_planner.cpp's
// ColumnPermutingSource and reconstruct.cpp's resolve_secondary_permutation
// — this bench tool has its own, independent copy of the same gather logic
// and needed the identical correction once a rank > 2 fixture existed to
// expose it.
std::uint64_t trailing_elems(const std::vector<std::uint64_t>& shape, std::uint32_t dim) {
    std::uint64_t t = 1;
    for (std::size_t d = static_cast<std::size_t>(dim) + 1; d < shape.size(); ++d) t *= shape[d];
    return t;
}

// Gather along axis `dim` of an (outer x perm_axis_len x trailing) row-major
// tensor: out[...perm applied to the perm_axis_len-sized axis `dim`...] =
// src[...], each gathered "unit" being `trailing` elements wide (1 for the
// 1-D/2-D case, kh*kw for a conv weight's non-last axis). `rows`/`cols` name
// the two axes this fixture set's tensors actually have: rows = shape[0],
// cols = shape[1] when dim==1 (irrelevant when dim==0).
std::vector<std::byte> gather_axis(std::span<const std::byte> src, std::uint64_t rows,
                                   std::uint64_t cols, std::uint64_t trailing,
                                   std::uint32_t elem_bytes, std::uint32_t dim,
                                   const std::vector<std::uint32_t>& perm) {
    const std::uint64_t unit_bytes = trailing * elem_bytes;
    std::vector<std::byte> out(src.size());
    if (dim == 0) {
        // trailing_elems(shape, 0) is ALREADY product(shape[1:]) -- the
        // whole per-row size (cols included, for rank > 2) -- so it IS
        // row_bytes directly. Multiplying by `cols` again here would
        // double-count cols for anything above rank 2 (harmless at rank 2,
        // where trailing == cols already).
        const std::uint64_t row_bytes = unit_bytes;
        for (std::uint64_t i = 0; i < rows; ++i)
            std::memcpy(out.data() + i * row_bytes,
                       src.data() + std::uint64_t{perm[i]} * row_bytes, row_bytes);
    } else {
        for (std::uint64_t i = 0; i < rows; ++i)
            for (std::uint64_t j = 0; j < cols; ++j)
                std::memcpy(out.data() + (i * cols + j) * unit_bytes,
                           src.data() + (i * cols + perm[j]) * unit_bytes, unit_bytes);
    }
    return out;
}

struct Sidecar {
    json topology;     // fixtures/*_topology.json
    json permutation;  // fixtures/*_permuted.permutation.json ({"groups": {name: [perm]}})
};

// base bytes for `name`, gathered into TARGET unit order using the topology's
// axis->group bindings and the permutation sidecar's per-group arrays.
// Identity (raw base bytes) when no sidecar is given, or for axes bound to a
// pinned group (absent from the sidecar's "groups", by construction —
// fixtures/permute.py only emits non-pinned groups).
std::vector<std::byte> align_to_target(const std::string& name, std::span<const std::byte> base,
                                       const std::vector<std::uint64_t>& shape,
                                       std::uint32_t elem_bytes, const Sidecar* sc) {
    std::vector<std::byte> cur(base.begin(), base.end());
    if (sc == nullptr) return cur;
    const auto tit = sc->topology["tensors"].find(name);
    if (tit == sc->topology["tensors"].end()) return cur;

    const std::uint64_t rows = shape.empty() ? 1 : shape[0];
    const std::uint64_t cols = shape.size() >= 2 ? shape[1] : 1;

    for (const auto& axis : tit.value()["axes"]) {
        const auto group = axis.at("group").get<std::string>();
        const auto git = sc->permutation["groups"].find(group);
        if (git == sc->permutation["groups"].end()) continue;  // pinned: identity
        const auto perm = git.value().get<std::vector<std::uint32_t>>();
        const auto block = axis.value("block", 1u);
        const auto expanded = expand_permutation(perm, block);
        const auto dim = axis.at("dim").get<std::uint32_t>();
        const auto trailing = trailing_elems(shape, dim);
        cur = gather_axis(cur, rows, cols, trailing, elem_bytes, dim, expanded);
    }
    return cur;
}

// ------------------------------------------------ zigzag encode (write side)
//
// Not exposed by residual_codec.hpp: only *_apply is ISA-dispatched, because
// only the read path is the hot loop ADR 0011 cares about (encode happens
// once per commit). This mirrors residual_scalar.cpp's decode formula in
// reverse — the standard zigzag bit trick, generalised from a fixed 64-bit
// width to whatever the dtype's element width is.

template <typename U>
void zigzag_encode_width(std::byte* dst, const std::byte* base, const std::byte* target,
                         std::size_t count) noexcept {
    constexpr U kTop = U(1) << (sizeof(U) * 8 - 1);
    for (std::size_t i = 0; i < count; ++i) {
        U b = 0, t = 0;
        std::memcpy(&b, base + i * sizeof(U), sizeof(U));
        std::memcpy(&t, target + i * sizeof(U), sizeof(U));
        const U delta_bits = static_cast<U>(t - b);  // wraparound: two's-complement bits of (t - b)
        const U sign_mask = (delta_bits & kTop) ? static_cast<U>(~U(0)) : U(0);
        const U z = static_cast<U>((delta_bits << 1) ^ sign_mask);
        std::memcpy(dst + i * sizeof(U), &z, sizeof(U));
    }
}

void zigzag_encode(std::byte* dst, const std::byte* base, const std::byte* target, std::size_t n,
                   std::uint32_t elem_bytes) noexcept {
    switch (elem_bytes) {
        case 1: zigzag_encode_width<std::uint8_t>(dst, base, target, n / 1); return;
        case 2: zigzag_encode_width<std::uint16_t>(dst, base, target, n / 2); return;
        case 4: zigzag_encode_width<std::uint32_t>(dst, base, target, n / 4); return;
        case 8: zigzag_encode_width<std::uint64_t>(dst, base, target, n / 8); return;
        default: return;
    }
}

// -------------------------------------------------------------- candidates

struct Candidate {
    std::string pair, residual, transform;
    double ratio = 0, compress_mb_s = 0, decompress_mb_s = 0;
};

std::vector<Candidate> six_candidates(const std::string& pair_label,
                                      std::span<const std::byte> base,
                                      std::span<const std::byte> target,
                                      std::uint32_t elem_bytes) {
    std::vector<std::byte> xor_resid(target.size()), zz_resid(target.size());
    codec::xor_encode_dispatch()(xor_resid.data(), base.data(), target.data(), target.size());
    zigzag_encode(zz_resid.data(), base.data(), target.data(), target.size(), elem_bytes);

    const std::pair<const char*, std::span<const std::byte>> residuals[] = {
        {"xor_after_permute", xor_resid},
        {"zigzag_after_permute", zz_resid},
    };
    const char* transforms[] = {"none", "byteplane", "bitshuffle"};

    std::vector<Candidate> out;
    for (const auto& [rname, rbytes] : residuals) {
        for (const char* tname : transforms) {
            std::vector<std::byte> transformed(rbytes.size());
            if (std::string_view(tname) == "none")
                std::memcpy(transformed.data(), rbytes.data(), rbytes.size());
            else if (std::string_view(tname) == "byteplane")
                codec::byteplane_split(rbytes, transformed, elem_bytes);
            else
                codec::bitshuffle(rbytes, transformed, elem_bytes);

            const auto t0 = std::chrono::steady_clock::now();
            auto comp = codec::compress_frame(transformed);
            const auto t1 = std::chrono::steady_clock::now();
            if (!comp) {
                std::cerr << "compress_frame failed for " << rname << "/" << tname << ": "
                         << comp.error().to_string() << "\n";
                continue;
            }

            constexpr int kReps = 20;
            std::vector<std::byte> decompressed(transformed.size());
            const auto t2 = std::chrono::steady_clock::now();
            for (int r = 0; r < kReps; ++r) {
                auto n = codec::decompress_frame(*comp, decompressed);
                if (!n) { std::cerr << "decompress_frame failed\n"; break; }
            }
            const auto t3 = std::chrono::steady_clock::now();

            const double compress_s = std::chrono::duration<double>(t1 - t0).count();
            const double decompress_s = std::chrono::duration<double>(t3 - t2).count() / kReps;

            Candidate c;
            c.pair = pair_label;
            c.residual = rname;
            c.transform = tname;
            c.ratio = static_cast<double>(comp->size()) / static_cast<double>(target.size());
            c.compress_mb_s = compress_s > 0 ? (target.size() / 1e6) / compress_s : 0.0;
            c.decompress_mb_s = decompress_s > 0 ? (target.size() / 1e6) / decompress_s : 0.0;
            out.push_back(c);
        }
    }
    return out;
}

// ------------------------------------------------------------- pair driver

struct PairSpec {
    std::string label, base_path, target_path;
    std::optional<std::string> topology_path, permutation_path;
};

void run_pair(const PairSpec& spec, std::vector<Candidate>& all_candidates, json& pairs_out) {
    Checkpoint base = load_checkpoint(spec.base_path);
    Checkpoint target = load_checkpoint(spec.target_path);

    std::optional<Sidecar> sc_storage;
    const Sidecar* sc = nullptr;
    if (spec.topology_path && spec.permutation_path) {
        sc_storage = Sidecar{read_json(*spec.topology_path), read_json(*spec.permutation_path)};
        sc = &*sc_storage;
    }

    std::vector<std::byte> all_base, all_target;
    std::uint32_t elem_bytes = 2;  // every tensor in this fixture set is F16
    std::size_t tensor_count = 0;

    for (const auto& [name, meta] : target.header.tensors) {
        const auto tit = base.header.tensors.find(name);
        if (tit == base.header.tensors.end()) continue;  // only in one checkpoint: skip
        const auto tbytes = target.tensor(name);
        const auto bbytes_raw = base.tensor(name);
        elem_bytes = core::dtype_size(meta.dtype);

        auto bbytes = align_to_target(name, bbytes_raw, meta.shape, elem_bytes, sc);
        if (bbytes.size() != tbytes.size()) continue;  // shape mismatch: not comparable

        all_base.insert(all_base.end(), bbytes.begin(), bbytes.end());
        all_target.insert(all_target.end(), tbytes.begin(), tbytes.end());
        ++tensor_count;
    }

    if (all_target.empty()) {
        std::cerr << "pair " << spec.label << ": no comparable tensors, skipping\n";
        return;
    }

    // Baseline per spec 12 §5's "how we would know we were wrong": plain
    // zstd of the target, no base/residual involved at all. If a candidate
    // doesn't beat this, alignment isn't helping for this pair.
    auto baseline = codec::compress_frame(all_target);
    const double baseline_ratio =
        baseline ? static_cast<double>(baseline->size()) / static_cast<double>(all_target.size())
                 : -1.0;

    auto candidates = six_candidates(spec.label, all_base, all_target, elem_bytes);
    all_candidates.insert(all_candidates.end(), candidates.begin(), candidates.end());

    pairs_out.push_back({
        {"label", spec.label},
        {"base", spec.base_path},
        {"target", spec.target_path},
        {"tensors_compared", tensor_count},
        {"naive_bytes", all_target.size()},
        {"baseline_zstd_ratio", baseline_ratio},
    });
}

// -------------------------------------------------------------- discovery

std::vector<PairSpec> discover_pairs(const fs::path& dir) {
    std::vector<PairSpec> out;
    if (!fs::exists(dir)) return out;

    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        const std::string suffix = "_step0.safetensors";
        if (name.size() <= suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string prefix = name.substr(0, name.size() - suffix.size());

        const auto step0 = dir / (prefix + "_step0.safetensors");
        const auto step1 = dir / (prefix + "_step1.safetensors");
        const auto permuted = dir / (prefix + "_permuted.safetensors");
        const auto topology = dir / (prefix + "_topology.json");
        const auto permutation = dir / (prefix + "_permuted.permutation.json");

        if (fs::exists(step1))
            out.push_back({prefix + "_finetune", step0.string(), step1.string(), std::nullopt,
                          std::nullopt});
        if (fs::exists(permuted) && fs::exists(topology) && fs::exists(permutation))
            out.push_back({prefix + "_permuted_only", step0.string(), permuted.string(),
                          topology.string(), permutation.string()});
    }
    return out;
}

// ------------------------------------------------------------ kernel-only

void run_kernel_only(std::size_t bytes) {
    std::mt19937_64 rng(42);
    std::vector<std::byte> base(bytes), resid(bytes), out(bytes);
    for (auto& b : base) b = std::byte(rng());
    for (auto& b : resid) b = std::byte(rng());

    const auto isa = codec::active_isa();
    std::cout << "isa: " << util::isa_name(isa) << "\n";

    constexpr int kReps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kReps; ++r)
        codec::xor_apply_dispatch()(out.data(), base.data(), resid.data(), bytes);
    auto t1 = std::chrono::steady_clock::now();
    const double xor_s = std::chrono::duration<double>(t1 - t0).count() / kReps;
    const double xor_gbps = (bytes / 1e9) / xor_s;

    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < kReps; ++r)
        codec::zigzag_apply_dispatch()(out.data(), base.data(), resid.data(), bytes, 2);
    auto t3 = std::chrono::steady_clock::now();
    const double zz_s = std::chrono::duration<double>(t3 - t2).count() / kReps;
    const double zz_gbps = (bytes / 1e9) / zz_s;

    std::cout << "xor_apply:    " << xor_gbps << " GB/s\n";
    std::cout << "zigzag_apply: " << zz_gbps << " GB/s\n";

    json j = {{"isa", std::string(util::isa_name(isa))},
             {"xor_apply_gbps", xor_gbps},
             {"zigzag_apply_gbps", zz_gbps},
             {"bytes", bytes}};
    std::cout << j.dump(2) << "\n";
}

// ------------------------------------------------------------------ CLI

struct Args {
    bool json = false;
    bool kernel_only = false;
    std::size_t bytes = 64ull << 20;
    std::optional<std::string> pair_a, pair_b, topology, permutation;
    std::string fixtures_dir = "fixtures/out";
};

std::optional<std::string> take_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) return std::nullopt;
    return std::string(argv[++i]);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") a.json = true;
        else if (arg == "--kernel-only") a.kernel_only = true;
        else if (arg == "--bytes") { if (auto v = take_value(argc, argv, i)) a.bytes = std::stoull(*v); }
        else if (arg == "--fixtures-dir") { if (auto v = take_value(argc, argv, i)) a.fixtures_dir = *v; }
        else if (arg == "--topology") a.topology = take_value(argc, argv, i);
        else if (arg == "--permutation") a.permutation = take_value(argc, argv, i);
        else if (arg == "--pair") {
            if (auto v = take_value(argc, argv, i)) {
                const auto comma = v->find(',');
                if (comma == std::string::npos) {
                    std::cerr << "--pair expects a,b\n";
                    std::exit(2);
                }
                a.pair_a = v->substr(0, comma);
                a.pair_b = v->substr(comma + 1);
            }
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    if (args.kernel_only) {
        run_kernel_only(args.bytes);
        return 0;
    }

    std::vector<PairSpec> specs;
    if (args.pair_a && args.pair_b) {
        specs.push_back({"pair", *args.pair_a, *args.pair_b, args.topology, args.permutation});
    } else {
        specs = discover_pairs(args.fixtures_dir);
        if (specs.empty()) {
            std::cerr << "no --pair given and no fixtures found under " << args.fixtures_dir
                     << " (run fixtures/gen_mlp.py and fixtures/permute.py first)\n";
            return 1;
        }
    }

    std::vector<Candidate> all_candidates;
    json pairs_out = json::array();
    for (const auto& spec : specs) {
        try {
            run_pair(spec, all_candidates, pairs_out);
        } catch (const std::exception& e) {
            std::cerr << "pair " << spec.label << " failed: " << e.what() << "\n";
        }
    }

    if (args.json) {
        json candidates_json = json::array();
        for (const auto& c : all_candidates) {
            candidates_json.push_back({{"pair", c.pair},
                                       {"residual", c.residual},
                                       {"transform", c.transform},
                                       {"ratio", c.ratio},
                                       {"compress_mb_s", c.compress_mb_s},
                                       {"decompress_mb_s", c.decompress_mb_s}});
        }
        json out = {{"isa", std::string(util::isa_name(codec::active_isa()))},
                   {"pairs", pairs_out},
                   {"candidates", candidates_json}};
        std::cout << out.dump(2) << "\n";
    } else {
        std::cout << "| Pair | Residual | Transform | Ratio | Compress MB/s | Decompress MB/s |\n";
        std::cout << "|---|---|---|---|---|---|\n";
        for (const auto& c : all_candidates) {
            std::cout << "| " << c.pair << " | `" << c.residual << "` | " << c.transform << " | "
                     << c.ratio << " | " << c.compress_mb_s << " | " << c.decompress_mb_s
                     << " |\n";
        }
    }
    return 0;
}
