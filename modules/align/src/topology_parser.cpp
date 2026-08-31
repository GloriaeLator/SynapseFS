#include <synapsefs/align/topology_parser.hpp>

#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <synapsefs/align/graph.hpp>

namespace sfs::align {

using json = nlohmann::json;

namespace {

// Every tensor the config's layer list actually names, so the post-pass can
// find what it DIDN'T touch and give those a singleton pinned group (SPEC 13
// §3's coverage rule -- num_batches_tracked and friends round-trip through
// the identical code path as everything else, with no special case in the
// reconstructor).
struct BuildState {
    AxisUnionFind uf;
    std::unordered_map<std::string, std::vector<std::uint32_t>> axis_handle;  // tensor -> per-dim handle, only for registered dims
    std::unordered_set<std::string> touched;
    std::vector<std::string> derived_blocks;  // diagnostic lines, e.g. "9.weight dim=1 block=64"
};

std::uint32_t register_axis(BuildState& st, const std::string& tensor, std::uint32_t dim,
                            std::uint64_t len) {
    auto& handles = st.axis_handle[tensor];
    if (handles.size() <= dim) handles.resize(dim + 1, std::numeric_limits<std::uint32_t>::max());
    if (handles[dim] == std::numeric_limits<std::uint32_t>::max()) {
        handles[dim] = st.uf.add(AxisKey{tensor, dim}, len);
    }
    st.touched.insert(tensor);
    return handles[dim];
}

std::pair<std::int64_t, std::int64_t> as_hw(const json& v) {
    if (v.is_array()) return {v.at(0).get<std::int64_t>(), v.at(1).get<std::int64_t>()};
    return {v.get<std::int64_t>(), v.get<std::int64_t>()};
}

std::pair<std::int64_t, std::int64_t> conv2d_output_hw(std::int64_t h, std::int64_t w, const json& layer) {
    auto [kh, kw] = as_hw(layer.at("kernel_size"));
    auto [sh, sw] = layer.contains("stride") ? as_hw(layer.at("stride")) : std::pair<std::int64_t, std::int64_t>{1, 1};
    auto [ph, pw] = layer.contains("padding") ? as_hw(layer.at("padding")) : std::pair<std::int64_t, std::int64_t>{0, 0};
    return {(h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1};
}

std::pair<std::int64_t, std::int64_t> maxpool2d_output_hw(std::int64_t h, std::int64_t w, const json& layer) {
    auto [kh, kw] = as_hw(layer.at("kernel_size"));
    auto [sh, sw] = layer.contains("stride") ? as_hw(layer.at("stride")) : std::pair<std::int64_t, std::int64_t>{kh, kw};
    return {(h - kh) / sh + 1, (w - kw) / sw + 1};
}

core::Result<const core::TensorMeta*> require_meta(const core::ITensorSource& src, const std::string& name) {
    const core::TensorMeta* m = src.meta(name);
    if (m == nullptr) {
        return SFS_ERR(TopologyIncomplete, "config.json references a tensor not in the checkpoint", name);
    }
    return m;
}

/// Walks the "layers" sequential list (SPEC 13's supported shape: linear,
/// conv2d, batchnorm2d, relu, maxpool2d, flatten, dropout -- a plain chain,
/// no skip connections, which is what both sample configs are). Tensors are
/// named by list index, matching nn.Sequential's own state_dict convention.
core::Status walk_layers(const core::ITensorSource& src, const json& layers,
                         const std::optional<std::vector<std::int64_t>>& input_shape,
                         const ParseOptions& opts, BuildState& st) {
    std::vector<std::size_t> param_indices;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string t = layers[i].value("type", "");
        if (t == "linear" || t == "conv2d") param_indices.push_back(i);
    }
    if (param_indices.empty()) {
        return SFS_ERR(TopologyParse, "config has no parameterized (linear/conv2d) layers", "");
    }
    const std::size_t last_param_idx = param_indices.back();

    std::optional<std::uint32_t> current_axis;  // running "output of the previous layer" handle
    std::vector<std::int64_t> spatial;          // (C,H,W) while convolutional, (F,) once flat, empty if unknown
    if (input_shape.has_value()) spatial = *input_shape;

    for (std::size_t idx = 0; idx < layers.size(); ++idx) {
        const json& layer = layers[idx];
        const std::string t = layer.value("type", "");
        const std::string prefix = std::to_string(idx);
        if (t.empty()) {
            return SFS_ERR(TopologyParse, "layer missing 'type'", prefix);
        }

        if (t == "linear" || t == "conv2d") {
            const std::string weight = prefix + ".weight";
            const core::TensorMeta* wmeta = SFS_TRY(require_meta(src, weight));
            if (wmeta->shape.size() < 2) {
                return SFS_ERR(TopologyParse, "linear/conv2d weight must have rank >= 2", weight);
            }

            const std::uint32_t out_handle = register_axis(st, weight, 0, wmeta->shape[0]);
            const std::uint32_t in_handle = register_axis(st, weight, 1, wmeta->shape[1]);

            if (current_axis.has_value()) {
                SFS_TRY_VOID(st.uf.unite(in_handle, *current_axis));
            } else if (opts.strict) {
                st.uf.pin(in_handle);  // first parameterized layer: data features, fixed order
            }

            const std::string bias = prefix + ".bias";
            if (src.meta(bias) != nullptr) {
                const core::TensorMeta* bmeta = src.meta(bias);
                const std::uint32_t bias_handle = register_axis(st, bias, 0, bmeta->shape.at(0));
                SFS_TRY_VOID(st.uf.unite(bias_handle, out_handle));  // Relation::Bias
            }

            if (idx == last_param_idx) {
                st.uf.pin(out_handle);  // classifier output: class identity is fixed
            }
            current_axis = out_handle;

            if (t == "linear") {
                spatial = {static_cast<std::int64_t>(wmeta->shape[0])};
            } else {
                if (spatial.size() == 3) {
                    auto [h2, w2] = conv2d_output_hw(spatial[1], spatial[2], layer);
                    spatial = {static_cast<std::int64_t>(wmeta->shape[0]), h2, w2};
                } else {
                    spatial.clear();
                }
            }

        } else if (t == "batchnorm2d" || t == "layernorm") {
            // layernorm shares batchnorm2d's handling: same
            // prefix.weight/prefix.bias affine-term naming, unioned into the
            // preceding layer's output group the same way (Relation::
            // NormFollows). It never has running_mean/running_var -- the
            // per-suffix meta() check below already tolerates a missing
            // tensor (originally for exporters that omit BN's running
            // stats), so those two iterations simply no-op here rather than
            // needing a separate branch.
            if (!current_axis.has_value()) {
                return SFS_ERR(TopologyParse, t + " with no preceding conv2d/linear", prefix);
            }
            for (const char* suffix : {".weight", ".bias", ".running_mean", ".running_var"}) {
                const std::string name = prefix + suffix;
                if (src.meta(name) == nullptr) continue;  // some exporters omit running stats; layernorm always does
                const std::uint64_t len = src.meta(name)->shape.at(0);
                const std::uint32_t h = register_axis(st, name, 0, len);
                SFS_TRY_VOID(st.uf.unite(h, *current_axis));  // Relation::NormFollows
            }

        } else if (t == "maxpool2d") {
            if (spatial.size() == 3) {
                auto [h2, w2] = maxpool2d_output_hw(spatial[1], spatial[2], layer);
                spatial = {spatial[0], h2, w2};
            }

        } else if (t == "flatten") {
            // No explicit expansion tracking needed here (unlike a design that
            // carries a running block factor by hand): uniting the conv's
            // channel axis directly with the linear's wider input axis is
            // enough, because AxisUnionFind::finalize derives block = len /
            // group_size from the raw axis lengths it already recorded
            // (Relation::Flatten). The union above, at the next linear layer,
            // is where that actually happens; this branch only needs to stop
            // tracking a spatial (C,H,W) shape once flatten has occurred.
            if (spatial.size() == 3) spatial = {spatial[0] * spatial[1] * spatial[2]};

        } else if (t == "relu" || t == "dropout" || t == "avgpool2d" || t == "adaptiveavgpool2d") {
            // no parameters, no shape/permutation effect

        } else if (opts.strict) {
            return SFS_ERR(TopologyParse, "unsupported layer type", prefix + ": " + t);
        }
    }
    return {};
}

}  // namespace

core::Result<core::Topology> parse_topology(const core::ITensorSource& src,
                                            std::span<const std::byte> config_json,
                                            const ParseOptions& opts, ParseDiagnostics* diag) {
    BuildState st;

    if (!config_json.empty()) {
        json cfg;
        try {
            cfg = json::parse(std::string(reinterpret_cast<const char*>(config_json.data()), config_json.size()));
        } catch (const json::parse_error& e) {
            return SFS_ERR(TopologyParse, std::string("config.json parse error: ") + e.what(), "");
        }
        if (!cfg.contains("layers")) {
            return SFS_ERR(TopologyParse, "config.json has no 'layers' array", "");
        }
        std::optional<std::vector<std::int64_t>> input_shape;
        if (cfg.contains("input_shape")) input_shape = cfg.at("input_shape").get<std::vector<std::int64_t>>();
        SFS_TRY_VOID(walk_layers(src, cfg.at("layers"), input_shape, opts, st));
    }
    // Empty config: nothing is unioned. Every tensor falls through to the
    // singleton-pinned pass below, which is CORRECT (nothing is claimed to be
    // alignable without evidence) even though it is not USEFUL -- SPEC 13
    // §4.1 scopes structural inference from names alone as a possible future
    // extension, not something this parser guesses at today.

    // Coverage rule (SPEC 13 §3): every tensor in the file gets a group,
    // including ones the config never named. Give each untouched tensor's
    // own axis 0 a fresh, pinned, singleton group.
    for (const auto& entry : src.buffer_layout()) {
        if (st.touched.contains(entry.tensor)) continue;
        const core::TensorMeta* m = src.meta(entry.tensor);
        if (m == nullptr || m->shape.empty()) continue;
        const std::uint32_t h = register_axis(st, entry.tensor, 0, m->shape[0]);
        st.uf.pin(h);
        if (diag != nullptr) diag->unmodelled_tensors.push_back(entry.tensor);
    }

    core::Topology topo = SFS_TRY(st.uf.finalize());

    std::unordered_map<std::string, std::vector<std::uint64_t>> shapes;
    for (const auto& entry : src.buffer_layout()) {
        if (const core::TensorMeta* m = src.meta(entry.tensor)) shapes[entry.tensor] = m->shape;
    }
    SFS_TRY_VOID(topo.validate(shapes));

    if (diag != nullptr) {
        diag->group_count = static_cast<std::uint32_t>(topo.groups.size());
        for (const auto& [tensor, axes] : topo.tensors) {
            for (const auto& b : axes.axes) {
                if (b.block > 1) {
                    diag->derived_blocks.push_back(tensor + " dim=" + std::to_string(b.dim) +
                                                   " block=" + std::to_string(b.block));
                }
            }
        }
    }
    return topo;
}

core::Result<core::Topology> parse_topology_file(const core::ITensorSource& src,
                                                 const std::filesystem::path& config,
                                                 const ParseOptions& opts) {
    std::ifstream in(config, std::ios::binary);
    if (!in) {
        return SFS_ERR(NoSuchFile, "cannot open config file", config.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    return parse_topology(src, std::as_bytes(std::span(text.data(), text.size())), opts);
}

}  // namespace sfs::align
