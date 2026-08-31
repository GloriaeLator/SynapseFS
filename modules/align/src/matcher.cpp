#include <synapsefs/align/matcher.hpp>

#include <chrono>
#include <cmath>
#include <vector>

#include <synapsefs/align/lap.hpp>
#include <synapsefs/align/norm_fold.hpp>
#include <synapsefs/align/propagate.hpp>
#include <synapsefs/core/topology.hpp>

#include "detail_read_rows.hpp"
#include "detail_reorder.hpp"
#include "sparse_match.hpp"

namespace sfs::align {

using core::Result;
using detail::dim0_binding;
using detail::read_rows_as_float;
using detail::reorder_if_solved;

namespace {

/// Feature matrix for `group`'s `n` units, gathered from `src` (row-major
/// [unit][feature]) plus its width, before per-unit L2 normalisation.
struct FeatureBlock {
    std::vector<float> data;
    std::uint32_t width = 0;
};

Result<FeatureBlock> build_features(core::ITensorSource& src, const core::Topology& topo,
                                    std::string_view group, std::uint32_t n, const PermutationMap& solved,
                                    const CostOptions& opts) {
    // Evidence gathering. `outgoing`: this tensor's OWN dim-0 is `group`, so
    // reading its rows directly gives one feature segment per unit.
    // `incoming`: `group` feeds some OTHER tensor's non-zero axis; since
    // ITensorSource only reads along dim 0, that tensor's full row set is
    // read once and the columns belonging to this group's units are sliced
    // out in memory. Both are bounded by GROUP size, not parameter count
    // (docs/adr/0008), so this is the same "n is small, D can be huge but is
    // read in the row direction only" shape as the outgoing case.
    struct Segment {
        std::vector<float> values;  // [unit][seg_width], row-major
        std::uint32_t width = 0;
    };
    std::vector<Segment> segments;

    for (const auto& [tensor, axes] : topo.tensors) {
        for (const auto& b : axes.axes) {
            if (b.group != group) continue;
            const core::TensorMeta* meta = src.meta(tensor);
            if (meta == nullptr) return SFS_ERR(Internal, "topology tensor missing from source", tensor);

            if (b.dim == 0) {
                // shape[0] == n * b.block (Topology::validate already checked
                // this); a group unit spans b.block consecutive raw rows.
                const std::uint64_t row_width =
                    meta->shape.size() > 1 ? meta->elem_count() / meta->shape[0] : 1;
                auto raw = SFS_TRY(
                    read_rows_as_float(src, tensor, 0, static_cast<std::uint64_t>(n) * b.block, row_width,
                                       meta->dtype));

                // A row's own content spans this tensor's OTHER axis (e.g. a
                // linear layer's input columns), which belongs to whatever
                // group feeds it -- not necessarily pinned. If that group has
                // already been solved this run, its columns must be brought
                // into a consistent frame before target's and base's rows are
                // compared, for the same reason incoming evidence needs it
                // (propagate.hpp). Missed on the first pass here: it only
                // showed up once a fixture had two non-identity hidden layers
                // in a row (test_known_permutation.cpp's single hidden layer
                // has an identity/pinned input axis, so this bug was invisible
                // there; caught via modules/align/tools/align_demo.cpp against
                // real two-hidden-layer fixtures).
                reorder_if_solved(raw, axes, row_width, solved);

                segments.push_back(Segment{std::move(raw), static_cast<std::uint32_t>(b.block * row_width)});
            } else if (opts.include_incoming) {
                const core::AxisBinding* owner = dim0_binding(axes);
                if (owner == nullptr) continue;
                const std::uint64_t other_rows = meta->shape[0];
                const std::uint64_t row_elems = meta->elem_count() / other_rows;
                auto full = SFS_TRY(read_rows_as_float(src, tensor, 0, other_rows, row_elems, meta->dtype));

                // A group's cost depends on its neighbours' permutations
                // (propagate.hpp): once `owner->group` has a solved
                // permutation, this side's row order must be realigned to
                // that reference frame before the columns below are sliced,
                // or every sweep after the first compares stale positions and
                // "convergence" is a no-op. Only applied when a caller passes
                // a non-empty `solved` (the target side, mid-sweep); the base
                // side stays in its own canonical order as the fixed frame.
                if (auto it = solved.find(owner->group); it != solved.end() && !it->second.empty()) {
                    // `solved[owner->group][i] = j` means target row i takes
                    // base row j's place (lap.cpp: assignment[target_row] =
                    // base_col from a cost matrix indexed [target][base]) --
                    // a forward/scatter map, same convention expand_permutation
                    // uses. To pull, for each destination row r, the target
                    // row that lands there, gather with its inverse.
                    const std::vector<std::uint32_t> full_perm =
                        core::expand_permutation(it->second, owner->block);
                    const std::vector<std::uint32_t> gather = core::invert_permutation(full_perm);
                    std::vector<float> reordered(full.size());
                    for (std::uint64_t r = 0; r < other_rows; ++r) {
                        for (std::uint64_t k = 0; k < row_elems; ++k) {
                            reordered[r * row_elems + k] = full[gather[r] * row_elems + k];
                        }
                    }
                    full = std::move(reordered);
                }

                // shape[b.dim] == n * b.block (validated); extract the
                // b.block-wide run of RAW positions for each of our n units
                // from every other-row. That is NOT the same as a b.block-wide
                // column slab: b.dim need not be the tensor's last dim (a
                // conv2d weight's in_channels axis, dim=1, is followed by
                // kh/kw), so each raw position along b.dim owns a `trailing`-
                // wide contiguous run within the row -- the product of every
                // dim after b.dim -- not a single scalar. For a rank-2 tensor
                // (a linear layer's weight) b.dim is the last dim, trailing
                // == 1, and this reduces to the original per-scalar slice.
                std::uint64_t trailing = 1;
                for (std::size_t d = static_cast<std::size_t>(b.dim) + 1; d < meta->shape.size(); ++d) {
                    trailing *= meta->shape[d];
                }

                Segment seg;
                seg.width = static_cast<std::uint32_t>(other_rows * b.block * trailing);
                seg.values.assign(static_cast<std::size_t>(n) * seg.width, 0.0F);
                for (std::uint64_t unit = 0; unit < n; ++unit) {
                    for (std::uint64_t r = 0; r < other_rows; ++r) {
                        for (std::uint64_t k = 0; k < b.block; ++k) {
                            const std::uint64_t raw_index = unit * b.block + k;
                            const std::uint64_t src_off = r * row_elems + raw_index * trailing;
                            const std::uint64_t dst_off = unit * seg.width + (r * b.block + k) * trailing;
                            for (std::uint64_t t = 0; t < trailing; ++t) {
                                seg.values[dst_off + t] = full[src_off + t];
                            }
                        }
                    }
                }
                segments.push_back(std::move(seg));
            }
        }
    }

    if (opts.include_norm_stats) {
        // One 1-wide segment per norm tensor (weight/bias/running_mean/
        // running_var are all already 1 scalar per unit), gathered once for
        // the whole group rather than per axis-binding.
        for (const auto& norm : find_norm_tensors(topo, group)) {
            const core::TensorMeta* nm = src.meta(norm.tensor);
            if (nm == nullptr) continue;
            auto vals = SFS_TRY(read_rows_as_float(src, norm.tensor, 0, n, 1, nm->dtype));
            segments.push_back(Segment{std::move(vals), 1});
        }
    }

    FeatureBlock block;
    for (const auto& s : segments) block.width += s.width;
    if (block.width == 0) return block;

    block.data.assign(static_cast<std::size_t>(n) * block.width, 0.0F);
    std::uint32_t offset = 0;
    for (const auto& s : segments) {
        for (std::uint32_t u = 0; u < n; ++u) {
            for (std::uint32_t k = 0; k < s.width; ++k) {
                block.data[static_cast<std::size_t>(u) * block.width + offset + k] =
                    s.values[static_cast<std::size_t>(u) * s.width + k];
            }
        }
        offset += s.width;
    }

    if (opts.normalize_units) {
        for (std::uint32_t u = 0; u < n; ++u) {
            float* row = &block.data[static_cast<std::size_t>(u) * block.width];
            float norm = 0.0F;
            for (std::uint32_t k = 0; k < block.width; ++k) norm += row[k] * row[k];
            norm = std::sqrt(norm) + 1e-8F;
            for (std::uint32_t k = 0; k < block.width; ++k) row[k] /= norm;
        }
    }
    return block;
}

}  // namespace

struct Matcher::Impl {
    core::ITensorSource& base;
    core::ITensorSource& target;
    const core::Topology& topo;
    MatchOptions opts;
    /// Permutations found so far, keyed by group. Read by build_features to
    /// realign incoming evidence and written by match_group as each group is
    /// solved; run() drives repeated passes over this shared state until it
    /// stops changing.
    PermutationMap solved;
};

Matcher::Matcher(core::ITensorSource& base, core::ITensorSource& target, const core::Topology& topo,
                MatchOptions opts)
    : impl_(std::make_unique<Impl>(Impl{base, target, topo, std::move(opts), {}})) {}

Matcher::~Matcher() = default;

Result<GroupMatch> Matcher::match_group(std::string_view group) {
    const core::PermGroup* g = impl_->topo.find_group(group);
    if (g == nullptr) return SFS_ERR(Internal, "unknown permutation group", std::string(group));

    GroupMatch gm;
    gm.group = std::string(group);

    if (g->pinned) {
        gm.identity = true;
        gm.alignable = true;
        impl_->solved[gm.group] = {};
        return gm;
    }

    const std::uint32_t n = g->size;

    if (n >= impl_->opts.sparse_crossover) {
        // A dense n x n cost matrix, and the greedy solver's O(n^2)
        // enumerate-and-sort, are not just slow at this size -- they do not
        // fit in the PS's memory ceiling (docs/adr/0011). Bypass
        // CostMatrix/ILapSolver entirely: fingerprint + sparse candidates +
        // Jacobi auction, mirroring cpp/src/dispatch.cpp's "synapse-forward"
        // branch.
        gm = SFS_TRY(match_group_sparse(impl_->target, impl_->base, impl_->topo, group, n, impl_->solved,
                                        impl_->opts.sparse, impl_->opts.confidence));
        impl_->solved[gm.group] = gm.identity ? std::vector<std::uint32_t>{} : gm.permutation;
        return gm;
    }

    FeatureBlock target_block = SFS_TRY(
        build_features(impl_->target, impl_->topo, group, n, impl_->solved, impl_->opts.cost));
    // Base is always the fixed reference frame, so its rows are never reordered.
    FeatureBlock base_block =
        SFS_TRY(build_features(impl_->base, impl_->topo, group, n, PermutationMap{}, impl_->opts.cost));

    if (target_block.width == 0 || target_block.width != base_block.width) {
        // Nothing to compare against (an all-pinned-neighbourhood group with
        // no evidence tensors of its own) -- identity is the only sane answer.
        gm.identity = true;
        gm.alignable = true;
        impl_->solved[gm.group] = {};
        return gm;
    }

    CostMatrix cost(n);
    cost.accumulate_tile(target_block.data, 0, n, base_block.data, 0, n,
                        target_block.width, impl_->opts.cost);

    const std::uint32_t crossover = impl_->opts.lap_crossover;
    auto solver = make_auto_solver(crossover);
    core::LapResult lap = SFS_TRY(solver->solve(cost.data(), n));

    const double identity_cost = cost.identity_cost();
    std::uint32_t distinct = 0;
    {
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t best_j = 0;
            float best_v = cost.at(i, 0);
            std::uint32_t ties = 0;
            for (std::uint32_t j = 1; j < n; ++j) {
                if (cost.at(i, j) < best_v) {
                    best_v = cost.at(i, j);
                    best_j = j;
                    ties = 0;
                } else if (cost.at(i, j) == best_v) {
                    ++ties;
                }
            }
            (void)best_j;
            if (ties == 0) ++distinct;
        }
    }

    Confidence conf = assess(lap.cost_raw, identity_cost, n, distinct, impl_->opts.confidence);

    gm.identity = false;
    gm.alignable = conf.verdict != Alignability::NotAlignable;
    gm.cost_raw = conf.cost_raw;
    gm.cost_normalized = conf.cost_normalized;
    gm.exact_solver = lap.exact;
    if (gm.alignable) {
        gm.permutation = std::move(lap.assignment);
    }
    impl_->solved[gm.group] = gm.identity ? std::vector<std::uint32_t>{} : gm.permutation;
    return gm;
}

Result<MatchReport> Matcher::run() {
    MatchReport report;
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::string> order = topological_group_order(impl_->topo);

    for (std::uint32_t sweep = 0; sweep < impl_->opts.max_sweeps; ++sweep) {
        const PermutationMap before = impl_->solved;
        for (const auto& group : order) {
            GroupMatch gm = SFS_TRY(match_group(group));
            report.groups[group] = std::move(gm);
        }
        report.sweeps = sweep + 1;
        // match_group updates impl_->solved as it goes (Gauss-Seidel, not
        // Jacobi: a group solved earlier in this same sweep is already visible
        // to the ones after it), so comparing the pre-sweep snapshot against
        // the now-current state is both the update and the convergence check.
        if (converged(before, impl_->solved)) break;
    }

    report.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return report;
}

}  // namespace sfs::align
