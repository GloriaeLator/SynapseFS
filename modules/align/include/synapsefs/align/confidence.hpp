#pragma once
/// \file confidence.hpp
/// "Not meaningfully alignable", and how we decide it.
///
/// The PS requires detecting and reporting this. Most implementations will not
/// have it, so it is worth a slide.
///
/// Two distinct situations, both handled:
///   * DEGENERATE symmetry — dead or duplicated units, many equally good
///     permutations. Any valid permutation is acceptable per the PS. We take
///     the solver's answer and record cost_normalized so the ambiguity is
///     visible rather than hidden.
///   * UNRELATED checkpoints — different runs, or a different model. High
///     normalised cost, alignable = false, and the manifest entry falls back to
///     mode: full.
///
/// Note what CANNOT be used to decide this: comparing model outputs. Permuting
/// reorders summation, float addition is not associative, and the CORRECT
/// answer differs at ~5e-05. Alignment is verified by reconstructing bytes.

#include <cstdint>
#include <string>

namespace sfs::align {

struct ConfidenceOptions {
    /// How much of the gap between a perfect match (normalized cost 0) and
    /// this pair's OWN random-assignment baseline (normalized cost of
    /// matching target to base units by chance, see CostMatrix::random_cost)
    /// the achieved cost must close to count as alignable, e.g. 0.9 means
    /// "must beat 90% of the way from perfect down to random chance".
    /// Replaces a single fixed cost-ratio constant: what a given normalized
    /// cost value MEANS depends on the metric and architecture, but "close to
    /// what a random pairing would score, for this exact pair" is comparable
    /// in the same frame everywhere, so the cutoff no longer needs
    /// per-metric recalibration the way a flat threshold did.
    double random_baseline_margin = 0.9;

    /// Below this fraction of units having a distinct best match, treat the
    /// group as degenerate: still alignable, but flagged.
    double distinct_match_floor = 0.5;
};

enum class Alignability : std::uint8_t { Aligned, Degenerate, NotAlignable };

struct Confidence {
    Alignability  verdict = Alignability::Aligned;
    double        cost_raw = 0.0;
    double        cost_normalized = 0.0;
    double        identity_cost = 0.0;
    double        random_cost = 0.0;
    double        random_normalized = 0.0;   ///< the effective threshold was random_normalized * margin
    std::uint32_t distinct_matches = 0;
    std::string   reason;   ///< human-facing, goes in the artifact and the log
};

/// `random_cost`: CostMatrix::random_cost() (dense path) or a Monte Carlo
/// estimate over a few random permutations via sparse_true_cost (sparse
/// path) -- see match_group/match_group_sparse.
[[nodiscard]] Confidence assess(double achieved_cost, double identity_cost, double random_cost,
                                std::uint32_t n, std::uint32_t distinct_matches,
                                const ConfidenceOptions& = {});

}  // namespace sfs::align
