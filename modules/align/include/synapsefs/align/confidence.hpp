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
    /// Achieved assignment cost divided by the identity assignment's cost.
    /// Above this, the group is reported not alignable. Calibrate on Day 3
    /// against known-unrelated pairs and record the value in the docs — a
    /// threshold nobody can justify is worse than no threshold.
    double normalized_cost_threshold = 0.85;

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
    std::uint32_t distinct_matches = 0;
    std::string   reason;   ///< human-facing, goes in the artifact and the log
};

[[nodiscard]] Confidence assess(double achieved_cost, double identity_cost,
                                std::uint32_t n, std::uint32_t distinct_matches,
                                const ConfidenceOptions& = {});

}  // namespace sfs::align
