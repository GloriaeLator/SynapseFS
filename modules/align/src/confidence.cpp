#include <synapsefs/align/confidence.hpp>

#include <cmath>
#include <sstream>

namespace sfs::align {

namespace {

// "cost divided by identity cost" (confidence.hpp), oriented so LOW = well
// aligned, HIGH = no better than doing nothing. That literal ratio only has
// that shape for a non-negative cost (L2, CosineDistance): achieved <=
// identity always (achieved is the true optimum), both >= 0, so
// achieved/identity is a genuine [0, 1] "how much of identity's cost
// survives" figure. cost.hpp's DEFAULT metric, NegInnerProduct, is typically
// <= 0, and achieved/identity is then >= 1 for a GOOD match (dividing a
// larger-magnitude negative by a smaller one) -- confirmed, not guessed, by
// modules/align/tests/test_known_permutation.cpp: a planted, perfectly
// recoverable permutation reported cost_normalized ~1.05 under the literal
// formula, which would call it not alignable. Since achieved <= identity
// holds for every metric regardless of sign, swapping the ratio when
// identity_cost is negative restores the same [~0, ~1]-with-1-as-baseline
// shape: identity/cost is small when cost is far more negative than
// identity (good), ~1 when cost barely improves on identity (bad). Applied
// identically to achieved_cost and random_cost so both land in the same
// frame and stay comparable to each other.
double normalize_against_identity(double cost, double identity_cost) {
    if (identity_cost > 0.0) {
        return cost / identity_cost;
    }
    if (identity_cost < 0.0) {
        return (cost != 0.0) ? identity_cost / cost : 0.0;
    }
    return 0.0;
}

}  // namespace

Confidence assess(double achieved_cost, double identity_cost, double random_cost, std::uint32_t n,
                  std::uint32_t distinct_matches, const ConfidenceOptions& opts) {
    Confidence c;
    c.cost_raw = achieved_cost;
    c.identity_cost = identity_cost;
    c.random_cost = random_cost;
    c.distinct_matches = distinct_matches;

    c.cost_normalized = normalize_against_identity(achieved_cost, identity_cost);
    c.random_normalized = normalize_against_identity(random_cost, identity_cost);

    // effective_threshold: how far achieved_cost must close the gap from a
    // perfect match (0) toward this pair's own random-assignment baseline
    // (random_normalized) to count as alignable -- see ConfidenceOptions::
    // random_baseline_margin.
    const double effective_threshold = c.random_normalized * opts.random_baseline_margin;

    std::ostringstream reason;
    if (c.cost_normalized > effective_threshold) {
        c.verdict = Alignability::NotAlignable;
        reason << "normalized cost " << c.cost_normalized << " exceeds " << (opts.random_baseline_margin * 100.0)
              << "% of this pair's random-assignment baseline (" << c.random_normalized << ")";
    } else if (n > 0 && static_cast<double>(distinct_matches) / static_cast<double>(n) <
                            opts.distinct_match_floor) {
        c.verdict = Alignability::Degenerate;
        reason << distinct_matches << "/" << n << " units have a distinct best match, below floor "
              << opts.distinct_match_floor;
    } else {
        c.verdict = Alignability::Aligned;
    }
    c.reason = reason.str();
    return c;
}

}  // namespace sfs::align
