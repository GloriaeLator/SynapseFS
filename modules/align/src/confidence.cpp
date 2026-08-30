#include <synapsefs/align/confidence.hpp>

#include <cmath>
#include <sstream>

namespace sfs::align {

Confidence assess(double achieved_cost, double identity_cost, std::uint32_t n,
                  std::uint32_t distinct_matches, const ConfidenceOptions& opts) {
    Confidence c;
    c.cost_raw = achieved_cost;
    c.identity_cost = identity_cost;
    c.distinct_matches = distinct_matches;

    // "achieved cost divided by identity cost" (confidence.hpp), oriented so
    // LOW = well aligned, HIGH = no better than doing nothing, matching
    // "above normalized_cost_threshold => not alignable" below. That literal
    // ratio only has that shape for a non-negative cost (L2, CosineDistance):
    // achieved <= identity always (achieved is the true optimum), both >= 0,
    // so achieved/identity is a genuine [0, 1] "how much of identity's cost
    // survives" figure. cost.hpp's DEFAULT metric, NegInnerProduct, is
    // typically <= 0, and achieved/identity is then >= 1 for a GOOD match
    // (dividing a larger-magnitude negative by a smaller one) -- confirmed,
    // not guessed, by modules/align/tests/test_known_permutation.cpp: a
    // planted, perfectly recoverable permutation reported cost_normalized
    // ~1.05 under the literal formula, which would call it not alignable.
    // Since achieved <= identity holds for every metric regardless of sign,
    // swapping the ratio when identity_cost is negative restores the same
    // [~0, ~1]-with-1-as-baseline shape: identity/achieved is small when
    // achieved is far more negative than identity (good), ~1 when achieved
    // barely improves on identity (bad).
    if (identity_cost > 0.0) {
        c.cost_normalized = achieved_cost / identity_cost;
    } else if (identity_cost < 0.0) {
        c.cost_normalized = (achieved_cost != 0.0) ? identity_cost / achieved_cost : 0.0;
    } else {
        c.cost_normalized = 0.0;
    }

    std::ostringstream reason;
    if (c.cost_normalized > opts.normalized_cost_threshold) {
        c.verdict = Alignability::NotAlignable;
        reason << "normalized cost " << c.cost_normalized << " exceeds threshold "
              << opts.normalized_cost_threshold;
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
