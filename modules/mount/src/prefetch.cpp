/// \file prefetch.cpp
/// Detects a sequential access pattern per open-file-handle and reports how
/// many frames ahead to fill. See prefetch.hpp: this must DETECT rather than
/// assume, because safetensors' load pattern is sequential but the other
/// realistic pattern -- random access into a memory-mapped file -- is not,
/// and prefetching there is pure waste against the peak-RSS budget.

#include <synapsefs/mount/prefetch.hpp>

namespace sfs::mount {

PrefetchState::PrefetchState(PrefetchOptions opts) : opts_(opts) {}

std::uint32_t PrefetchState::observe(std::uint64_t offset, std::uint64_t length) noexcept {
    if (!opts_.enabled) return 0;

    // In order iff this read starts exactly where the last one ended. A gap
    // or a rewind breaks the streak; it does not immediately disable
    // prefetch (a single skip -- e.g. crossing the header/buffer boundary --
    // shouldn't cost the whole streak), it just resets the counter.
    const bool in_order = (offset == next_expected_);

    if (in_order) {
        if (streak_ < opts_.trigger_streak) ++streak_;
    } else {
        streak_ = 0;
    }

    next_expected_ = offset + length;

    if (!sequential()) return 0;
    return opts_.max_ahead_frames;
}

void PrefetchState::reset() noexcept {
    next_expected_ = 0;
    streak_        = 0;
}

}  // namespace sfs::mount
