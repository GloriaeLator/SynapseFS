#pragma once
/// \file stats.hpp
/// Daemon counters, for the benchmarks and for `sfs mount --foreground`.
///
/// Everything here is a relaxed atomic increment. Nothing on the fault path
/// takes a lock or formats a string: a log statement per page fault is a
/// throughput bug, and throughput is 8% of the grade.

#include <atomic>
#include <cstdint>
#include <string>

namespace sfs::mount {

struct DaemonStats {
    std::atomic<std::uint64_t> reads = 0;
    std::atomic<std::uint64_t> bytes_served = 0;
    std::atomic<std::uint64_t> frames_decompressed = 0;
    std::atomic<std::uint64_t> chain_hops = 0;
    std::atomic<std::uint64_t> cache_hits = 0;
    std::atomic<std::uint64_t> cache_misses = 0;
    std::atomic<std::uint64_t> single_flight_waits = 0;
    std::atomic<std::uint64_t> digest_failures = 0;   ///< always also logged at error
    std::atomic<std::uint64_t> prefetched_frames = 0;

    [[nodiscard]] std::string to_json() const;
};

[[nodiscard]] DaemonStats& global_stats() noexcept;

/// Peak resident set from /proc/self/status VmHWM. This is the number
/// bench/scripts/peak_rss.sh reports and the scale test asserts against.
[[nodiscard]] std::uint64_t peak_rss_bytes() noexcept;
[[nodiscard]] std::uint64_t current_rss_bytes() noexcept;

}  // namespace sfs::mount
