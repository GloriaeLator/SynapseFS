/// \file stats.cpp
/// Relaxed-atomic counters, serialized on demand -- never on the fault path.
/// Peak RSS comes from /proc/self/status VmHWM, which is what
/// bench/scripts/peak_rss.sh and the scale test both key off.

#include <synapsefs/mount/stats.hpp>

#include <cstdio>
#include <format>

namespace sfs::mount {

namespace {

// Parses a line of the form "VmHWM:    12345 kB" from /proc/self/status.
// Returns bytes (the file reports kB), or 0 if the field is absent -- this
// happens on non-Linux fallbacks in tests that don't care about the number.
std::uint64_t read_status_field_bytes(std::string_view field) noexcept {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) return 0;

    char line[256];
    std::uint64_t result = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        std::string_view sv(line);
        if (sv.substr(0, field.size()) == field) {
            unsigned long long kb = 0;
            // Format is "<Field>:<ws>*<digits> kB\n" -- std::sscanf is fine
            // here since this runs at most once per stats query, never on
            // the read path.
            if (std::sscanf(line + field.size(), "%llu", &kb) == 1) {
                result = static_cast<std::uint64_t>(kb) * 1024ull;
            }
            break;
        }
    }
    std::fclose(f);
    return result;
}

}  // namespace

std::string DaemonStats::to_json() const {
    return std::format(
        "{{"
        "\"reads\":{},"
        "\"bytes_served\":{},"
        "\"frames_decompressed\":{},"
        "\"chain_hops\":{},"
        "\"cache_hits\":{},"
        "\"cache_misses\":{},"
        "\"single_flight_waits\":{},"
        "\"digest_failures\":{},"
        "\"prefetched_frames\":{},"
        "\"peak_rss_bytes\":{},"
        "\"current_rss_bytes\":{}"
        "}}",
        reads.load(std::memory_order_relaxed), bytes_served.load(std::memory_order_relaxed),
        frames_decompressed.load(std::memory_order_relaxed),
        chain_hops.load(std::memory_order_relaxed), cache_hits.load(std::memory_order_relaxed),
        cache_misses.load(std::memory_order_relaxed),
        single_flight_waits.load(std::memory_order_relaxed),
        digest_failures.load(std::memory_order_relaxed),
        prefetched_frames.load(std::memory_order_relaxed), peak_rss_bytes(), current_rss_bytes());
}

DaemonStats& global_stats() noexcept {
    static DaemonStats stats;
    return stats;
}

std::uint64_t peak_rss_bytes() noexcept { return read_status_field_bytes("VmHWM:"); }
std::uint64_t current_rss_bytes() noexcept { return read_status_field_bytes("VmRSS:"); }

}  // namespace sfs::mount
