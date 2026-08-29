#pragma once
/// \file blockcache.hpp
/// Bounded LRU of DECOMPRESSED FRAMES, with single-flight fill.
///
/// This is the only genuinely subtle piece of concurrency in the project.
/// docs/spec/16-consistency.md §5:
///
///   * Immutability. A mounted commit never changes, so there is no
///     invalidation problem — only a fill problem.
///   * Single flight. Two readers faulting the same frame must not both
///     decompress it. The first arrival fills; the rest wait.
///   * Publication. An entry becomes visible only when fully populated,
///     published with release semantics and read with acquire. A reader never
///     observes a half-filled frame.
///   * Eviction. An entry with a non-zero reader refcount is never evicted.
///     Under pressure the daemon serves correctly and slowly, never wrongly.
///
/// Peak RSS is bounded by cache_bytes + frame_bytes * depth * readers, and is
/// NOT a function of checkpoint size. That claim is what bench/scripts/peak_rss.sh
/// exists to support.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>

namespace sfs::mount {

struct FrameKey {
    core::Oid     artifact;
    std::uint32_t tensor_index = 0;
    std::uint32_t frame_index  = 0;
    friend bool operator==(const FrameKey&, const FrameKey&) noexcept = default;
};

struct FrameKeyHash {
    [[nodiscard]] std::size_t operator()(const FrameKey&) const noexcept;
};

/// Borrowed view of a cached frame. Holds a refcount for its lifetime, so the
/// entry cannot be evicted while a reader is copying out of it.
class FrameLease {
public:
    FrameLease() = default;
    ~FrameLease();
    FrameLease(FrameLease&&) noexcept;
    FrameLease& operator=(FrameLease&&) noexcept;
    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] bool valid() const noexcept;

private:
    friend class FrameCache;
    struct Entry;
    Entry* entry_ = nullptr;
    class FrameCache* owner_ = nullptr;
};

class FrameCache {
public:
    explicit FrameCache(std::uint64_t budget_bytes);
    ~FrameCache();

    /// Look up, or fill exactly once. `fill` is invoked by whichever thread
    /// arrives first; the others block until it publishes.
    [[nodiscard]] core::Result<FrameLease> get_or_fill(
        const FrameKey&, std::uint64_t size_hint,
        const std::function<core::Status(std::span<std::byte>)>& fill);

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t single_flight_waits = 0;
        std::uint64_t evictions = 0;
        std::uint64_t bytes_resident = 0;
        std::uint64_t bytes_budget = 0;
    };
    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfs::mount
