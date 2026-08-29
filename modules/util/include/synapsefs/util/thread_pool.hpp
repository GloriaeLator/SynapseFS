#pragma once
/// \file thread_pool.hpp
/// Fixed-size worker pool for the aligner's tiled cost accumulation and for
/// per-frame residual encoding.
///
/// Not used by the mount daemon: libfuse runs its own threads, and adding a
/// second pool underneath it makes peak RSS and latency harder to reason about
/// for no benefit.

#include <cstddef>
#include <functional>
#include <future>
#include <span>
#include <vector>

namespace sfs::util {

class ThreadPool {
public:
    /// `threads == 0` means std::thread::hardware_concurrency().
    explicit ThreadPool(std::size_t threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;

    /// Fire and forget. Exceptions escaping `fn` are caught and logged; they
    /// must not cross a thread boundary.
    void post(std::function<void()> fn);

    /// Run `fn(i)` for i in [0, n) and block until all have completed.
    /// This is the shape the aligner actually uses: a tile per index, joined
    /// before the group's LAP solve.
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

    /// Block until every posted task has run.
    void drain();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Process-wide pool, created on first use. Sized from
/// SFS_THREADS if set, else hardware_concurrency.
[[nodiscard]] ThreadPool& default_pool();

}  // namespace sfs::util
