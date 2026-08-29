#pragma once
/// \file scope_guard.hpp
/// RAII cleanup for the many places we hold a raw fd, a mapping or a lock
/// across an early return. We do not use exceptions across module boundaries
/// (docs/interfaces/errors.md), so every error path is an explicit return and
/// every one of them has to release what it holds.

#include <concepts>
#include <utility>

namespace sfs::util {

template <std::invocable F>
class ScopeGuard {
public:
    explicit ScopeGuard(F f) : fn_(std::move(f)) {}

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&& o) noexcept : fn_(std::move(o.fn_)), active_(o.active_) {
        o.active_ = false;
    }
    ScopeGuard& operator=(ScopeGuard&&) = delete;

    ~ScopeGuard() {
        if (active_) fn_();
    }

    /// Cancel the cleanup — the happy path took ownership.
    void dismiss() noexcept { active_ = false; }

private:
    F    fn_;
    bool active_ = true;
};

template <std::invocable F>
[[nodiscard]] ScopeGuard<F> on_scope_exit(F f) {
    return ScopeGuard<F>(std::move(f));
}

}  // namespace sfs::util

#define SFS_CONCAT_(a, b) a##b
#define SFS_CONCAT(a, b) SFS_CONCAT_(a, b)
/// SFS_DEFER { close(fd); };
#define SFS_DEFER \
    auto SFS_CONCAT(sfs_guard_, __LINE__) = ::sfs::util::on_scope_exit([&]() noexcept
