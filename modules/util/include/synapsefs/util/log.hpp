#pragma once
/// \file log.hpp
/// Minimal levelled logging. One line per event, structured key=value tail.
///
/// Rule for the mount's fault path: nothing above Trace. A log statement per
/// page fault is a throughput bug, and throughput is 8% of the grade.

#include <cstdint>
#include <format>
#include <string_view>

namespace sfs::util {

enum class LogLevel : std::uint8_t {
    Error = 0,
    Warn  = 1,
    Info  = 2,
    Debug = 3,
    Trace = 4,
};

void set_log_level(LogLevel) noexcept;
[[nodiscard]] LogLevel log_level() noexcept;

/// Writes to stderr, so that `sfs log --json | jq` stays clean.
void log_write(LogLevel, std::string_view module, std::string_view message) noexcept;

template <class... Args>
void log_at(LogLevel lvl, std::string_view module, std::format_string<Args...> fmt,
            Args&&... args) {
    if (static_cast<std::uint8_t>(lvl) > static_cast<std::uint8_t>(log_level())) return;
    log_write(lvl, module, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace sfs::util

#define SFS_LOG_E(mod, ...) ::sfs::util::log_at(::sfs::util::LogLevel::Error, mod, __VA_ARGS__)
#define SFS_LOG_W(mod, ...) ::sfs::util::log_at(::sfs::util::LogLevel::Warn,  mod, __VA_ARGS__)
#define SFS_LOG_I(mod, ...) ::sfs::util::log_at(::sfs::util::LogLevel::Info,  mod, __VA_ARGS__)
#define SFS_LOG_D(mod, ...) ::sfs::util::log_at(::sfs::util::LogLevel::Debug, mod, __VA_ARGS__)
#define SFS_LOG_T(mod, ...) ::sfs::util::log_at(::sfs::util::LogLevel::Trace, mod, __VA_ARGS__)
