#include <synapsefs/util/log.hpp>

#include <atomic>
#include <cstdio>
#include <ctime>

namespace sfs::util {

namespace {
std::atomic<LogLevel> g_level{LogLevel::Info};

const char* level_name(LogLevel lvl) noexcept {
    switch (lvl) {
        case LogLevel::Error: return "error";
        case LogLevel::Warn:  return "warn";
        case LogLevel::Info:  return "info";
        case LogLevel::Debug: return "debug";
        case LogLevel::Trace: return "trace";
    }
    return "?";
}
}  // namespace

void set_log_level(LogLevel lvl) noexcept { g_level.store(lvl, std::memory_order_relaxed); }
LogLevel log_level() noexcept { return g_level.load(std::memory_order_relaxed); }

void log_write(LogLevel lvl, std::string_view module, std::string_view message) noexcept {
    // stderr only: `sfs log --json | jq` and similar pipelines must stay
    // clean on stdout.
    std::fprintf(stderr, "[%s] %.*s: %.*s\n", level_name(lvl), static_cast<int>(module.size()),
                module.data(), static_cast<int>(message.size()), message.data());
}

}  // namespace sfs::util
