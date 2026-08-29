#pragma once
/// \file file.hpp
/// Owning file descriptor and thin pread/pwrite wrappers.
///
/// Everything returns std::expected<T, std::error_code>; `util` deliberately
/// does not depend on `core`, so it speaks errno rather than sfs::core::Error.
/// `core` wraps these into ErrKind::Io at the boundary.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <system_error>

namespace sfs::util {

/// Non-copyable, movable owning fd. Closes on destruction.
class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) noexcept : fd_(fd) {}
    ~Fd();

    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&&) noexcept;
    Fd& operator=(Fd&&) noexcept;

    [[nodiscard]] int  get() const noexcept { return fd_; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    [[nodiscard]] int  release() noexcept;
    void reset(int fd = -1) noexcept;

private:
    int fd_ = -1;
};

enum class OpenMode : std::uint8_t { Read, Write, ReadWrite };

[[nodiscard]] std::expected<Fd, std::error_code> open_file(const std::filesystem::path&,
                                                           OpenMode,
                                                           bool create = false,
                                                           int mode = 0644);

/// Positional read. Retries on EINTR; a short read is reported as such, not as
/// an error — callers loop.
[[nodiscard]] std::expected<std::size_t, std::error_code> pread_all(int fd,
                                                                    std::span<std::byte> out,
                                                                    std::uint64_t offset);

[[nodiscard]] std::expected<std::size_t, std::error_code> pwrite_all(
    int fd, std::span<const std::byte> data, std::uint64_t offset);

[[nodiscard]] std::expected<void, std::error_code> fsync_fd(int fd);

/// fsync of a DIRECTORY. Separate from fsync_fd only so that call sites read
/// as the deliberate act they are: without this, a rename can be lost across a
/// power failure while its data survives. See docs/spec/11-repo-layout.md §3.1.
[[nodiscard]] std::expected<void, std::error_code> fsync_dir(const std::filesystem::path&);

[[nodiscard]] std::expected<std::uint64_t, std::error_code> file_size(int fd);

}  // namespace sfs::util
