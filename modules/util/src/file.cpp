#include <synapsefs/util/file.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

namespace sfs::util {

namespace {
std::error_code errno_ec() { return std::make_error_code(static_cast<std::errc>(errno)); }
}  // namespace

Fd::~Fd() { reset(); }

Fd::Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

Fd& Fd::operator=(Fd&& o) noexcept {
    if (this != &o) {
        reset();
        fd_ = o.fd_;
        o.fd_ = -1;
    }
    return *this;
}

int Fd::release() noexcept {
    int f = fd_;
    fd_ = -1;
    return f;
}

void Fd::reset(int fd) noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
}

std::expected<Fd, std::error_code> open_file(const std::filesystem::path& path, OpenMode mode,
                                             bool create, int perm) {
    int flags = 0;
    switch (mode) {
        case OpenMode::Read:      flags = O_RDONLY; break;
        case OpenMode::Write:     flags = O_WRONLY; break;
        case OpenMode::ReadWrite: flags = O_RDWR;   break;
    }
    flags |= O_CLOEXEC;
    if (create) flags |= O_CREAT;

    int fd = create ? ::open(path.c_str(), flags, perm) : ::open(path.c_str(), flags);
    if (fd < 0) return std::unexpected(errno_ec());
    return Fd(fd);
}

std::expected<std::size_t, std::error_code> pread_all(int fd, std::span<std::byte> out,
                                                       std::uint64_t offset) {
    std::size_t done = 0;
    while (done < out.size()) {
        ssize_t n = ::pread(fd, out.data() + done, out.size() - done,
                            static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return std::unexpected(errno_ec());
        }
        if (n == 0) break;  // short read: EOF
        done += static_cast<std::size_t>(n);
    }
    return done;
}

std::expected<std::size_t, std::error_code> pwrite_all(int fd, std::span<const std::byte> data,
                                                        std::uint64_t offset) {
    std::size_t done = 0;
    while (done < data.size()) {
        ssize_t n = ::pwrite(fd, data.data() + done, data.size() - done,
                             static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return std::unexpected(errno_ec());
        }
        if (n == 0) break;
        done += static_cast<std::size_t>(n);
    }
    return done;
}

std::expected<void, std::error_code> fsync_fd(int fd) {
    if (::fsync(fd) != 0) return std::unexpected(errno_ec());
    return {};
}

std::expected<void, std::error_code> fsync_dir(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return std::unexpected(errno_ec());
    int rc = ::fsync(fd);
    ::close(fd);
    if (rc != 0) return std::unexpected(errno_ec());
    return {};
}

std::expected<std::uint64_t, std::error_code> file_size(int fd) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) return std::unexpected(errno_ec());
    return static_cast<std::uint64_t>(st.st_size);
}

}  // namespace sfs::util
