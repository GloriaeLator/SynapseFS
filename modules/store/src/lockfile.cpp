#include <synapsefs/store/lockfile.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <thread>

namespace sfs::store {

namespace {
std::error_code errno_ec() { return std::make_error_code(static_cast<std::errc>(errno)); }
}  // namespace

RepoLock::~RepoLock() { release(); }

RepoLock::RepoLock(RepoLock&& o) noexcept : fd_(std::move(o.fd_)) {}

RepoLock& RepoLock::operator=(RepoLock&& o) noexcept {
    if (this != &o) {
        release();
        fd_ = std::move(o.fd_);
    }
    return *this;
}

void RepoLock::release() noexcept {
    if (fd_.valid()) {
        ::flock(fd_.get(), LOCK_UN);
        fd_.reset();
    }
}

Result<RepoLock> RepoLock::acquire(const std::filesystem::path& lock_path, LockMode mode,
                                   std::chrono::milliseconds wait) {
    int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return SFS_ERR(Io, "cannot open lock file", lock_path.string());

    int op = (mode == LockMode::Exclusive) ? LOCK_EX : LOCK_SH;

    auto try_once = [&]() { return ::flock(fd, op | LOCK_NB); };

    if (try_once() == 0) {
        RepoLock l;
        l.fd_.reset(fd);
        return l;
    }
    if (errno != EWOULDBLOCK) {
        auto ec = errno_ec();
        ::close(fd);
        return SFS_ERR(Io, ec.message(), lock_path.string());
    }

    if (wait.count() > 0) {
        auto deadline = std::chrono::steady_clock::now() + wait;
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (try_once() == 0) {
                RepoLock l;
                l.fd_.reset(fd);
                return l;
            }
            if (errno != EWOULDBLOCK) {
                ::close(fd);
                return SFS_ERR(Io, "flock failed", lock_path.string());
            }
        }
    }

    ::close(fd);
    return SFS_ERR(RepositoryLocked, "repository is locked by another process",
                   lock_path.string());
}

}  // namespace sfs::store
