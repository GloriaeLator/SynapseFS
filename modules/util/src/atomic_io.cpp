#include <synapsefs/util/atomic_io.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <random>
#include <cstring>

#include <synapsefs/util/file.hpp>

namespace fs = std::filesystem;

namespace sfs::util {

namespace {

std::error_code errno_ec() { return std::make_error_code(static_cast<std::errc>(errno)); }

fs::path make_temp_name(const fs::path& dir) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "tmp.%llx", static_cast<unsigned long long>(dist(rng)));
    return dir / buf;
}

/// Core sequence, parameterised over a writer callback so both the
/// single-buffer and multi-part entry points share one implementation.
///   tmp file (O_CREAT|O_EXCL) -> write -> fsync -> rename -> fsync(parent)
template <class WriteFn>
std::expected<void, std::error_code> atomic_write_impl(const fs::path& dest, WriteFn&& write_fn,
                                                        const AtomicWriteOptions& opts) {
    if (!opts.overwrite) {
        std::error_code ec;
        if (fs::exists(dest, ec)) return {};  // content-addressed: identical bytes, no-op
    }

    fs::path temp_dir = !opts.temp_dir.empty() ? opts.temp_dir : dest.parent_path();
    std::error_code mkec;
    fs::create_directories(temp_dir, mkec);
    fs::create_directories(dest.parent_path(), mkec);

    fs::path tmp;
    int fd = -1;
    for (int attempt = 0; attempt < 8; ++attempt) {
        tmp = make_temp_name(temp_dir);
        fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, opts.mode);
        if (fd >= 0) break;
        if (errno != EEXIST) return std::unexpected(errno_ec());
    }
    if (fd < 0) return std::unexpected(errno_ec());

    auto cleanup = [&] {
        ::close(fd);
        std::error_code rmec;
        fs::remove(tmp, rmec);
    };

    if (auto r = write_fn(fd); !r) {
        cleanup();
        return r;
    }

    if (opts.fsync_contents && ::fsync(fd) != 0) {
        auto ec = errno_ec();
        cleanup();
        return std::unexpected(ec);
    }
    ::close(fd);
    fd = -1;

    if (::rename(tmp.c_str(), dest.c_str()) != 0) {
        auto ec = errno_ec();
        std::error_code rmec;
        fs::remove(tmp, rmec);
        return std::unexpected(ec);
    }

    if (opts.fsync_parent) {
        if (auto r = fsync_dir(dest.parent_path()); !r) return std::unexpected(r.error());
    }
    return {};
}

}  // namespace

std::expected<void, std::error_code> atomic_write(const fs::path& dest,
                                                   std::span<const std::byte> data,
                                                   const AtomicWriteOptions& opts) {
    return atomic_write_impl(
        dest,
        [&](int fd) -> std::expected<void, std::error_code> {
            std::size_t off = 0;
            while (off < data.size()) {
                ssize_t n = ::write(fd, data.data() + off, data.size() - off);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    return std::unexpected(errno_ec());
                }
                off += static_cast<std::size_t>(n);
            }
            return {};
        },
        opts);
}

std::expected<void, std::error_code> atomic_write_v(
    const fs::path& dest, std::span<const std::span<const std::byte>> parts,
    const AtomicWriteOptions& opts) {
    return atomic_write_impl(
        dest,
        [&](int fd) -> std::expected<void, std::error_code> {
            for (auto part : parts) {
                std::size_t off = 0;
                while (off < part.size()) {
                    ssize_t n = ::write(fd, part.data() + off, part.size() - off);
                    if (n < 0) {
                        if (errno == EINTR) continue;
                        return std::unexpected(errno_ec());
                    }
                    off += static_cast<std::size_t>(n);
                }
            }
            return {};
        },
        opts);
}

std::expected<std::vector<std::byte>, std::error_code> read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(std::make_error_code(std::errc::no_such_file_or_directory));
    f.seekg(0, std::ios::end);
    auto size = f.tellg();
    if (size < 0) return std::unexpected(std::make_error_code(std::errc::io_error));
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(size));
    if (!buf.empty() && !f.read(reinterpret_cast<char*>(buf.data()),
                                static_cast<std::streamsize>(buf.size()))) {
        return std::unexpected(std::make_error_code(std::errc::io_error));
    }
    return buf;
}

std::expected<bool, std::error_code> atomic_replace_if(const fs::path& dest,
                                                        std::string_view expected,
                                                        std::string_view desired,
                                                        const AtomicWriteOptions& opts) {
    // Compare-and-swap on a single-line file. `expected` empty means "must
    // not currently exist". This is advisory at the filesystem level; callers
    // that need true concurrency safety hold RepoLock (store/lockfile.hpp)
    // around the read-compare-write sequence.
    std::error_code ec;
    bool exists = fs::exists(dest, ec);
    if (expected.empty()) {
        if (exists) return false;
    } else {
        if (!exists) return false;
        auto current = read_file(dest);
        if (!current) return std::unexpected(current.error());
        std::string_view cur_sv(reinterpret_cast<const char*>(current->data()), current->size());
        // Ignore a single trailing newline when comparing.
        if (!cur_sv.empty() && cur_sv.back() == '\n') cur_sv.remove_suffix(1);
        if (cur_sv != expected) return false;
    }

    std::vector<std::byte> bytes(desired.size());
    ::memcpy(bytes.data(), desired.data(), desired.size());
    AtomicWriteOptions o = opts;
    o.overwrite = true;
    if (auto r = atomic_write(dest, bytes, o); !r) return std::unexpected(r.error());
    return true;
}

std::expected<std::size_t, std::error_code> purge_temp_dir(const fs::path& temp_dir) {
    std::error_code ec;
    if (!fs::exists(temp_dir, ec)) {
        fs::create_directories(temp_dir, ec);
        return std::size_t{0};
    }
    std::size_t removed = 0;
    for (const auto& entry : fs::directory_iterator(temp_dir, ec)) {
        std::error_code rmec;
        // Everything in tmp/ is garbage by construction: it is only ever
        // renamed OUT of, never read. A crash mid-write leaves an orphan
        // here, never a half-written object under its real name.
        auto count = fs::remove_all(entry.path(), rmec);
        if (!rmec) removed += static_cast<std::size_t>(count);
    }
    return removed;
}

}  // namespace sfs::util
