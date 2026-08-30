#include <synapsefs/util/mmap.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

#include <synapsefs/util/bits.hpp>

namespace sfs::util {

namespace {
std::error_code errno_ec() { return std::make_error_code(static_cast<std::errc>(errno)); }
constexpr std::size_t kPageSize = 4096;
}  // namespace

Mmap::~Mmap() {
    if (map_base_) ::munmap(const_cast<std::byte*>(map_base_), map_size_);
}

Mmap::Mmap(Mmap&& o) noexcept
    : data_(o.data_), size_(o.size_), map_size_(o.map_size_), map_base_(o.map_base_) {
    o.data_ = nullptr;
    o.size_ = 0;
    o.map_size_ = 0;
    o.map_base_ = nullptr;
}

Mmap& Mmap::operator=(Mmap&& o) noexcept {
    if (this != &o) {
        if (map_base_) ::munmap(const_cast<std::byte*>(map_base_), map_size_);
        data_ = o.data_;
        size_ = o.size_;
        map_size_ = o.map_size_;
        map_base_ = o.map_base_;
        o.data_ = nullptr;
        o.size_ = 0;
        o.map_size_ = 0;
        o.map_base_ = nullptr;
    }
    return *this;
}

std::expected<Mmap, std::error_code> Mmap::open_read(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::unexpected(errno_ec());
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        auto ec = errno_ec();
        ::close(fd);
        return std::unexpected(ec);
    }
    auto result = map_fd(fd, 0, static_cast<std::size_t>(st.st_size));
    ::close(fd);
    return result;
}

std::expected<Mmap, std::error_code> Mmap::map_fd(int fd, std::uint64_t offset,
                                                   std::size_t length) {
    if (length == 0) {
        // A zero-length mapping is valid content (an empty object): return a
        // handle with no backing pages rather than failing mmap(2), which
        // rejects length == 0 outright.
        Mmap m;
        m.data_ = nullptr;
        m.size_ = 0;
        return m;
    }

    std::uint64_t aligned_off = align_down<std::uint64_t>(offset, kPageSize);
    std::uint64_t misalign = offset - aligned_off;
    std::size_t map_len = static_cast<std::size_t>(misalign) + length;

    void* base = ::mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd,
                        static_cast<off_t>(aligned_off));
    if (base == MAP_FAILED) return std::unexpected(errno_ec());

    Mmap m;
    m.map_base_ = static_cast<const std::byte*>(base);
    m.map_size_ = map_len;
    m.data_ = m.map_base_ + misalign;
    m.size_ = length;
    return m;
}

std::expected<void, std::error_code> Mmap::advise(MapAdvice advice) noexcept {
    if (!map_base_) return {};
    return advise_range(0, size_, advice);
}

std::expected<void, std::error_code> Mmap::advise_range(std::uint64_t offset, std::size_t len,
                                                         MapAdvice advice) noexcept {
    if (!map_base_) return {};
    int flag = MADV_NORMAL;
    switch (advice) {
        case MapAdvice::Normal:     flag = MADV_NORMAL;     break;
        case MapAdvice::Sequential: flag = MADV_SEQUENTIAL; break;
        case MapAdvice::Random:     flag = MADV_RANDOM;     break;
        case MapAdvice::WillNeed:   flag = MADV_WILLNEED;   break;
        case MapAdvice::DontNeed:   flag = MADV_DONTNEED;   break;
    }
    std::uint64_t misalign = static_cast<std::uint64_t>(data_ - map_base_);
    std::uint64_t abs_off = misalign + offset;
    std::uint64_t aligned = align_down<std::uint64_t>(abs_off, kPageSize);
    std::size_t adjusted_len = len + static_cast<std::size_t>(abs_off - aligned);
    if (aligned + adjusted_len > map_size_) adjusted_len = map_size_ - aligned;

    if (::madvise(const_cast<std::byte*>(map_base_) + aligned, adjusted_len, flag) != 0)
        return std::unexpected(errno_ec());
    return {};
}

}  // namespace sfs::util
