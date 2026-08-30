#include "simple_st_source.hpp"

#include <cstring>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <synapsefs/core/dtype.hpp>

namespace sfs::align::tools {

using json = nlohmann::json;

core::Result<SimpleStSource> SimpleStSource::open(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return SFS_ERR(NoSuchFile, "cannot open safetensors file", path.string());
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return SFS_ERR(Io, "fstat failed", path.string());
    }
    const auto size = static_cast<std::uint64_t>(st.st_size);
    if (size < 8) {
        ::close(fd);
        return SFS_ERR(NotSafetensors, "file too small for an 8-byte header length", path.string());
    }

    // MAP_PRIVATE + PROT_READ: a read-only, copy-on-write mapping. The fd can
    // be closed immediately after mmap succeeds -- the mapping keeps the
    // underlying file open on POSIX regardless.
    void* region = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (region == MAP_FAILED) {
        return SFS_ERR(Io, "mmap failed", path.string());
    }

    SimpleStSource src;
    src.data_ = static_cast<const std::byte*>(region);
    src.size_ = size;

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, src.data_, 8);
    if (header_len > size - 8) {
        ::munmap(region, size);
        return SFS_ERR(NotSafetensors, "header length exceeds file size", path.string());
    }
    src.data_start_ = 8 + header_len;

    json header;
    try {
        header = json::parse(reinterpret_cast<const char*>(src.data_ + 8),
                             reinterpret_cast<const char*>(src.data_ + 8 + header_len));
    } catch (const json::parse_error& e) {
        ::munmap(region, size);
        return SFS_ERR(NotSafetensors, std::string("safetensors header parse error: ") + e.what(),
                      path.string());
    }

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const json& entry = it.value();

        core::TensorMeta m;
        m.shape_owner = it.key();
        for (const auto& dim : entry.at("shape")) m.shape.push_back(dim.get<std::uint64_t>());

        const std::string dtype_str = entry.at("dtype").get<std::string>();
        auto dtype = core::dtype_from_string(dtype_str);
        if (!dtype) {
            ::munmap(region, size);
            return SFS_ERR(UnsupportedDType, "unrecognised safetensors dtype", dtype_str);
        }
        m.dtype = *dtype;

        const auto& offsets = entry.at("data_offsets");
        const std::uint64_t rel_start = offsets.at(0).get<std::uint64_t>();
        const std::uint64_t rel_end = offsets.at(1).get<std::uint64_t>();
        if (src.data_start_ + rel_end > size) {
            ::munmap(region, size);
            return SFS_ERR(NotSafetensors, "tensor data_offsets exceed file size", it.key());
        }
        m.data_off = src.data_start_ + rel_start;
        m.nbytes = rel_end - rel_start;

        core::BufferEntry buf;
        buf.tensor = it.key();
        buf.off = rel_start;
        buf.nbytes = m.nbytes;
        src.layout_.push_back(buf);

        src.metas_.emplace(it.key(), std::move(m));
    }

    return src;
}

SimpleStSource::SimpleStSource(SimpleStSource&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      data_start_(other.data_start_),
      metas_(std::move(other.metas_)),
      layout_(std::move(other.layout_)) {}

SimpleStSource& SimpleStSource::operator=(SimpleStSource&& other) noexcept {
    if (this == &other) return *this;
    if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    data_start_ = other.data_start_;
    metas_ = std::move(other.metas_);
    layout_ = std::move(other.layout_);
    return *this;
}

SimpleStSource::~SimpleStSource() {
    if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
}

std::span<const std::byte> SimpleStSource::header_bytes() const { return {data_, data_start_}; }

std::span<const core::BufferEntry> SimpleStSource::buffer_layout() const { return layout_; }

const core::TensorMeta* SimpleStSource::meta(std::string_view name) const {
    auto it = metas_.find(std::string(name));
    return it == metas_.end() ? nullptr : &it->second;
}

std::uint64_t SimpleStSource::total_bytes() const { return size_; }

core::Result<std::size_t> SimpleStSource::read_units(std::string_view name, std::uint64_t first,
                                                     std::uint64_t count, std::span<std::byte> out) {
    auto it = metas_.find(std::string(name));
    if (it == metas_.end()) {
        return SFS_ERR(TensorNotInBufferLayout, "no such tensor", std::string(name));
    }
    const core::TensorMeta& m = it->second;
    const std::uint64_t row_bytes = m.shape.empty() ? m.nbytes : SFS_TRY(m.unit_bytes(0));
    const std::uint64_t want = count * row_bytes;
    if (want > out.size()) {
        return SFS_ERR(Internal, "output span too small for requested units", std::string(name));
    }
    const std::uint64_t src_off = m.data_off + first * row_bytes;
    if (src_off + want > size_) {
        return SFS_ERR(Internal, "requested units exceed tensor bounds", std::string(name));
    }
    // The kernel pages this range in from the file/page cache on first touch
    // (and only this range, not the whole file) -- this memcpy is the only
    // place bytes actually move into process-owned memory.
    std::memcpy(out.data(), data_ + src_off, want);
    return want;
}

}  // namespace sfs::align::tools
