#include <synapsefs/stio/st_source.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <synapsefs/util/file.hpp>

namespace sfs::stio {

struct StSource::Impl {
    util::Fd fd;
    util::Mmap mapping;   // only populated when options.use_mmap
    bool use_mmap = false;

    std::vector<std::byte> header_prefix;   // [8-byte len][JSON], read eagerly
    format::StHeader header;
    std::vector<core::BufferEntry> layout;
    std::uint64_t total_bytes = 0;
};

StSource::StSource() : impl_(std::make_unique<Impl>()) {}
StSource::~StSource() = default;

Result<std::unique_ptr<StSource>> StSource::open(const std::filesystem::path& path,
                                                 StSourceOptions opts) {
    auto fd_r = util::open_file(path, util::OpenMode::Read);
    if (!fd_r) return SFS_ERR(NoSuchFile, "cannot open safetensors file", path.string());

    auto size_r = util::file_size(fd_r->get());
    if (!size_r) return SFS_ERR(Io, "cannot stat safetensors file", path.string());
    std::uint64_t total = *size_r;
    if (total < 8) return SFS_ERR(NotSafetensors, "file too small to be safetensors", path.string());

    // Read the length prefix, then exactly the header JSON — never more.
    std::array<std::byte, 8> len_buf{};
    auto n8 = util::pread_all(fd_r->get(), len_buf, 0);
    if (!n8 || *n8 != 8)
        return SFS_ERR(NotSafetensors, "cannot read 8-byte header length", path.string());

    auto len_r = format::read_header_len(len_buf);
    if (!len_r) return std::unexpected(len_r.error());
    std::uint64_t json_len = *len_r;
    if (json_len == 0 || 8 + json_len > total)
        return SFS_ERR(NotSafetensors, "implausible header length", path.string());

    auto self = std::unique_ptr<StSource>(new StSource());
    self->impl_->fd = std::move(*fd_r);
    self->impl_->total_bytes = total;
    self->impl_->use_mmap = opts.use_mmap;

    self->impl_->header_prefix.resize(static_cast<std::size_t>(8 + json_len));
    auto n_full = util::pread_all(self->impl_->fd.get(), self->impl_->header_prefix, 0);
    if (!n_full || *n_full != self->impl_->header_prefix.size())
        return SFS_ERR(NotSafetensors, "cannot read full header", path.string());

    auto hdr_r = format::parse_st_header(self->impl_->header_prefix);
    if (!hdr_r) return std::unexpected(hdr_r.error());
    self->impl_->header = std::move(*hdr_r);
    self->impl_->layout = self->impl_->header.buffer_layout();

    if (auto st = format::validate_buffer_layout(self->impl_->layout,
                                                 self->impl_->header.header_extent, total);
        !st) {
        return std::unexpected(st.error());
    }

    if (opts.use_mmap) {
        auto m = util::Mmap::open_read(path);
        if (!m) return SFS_ERR(Io, "mmap open failed", path.string());
        self->impl_->mapping = std::move(*m);
    }

    return self;
}

std::span<const std::byte> StSource::header_bytes() const {
    return impl_->header_prefix;
}

std::span<const core::BufferEntry> StSource::buffer_layout() const { return impl_->layout; }

const core::TensorMeta* StSource::meta(std::string_view name) const {
    auto it = impl_->header.tensors.find(std::string(name));
    return it == impl_->header.tensors.end() ? nullptr : &it->second;
}

std::uint64_t StSource::total_bytes() const { return impl_->total_bytes; }

const format::StHeader& StSource::header() const noexcept { return impl_->header; }

Result<std::size_t> StSource::read_raw(std::uint64_t data_offset, std::span<std::byte> out) {
    if (impl_->use_mmap) {
        auto bytes = impl_->mapping.bytes();
        if (data_offset >= bytes.size()) return std::size_t{0};
        std::size_t take = std::min<std::size_t>(out.size(), bytes.size() - data_offset);
        std::memcpy(out.data(), bytes.data() + data_offset, take);
        return take;
    }
    auto n = util::pread_all(impl_->fd.get(), out, data_offset);
    if (!n) return SFS_ERR(Io, "read failed", std::to_string(data_offset));
    return *n;
}

Result<std::size_t> StSource::read_units(std::string_view name, std::uint64_t first,
                                         std::uint64_t count, std::span<std::byte> out) {
    const core::TensorMeta* m = meta(name);
    if (!m) return SFS_ERR(TensorNotInBufferLayout, "no such tensor", std::string(name));

    // Output units are rows along axis 0 by convention here: row_stride =
    // nbytes / shape[0]. Callers that need a different axis go through
    // row_iter's UnitReader, which is axis-aware; this is the raw primitive
    // it is built on.
    auto ub = m->unit_bytes(0);
    if (!ub) return std::unexpected(ub.error());
    std::uint64_t stride = *ub;

    std::uint64_t offset = m->data_off + first * stride;
    std::size_t want = static_cast<std::size_t>(count * stride);
    if (out.size() < want)
        return SFS_ERR(Internal, "output buffer too small for requested units");

    return read_raw(offset, out.subspan(0, want));
}

}  // namespace sfs::stio
