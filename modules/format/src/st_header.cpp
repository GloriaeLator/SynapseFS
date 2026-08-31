#include <synapsefs/format/st_header.hpp>

#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

#include <synapsefs/core/endian.hpp>

namespace sfs::format {

using json = nlohmann::json;

std::vector<BufferEntry> StHeader::buffer_layout() const {
    // Every tensor, ordered by data offset — safetensors does not guarantee
    // JSON key order matches data order, and callers need offset order to
    // build a gapless, binary-searchable layout.
    std::vector<const std::pair<const std::string, TensorMeta>*> ordered;
    ordered.reserve(tensors.size());
    for (const auto& kv : tensors) ordered.push_back(&kv);
    std::sort(ordered.begin(), ordered.end(), [](auto* a, auto* b) {
        return a->second.data_off < b->second.data_off;
    });

    std::vector<BufferEntry> out;
    out.reserve(ordered.size());
    for (auto* kv : ordered) {
        BufferEntry e;
        e.tensor = kv->first;
        e.off = kv->second.data_off;
        e.nbytes = kv->second.nbytes;
        e.group = kv->first;  // no topology parsed yet: singleton group per tensor
        out.push_back(std::move(e));
    }
    return out;
}

Result<std::uint64_t> read_header_len(std::span<const std::byte> first8) {
    if (first8.size() < 8)
        return SFS_ERR(NotSafetensors, "file too small to contain an 8-byte header length");
    return core::load_le<std::uint64_t>(first8.data());
}

Result<StHeader> parse_st_header(std::span<const std::byte> file_prefix) {
    auto len_r = read_header_len(file_prefix);
    if (!len_r) return std::unexpected(len_r.error());
    std::uint64_t n = *len_r;

    if (n == 0 || file_prefix.size() < 8 + n)
        return SFS_ERR(NotSafetensors, "implausible or truncated header length");

    StHeader hdr;
    hdr.header_extent = 8 + n;

    std::string_view header_json(reinterpret_cast<const char*>(file_prefix.data() + 8),
                                 static_cast<std::size_t>(n));
    json j;
    try {
        j = json::parse(header_json);
    } catch (const std::exception& e) {
        return SFS_ERR(NotSafetensors, std::string("header JSON parse failed: ") + e.what());
    }
    if (!j.is_object())
        return SFS_ERR(NotSafetensors, "safetensors header is not a JSON object");

    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") {
            for (auto mit = it.value().begin(); it.value().is_object() && mit != it.value().end();
                ++mit) {
                if (mit.value().is_string())
                    hdr.metadata[mit.key()] = mit.value().get<std::string>();
            }
            continue;
        }
        const auto& v = it.value();
        if (!v.contains("dtype") || !v.contains("data_offsets")) continue;

        TensorMeta t;
        t.shape_owner = it.key();
        auto dt = core::dtype_from_string(v.at("dtype").get<std::string>());
        if (!dt) return std::unexpected(dt.error());
        t.dtype = *dt;

        if (v.contains("shape"))
            for (const auto& d : v.at("shape")) t.shape.push_back(d.get<std::uint64_t>());

        const auto& off = v.at("data_offsets");
        if (!off.is_array() || off.size() != 2)
            return SFS_ERR(NotSafetensors, "tensor has malformed data_offsets", it.key());

        std::uint64_t begin = off[0].get<std::uint64_t>();
        std::uint64_t end = off[1].get<std::uint64_t>();
        if (begin > end)
            return SFS_ERR(NotSafetensors, "tensor data_offsets begin > end", it.key());

        t.data_off = hdr.header_extent + begin;
        t.nbytes = end - begin;
        if (hdr.header_extent + end > file_prefix.size() && file_prefix.size() >= hdr.header_extent) {
            // Caller may have passed only the header prefix; a full bounds
            // check against total file size happens in validate_buffer_layout
            // once the caller knows the real file size.
        }
        hdr.tensors.emplace(it.key(), std::move(t));
    }

    return hdr;
}

core::Status validate_buffer_layout(std::span<const BufferEntry> entries, std::uint64_t header_extent,
                              std::uint64_t total_bytes) {
    std::uint64_t cursor = header_extent;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.off != cursor)
            return SFS_ERR(TensorNotInBufferLayout,
                           "gap or overlap at buffer entry " + std::to_string(i), e.tensor);
        if (e.nbytes == 0)
            return SFS_ERR(TensorNotInBufferLayout, "zero-length buffer entry", e.tensor);
        cursor += e.nbytes;
    }
    if (cursor != total_bytes)
        return SFS_ERR(TensorNotInBufferLayout,
                       "buffer entries cover " + std::to_string(cursor) + " but total is " +
                           std::to_string(total_bytes));
    return {};
}

}  // namespace sfs::format
