#include <synapsefs/format/manifest.hpp>

#include <cstring>

#include <nlohmann/json.hpp>

namespace sfs::format {

using json = nlohmann::json;

Status Manifest::validate() const {
    if (buffer.empty())
        return SFS_ERR(MalformedObject, "manifest has no buffer entries");

    // buffer[0].off == 0, contiguous, sum + header extent (implicitly
    // buffer[0].off) == total_bytes.
    std::uint64_t cursor = buffer[0].off;
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const auto& e = buffer[i];
        if (e.off != cursor)
            return SFS_ERR(TensorNotInBufferLayout,
                           "gap or overlap at buffer entry " + std::to_string(i), e.tensor);
        if (e.nbytes == 0)
            return SFS_ERR(TensorNotInBufferLayout, "zero-length buffer entry", e.tensor);
        if (!find_group(e.group))
            return SFS_ERR(MalformedObject, "buffer entry references unknown group", e.group);
        cursor += e.nbytes;
    }
    if (cursor != file.total_bytes)
        return SFS_ERR(TensorNotInBufferLayout,
                       "buffer covers " + std::to_string(cursor) + " but file.total_bytes is " +
                           std::to_string(file.total_bytes));

    for (const auto& [name, g] : groups) {
        if (g.mode == GroupMode::Full) {
            if (!g.block)
                return SFS_ERR(MalformedObject, "Full group missing block", name);
            if (g.chain_depth != 0)
                return SFS_ERR(MalformedObject, "Full group must have chain_depth 0", name);
        } else {
            if (!g.base || !g.diff_block)
                return SFS_ERR(MalformedObject, "Delta group missing base or diff_block", name);
        }
    }
    return {};
}

const GroupEntry* Manifest::find_group(std::string_view name) const noexcept {
    auto it = groups.find(std::string(name));
    return it == groups.end() ? nullptr : &it->second;
}

std::uint32_t Manifest::max_chain_depth() const noexcept {
    std::uint32_t m = 0;
    for (const auto& [_, g] : groups) m = std::max(m, g.chain_depth);
    return m;
}

std::vector<std::byte> Manifest::to_canonical_json() const {
    json j;
    j["format_version"] = format_version;
    j["hash_algo"] = hash_algo;

    json fj;
    fj["name"] = file.name;
    fj["header_block"] = file.header_block.to_string();
    fj["total_bytes"] = file.total_bytes;
    fj["sha256"] = file.sha256;
    j["file"] = fj;

    json buf = json::array();
    for (const auto& e : buffer) {
        json be;
        be["tensor"] = e.tensor;
        be["off"] = e.off;
        be["nbytes"] = e.nbytes;
        be["group"] = e.group;
        buf.push_back(std::move(be));
    }
    j["buffer"] = buf;

    // `groups` is a std::map, so iteration is already key-sorted, matching
    // the ordered JSON output nlohmann::json's default map-backed type
    // produces on dump().
    json gj = json::object();
    for (const auto& [name, g] : groups) {
        json ge;
        ge["mode"] = g.mode == GroupMode::Full ? "full" : "delta";
        if (g.block) ge["block"] = g.block->to_string();
        if (g.base) {
            json bj;
            bj["commit"] = g.base->commit.to_string();
            bj["group"] = g.base->group;
            ge["base"] = bj;
        }
        if (g.diff_block) ge["diff_block"] = g.diff_block->to_string();
        ge["chain_depth"] = g.chain_depth;
        gj[name] = ge;
    }
    j["groups"] = gj;

    std::string s = j.dump();
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

Oid Manifest::oid() const {
    auto bytes = to_canonical_json();
    return core::compute_oid(core::ObjectKind::Manifest, bytes);
}

Result<Manifest> Manifest::parse(std::span<const std::byte> bytes) {
    std::string_view sv(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    json j;
    try {
        j = json::parse(sv);
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("manifest JSON parse failed: ") + e.what());
    }

    Manifest m;
    try {
        m.format_version = j.at("format_version").get<std::uint32_t>();
        m.hash_algo = j.at("hash_algo").get<std::string>();

        const auto& fj = j.at("file");
        m.file.name = fj.at("name").get<std::string>();
        auto hb = Oid::parse(fj.at("header_block").get<std::string>());
        if (!hb) return std::unexpected(hb.error());
        m.file.header_block = *hb;
        m.file.total_bytes = fj.at("total_bytes").get<std::uint64_t>();
        m.file.sha256 = fj.value("sha256", std::string{});

        for (const auto& be : j.at("buffer")) {
            BufferEntry e;
            e.tensor = be.at("tensor").get<std::string>();
            e.off = be.at("off").get<std::uint64_t>();
            e.nbytes = be.at("nbytes").get<std::uint64_t>();
            e.group = be.at("group").get<std::string>();
            m.buffer.push_back(std::move(e));
        }

        for (auto it = j.at("groups").begin(); it != j.at("groups").end(); ++it) {
            GroupEntry g;
            std::string mode = it.value().at("mode").get<std::string>();
            g.mode = (mode == "full") ? GroupMode::Full : GroupMode::Delta;
            if (it.value().contains("block")) {
                auto b = Oid::parse(it.value().at("block").get<std::string>());
                if (!b) return std::unexpected(b.error());
                g.block = *b;
            }
            if (it.value().contains("base")) {
                DeltaBase db;
                auto c = Oid::parse(it.value().at("base").at("commit").get<std::string>());
                if (!c) return std::unexpected(c.error());
                db.commit = *c;
                db.group = it.value().at("base").at("group").get<std::string>();
                g.base = db;
            }
            if (it.value().contains("diff_block")) {
                auto d = Oid::parse(it.value().at("diff_block").get<std::string>());
                if (!d) return std::unexpected(d.error());
                g.diff_block = *d;
            }
            g.chain_depth = it.value().value("chain_depth", 0u);
            m.groups.emplace(it.key(), std::move(g));
        }
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("manifest field missing/malformed: ") + e.what());
    }

    if (auto st = m.validate(); !st) return std::unexpected(st.error());

    auto reserialized = m.to_canonical_json();
    if (reserialized.size() != bytes.size() ||
        std::memcmp(reserialized.data(), bytes.data(), bytes.size()) != 0) {
        return SFS_ERR(CanonicalizationMismatch, "manifest does not round-trip to identical bytes");
    }

    return m;
}

}  // namespace sfs::format
