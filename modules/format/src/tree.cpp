#include <synapsefs/format/tree.hpp>

#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

namespace sfs::format {

using json = nlohmann::json;

bool is_valid_tree_entry_name(std::string_view name) noexcept {
    if (name.empty()) return false;
    if (name == "." || name == "..") return false;
    for (char c : name) {
        const auto u = static_cast<unsigned char>(c);
        // Path separators would let a tree name a location outside the
        // checkout directory; control bytes (including NUL) would let one
        // name print as another in every tool that shows it to a human.
        if (c == '/' || c == '\\') return false;
        if (u < 0x20 || u == 0x7f) return false;
    }
    return true;
}

Status Tree::validate() const {
    if (format_version != kTreeFormatVersion)
        return SFS_ERR(UnsupportedFormatVersion, "tree format_version must be " +
                                                     std::to_string(kTreeFormatVersion),
                       std::to_string(format_version));

    if (entries.empty())
        return SFS_ERR(MalformedObject, "tree has no entries");

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (!is_valid_tree_entry_name(e.name))
            return SFS_ERR(MalformedObject, "invalid tree entry name", e.name);
        if (e.manifest.is_null())
            return SFS_ERR(MalformedObject, "tree entry addresses the null oid", e.name);
        // Strictly ascending covers sortedness and uniqueness in one pass.
        if (i > 0 && !(entries[i - 1].name < e.name))
            return SFS_ERR(MalformedObject,
                           entries[i - 1].name == e.name
                               ? "duplicate tree entry name"
                               : "tree entries are not sorted by name",
                           e.name);
    }
    return {};
}

Result<Tree> Tree::make(std::vector<TreeEntry> entries) {
    std::sort(entries.begin(), entries.end(),
              [](const TreeEntry& a, const TreeEntry& b) { return a.name < b.name; });
    Tree t;
    t.entries = std::move(entries);
    if (auto st = t.validate(); !st) return std::unexpected(st.error());
    return t;
}

const TreeEntry* Tree::find(std::string_view name) const noexcept {
    // validate() guarantees sorted-by-name, so this is a binary search.
    auto it = std::lower_bound(entries.begin(), entries.end(), name,
                               [](const TreeEntry& e, std::string_view n) { return e.name < n; });
    if (it == entries.end() || it->name != name) return nullptr;
    return &*it;
}

std::vector<std::byte> Tree::to_canonical_json() const {
    // Emitted in stored order on purpose — see the sortedness note in
    // tree.hpp. An unsorted Tree serialises unsorted, which is exactly what
    // makes parse()'s round-trip check reject it instead of quietly
    // producing a different address than the writer expected.
    json ej = json::array();
    for (const auto& e : entries) {
        json entry;
        entry["name"] = e.name;
        entry["manifest"] = e.manifest.to_string();
        ej.push_back(std::move(entry));
    }

    json j;
    j["format_version"] = format_version;
    j["entries"] = std::move(ej);

    std::string s = j.dump();
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

Oid Tree::oid() const {
    auto bytes = to_canonical_json();
    return core::compute_oid(core::ObjectKind::Tree, bytes);
}

Result<Tree> Tree::parse(std::span<const std::byte> bytes) {
    std::string_view sv(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    json j;
    try {
        j = json::parse(sv);
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("tree JSON parse failed: ") + e.what());
    }

    Tree t;
    try {
        t.format_version = j.at("format_version").get<std::uint32_t>();

        const auto& ej = j.at("entries");
        if (!ej.is_array())
            return SFS_ERR(MalformedObject, "tree entries is not an array");

        t.entries.reserve(ej.size());
        for (const auto& e : ej) {
            TreeEntry te;
            te.name = e.at("name").get<std::string>();
            auto m = Oid::parse(e.at("manifest").get<std::string>());
            if (!m) return std::unexpected(m.error());
            te.manifest = *m;
            t.entries.push_back(std::move(te));
        }
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("tree field missing/malformed: ") + e.what());
    }

    if (auto st = t.validate(); !st) return std::unexpected(st.error());

    auto reserialized = t.to_canonical_json();
    if (reserialized.size() != bytes.size() ||
        std::memcmp(reserialized.data(), bytes.data(), bytes.size()) != 0) {
        return SFS_ERR(CanonicalizationMismatch, "tree does not round-trip to identical bytes");
    }

    return t;
}

}  // namespace sfs::format
